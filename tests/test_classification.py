"""Tests for grouped classifier data, artifacts, metrics, and ParSNIP adapters."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from astropy.table import Table
import numpy as np
import pandas as pd

from warptemplate import classification as workflow


RAW_CLASSES = (
    "SLSN-I",
    "SLSN-II",
    "SN IIP",
    "SN IIb",
    "SN IIn",
    "SN Ib",
    "SN Ic",
    "SN Ic-BL",
)


def make_synthetic_sample() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a compact sample with sufficient provenance groups for three folds."""
    truth_rows = []
    observation_rows = []
    object_number = 0
    for raw_label in RAW_CLASSES:
        for group_number in range(3):
            object_id = f"object-{object_number:03d}"
            object_number += 1
            truth_rows.append(
                {
                    "object_id": object_id,
                    "fitclass": raw_label,
                    "template_key": f"{raw_label}|basis-{group_number}|0",
                    "basis_sn": f"basis-{group_number}",
                    "z": 0.05 + 0.01 * group_number,
                    "t0": 60000.0,
                    "ra": 10.0,
                    "dec": -5.0,
                }
            )
            for epoch, time in enumerate((60000.0, 60000.5, 60001.0)):
                observation_rows.append(
                    {
                        "object_id": object_id,
                        "mjd": time,
                        "band": ("lsstg", "lsstr", "ztfg")[epoch],
                        "flux": 100.0 + epoch,
                        "fluxerr": 5.0,
                        "zp": 30.0 if epoch < 2 else 26.0,
                    }
                )
    # This extra object verifies the minimum grouped-epoch exclusion path.
    truth_rows.append(
        {
            "object_id": "excluded-object",
            "fitclass": "SN Ic",
            "template_key": "SN Ic|excluded|0",
            "basis_sn": "excluded",
            "z": 0.1,
            "t0": 60000.0,
            "ra": 10.0,
            "dec": -5.0,
        }
    )
    for time in (60000.0, 60000.1):
        observation_rows.append(
            {
                "object_id": "excluded-object",
                "mjd": time,
                "band": "lsstg",
                "flux": 10.0,
                "fluxerr": 2.0,
                "zp": 30.0,
            }
        )
    return pd.DataFrame(truth_rows), pd.DataFrame(observation_rows)


def make_representations(objects_per_class: int = 8) -> Table:
    """Create separable synthetic ParSNIP representations for LightGBM tests."""
    rng = np.random.default_rng(20260721)
    keys = [
        "color",
        "color_error",
        "s1",
        "s1_error",
        "s2",
        "s2_error",
        "s3",
        "s3_error",
        "luminosity",
        "luminosity_error",
        "reference_time_error",
    ]
    rows = []
    for class_index, label in enumerate(workflow.FINAL_CLASSES):
        for object_index in range(objects_per_class):
            row = {
                "object_id": f"{class_index}-{object_index}",
                "type": label,
            }
            for feature_index, key in enumerate(keys):
                row[key] = class_index + 0.01 * feature_index + rng.normal(0, 0.02)
            rows.append(row)
    return Table.from_pandas(pd.DataFrame(rows))


class TaxonomyAndFluxTests(unittest.TestCase):
    """Verify scientifically delicate taxonomy and flux transformations."""

    def test_only_slsn_subtypes_merge(self) -> None:
        """SLSN subtypes should merge while all six other labels remain distinct."""
        merged = workflow.merge_fitclasses(RAW_CLASSES).tolist()
        self.assertEqual(merged[:2], ["SLSN", "SLSN"])
        self.assertEqual(merged[2:], list(RAW_CLASSES[2:]))

    def test_zeropoint_conversion_preserves_magnitude_and_snr(self) -> None:
        """Flux scaling should preserve AB magnitude, S/N, and error scaling."""
        source = pd.DataFrame({"flux": [100.0], "fluxerr": [4.0], "zp": [30.0]})
        converted = workflow.rescale_flux_to_zeropoint(source, 25.0)
        magnitude_before = source.loc[0, "zp"] - 2.5 * np.log10(source.loc[0, "flux"])
        magnitude_after = converted.loc[0, "zp"] - 2.5 * np.log10(converted.loc[0, "flux"])
        self.assertAlmostEqual(magnitude_before, magnitude_after)
        self.assertAlmostEqual(
            source.loc[0, "flux"] / source.loc[0, "fluxerr"],
            converted.loc[0, "flux"] / converted.loc[0, "fluxerr"],
        )
        self.assertAlmostEqual(converted.loc[0, "fluxerr"], 0.04)


class GroupedSplitTests(unittest.TestCase):
    """Verify epoch filtering and leakage-free deterministic splits."""

    def setUp(self) -> None:
        """Construct one fresh synthetic sample per test."""
        self.truth, self.observations = make_synthetic_sample()

    def test_nightly_grouping_selects_lowest_error_duplicate(self) -> None:
        """Repeated bands in one grouped epoch should keep the lowest-error row."""
        rows = pd.DataFrame(
            {
                "object_id": ["a", "a", "a", "a"],
                "mjd": [1.0, 1.1, 1.2, 1.5],
                "band": ["lsstg", "lsstr", "lsstg", "lsstg"],
                "flux": [10.0, 20.0, 30.0, 40.0],
                "fluxerr": [3.0, 2.0, 1.0, 4.0],
            }
        )
        grouped = workflow.group_observing_epochs(rows)
        first_epoch_g = grouped[(grouped["grouped_mjd"] == 1.0) & (grouped["band"] == "lsstg")]
        self.assertEqual(len(first_epoch_g), 1)
        self.assertEqual(first_epoch_g.iloc[0]["flux"], 30.0)
        self.assertTrue(np.all(np.diff(grouped["grouped_mjd"]) >= 0))

    def test_both_group_strategies_are_complete_and_leakage_free(self) -> None:
        """Each retained object and active group should belong to one partition."""
        for strategy in ("template_key", "basis_sn"):
            manifest, excluded = workflow.create_grouped_split_manifest(
                self.truth,
                self.observations,
                strategy=strategy,
                n_splits=3,
            )
            workflow.validate_split_manifest(manifest)
            self.assertEqual(len(manifest), len(self.truth) - 1)
            self.assertEqual(excluded["object_id"].tolist(), ["excluded-object"])
            self.assertFalse(manifest["object_id"].duplicated().any())
            self.assertEqual(manifest.groupby("group_id")["split"].nunique().max(), 1)

    def test_lcdata_conversion_is_finite_ordered_and_keeps_bands(self) -> None:
        """Converted light curves should satisfy ParSNIP's core input invariants."""
        retained_ids = self.truth.loc[self.truth["object_id"] != "excluded-object", "object_id"][:3]
        dataset = workflow.to_lcdata(self.truth, self.observations, retained_ids.tolist())
        self.assertEqual(len(dataset), 3)
        for curve in dataset.light_curves:
            self.assertTrue(np.isfinite(curve["flux"]).all())
            self.assertTrue(np.isfinite(curve["fluxerr"]).all())
            self.assertTrue((curve["fluxerr"] > 0).all())
            self.assertTrue((np.diff(curve["time"]) >= 0).all())
            self.assertTrue(set(curve["band"]).issubset(workflow.EXPECTED_BANDS))

    def test_schema6_ensemble_loading_and_streaming_epoch_counts(self) -> None:
        """Nested realization trees should load selectively with bounded-memory counts."""
        with tempfile.TemporaryDirectory() as directory:
            sample_dir = Path(directory) / "schema6-sample"
            sample_dir.mkdir()
            (sample_dir / "ensemble_manifest.json").write_text(
                json.dumps({"configuration": {"simulation_schema": 6}})
            )
            object_ids = self.truth["object_id"].tolist()
            for realization_index, selected_ids in enumerate(
                (object_ids[::2], object_ids[1::2])
            ):
                child = sample_dir / f"schema6-sample__r{realization_index:03d}"
                truth_path = child / "truth" / "fitclass=all" / "batch.parquet"
                observation_path = child / "observations" / "fitclass=all" / "batch.parquet"
                truth_path.parent.mkdir(parents=True)
                observation_path.parent.mkdir(parents=True)
                self.truth[self.truth["object_id"].isin(selected_ids)].to_parquet(
                    truth_path, index=False
                )
                self.observations[
                    self.observations["object_id"].isin(selected_ids)
                ].to_parquet(observation_path, index=False)

            loaded_truth = workflow.load_sample_truth(sample_dir)
            selected = workflow.load_sample_observations(
                sample_dir,
                columns=["object_id", "mjd", "band"],
                object_ids=[object_ids[0]],
            )
            streamed_counts = workflow.grouped_epoch_counts_from_sample(sample_dir)
            in_memory_counts = workflow.grouped_epoch_counts(self.observations)
            dataset = workflow.to_lcdata(
                loaded_truth, sample_dir, object_ids=[object_ids[0]]
            )

            self.assertEqual(len(loaded_truth), len(self.truth))
            self.assertEqual(selected["object_id"].unique().tolist(), [object_ids[0]])
            self.assertEqual(len(dataset), 1)
            self.assertEqual(str(dataset.meta["object_id"][0]), object_ids[0])
            pd.testing.assert_series_equal(
                streamed_counts.sort_index(), in_memory_counts.sort_index()
            )
            self.assertEqual(
                workflow.load_sample_manifest(sample_dir)["configuration"]["simulation_schema"],
                6,
            )


class ExperimentAndMetricTests(unittest.TestCase):
    """Verify standard predictions, metrics, comparison, and artifact safeguards."""

    def setUp(self) -> None:
        """Build a synthetic split and perfectly classified probability table."""
        truth, observations = make_synthetic_sample()
        self.manifest, _ = workflow.create_grouped_split_manifest(
            truth, observations, strategy="template_key", n_splits=3
        )
        test = self.manifest[self.manifest["split"] == "test"].reset_index(drop=True)
        probabilities = np.full((len(test), len(workflow.FINAL_CLASSES)), 0.01)
        for row, label in enumerate(test["final_label"]):
            probabilities[row, workflow.FINAL_CLASSES.index(label)] = 0.94
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        classifications = Table({"object_id": test["object_id"].to_numpy()})
        for index, label in enumerate(workflow.FINAL_CLASSES):
            classifications[label] = probabilities[:, index]
        self.config = workflow.ExperimentConfig(
            training_sample="sample-a", evaluation_sample="sample-a"
        )
        self.predictions = workflow.standardize_predictions(
            classifications, self.manifest, self.config, partition="test"
        )

    def test_probabilities_and_metrics_are_well_formed(self) -> None:
        """Standard probabilities should be ordered, normalized, finite, and scoreable."""
        probability_columns = [f"prob_{label}" for label in workflow.FINAL_CLASSES]
        probabilities = self.predictions[probability_columns].to_numpy()
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        metrics = workflow.compute_classification_metrics(self.predictions)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(list(metrics["per_class"]), list(workflow.FINAL_CLASSES))

    def test_class_weights_have_equal_aggregate_contribution(self) -> None:
        """Inverse-frequency weights should sum equally within every class."""
        labels = np.asarray(["a"] * 2 + ["b"] * 5 + ["c"] * 9)
        weights = workflow.inverse_frequency_weights(labels)
        sums = [weights[labels == label].sum() for label in np.unique(labels)]
        np.testing.assert_allclose(sums, sums[0])

    def test_frozen_artifact_requires_explicit_overwrite(self) -> None:
        """A second test artifact write should fail unless overwrite is explicit."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.parquet"
            workflow.write_table_once(self.predictions, path)
            with self.assertRaises(FileExistsError):
                workflow.write_table_once(self.predictions, path)
            workflow.write_table_once(self.predictions, path, overwrite=True)

    def test_comparison_rejects_incompatible_evaluation_identity(self) -> None:
        """Direct comparisons must share evaluation sample and split strategy."""
        incompatible_sample = workflow.ExperimentConfig(
            training_sample="sample-b", evaluation_sample="sample-b"
        )
        with self.assertRaises(ValueError):
            workflow.assert_comparable_experiments(self.config, incompatible_sample)
        incompatible_split = workflow.ExperimentConfig(
            training_sample="sample-a",
            evaluation_sample="sample-a",
            split_strategy="basis_sn",
        )
        with self.assertRaises(ValueError):
            workflow.assert_comparable_experiments(self.config, incompatible_split)

    def test_cross_sample_group_leakage_is_detected(self) -> None:
        """Training on a frozen test provenance group should be rejected."""
        future = self.manifest.copy()
        frozen_test_group = self.manifest.loc[self.manifest["split"] == "test", "group_id"].iloc[0]
        future.loc[future.index[0], ["group_id", "split"]] = [frozen_test_group, "train"]
        with self.assertRaises(ValueError):
            workflow.assert_no_held_out_group_leakage(future, self.manifest)

    def test_named_run_metadata_rejects_changed_configuration(self) -> None:
        """A readable run name must not hide a changed scientific configuration."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.json"
            named = workflow.ExperimentConfig(
                training_sample="sample-a",
                evaluation_sample="sample-a",
                run_id="readable-baseline",
            )
            workflow.ensure_run_metadata(named, self.manifest, path)
            workflow.load_run_metadata(path, named)
            changed = workflow.ExperimentConfig(
                training_sample="sample-a",
                evaluation_sample="sample-a",
                redshift_mode="photometry_only",
                run_id="readable-baseline",
            )
            with self.assertRaises(ValueError):
                workflow.load_run_metadata(path, changed)


class ModelRoundTripTests(unittest.TestCase):
    """Exercise lightweight classifier and ParSNIP checkpoint round trips."""

    def test_lightgbm_classifier_saves_reloads_and_predicts(self) -> None:
        """ParSNIP's supervised classifier should survive a pickle round trip."""
        try:
            import parsnip
        except ImportError as error:
            self.skipTest(str(error))
        representations = make_representations()
        classifier = workflow.refit_parsnip_classifier(
            representations[:42], representations[42:], min_child_weight=1
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models" / "classifier.pkl"
            classifier.write(str(path))
            reloaded = parsnip.Classifier.load(str(path))
            classified = reloaded.classify(representations)
        probabilities = np.column_stack(
            [np.asarray(classified[label], dtype=float) for label in workflow.FINAL_CLASSES]
        )
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)

    def test_parsnip_checkpoint_saves_reloads_and_predicts(self) -> None:
        """A tiny ParSNIP network should train one batch, checkpoint, and infer."""
        try:
            import parsnip
        except ImportError as error:
            self.skipTest(str(error))
        truth, observations = make_synthetic_sample()
        object_ids = truth.loc[truth["object_id"] != "excluded-object", "object_id"][:2].tolist()
        dataset = workflow.to_lcdata(truth, observations, object_ids)
        settings = {
            "spectrum_bins": 20,
            "band_oversampling": 3,
            "time_window": 30,
            "time_pad": 5,
            "batch_size": 2,
            "encode_conv_architecture": [32],
            "encode_conv_dilations": [1],
            "encode_fc_architecture": [4],
            "encode_time_architecture": [4],
            "encode_latent_prepool_architecture": [4],
            "encode_latent_postpool_architecture": [4],
            "decode_architecture": [4],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models" / "tiny.pt"
            model = parsnip.ParsnipModel(
                str(path), list(workflow.EXPECTED_BANDS), device="cpu", threads=1, settings=settings
            )
            preprocessed = model.preprocess(dataset, verbose=False)
            batch = next(iter(model.get_data_loader(preprocessed, augment=False)))
            model.train()
            model.optimizer.zero_grad()
            loss = model.loss_function(model.forward(batch))
            loss.backward()
            model.optimizer.step()
            model.save()
            reloaded = parsnip.load_model(str(path), device="cpu", threads=1)
            predictions = reloaded.predict_dataset(dataset)
        self.assertEqual(len(predictions), len(object_ids))
        self.assertTrue(np.isfinite(np.asarray(predictions["s1"], dtype=float)).all())


class NotebookStyleTests(unittest.TestCase):
    """Enforce repository notebook-cell commentary rules."""

    def test_every_code_cell_starts_with_purpose_comment(self) -> None:
        """Code cells must be documented, compilable, clean, and kernel-first."""
        notebook_root = Path(__file__).parents[1] / "notebooks"
        notebook_paths = (
            notebook_root / "template_usage" / "skysurvey_warp_sample.ipynb",
            notebook_root / "classification" / "train_parsnip_classifier.ipynb",
            notebook_root / "classification" / "train_supernnova_classifier.ipynb",
        )
        for notebook_path in notebook_paths:
            notebook = json.loads(notebook_path.read_text())
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                source = "".join(cell["source"])
                first_line = source.splitlines()[0]
                self.assertTrue(
                    first_line.startswith("#"),
                    f"{notebook_path.name} code cell {index} lacks a purpose comment",
                )
                self.assertFalse(cell.get("outputs"))
                self.assertIsNone(cell.get("execution_count"))
                compile(source, f"{notebook_path.name}-cell-{index}", "exec")

        combined_sources = {}
        for notebook_path in notebook_paths:
            combined_sources[notebook_path.name] = "\n".join(
                "".join(cell["source"])
                for cell in json.loads(notebook_path.read_text())["cells"]
                if cell["cell_type"] == "code"
            )
        parsnip_source = combined_sources["train_parsnip_classifier.ipynb"]
        supernnova_source = combined_sources["train_supernnova_classifier.ipynb"]
        for source in combined_sources.values():
            self.assertIn("import warptemplate", source)
            self.assertIn(
                'DATA_ROOT = Path(warptemplate.__file__).resolve().parents[2] / "data"',
                source,
            )
            self.assertNotIn("sys.path", source)
            self.assertNotIn("sys.modules", source)
            self.assertNotIn("search_roots", source)
            self.assertNotIn(".exists()", source)
            self.assertNotIn("FORCE_", source)
            self.assertNotIn("ALLOW_TEST_OVERWRITE", source)
        self.assertIn("import parsnip", parsnip_source)
        self.assertIn("from warptemplate import classification", parsnip_source)
        self.assertIn("from warptemplate import supernnova_backend", supernnova_source)
        self.assertNotIn('\nimport supernnova\n', f'\n{supernnova_source}\n')


if __name__ == "__main__":
    unittest.main()
