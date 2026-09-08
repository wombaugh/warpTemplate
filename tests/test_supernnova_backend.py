"""Tests for the Warp-specific SuperNNova sequence and training backend."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd

from warptemplate import classification as workflow
from warptemplate import supernnova_backend as backend
from test_classification import make_synthetic_sample


def make_role_manifest(truth: pd.DataFrame) -> pd.DataFrame:
    """Assign synthetic provenance groups to deterministic train/validation/test roles."""
    retained = truth[truth["object_id"] != "excluded-object"].copy()
    retained["final_label"] = workflow.merge_fitclasses(retained["fitclass"]).to_numpy()
    retained["group_number"] = retained["template_key"].str.extract(r"basis-(\d+)").astype(int)
    retained["role"] = retained["group_number"].map(
        {0: "train", 1: "validation", 2: "test"}
    )
    retained["group_id"] = retained["template_key"]
    retained["survey_realization_id"] = "r000"
    return retained[
        ["object_id", "role", "final_label", "group_id", "survey_realization_id"]
    ]


def write_synthetic_ensemble(
    root: Path,
    truth: pd.DataFrame,
    observations: pd.DataFrame,
) -> Path:
    """Persist one compact schema-6-style realization below a sample directory."""
    sample_dir = root / "sample"
    child = sample_dir / "sample__r000"
    truth_path = child / "truth" / "fitclass=all" / "batch.parquet"
    observation_path = child / "observations" / "fitclass=all" / "batch.parquet"
    truth_path.parent.mkdir(parents=True)
    observation_path.parent.mkdir(parents=True)
    truth.assign(survey_realization_id="r000").to_parquet(truth_path, index=False)
    observations.assign(survey_realization_id="r000").to_parquet(
        observation_path, index=False
    )
    return sample_dir


class SequenceAdapterTests(unittest.TestCase):
    """Verify feature construction and scientifically delicate transformations."""

    def setUp(self) -> None:
        """Construct reusable synthetic truth and observations."""
        self.truth, self.observations = make_synthetic_sample()
        self.retained = self.truth[self.truth["object_id"] != "excluded-object"]

    def test_fixed_feature_order_and_sequence_invariants(self) -> None:
        """Sequences should retain bands and time while excluding object context."""
        object_id = self.retained.iloc[0]["object_id"]
        sequences = backend.build_supernnova_sequences(
            self.truth, self.observations, [object_id]
        )
        sequence, times = sequences[object_id]
        self.assertEqual(sequence.shape[1], len(backend.ALL_FEATURES))
        self.assertEqual(
            backend.feature_names("photometry_only"), backend.BASE_PHOTOMETRY_FEATURES
        )
        self.assertEqual(
            backend.feature_names("photometry_plus_truth_z"),
            (*backend.BASE_PHOTOMETRY_FEATURES, *backend.REDSHIFT_FEATURES),
        )
        self.assertEqual(
            backend.feature_names("photometry_only", engineered_features=True),
            backend.PHOTOMETRY_FEATURES,
        )
        self.assertFalse(
            any("HOSTGAL" in name for name in backend.feature_names("photometry_only"))
        )
        self.assertTrue(np.all(np.diff(times) >= 0))
        delta = sequence[:, backend.ALL_FEATURES.index("delta_time")]
        self.assertEqual(delta[0], 0.0)
        self.assertTrue((delta >= 0).all())
        presence = sequence[
            :,
            len(backend.FLUX_FEATURES)
            + len(backend.FLUXERR_FEATURES) : len(backend.FLUX_FEATURES)
            + len(backend.FLUXERR_FEATURES)
            + len(backend.PRESENCE_FEATURES),
        ]
        self.assertTrue(set(np.unique(presence)).issubset({0.0, 1.0}))
        self.assertNotIn("HOSTGAL_SPECZ", backend.ALL_FEATURES)
        band_fraction = sequence[:, backend.ALL_FEATURES.index("observed_band_fraction")]
        np.testing.assert_allclose(band_fraction, presence.mean(axis=1))
        self.assertTrue(np.isfinite(sequence).all())

    def test_inverse_variance_duplicate_aggregation(self) -> None:
        """Repeated same-band observations should combine with inverse variance."""
        observations = pd.DataFrame(
            {
                "object_id": ["object", "object"],
                "mjd": [1.0, 1.1],
                "band": ["lsstr", "lsstr"],
                "flux": [10.0, 20.0],
                "fluxerr": [2.0, 1.0],
            }
        )
        grouped = workflow.group_observing_epochs(
            observations, duplicate_strategy="inverse_variance"
        )
        self.assertEqual(len(grouped), 1)
        self.assertAlmostEqual(grouped.loc[0, "flux"], 18.0)
        self.assertAlmostEqual(grouped.loc[0, "fluxerr"], np.sqrt(0.8))

    def test_requested_object_order_is_preserved(self) -> None:
        """Explicit object selections should define deterministic sequence order."""
        object_ids = list(self.retained["object_id"].astype(str).iloc[:3])[::-1]
        sequences = backend.build_supernnova_sequences(
            self.truth, self.observations, object_ids
        )
        self.assertEqual(list(sequences), object_ids)

    def test_zeropoint_275_preserves_magnitude_and_snr(self) -> None:
        """The backend zeropoint convention should preserve magnitude and S/N."""
        source = pd.DataFrame({"flux": [50.0], "fluxerr": [2.5], "zp": [30.0]})
        converted = workflow.rescale_flux_to_zeropoint(
            source, backend.SUPERNOVA_ZEROPOINT
        )
        before = source.loc[0, "zp"] - 2.5 * np.log10(source.loc[0, "flux"])
        after = converted.loc[0, "zp"] - 2.5 * np.log10(converted.loc[0, "flux"])
        self.assertAlmostEqual(before, after)
        self.assertAlmostEqual(
            source.loc[0, "flux"] / source.loc[0, "fluxerr"],
            converted.loc[0, "flux"] / converted.loc[0, "fluxerr"],
        )

    def test_smoke_roles_are_group_disjoint(self) -> None:
        """Smoke training, validation, and test roles must not share groups."""
        split_manifest, _ = workflow.create_grouped_split_manifest(
            self.truth,
            self.observations,
            strategy="template_key",
            n_splits=3,
        )
        # Re-label the three available folds to the smoke fold numbers expected by
        # the production helper while preserving their disjoint provenance.
        split_manifest["fold"] = split_manifest["fold"].map({0: 2, 1: 3, 2: 4})
        split_manifest["split"] = "train"
        roles = backend.build_execution_role_manifest(
            split_manifest, "smoke", objects_per_class=1
        )
        group_sets = {
            role: set(roles.loc[roles["role"] == role, "group_id"])
            for role in backend.SPLIT_CODES
        }
        self.assertFalse(group_sets["train"] & group_sets["validation"])
        self.assertFalse(group_sets["train"] & group_sets["test"])
        self.assertFalse(group_sets["validation"] & group_sets["test"])


class DatabaseAndTrainingTests(unittest.TestCase):
    """Exercise HDF5 compatibility, normalization isolation, and tiny RNN runs."""

    def setUp(self) -> None:
        """Construct reusable synthetic sample tables."""
        self.truth, self.observations = make_synthetic_sample()
        self.role_manifest = make_role_manifest(self.truth)

    def _prepare_database(
        self,
        directory: Path,
        observations: pd.DataFrame | None = None,
        filename: str = "database.h5",
    ) -> Path:
        """Write one test database and return its path."""
        sample_dir = write_synthetic_ensemble(
            directory, self.truth, observations if observations is not None else self.observations
        )
        database_path = directory / filename
        backend.prepare_supernnova_database(
            sample_dir,
            self.truth.assign(survey_realization_id="r000"),
            self.role_manifest,
            database_path,
        )
        return database_path

    def test_normalization_uses_training_sequences_only(self) -> None:
        """Changing held-out fluxes must not change persisted normalization."""
        with (
            tempfile.TemporaryDirectory() as first_dir,
            tempfile.TemporaryDirectory() as second_dir,
        ):
            altered = self.observations.copy()
            train_id = self.role_manifest.loc[
                self.role_manifest["role"] == "train", "object_id"
            ].iloc[0]
            altered.loc[altered["object_id"] == train_id, "flux"] = -1e9
            first_path = self._prepare_database(Path(first_dir), altered)
            held_out_ids = set(
                self.role_manifest.loc[
                    self.role_manifest["role"] != "train", "object_id"
                ]
            )
            held_out_altered = altered.copy()
            held_out_altered.loc[
                held_out_altered["object_id"].isin(held_out_ids), "flux"
            ] *= 1e6
            second_path = self._prepare_database(Path(second_dir), held_out_altered)
            with h5py.File(first_path, "r") as first, h5py.File(
                second_path, "r"
            ) as second:
                summary = json.loads(first.attrs["summary_json"])
                second_summary = json.loads(second.attrs["summary_json"])
                self.assertEqual(summary["normalization"], second_summary["normalization"])
                self.assertNotEqual(
                    summary["database_identity"], second_summary["database_identity"]
                )
                self.assertEqual(
                    summary["normalization"]["FLUXCAL"]["min"],
                    backend.FLUX_NORMALIZATION_FLOOR,
                )

    def test_missing_bands_remain_neutral_after_normalization(self) -> None:
        """Structural missing-band zeros must not masquerade as measurements."""
        with tempfile.TemporaryDirectory() as directory:
            database_path = self._prepare_database(Path(directory))
            config = backend.SuperNNovaTrainingConfig(epochs=1, threads=1)
            store = backend.HDF5SequenceStore(database_path, config)
            records, _ = store.fetch([store.indices("train")[0]])
            sequence = records[0][0]
            presence = sequence[:, 2 * len(backend.FLUX_FEATURES) : 3 * len(backend.FLUX_FEATURES)]
            flux = sequence[:, : len(backend.FLUX_FEATURES)]
            flux_error = sequence[:, len(backend.FLUX_FEATURES) : 2 * len(backend.FLUX_FEATURES)]
            self.assertTrue(np.all(flux[presence == 0] == 0.0))
            self.assertTrue(np.all(flux_error[presence == 0] == 0.0))
            store.close()

            truth_config = backend.SuperNNovaTrainingConfig(
                redshift_mode="photometry_plus_truth_z", epochs=1, threads=1
            )
            truth_store = backend.HDF5SequenceStore(database_path, truth_config)
            truth_index = truth_store.indices("train")[0]
            truth_records, _ = truth_store.fetch([truth_index])
            np.testing.assert_allclose(
                truth_records[0][0][:, -1],
                truth_store.redshift[truth_index],
                rtol=1e-6,
            )
            truth_store.close()

    def test_database_run_refuses_existing_output_and_load_is_explicit(self) -> None:
        """Database creation and loading should be separate explicit operations."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_dir = write_synthetic_ensemble(root, self.truth, self.observations)
            database_path = root / "database.h5"
            truth = self.truth.assign(survey_realization_id="r000")
            backend.prepare_supernnova_database(
                sample_dir, truth, self.role_manifest, database_path
            )
            summary = backend.load_supernnova_database_summary(database_path)
            self.assertEqual(summary["objects"], len(self.role_manifest))
            with self.assertRaises(FileExistsError):
                backend.prepare_supernnova_database(
                    sample_dir, truth, self.role_manifest, database_path
                )
            with self.assertRaises(FileNotFoundError):
                backend.load_supernnova_database_summary(root / "missing.h5")

    def test_minimum_epoch_rule_applies_to_held_out_roles(self) -> None:
        """Validation and test objects should obey the same eligibility rule as train."""
        validation_id = self.role_manifest.loc[
            self.role_manifest["role"] == "validation", "object_id"
        ].iloc[0]
        selected = self.observations[self.observations["object_id"] == validation_id]
        retained_times = np.sort(selected["mjd"].unique())[:2]
        shortened = self.observations[
            (self.observations["object_id"] != validation_id)
            | self.observations["mjd"].isin(retained_times)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database_path = self._prepare_database(root, observations=shortened)
            with h5py.File(database_path, "r") as handle:
                self.assertNotIn(validation_id, set(handle["SNID"][:].astype(str)))
                summary = json.loads(handle.attrs["summary_json"])
                self.assertEqual(summary["skipped_short_objects"], 1)

    def test_database_round_trips_through_installed_supernnova_loader(self) -> None:
        """The generated HDF5 schema should load through kernel utilities."""
        try:
            from supernnova.utils import training_utils
        except ModuleNotFoundError as error:
            self.skipTest(f"optional SuperNNova dependency is unavailable: {error.name}")

        with tempfile.TemporaryDirectory() as directory:
            database_path = self._prepare_database(Path(directory))
            config = backend.SuperNNovaTrainingConfig(
                redshift_mode="photometry_only", epochs=1, threads=1
            )
            settings = backend.build_supernnova_settings(database_path, config)
            train, validation = training_utils.load_HDF5(settings, test=False)
            self.assertEqual(len(train), int((self.role_manifest["role"] == "train").sum()))
            self.assertEqual(
                len(validation),
                int((self.role_manifest["role"] == "validation").sum()),
            )
            self.assertEqual(
                train[0][0].shape[1], len(backend.BASE_PHOTOMETRY_FEATURES)
            )
            self.assertTrue(np.isfinite(train[0][0]).all())

    def test_both_tiny_models_train_reload_and_predict(self) -> None:
        """Both redshift variants should survive training and checkpoint reload."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database_path = self._prepare_database(root)
            for redshift_mode in backend.REDSHIFT_MODES:
                config = backend.SuperNNovaTrainingConfig(
                    redshift_mode=redshift_mode,
                    epochs=1,
                    hidden_dim=8,
                    num_layers=1,
                    dropout=0.0,
                    batch_size=8,
                    device="cpu",
                    threads=1,
                )
                output_dir = root / redshift_mode
                history = backend.train_supernnova(
                    database_path, output_dir, config, show_progress=False
                )
                self.assertEqual(len(history["epochs"]), 1)
                self.assertTrue((output_dir / "complete.json").exists())
                with self.assertRaises(FileExistsError):
                    backend.train_supernnova(
                        database_path, output_dir, config, show_progress=False
                    )
                self.assertEqual(
                    backend.load_supernnova_training_history(output_dir), history
                )
                model, loaded_config = backend.load_supernnova_checkpoint(
                    output_dir / "best.pt", device="cpu"
                )
                self.assertEqual(loaded_config.redshift_mode, redshift_mode)
                self.assertIsNotNone(model)
                result = backend.predict_supernnova(
                    output_dir / "best.pt",
                    database_path,
                    role="test",
                    device="cpu",
                )
                probabilities = np.column_stack(
                    [
                        np.asarray(result.classifications[label], dtype=float)
                        for label in workflow.FINAL_CLASSES
                    ]
                )
                self.assertTrue(np.isfinite(probabilities).all())
                np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
                self.assertEqual(probabilities.shape[1], len(workflow.FINAL_CLASSES))
                partial = backend.predict_supernnova(
                    output_dir / "best.pt",
                    database_path,
                    role="test",
                    cutoff_days=-1e6,
                    device="cpu",
                )
                self.assertEqual(partial.eligible_objects, 0)
                self.assertEqual(
                    partial.ineligible_objects,
                    int((self.role_manifest["role"] == "test").sum()),
                )
                first_epoch = backend.predict_supernnova(
                    output_dir / "best.pt",
                    database_path,
                    role="test",
                    cutoff_days=0.0,
                    cutoff_reference="first_observation",
                    device="cpu",
                )
                self.assertEqual(
                    first_epoch.eligible_objects,
                    int((self.role_manifest["role"] == "test").sum()),
                )
                self.assertFalse((output_dir / "best.pt.tmp").exists())

    def test_interrupted_training_matches_uninterrupted_training(self) -> None:
        """Restored RNG and scheduler states should make CPU resume exact."""
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database_path = self._prepare_database(root)
            config = backend.SuperNNovaTrainingConfig(
                epochs=3,
                hidden_dim=8,
                num_layers=1,
                dropout=0.1,
                batch_size=8,
                device="cpu",
                threads=1,
                calibrate_probabilities=False,
            )
            uninterrupted = root / "uninterrupted"
            resumed = root / "resumed"
            direct_history = backend.train_supernnova(
                database_path, uninterrupted, config, show_progress=False
            )
            backend.train_supernnova(
                database_path,
                resumed,
                config,
                show_progress=False,
                max_epochs_this_call=1,
            )
            self.assertFalse((resumed / "complete.json").exists())
            resumed_history = backend.train_supernnova(
                database_path,
                resumed,
                config,
                action="resume",
                show_progress=False,
            )
            self.assertEqual(direct_history["epochs"], resumed_history["epochs"])
            direct_payload = torch.load(
                uninterrupted / "best.pt", map_location="cpu", weights_only=False
            )
            resumed_payload = torch.load(
                resumed / "best.pt", map_location="cpu", weights_only=False
            )
            for name, tensor in direct_payload["model_state"].items():
                self.assertTrue(torch.equal(tensor, resumed_payload["model_state"][name]))

    def test_resume_requires_an_incomplete_checkpoint(self) -> None:
        """Resume should fail normally when no paused checkpoint is present."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database_path = self._prepare_database(root)
            config = backend.SuperNNovaTrainingConfig(
                epochs=1,
                hidden_dim=8,
                num_layers=1,
                batch_size=8,
                device="cpu",
                threads=1,
            )
            with self.assertRaises(FileNotFoundError):
                backend.train_supernnova(
                    database_path,
                    root / "missing-run",
                    config,
                    action="resume",
                    show_progress=False,
                )

    def test_attention_engineered_causal_variant_trains(self) -> None:
        """Opt-in engineered features, attention, and causal recurrence should compose."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database_path = self._prepare_database(root)
            config = backend.SuperNNovaTrainingConfig(
                redshift_mode="photometry_plus_truth_z",
                epochs=1,
                hidden_dim=8,
                num_layers=1,
                dropout=0.0,
                batch_size=8,
                bidirectional=False,
                rnn_output_option="attention",
                engineered_features=True,
                device="cpu",
                threads=1,
            )
            output_dir = root / "attention"
            backend.train_supernnova(
                database_path, output_dir, config, show_progress=False
            )
            model, _ = backend.load_supernnova_checkpoint(
                output_dir / "best.pt", device="cpu"
            )
            self.assertEqual(model.rnn_layer.input_size, len(backend.PHOTOMETRY_FEATURES))
            self.assertEqual(model.output_layer.in_features, config.hidden_dim + 1)


class PairedComparisonTests(unittest.TestCase):
    """Verify group-paired comparison of standardized backend predictions."""

    def test_second_minus_first_bootstrap_uses_common_objects(self) -> None:
        """A better second classifier should have a positive accuracy difference."""
        rows = []
        groups = []
        for class_index, label in enumerate(workflow.FINAL_CLASSES):
            for object_index in range(2):
                rows.append((f"{class_index}-{object_index}", label, class_index))
                groups.append(f"group-{object_index}")
        first = pd.DataFrame(
            {
                "object_id": [row[0] for row in rows],
                "true_class": [row[1] for row in rows],
                "predicted_class": [workflow.FINAL_CLASSES[0]] * len(rows),
            }
        )
        second = first.copy()
        second["predicted_class"] = second["true_class"]
        for class_index, label in enumerate(workflow.FINAL_CLASSES):
            first[f"prob_{label}"] = 1.0 / len(workflow.FINAL_CLASSES)
            second[f"prob_{label}"] = [
                0.94 if row[2] == class_index else 0.01 for row in rows
            ]
        split_manifest = pd.DataFrame(
            {"object_id": first["object_id"], "group_id": groups}
        )
        comparison = backend.paired_group_bootstrap_differences(
            first, second, split_manifest, repeats=50, seed=20260721
        )
        self.assertEqual(comparison["direction"], "second_minus_first")
        self.assertEqual(comparison["common_objects"], len(rows))
        self.assertGreater(
            comparison["intervals"]["balanced_accuracy"]["median"], 0.0
        )

    def test_temperature_scaling_softens_overconfident_errors(self) -> None:
        """Validation calibration should raise temperature for confident mistakes."""
        class_count = len(workflow.FINAL_CLASSES)
        labels = np.arange(class_count)
        probabilities = np.full((class_count, class_count), 0.01 / (class_count - 1))
        probabilities[np.arange(class_count), (labels + 1) % class_count] = 0.99
        temperature = backend.fit_temperature_scaling(labels, probabilities)
        self.assertGreater(temperature, 1.0)


if __name__ == "__main__":
    unittest.main()
