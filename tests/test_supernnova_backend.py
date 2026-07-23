"""Tests for the Warp-specific SuperNNova sequence and training backend."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd

from warpTemplate import classification as workflow
from warpTemplate import supernnova_backend as backend
from warpTemplate.tests.test_classification import make_synthetic_sample


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
        """Sequences should retain nine bands, flags, time order, and exact redshift."""
        object_id = self.retained.iloc[0]["object_id"]
        sequences = backend.build_supernnova_sequences(
            self.truth, self.observations, [object_id]
        )
        sequence, times = sequences[object_id]
        self.assertEqual(sequence.shape[1], 30)
        self.assertEqual(
            backend.feature_names("photometry_only"), backend.PHOTOMETRY_FEATURES
        )
        self.assertEqual(
            backend.feature_names("photometry_plus_truth_z"), backend.ALL_FEATURES
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
            + len(backend.FLUXERR_FEATURES) : len(backend.PHOTOMETRY_FEATURES)
            - 1,
        ]
        self.assertTrue(set(np.unique(presence)).issubset({0.0, 1.0}))
        self.assertTrue(
            np.all(
                sequence[:, backend.ALL_FEATURES.index("HOSTGAL_SPECZ")]
                == self.retained.iloc[0]["z"]
            )
        )
        self.assertTrue(
            np.all(
                sequence[:, backend.ALL_FEATURES.index("HOSTGAL_SPECZ_ERR")] == 0.0
            )
        )

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
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
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
                self.assertEqual(
                    first.attrs["summary_json"], second.attrs["summary_json"]
                )
                summary = json.loads(first.attrs["summary_json"])
                self.assertEqual(
                    summary["normalization"]["FLUXCAL"]["min"],
                    backend.FLUX_NORMALIZATION_FLOOR,
                )

    def test_database_round_trips_through_installed_supernnova_loader(self) -> None:
        """The generated HDF5 schema should load through kernel utilities."""
        from supernnova.utils import training_utils

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
            self.assertEqual(train[0][0].shape[1], len(backend.PHOTOMETRY_FEATURES))
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
                cached_history = backend.train_supernnova(
                    database_path, output_dir, config, show_progress=False
                )
                self.assertEqual(cached_history, history)
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


if __name__ == "__main__":
    unittest.main()
