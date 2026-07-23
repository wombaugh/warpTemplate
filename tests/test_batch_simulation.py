"""Tests for resource-bounded sample specifications and deterministic helpers."""

from __future__ import annotations

from copy import deepcopy
import json
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

from warpTemplate.batch_simulation import (
    WarpSampleSpec,
    WarpSimulationRunner,
    _WarpBatchTargets,
    _WarpTargetSampler,
    _stable_seed,
    allocate_group_balanced_counts,
)
from test_observer_population import FakeSurvey


class PartiallyObservedSurvey(FakeSurvey):
    """Synthetic survey that observes only the first target in each batch."""

    def radec_to_fieldid(self, radec):
        """Map the first coordinate to an observed field and all others away."""

        fieldids = pd.DataFrame({"fieldid": 2}, index=radec.index)
        if len(fieldids):
            fieldids.iloc[0, 0] = 1
        return fieldids


class UnobservedSurvey(FakeSurvey):
    """Synthetic survey whose only log field matches no target."""

    def radec_to_fieldid(self, radec, observed_fields=False):
        """Return no observed field match for any supplied coordinate."""

        return pd.DataFrame(
            {"fieldid": pd.Series(dtype=np.int64)},
            index=radec.index[:0],
        )


class ProvenanceSurvey(FakeSurvey):
    """Synthetic survey with an explicit, mutable log fingerprint."""

    def __init__(self, fingerprint):
        """Attach one stable fingerprint to otherwise identical survey data."""

        super().__init__()
        self.warp_provenance = {
            "name": "ztf",
            "log_fingerprint": fingerprint,
        }


class NamedSkyArea:
    """Small identity-bearing sky-area placeholder for precedence tests."""

    def __init__(self, name):
        """Store one stable printable label."""

        self.name = str(name)

    def __repr__(self):
        """Return a stable manifest representation."""

        return f"NamedSkyArea({self.name!r})"


class LoaderBackedSourceCache:
    """Provide test sources through the required cache interface."""

    def __init__(self, loader):
        """Store the coefficient loader and count cache reads."""

        self.loader = loader
        self.load_source = Mock(side_effect=loader.build_uncolored_source)


class WarpSampleSpecTest(unittest.TestCase):
    """Validate independent class/redshift modes and stable seeds."""

    def test_group_balancing_gives_merged_slsn_one_seventh(self):
        """A merged SLSN parent should get one quota split across its subtypes."""

        groups = {
            "SLSN": ("SLSN-I", "SLSN-II"),
            "SN IIP": ("SN IIP",),
            "SN IIb": ("SN IIb",),
            "SN IIn": ("SN IIn",),
            "SN Ib": ("SN Ib",),
            "SN Ic": ("SN Ic",),
            "SN Ic-BL": ("SN Ic-BL",),
        }
        allocations = allocate_group_balanced_counts(
            100_000, groups, shard_sizes=[2_778] * 28 + [2_777] * 8
        )
        aggregate = {
            raw_class: sum(allocation[raw_class] for allocation in allocations)
            for members in groups.values()
            for raw_class in members
        }
        final_counts = {
            parent: sum(aggregate[member] for member in members)
            for parent, members in groups.items()
        }

        self.assertEqual(sum(map(sum, (item.values() for item in allocations))), 100_000)
        self.assertLessEqual(max(final_counts.values()) - min(final_counts.values()), 1)
        self.assertEqual(final_counts["SLSN"], 14_286)
        self.assertEqual(aggregate["SLSN-I"], 7_143)
        self.assertEqual(aggregate["SLSN-II"], 7_143)

    def test_balanced_classes_and_binned_redshifts_are_independent(self):
        """Balanced class allocation must combine with explicitly binned redshifts."""

        spec = WarpSampleSpec(
            run_name="balanced_binned",
            active_fitclasses=["SN IIb", "SN Ib"],
            size=100,
            class_sampling="balanced",
            redshift_sampling="binned",
            redshift_bins=[0.0, 0.01, 0.02],
            zmax=0.02,
        )
        self.assertEqual(spec.class_sampling, "balanced")
        self.assertEqual(spec.redshift_sampling, "binned")
        self.assertEqual(spec.batch_size, 10_000)
        self.assertEqual(spec.max_sources_per_batch, 512)

    def test_invalid_weighted_sample_is_rejected(self):
        """Weighted class sampling must provide explicit weights."""

        with self.assertRaises(ValueError):
            WarpSampleSpec(
                run_name="invalid",
                active_fitclasses=["SN IIb"],
                size=10,
                class_sampling="weighted",
            )

    def test_survey_options_are_manifest_serializable(self):
        """Paths and filter tuples in survey options must serialize deterministically."""

        spec = WarpSampleSpec(
            run_name="survey_options",
            active_fitclasses=["SN IIP"],
            size=1,
            survey_name="lsst",
            survey_options={
                "lsst_path": Path("data/example.db"),
                "lsst_filters": ("g", "r"),
            },
        )
        serialized = spec.to_dict()["survey_options"]
        self.assertEqual(serialized["lsst_path"], "data/example.db")
        self.assertEqual(serialized["lsst_filters"], ["g", "r"])

    def test_nonpositive_source_limit_is_rejected(self):
        """The explicit per-batch source limit must remain positive."""

        with self.assertRaises(ValueError):
            WarpSampleSpec(
                run_name="invalid_sources",
                active_fitclasses=["SN IIP"],
                size=1,
                max_sources_per_batch=0,
            )

    def test_explicit_counts_do_not_require_a_redundant_total_size(self):
        """Explicit class counts must also define a valid fixed-size time window."""

        spec = WarpSampleSpec(
            run_name="counts",
            active_fitclasses=["SN IIP"],
            class_sampling="counts",
            class_counts={"SN IIP": 3},
            color_mode=None,
            batch_size=2,
        )
        batches = list(
            _WarpTargetSampler("data/warpcoeff_v3").iter_batches(spec)
        )
        self.assertEqual(sum(len(batch) for batch in batches), 3)

    def test_stable_seed_is_repeatable_and_partition_specific(self):
        """Derived seeds must be repeatable without relying on Python hash state."""

        first = _stable_seed(12, "SN IIb", "events")
        self.assertEqual(first, _stable_seed(12, "SN IIb", "events"))
        self.assertNotEqual(first, _stable_seed(12, "SN IIb", "templates"))
        self.assertIsInstance(np.random.default_rng(first), np.random.Generator)

    def test_noise_is_independent_of_batch_partitioning(self):
        """Per-object noise must not depend on neighboring targets or batch IDs."""

        original = pd.DataFrame(
            {
                "object_id": ["run:a", "run:a", "run:b"],
                "flux": [1.0, 2.0, 3.0],
                "fluxerr": [0.1, 0.2, 0.3],
            }
        )
        together = original.copy()
        WarpSimulationRunner._scatter_observations(together, 42)
        separate = []
        for object_id in original["object_id"].unique():
            part = original.loc[original["object_id"].eq(object_id)].copy()
            WarpSimulationRunner._scatter_observations(part, 42)
            separate.append(part)
        partitioned = pd.concat(separate).sort_index()
        np.testing.assert_allclose(together["flux"], partitioned["flux"])

    def test_runner_keeps_fluxerr_and_applies_noise_only_when_requested(self):
        """SkySurvey stays noiseless; the runner conditionally scatters flux."""

        def iter_test_batch(spec):
            """Yield one compact row without asking the source cache for a model."""

            yield pd.DataFrame(
                {
                    "fitclass": ["SN IIP"],
                    "template_key": ["entry"],
                    "object_id": [f"{spec.run_name}:SN IIP:000000000000"],
                    "ra": [0.0],
                    "dec": [0.0],
                }
            )

        def simulated_dataset(*_args, **_kwargs):
            """Return one deterministic noiseless SkySurvey-style observation."""

            return SimpleNamespace(
                data=pd.DataFrame(
                    {"flux": [1.0], "fluxerr": [0.25]},
                    index=pd.Index([0], name="index"),
                )
            )

        with tempfile.TemporaryDirectory() as directory:
            runner = WarpSimulationRunner(
                "data/warpcoeff_v3",
                source_cache_dir=Path(directory) / "source-cache",
            )
            runner.source_cache.build = Mock()
            runner.source_cache.describe = Mock(
                return_value={"fingerprint": "stable-cache"}
            )
            runner.sampler.iter_batches = iter_test_batch
            results = {}
            with patch(
                "skysurvey.DataSet.from_targets_and_survey",
                side_effect=simulated_dataset,
            ) as skysurvey_call:
                for incl_error in (False, True):
                    spec = WarpSampleSpec(
                        run_name=f"noise_{incl_error}",
                        active_fitclasses=["SN IIP"],
                        size=1,
                        class_sampling="balanced",
                        color_mode=None,
                        incl_error=incl_error,
                        seed=84,
                    )
                    manifest = runner.run(
                        spec,
                        directory,
                        survey=FakeSurvey(),
                    )
                    batch = manifest["batches"]["00000000"]
                    results[incl_error] = pd.read_parquet(
                        Path(directory)
                        / spec.run_name
                        / batch["observations"]
                    )

            self.assertEqual(skysurvey_call.call_count, 2)
            self.assertTrue(
                all(
                    call.kwargs["incl_error"] is False
                    for call in skysurvey_call.call_args_list
                )
            )

        self.assertEqual(float(results[False]["flux"].iloc[0]), 1.0)
        expected_noise = np.random.default_rng(
            _stable_seed(
                84,
                "noise_True:SN IIP:000000000000",
                "noise",
            )
        ).normal(loc=0.0, scale=0.25)
        self.assertAlmostEqual(
            float(results[True]["flux"].iloc[0]),
            1.0 + float(expected_noise),
        )
        np.testing.assert_allclose(results[False]["fluxerr"], 0.25)
        np.testing.assert_allclose(results[True]["fluxerr"], 0.25)

    def test_streamed_target_draw_is_independent_of_batch_size(self):
        """Operational chunking must not change any target-level random draw."""

        coefficient_dir = Path("data/warpcoeff_v3")
        if not coefficient_dir.exists():
            self.skipTest("repository coefficient library is unavailable")
        common = dict(
            run_name="batch_invariant",
            active_fitclasses=["SN IIP"],
            size=12,
            zmax=0.02,
            tstart=60000.0,
            tstop=60010.0,
            class_sampling="balanced",
            redshift_sampling="binned",
            redshift_bins=[0.0, 0.01, 0.02],
            color_mode="draw",
            seed=2026,
        )
        small = _WarpTargetSampler(coefficient_dir).draw(
            WarpSampleSpec(**common, batch_size=2)
        )
        large = _WarpTargetSampler(coefficient_dir).draw(
            WarpSampleSpec(**common, batch_size=7)
        )
        sort_key = ["object_id"]
        pd.testing.assert_frame_equal(
            small.sort_values(sort_key).reset_index(drop=True),
            large.sort_values(sort_key).reset_index(drop=True),
        )

    def test_streamed_batches_contain_no_model_objects(self):
        """Drawing a batch must retain only serializable target information."""

        coefficient_dir = Path("data/warpcoeff_v3")
        if not coefficient_dir.exists():
            self.skipTest("repository coefficient library is unavailable")
        spec = WarpSampleSpec(
            run_name="descriptor_only",
            active_fitclasses=["SN IIP"],
            size=3,
            class_sampling="balanced",
            color_mode=None,
            batch_size=2,
        )
        batch = next(_WarpTargetSampler(coefficient_dir).iter_batches(spec))
        self.assertNotIn("model", batch.columns)
        self.assertFalse(
            batch.apply(
                lambda column: column.map(
                    lambda value: value.__class__.__name__ == "Model"
                ).any()
            ).any()
        )

    def test_truth_columns_separate_raw_and_effective_entry_probabilities(self):
        """Schema-6 truth must expose the actual draw probability and distance."""

        coefficient_dir = Path("data/warpcoeff_v3")
        if not coefficient_dir.exists():
            self.skipTest("repository coefficient library is unavailable")
        spec = WarpSampleSpec(
            run_name="truth_schema",
            active_fitclasses=["SN IIP"],
            size=30,
            class_sampling="balanced",
            redshift_sampling="uniform",
            color_mode=None,
            seed=27,
        )
        sampler = _WarpTargetSampler(coefficient_dir)
        drawn = sampler.draw(spec)
        entries = sampler.loader.get_entry_probabilities(
            "SN IIP", min_fit_quality=spec.min_fit_quality
        )
        probability_sum = sum(probability for _, probability in entries)
        raw_weights = {
            descriptor.template_key: descriptor.template_prob
            for descriptor, _ in entries
        }
        effective_probabilities = {
            descriptor.template_key: probability / probability_sum
            for descriptor, probability in entries
        }

        self.assertNotIn("magobs", drawn)
        self.assertIn("distance_modulus", drawn)
        self.assertIn("entry_sampling_probability", drawn)
        np.testing.assert_allclose(
            drawn["distance_modulus"],
            sampler.cosmology.distmod(drawn["z"].to_numpy()).value,
        )
        np.testing.assert_allclose(
            drawn["template_prob"],
            drawn["template_key"].map(raw_weights),
        )
        np.testing.assert_allclose(
            drawn["entry_sampling_probability"],
            drawn["template_key"].map(effective_probabilities),
        )

    def test_grouped_sampler_supports_every_color_mode(self):
        """Every color mode must produce its intended lightweight event parameter."""

        coefficient_dir = Path("data/warpcoeff_v3")
        if not coefficient_dir.exists():
            self.skipTest("repository coefficient library is unavailable")
        for color_mode in (None, "harmonize", "draw", "target"):
            with self.subTest(color_mode=color_mode):
                spec = WarpSampleSpec(
                    run_name=f"color_{color_mode}",
                    active_fitclasses=["SN IIP"],
                    size=4,
                    class_sampling="balanced",
                    color_mode=color_mode,
                    target_peak_color=0.2 if color_mode == "target" else None,
                    batch_size=2,
                    seed=18,
                )
                drawn = _WarpTargetSampler(coefficient_dir).draw(spec)
                if color_mode is None:
                    self.assertTrue(drawn["samplecorr_ebv"].isna().all())
                else:
                    self.assertTrue(
                        np.isfinite(drawn["samplecorr_ebv"].astype(float)).all()
                    )
                if color_mode == "target":
                    np.testing.assert_allclose(drawn["target_peak_color"], 0.2)

    def test_runner_uses_skysurvey_and_resumes_completed_parquet_batches(self):
        """A real Warp model must pass through unmodified SkySurvey and resume safely."""

        coefficient_dir = Path("data/warpcoeff_v3")
        if not coefficient_dir.exists():
            self.skipTest("repository coefficient library is unavailable")
        spec = WarpSampleSpec(
            run_name="integration",
            active_fitclasses=["SN IIP"],
            size=2,
            zmax=0.01,
            tstart=60000.0,
            tstop=60001.0,
            class_sampling="balanced",
            redshift_sampling="uniform",
            color_mode=None,
            survey_realization_id="season-00",
            batch_size=1,
            incl_error=False,
            seed=9,
        )
        with tempfile.TemporaryDirectory() as directory:
            runner = WarpSimulationRunner(
                coefficient_dir,
                source_cache_dir=Path(directory) / "source-cache",
            )
            with patch("builtins.print") as print_mock:
                first = runner.run(
                    spec,
                    FakeSurvey(),
                    directory,
                    progress_every_batches=1,
                    retain_coefficient_cache=True,
                )
            self.assertIn("SN IIP", runner.loader._cache)
            second = runner.run(
                spec,
                FakeSurvey(),
                directory,
                resume=True,
                retain_coefficient_cache=True,
            )
            persisted_truth = pd.read_parquet(
                Path(directory)
                / spec.run_name
                / first["batches"]["00000000"]["truth"]
            )
            persisted_observations = pd.read_parquet(
                Path(directory)
                / spec.run_name
                / first["batches"]["00000000"]["observations"]
            )
            self.assertEqual(first["status"], "complete")
            self.assertEqual(first, second)
            self.assertEqual(first["truth_rows"], 2)
            self.assertEqual(first["observation_rows"], 2)
            self.assertEqual(first["schema_version"], 6)
            self.assertEqual(first["software"]["color_engine"], "dynamic-ccm89-v1")
            self.assertEqual(first["batching"]["scope"], "fitclass-local")
            self.assertIn("truth_schema", first)
            self.assertIn("timing_seconds", first["batches"]["00000000"])
            self.assertNotIn("runner", first)
            self.assertNotIn("source_pool", first["batches"]["00000000"])
            self.assertNotIn("magobs", persisted_truth)
            self.assertIn("distance_modulus", persisted_truth)
            self.assertIn("entry_sampling_probability", persisted_truth)
            self.assertEqual(
                persisted_truth["survey_realization_id"].unique().tolist(),
                ["season-00"],
            )
            self.assertEqual(
                persisted_observations[
                    "survey_realization_id"
                ].unique().tolist(),
                ["season-00"],
            )
            self.assertTrue(
                any(
                    "[Warp progress]" in str(call.args[0])
                    for call in print_mock.call_args_list
                )
            )
            runner.clear_coefficient_cache()
            self.assertFalse(runner.loader._cache)
            with self.assertRaisesRegex(ValueError, "must be positive"):
                runner.run(
                    spec,
                    FakeSurvey(),
                    directory,
                    progress_every_batches=0,
                )

    def test_runner_can_resolve_the_survey_from_the_spec(self):
        """The new runner form must ask SurveyFactory when no override is supplied."""

        spec = WarpSampleSpec(
            run_name="automatic_survey",
            active_fitclasses=["SN IIP"],
            size=1,
            zmax=0.01,
            tstart=60000.0,
            tstop=60001.0,
            class_sampling="balanced",
            redshift_sampling="uniform",
            color_mode=None,
            batch_size=1,
            incl_error=False,
            seed=13,
        )
        with tempfile.TemporaryDirectory() as directory:
            runner = WarpSimulationRunner(
                "data/warpcoeff_v3",
                source_cache_dir=Path(directory) / "source-cache",
            )
            with patch(
                "warpTemplate.survey_factory.SurveyFactory.from_spec",
                return_value=FakeSurvey(),
            ) as factory:
                manifest = runner.run(spec, directory)
        factory.assert_called_once_with(spec)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["survey"]["name"], "ztf")

    def test_explicit_survey_override_skips_automatic_loading(self):
        """An explicit survey must remain supported in the new keyword API."""

        spec = WarpSampleSpec(
            run_name="explicit_survey",
            active_fitclasses=["SN IIP"],
            size=1,
            zmax=0.01,
            tstart=60000.0,
            tstop=60001.0,
            class_sampling="balanced",
            redshift_sampling="uniform",
            color_mode=None,
            batch_size=1,
            incl_error=False,
            seed=14,
        )
        with tempfile.TemporaryDirectory() as directory:
            runner = WarpSimulationRunner(
                "data/warpcoeff_v3",
                source_cache_dir=Path(directory) / "source-cache",
            )
            with patch(
                "warpTemplate.survey_factory.SurveyFactory.from_spec",
                side_effect=AssertionError("automatic loader should not run"),
            ):
                manifest = runner.run(
                    spec,
                    directory,
                    survey=FakeSurvey(),
                )
        self.assertEqual(manifest["status"], "complete")

    def test_skyarea_precedence_distinguishes_none_full_and_explicit(self):
        """Only None adopts the factory survey's simulation footprint."""

        survey_area = NamedSkyArea("survey")
        explicit_area = NamedSkyArea("explicit")
        survey = FakeSurvey()
        survey.simulation_skyarea = survey_area
        cases = [
            ("none", None, survey_area),
            ("full", "full", "full"),
            ("explicit", explicit_area, explicit_area),
        ]
        with tempfile.TemporaryDirectory() as directory:
            runner = WarpSimulationRunner(
                "data/warpcoeff_v3",
                source_cache_dir=Path(directory) / "source-cache",
            )
            runner.source_cache.build = Mock()
            runner.source_cache.describe = Mock(
                return_value={"fingerprint": "stable-cache"}
            )
            for label, requested, expected in cases:
                with self.subTest(label=label):
                    captured = []

                    def iter_empty_batches(resolved_spec):
                        """Capture the post-survey sampling specification."""

                        captured.append(resolved_spec)
                        if False:
                            yield pd.DataFrame()

                    runner.sampler.iter_batches = iter_empty_batches
                    runner.run(
                        WarpSampleSpec(
                            run_name=f"skyarea_{label}",
                            active_fitclasses=["SN IIP"],
                            size=0,
                            skyarea=requested,
                            class_sampling="balanced",
                            color_mode=None,
                        ),
                        directory,
                        survey=survey,
                    )
                    if label == "full":
                        self.assertEqual(captured[0].skyarea, expected)
                    else:
                        self.assertIs(captured[0].skyarea, expected)

    def test_complete_fast_return_rebuilds_and_compares_survey_provenance(self):
        """A complete run must not hide changes to automatically loaded logs."""

        spec = WarpSampleSpec(
            run_name="changed_complete_survey",
            active_fitclasses=["SN IIP"],
            size=0,
            class_sampling="balanced",
            color_mode=None,
        )
        with tempfile.TemporaryDirectory() as directory:
            runner = WarpSimulationRunner(
                "data/warpcoeff_v3",
                source_cache_dir=Path(directory) / "source-cache",
            )
            runner.source_cache.build = Mock()
            runner.source_cache.describe = Mock(
                return_value={"fingerprint": "stable-cache"}
            )
            with patch(
                "warpTemplate.survey_factory.SurveyFactory.from_spec",
                return_value=ProvenanceSurvey("first-log"),
            ):
                first = runner.run(spec, directory)
            self.assertEqual(first["status"], "complete")

            with patch(
                "warpTemplate.survey_factory.SurveyFactory.from_spec",
                return_value=ProvenanceSurvey("changed-log"),
            ) as factory:
                with self.assertRaisesRegex(
                    ValueError, "survey provenance changed"
                ):
                    runner.run(spec, directory, resume=True)
            factory.assert_called_once_with(spec)

    def test_explicit_survey_fallback_provenance_hashes_log_contents(self):
        """Equal row counts must not hide changed explicit survey data."""

        spec = WarpSampleSpec(
            run_name="changed_explicit_survey",
            active_fitclasses=["SN IIP"],
            size=0,
            class_sampling="balanced",
            color_mode=None,
        )
        survey = FakeSurvey()
        with tempfile.TemporaryDirectory() as directory:
            runner = WarpSimulationRunner(
                "data/warpcoeff_v3",
                source_cache_dir=Path(directory) / "source-cache",
            )
            runner.source_cache.build = Mock()
            runner.source_cache.describe = Mock(
                return_value={"fingerprint": "stable-cache"}
            )
            first = runner.run(spec, directory, survey=survey)
            self.assertEqual(first["status"], "complete")
            self.assertIn("data_fingerprint", first["survey"])

            survey.data.loc[0, "mjd"] += 1.0
            with self.assertRaisesRegex(
                ValueError,
                "survey provenance changed",
            ):
                runner.run(spec, directory, survey=survey, resume=True)

    def test_changed_population_config_cannot_resume_schema_five_run(self):
        """Rate and magnitude priors are part of the deterministic run contract."""

        spec = WarpSampleSpec(
            run_name="changed_population_config",
            active_fitclasses=["SN IIP"],
            size=0,
            class_sampling="balanced",
            color_mode=None,
        )
        with tempfile.TemporaryDirectory() as directory:
            runner = WarpSimulationRunner(
                "data/warpcoeff_v3",
                source_cache_dir=Path(directory) / "source-cache",
            )
            runner.source_cache.build = Mock()
            runner.source_cache.describe = Mock(
                return_value={"fingerprint": "stable-cache"}
            )
            first = runner.run(spec, directory, survey=FakeSurvey())
            self.assertIn("population_config", first)

            changed = deepcopy(runner.sampler.rate_config)
            changed["notes"] = f"{changed.get('notes', '')} changed"
            runner.sampler.rate_config = changed
            with self.assertRaisesRegex(
                ValueError,
                "population rate or magnitude configuration changed",
            ):
                runner.run(
                    spec,
                    directory,
                    survey=FakeSurvey(),
                    resume=True,
                )

    def test_fitclass_end_is_a_hard_batch_boundary(self):
        """Targets from distinct fitclasses must never share one batch."""

        spec = WarpSampleSpec(
            run_name="fitclass_batches",
            active_fitclasses=["SN IIP", "SN Ib"],
            class_sampling="counts",
            class_counts={"SN IIP": 3, "SN Ib": 2},
            color_mode=None,
            batch_size=10,
            max_sources_per_batch=10,
            seed=53,
        )
        batches = list(
            _WarpTargetSampler("data/warpcoeff_v3").iter_batches(spec)
        )
        self.assertEqual([len(batch) for batch in batches], [3, 2])
        self.assertEqual(
            [batch["fitclass"].iloc[0] for batch in batches],
            ["SN IIP", "SN Ib"],
        )
        self.assertTrue(
            all(batch["fitclass"].nunique() == 1 for batch in batches)
        )

    def test_batches_group_entries_and_respect_source_limit(self):
        """Batches must stop at both their row and selected-source limits."""

        source_limited_spec = WarpSampleSpec(
            run_name="source_groups",
            active_fitclasses=["SN IIP"],
            size=200,
            class_sampling="balanced",
            color_mode=None,
            batch_size=37,
            max_sources_per_batch=3,
            seed=61,
        )
        sampler = _WarpTargetSampler("data/warpcoeff_v3")
        batches = list(sampler.iter_batches(source_limited_spec))
        self.assertEqual(sum(map(len, batches)), 200)
        self.assertTrue(all(len(batch) <= 37 for batch in batches))
        self.assertTrue(
            all(batch["template_key"].nunique() <= 3 for batch in batches)
        )
        self.assertTrue(all(batch["fitclass"].nunique() == 1 for batch in batches))
        self.assertTrue(
            any(
                len(batch) < source_limited_spec.batch_size
                and batch["template_key"].nunique()
                == source_limited_spec.max_sources_per_batch
                for batch in batches[:-1]
            )
        )

        # Direct count drawing visits every selected entry in one stable block.
        combined = pd.concat(batches, ignore_index=True)
        self.assertTrue(
            all(
                len(np.flatnonzero(combined["template_key"].eq(key)))
                == np.flatnonzero(combined["template_key"].eq(key))[-1]
                - np.flatnonzero(combined["template_key"].eq(key))[0]
                + 1
                for key in combined["template_key"].unique()
            )
        )

        row_limited_spec = WarpSampleSpec(
            run_name="row_groups",
            active_fitclasses=["SN IIP"],
            size=20,
            class_sampling="balanced",
            color_mode=None,
            batch_size=4,
            max_sources_per_batch=512,
            seed=61,
        )
        row_limited = list(sampler.iter_batches(row_limited_spec))
        self.assertEqual([len(batch) for batch in row_limited], [4] * 5)

    def test_one_large_entry_spans_multiple_row_limited_batches(self):
        """A selected entry larger than the row limit must continue next batch."""

        sampler = _WarpTargetSampler("data/warpcoeff_v3")
        only_entry = sampler.loader.get_entry_probabilities(
            "SN IIP",
            min_fit_quality="bronze",
        )[0]
        spec = WarpSampleSpec(
            run_name="large_entry",
            active_fitclasses=["SN IIP"],
            size=10,
            class_sampling="balanced",
            color_mode=None,
            batch_size=4,
            max_sources_per_batch=1,
            seed=71,
        )
        with patch.object(
            sampler.loader,
            "get_entry_probabilities",
            return_value=[only_entry],
        ):
            batches = list(sampler.iter_batches(spec))

        self.assertEqual([len(batch) for batch in batches], [4, 4, 2])
        self.assertTrue(
            all(batch["template_key"].nunique() == 1 for batch in batches)
        )
        self.assertEqual(
            pd.concat(batches)["template_key"].nunique(),
            1,
        )

    def test_sources_are_shared_only_inside_one_batch(self):
        """A source is reused within a batch and rebuilt in a new batch."""

        coefficient_dir = Path("data/warpcoeff_v3")
        spec = WarpSampleSpec(
            run_name="batch_local_sources",
            active_fitclasses=["SN IIP"],
            size=200,
            class_sampling="balanced",
            color_mode=None,
            batch_size=200,
            seed=44,
        )
        sampler = _WarpTargetSampler(coefficient_dir)
        drawn = next(sampler.iter_batches(spec))
        repeated_key = drawn["template_key"].value_counts().index[0]
        repeated = drawn.loc[drawn["template_key"].eq(repeated_key)].head(2)
        self.assertEqual(len(repeated), 2)
        source_cache = LoaderBackedSourceCache(sampler.loader)

        first = _WarpBatchTargets(
            repeated,
            sampler.loader,
            sampler.cosmology,
            source_cache=source_cache,
        )
        for index in repeated.index:
            first.get_target_template(index, as_model=True)
        self.assertEqual(first.loaded_source_count, 1)
        self.assertEqual(source_cache.load_source.call_count, 1)

        second = _WarpBatchTargets(
            repeated.iloc[:1],
            sampler.loader,
            sampler.cosmology,
            source_cache=source_cache,
        )
        second.get_target_template(repeated.index[0], as_model=True)
        self.assertEqual(second.loaded_source_count, 1)
        self.assertEqual(source_cache.load_source.call_count, 2)

    def test_batch_targets_load_sources_only_when_skysurvey_requests_models(self):
        """Creating compact targets alone must not prepare any dense source."""

        sampler = _WarpTargetSampler("data/warpcoeff_v3")
        spec = WarpSampleSpec(
            run_name="lazy_sources",
            active_fitclasses=["SN IIP"],
            size=3,
            class_sampling="balanced",
            color_mode=None,
        )
        batch = next(sampler.iter_batches(spec))
        targets = _WarpBatchTargets(
            batch,
            sampler.loader,
            sampler.cosmology,
            source_cache=LoaderBackedSourceCache(sampler.loader),
        )
        self.assertEqual(targets.loaded_source_count, 0)

    def test_unobserved_targets_do_not_load_sources_in_skysurvey(self):
        """SkySurvey field selection must happen before lazy source creation."""

        spec = WarpSampleSpec(
            run_name="partially_observed",
            active_fitclasses=["SN IIP"],
            size=4,
            zmax=0.01,
            tstart=60000.0,
            tstop=60001.0,
            class_sampling="balanced",
            redshift_sampling="uniform",
            color_mode=None,
            batch_size=4,
            incl_error=False,
            seed=31,
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = WarpSimulationRunner(
                "data/warpcoeff_v3",
                source_cache_dir=Path(directory) / "source-cache",
            ).run(
                spec, PartiallyObservedSurvey(), directory
            )
        batch = manifest["batches"]["00000000"]
        self.assertEqual(batch["loaded_source_count"], 1)
        self.assertEqual(batch["observation_rows"], 1)
        self.assertEqual(batch["truth_rows"], 4)
        self.assertEqual(manifest["truth_rows"], 4)
        self.assertLess(batch["loaded_source_count"], batch["template_count"])

    def test_fully_unobserved_batch_persists_truth_and_empty_observations(self):
        """A zero-field-match batch must not enter SkySurvey's empty concat path."""

        spec = WarpSampleSpec(
            run_name="fully_unobserved",
            active_fitclasses=["SN IIP"],
            size=3,
            zmax=0.01,
            tstart=60_000.0,
            tstop=60_001.0,
            skyarea="full",
            class_sampling="balanced",
            redshift_sampling="uniform",
            color_mode=None,
            batch_size=3,
            incl_error=True,
            seed=32,
        )
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / spec.run_name
            manifest = WarpSimulationRunner(
                "data/warpcoeff_v3",
                source_cache_dir=Path(directory) / "source-cache",
            ).run(
                spec,
                UnobservedSurvey(),
                directory,
            )
            batch = manifest["batches"]["00000000"]
            truth = pd.read_parquet(run_dir / batch["truth"])
            observations = pd.read_parquet(run_dir / batch["observations"])

        self.assertEqual(len(truth), 3)
        self.assertTrue(observations.empty)
        self.assertIn("object_id", observations)
        self.assertIn("fluxerr", observations)
        self.assertEqual(batch["loaded_source_count"], 0)
        self.assertEqual(manifest["observation_rows"], 0)

    def test_phase_empty_batch_loads_source_but_writes_no_observations(self):
        """Field matching precedes lazy loading, while phase filtering follows it."""

        spec = WarpSampleSpec(
            run_name="phase_empty",
            active_fitclasses=["SN IIP"],
            size=1,
            zmax=0.01,
            tstart=61_000.0,
            tstop=61_001.0,
            class_sampling="balanced",
            redshift_sampling="uniform",
            color_mode=None,
            batch_size=1,
            phase_range=(-1.0, 1.0),
            incl_error=False,
            seed=33,
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = WarpSimulationRunner(
                "data/warpcoeff_v3",
                source_cache_dir=Path(directory) / "source-cache",
            ).run(
                spec,
                FakeSurvey(),
                directory,
            )

        batch = manifest["batches"]["00000000"]
        self.assertEqual(batch["truth_rows"], 1)
        self.assertEqual(batch["loaded_source_count"], 1)
        self.assertEqual(batch["observation_rows"], 0)

    def test_partial_resume_skips_complete_batches_and_recomputes_interrupted_one(self):
        """An orphan truth partition must not make an interrupted batch resumable."""

        spec = WarpSampleSpec(
            run_name="partial_resume",
            active_fitclasses=["SN IIP"],
            size=2,
            class_sampling="balanced",
            color_mode=None,
            incl_error=False,
        )
        batches = [
            pd.DataFrame(
                {
                    "fitclass": ["SN IIP"],
                    "template_key": [f"entry-{index}"],
                    "object_id": [f"partial_resume:SN IIP:{index:012d}"],
                    "ra": [0.0],
                    "dec": [0.0],
                }
            )
            for index in range(2)
        ]

        def iter_test_batches(_spec):
            """Yield fresh copies of both deterministic batches."""

            for batch in batches:
                yield batch.copy()

        simulated_object_ids = []

        def simulate_batch(targets, _survey, **_kwargs):
            """Return one observation while recording every simulation attempt."""

            simulated_object_ids.append(targets.data["object_id"].iloc[0])
            return SimpleNamespace(
                data=pd.DataFrame(
                    {"flux": [1.0], "fluxerr": [0.1]},
                    index=pd.Index([0], name="index"),
                )
            )

        with tempfile.TemporaryDirectory() as directory:
            runner = WarpSimulationRunner(
                "data/warpcoeff_v3",
                source_cache_dir=Path(directory) / "source-cache",
            )
            runner.source_cache.build = Mock()
            runner.source_cache.describe = Mock(
                return_value={"fingerprint": "stable-cache"}
            )
            runner.sampler.iter_batches = iter_test_batches
            original_atomic = runner._atomic_parquet
            write_count = 0

            def fail_on_second_observation(frame, path):
                """Leave the second truth file orphaned, then emulate interruption."""

                nonlocal write_count
                write_count += 1
                if write_count == 4:
                    raise RuntimeError("interrupted observation write")
                original_atomic(frame, path)

            run_dir = Path(directory) / spec.run_name
            with patch(
                "skysurvey.DataSet.from_targets_and_survey",
                side_effect=simulate_batch,
            ):
                with patch.object(
                    runner,
                    "_atomic_parquet",
                    side_effect=fail_on_second_observation,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "interrupted observation write"
                    ):
                        runner.run(spec, directory, survey=FakeSurvey())

                partial_manifest = json.loads(
                    (run_dir / "manifest.json").read_text(encoding="utf-8")
                )
                orphan_truth = (
                    run_dir
                    / "truth"
                    / "fitclass=SN IIP"
                    / "batch-00000001.parquet"
                )
                missing_observations = (
                    run_dir
                    / "observations"
                    / "fitclass=SN IIP"
                    / "batch-00000001.parquet"
                )
                self.assertEqual(list(partial_manifest["batches"]), ["00000000"])
                self.assertTrue(orphan_truth.exists())
                self.assertFalse(missing_observations.exists())

                resumed = runner.run(
                    spec, directory, survey=FakeSurvey(), resume=True
                )

        self.assertEqual(resumed["status"], "complete")
        self.assertEqual(len(resumed["batches"]), 2)
        self.assertEqual(
            simulated_object_ids,
            [
                "partial_resume:SN IIP:000000000000",
                "partial_resume:SN IIP:000000000001",
                "partial_resume:SN IIP:000000000001",
            ],
        )

    def test_schema_five_run_cannot_resume(self):
        """Schema-5 output lacks explicit survey-realization identity."""

        spec = WarpSampleSpec(
            run_name="old_schema",
            active_fitclasses=["SN IIP"],
            size=1,
            class_sampling="balanced",
        )
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / spec.run_name
            run_dir.mkdir()
            (run_dir / "manifest.json").write_text(
                json.dumps({"schema_version": 5}), encoding="utf-8"
            )
            runner = WarpSimulationRunner(
                "data/warpcoeff_v3",
                source_cache_dir=Path(directory) / "source-cache",
            )
            runner.source_cache.build = Mock()
            runner.source_cache.describe = Mock(return_value={})
            with self.assertRaisesRegex(ValueError, "incompatible"):
                runner.run(spec, FakeSurvey(), directory, resume=True)


if __name__ == "__main__":
    unittest.main()
