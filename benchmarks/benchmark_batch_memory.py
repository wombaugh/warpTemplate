"""Measure peak memory of the real batch runner with a synthetic survey."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import tempfile
import time

import numpy as np
import pandas as pd


class _OneFieldIds:
    """Describe the single integer field column required by SkySurvey."""

    names = ["fieldid"]


class SyntheticYearSurvey:
    """One-field survey with a configurable number of year-long visits."""

    def __init__(self, visits: int, observed_targets_per_batch: int | None) -> None:
        """Create evenly spaced observations accepted by SkySurvey."""

        self.fieldids = _OneFieldIds()
        self.observed_targets_per_batch = observed_targets_per_batch
        self.data = pd.DataFrame(
            {
                "mjd": np.linspace(60000.0, 60365.25, visits),
                "band": "bessellb",
                "skynoise": 0.1,
                "gain": 1.0,
                "zp": 25.0,
                "fieldid": 1,
            }
        )

    def radec_to_fieldid(self, radec: pd.DataFrame) -> pd.DataFrame:
        """Map every generated target to the observed synthetic field."""

        if self.observed_targets_per_batch is None:
            return pd.DataFrame({"fieldid": 1}, index=radec.index)
        fieldids = pd.DataFrame({"fieldid": 2}, index=radec.index)
        fieldids.iloc[: self.observed_targets_per_batch, 0] = 1
        return fieldids


def _peak_rss_mb() -> float:
    """Return Linux peak resident memory in MiB."""

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def benchmark(
    coefficient_dir: Path,
    *,
    events: int,
    visits: int,
    batch_size: int,
    max_sources_per_batch: int,
    observed_targets_per_batch: int | None = None,
) -> dict[str, float | int | str | None]:
    """Run unchanged SkySurvey and return one machine-readable memory record."""

    from warptemplate import WarpSampleSpec, WarpSimulationRunner

    spec = WarpSampleSpec(
        run_name="batch_memory_benchmark",
        active_fitclasses=["SN IIP"],
        size=events,
        class_sampling="balanced",
        redshift_sampling="uniform",
        color_mode=None,
        tstart=60000.0,
        tstop=60365.25,
        phase_range=None,
        incl_error=False,
        batch_size=batch_size,
        max_sources_per_batch=max_sources_per_batch,
        seed=20260715,
    )
    survey = SyntheticYearSurvey(visits, observed_targets_per_batch)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as output_dir:
        manifest = WarpSimulationRunner(
            coefficient_dir,
            source_cache_dir=Path(output_dir) / "source-cache",
        ).run(
            spec, survey, output_dir, resume=False
        )
    elapsed = time.perf_counter() - started
    return {
        "events": events,
        "visits_per_target": visits,
        "observed_targets_per_batch": observed_targets_per_batch,
        "batch_size": batch_size,
        "max_sources_per_batch": max_sources_per_batch,
        "truth_rows": int(manifest["truth_rows"]),
        "observation_rows": int(manifest["observation_rows"]),
        "batches": len(manifest["batches"]),
        "elapsed_seconds": elapsed,
        "peak_rss_mb": _peak_rss_mb(),
        "manifest_schema": int(manifest["schema_version"]),
    }


def main() -> None:
    """Parse benchmark options, run the simulation, and enforce a RAM ceiling."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=10_000)
    parser.add_argument("--visits", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--max-sources", type=int, default=512)
    parser.add_argument(
        "--observed-targets-per-batch",
        type=int,
        help="observe only this many leading targets per batch",
    )
    parser.add_argument("--max-rss-mb", type=float, default=4096.0)
    parser.add_argument(
        "--coefficients", type=Path, default=Path("data/warpcoeff_v4")
    )
    arguments = parser.parse_args()
    if min(
        arguments.events,
        arguments.visits,
        arguments.batch_size,
        arguments.max_sources,
    ) <= 0:
        parser.error("event, visit, batch, and source counts must be positive")
    if (
        arguments.observed_targets_per_batch is not None
        and arguments.observed_targets_per_batch <= 0
    ):
        parser.error("--observed-targets-per-batch must be positive")
    if (
        arguments.observed_targets_per_batch is not None
        and arguments.observed_targets_per_batch > arguments.batch_size
    ):
        parser.error("--observed-targets-per-batch cannot exceed --batch-size")

    # Running from the repository root should import the local package.
    repository = Path(__file__).resolve().parents[1]
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    result = benchmark(
        arguments.coefficients,
        events=arguments.events,
        visits=arguments.visits,
        batch_size=arguments.batch_size,
        max_sources_per_batch=arguments.max_sources,
        observed_targets_per_batch=arguments.observed_targets_per_batch,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["peak_rss_mb"] > arguments.max_rss_mb:
        raise SystemExit(
            f"peak RSS {result['peak_rss_mb']:.1f} MiB exceeds "
            f"{arguments.max_rss_mb:.1f} MiB"
        )


if __name__ == "__main__":
    main()
