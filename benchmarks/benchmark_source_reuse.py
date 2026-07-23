"""Benchmark event-model instantiation from one shared neutral Warp source."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import resource
import sys
import time


def _peak_rss_mb() -> float:
    """Return Linux peak resident memory in MiB."""

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def benchmark(
    coefficient_dir: Path,
    *,
    fitclass: str,
    event_count: int,
    fresh_event_count: int = 200,
) -> dict[str, float | int | str]:
    """Compare shared-source models with rebuilding a source for every event."""

    from warpTemplate.loaders import WarpfitTemplateLoader

    loader = WarpfitTemplateLoader(str(coefficient_dir))
    descriptor = loader.get_entry_probabilities(fitclass)[0][0]
    prepared_started = time.perf_counter()
    source = loader.build_uncolored_source(descriptor)
    preparation_seconds = time.perf_counter() - prepared_started

    # Vary colour without rebuilding the shared two-dimensional interpolation.
    started = time.perf_counter()
    for index in range(event_count):
        event_descriptor = replace(
            descriptor,
            color_mode="draw",
            samplecorr_ebv=((index % 101) - 50) / 100.0,
        )
        model = loader.materialize_descriptor(
            event_descriptor, source=source
        )["model"]
        model.set(z=0.05, t0=60000.0 + (index % 365))
        del model
    model_seconds = time.perf_counter() - started

    # A small normalized baseline is enough because every iteration repeats the
    # expensive two-dimensional warp-grid and spline construction.
    baseline_started = time.perf_counter()
    for index in range(fresh_event_count):
        event_descriptor = replace(
            descriptor,
            color_mode="draw",
            samplecorr_ebv=((index % 101) - 50) / 100.0,
        )
        model = loader.materialize_descriptor(event_descriptor)["model"]
        model.set(z=0.05, t0=60000.0 + (index % 365))
        del model
    baseline_seconds = time.perf_counter() - baseline_started
    shared_seconds_per_event = model_seconds / event_count
    fresh_seconds_per_event = baseline_seconds / fresh_event_count
    return {
        "fitclass": fitclass,
        "events": event_count,
        "fresh_baseline_events": fresh_event_count,
        "source_preparation_seconds": preparation_seconds,
        "model_instantiation_seconds": model_seconds,
        "events_per_second": event_count / model_seconds,
        "shared_seconds_per_event": shared_seconds_per_event,
        "fresh_rebuild_seconds": baseline_seconds,
        "fresh_seconds_per_event": fresh_seconds_per_event,
        "model_creation_speedup": fresh_seconds_per_event
        / shared_seconds_per_event,
        "peak_rss_mb": _peak_rss_mb(),
    }


def main() -> None:
    """Parse arguments and print one machine-readable benchmark record."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, required=True)
    parser.add_argument(
        "--fresh-events",
        type=int,
        default=200,
        help="number of per-event source rebuilds used as the normalized baseline",
    )
    parser.add_argument("--fitclass", default="SN IIP")
    parser.add_argument(
        "--coefficients", type=Path, default=Path("data/warpcoeff_v3")
    )
    arguments = parser.parse_args()
    if arguments.events <= 0:
        parser.error("--events must be positive")
    if arguments.fresh_events <= 0:
        parser.error("--fresh-events must be positive")

    # Running from the repository root should always import the local package.
    repository = Path(__file__).resolve().parents[2]
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    print(
        json.dumps(
            benchmark(
                arguments.coefficients,
                fitclass=arguments.fitclass,
                event_count=arguments.events,
                fresh_event_count=arguments.fresh_events,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
