from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np


def make_lightcurve_table(
    model,
    bands: Sequence[str] = ("ztfg", "ztfr", "ztfi"),
    times: Optional[Iterable[float]] = None,
    zp: float = 25.0,
    zpsys: str = "ab",
    skip_failed_bands: bool = True,
):
    """Evaluate a sncosmo model into an astropy light-curve table."""
    from astropy.table import Table

    if times is None:
        times = np.linspace(model.mintime(), model.maxtime(), 160)
    else:
        times = np.asarray(list(times), dtype=float)

    rows = []
    skipped = {}
    for band in bands:
        try:
            flux = model.bandflux(band, times, zp=zp, zpsys=zpsys)
        except Exception as exc:
            if not skip_failed_bands:
                raise
            skipped[band] = str(exc)
            continue

        for time, value in zip(times, flux):
            rows.append((float(time), band, float(value), zp, zpsys))

    table = Table(rows=rows, names=("time", "band", "flux", "zp", "zpsys"))
    table.meta["skipped_bands"] = skipped
    return table


def plot_lightcurve_table(tab, title: Optional[str] = None, ax=None):
    """Plot a table produced by make_lightcurve_table."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))

    for band in sorted(set(tab["band"])):
        cut = tab["band"] == band
        ax.plot(tab["time"][cut], tab["flux"][cut], label=band)

    ax.axhline(0, color="0.8", lw=1)
    ax.set_xlabel("MJD / observer-frame time")
    ax.set_ylabel("Flux")
    if title:
        ax.set_title(title)
    ax.legend()
    ax.figure.tight_layout()
    return ax.figure, ax


def measure_peak_color(
    model,
    color1: str,
    color2: str,
    phase: float = 0.0,
    magsys: str = "ab",
) -> float:
    """Return color1 - color2 at the requested model phase."""
    return float(
        model.bandmag(color1, magsys, phase)
        - model.bandmag(color2, magsys, phase)
    )


def first_available_source(candidates: Sequence[str]):
    """Return the first sncosmo source name that can be loaded."""
    import sncosmo

    failures = {}
    for name in candidates:
        try:
            return name, sncosmo.get_source(name)
        except Exception as exc:
            failures[name] = str(exc)
    raise RuntimeError(f"None of the source candidates could be loaded: {failures}")


def make_identity_warpdata(
    source_name: str,
    phase_min: float = -20.0,
    phase_max: float = 100.0,
    n_phase: int = 80,
    n_wave: int = 80,
):
    """Build an identity warp grid for smoke-testing WarpedTimeSeriesSource."""
    import sncosmo

    source = sncosmo.get_source(source_name)
    phase_lo = max(float(source.minphase()), float(phase_min))
    phase_hi = min(float(source.maxphase()), float(phase_max))
    wave_lo = float(source.minwave())
    wave_hi = float(source.maxwave())

    phase = np.linspace(phase_lo, phase_hi, n_phase)
    wave = np.linspace(wave_lo, wave_hi, n_wave)
    flux = np.ones((len(phase), len(wave)))
    return {"corrmodel": {"phase": phase, "wave": wave, "flux": flux}}


__all__ = [
    "first_available_source",
    "make_identity_warpdata",
    "make_lightcurve_table",
    "measure_peak_color",
    "plot_lightcurve_table",
]
