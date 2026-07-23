"""Observer-frame count, redshift, time, and sky-area helpers."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from astropy import time
from astropy.cosmology import Planck18
from skysurvey.tools.utils import parse_skyarea, surface_of_skyarea

from .population import MissingRateError


FULL_SKY_DEG2 = 4.0 * np.pi * (180.0 / np.pi) ** 2


def _as_mjd(value: Any) -> Optional[float]:
    """Convert a SkySurvey-style time input to MJD while preserving None."""

    if value is None:
        return None
    if isinstance(value, str):
        return float(time.Time(value).mjd)
    if isinstance(value, time.Time):
        return float(value.mjd)
    return float(value)


def _normalize_time_window(
    *,
    size: Optional[int],
    nyears: Optional[float],
    tstart: Any,
    tstop: Any,
    default_tstart: float,
    default_tstop: float,
) -> tuple[float, float, float]:
    """Resolve tstart, tstop, and observer-frame nyears consistently."""

    tstart = _as_mjd(tstart)
    tstop = _as_mjd(tstop)

    if nyears is not None:
        nyears = float(nyears)
        if nyears <= 0:
            raise ValueError("nyears must be positive")
        if tstart is None and tstop is None:
            tstart = float(default_tstart)
        if tstart is None:
            tstart = float(tstop) - 365.25 * nyears
        tstop = float(tstart) + 365.25 * nyears
    elif tstart is not None and tstop is not None:
        nyears = (float(tstop) - float(tstart)) / 365.25
    elif size is not None:
        tstart = float(default_tstart) if tstart is None else float(tstart)
        tstop = float(default_tstop) if tstop is None else float(tstop)
        nyears = (tstop - tstart) / 365.25
    else:
        raise ValueError(
            "rate-normalized drawing requires nyears or both tstart and tstop"
        )

    if tstop <= tstart:
        raise ValueError("tstop must be greater than tstart")
    if nyears <= 0:
        raise ValueError("the observer-frame duration must be positive")
    return float(tstart), float(tstop), float(nyears)


def _sky_fraction(
    skyarea: Any = None,
    sky_fraction: Optional[float] = None,
) -> float:
    """Return the simulated fraction of the full sky."""

    if sky_fraction is not None:
        sky_fraction = float(sky_fraction)
        if not 0.0 < sky_fraction <= 1.0:
            raise ValueError("sky_fraction must be in the interval (0, 1]")
        if skyarea is not None and not (
            isinstance(skyarea, str) and skyarea == "full"
        ):
            raise ValueError("provide either skyarea or sky_fraction, not both")
        return sky_fraction

    if hasattr(skyarea, "area_deg2"):
        area_deg2 = float(skyarea.area_deg2)
        fraction = area_deg2 / FULL_SKY_DEG2
        if not 0.0 < fraction <= 1.0:
            raise ValueError(
                "skyarea must cover a positive area no larger than the full sky"
            )
        return fraction

    skyarea = parse_skyarea(skyarea)
    if skyarea is None or (isinstance(skyarea, str) and skyarea == "full"):
        return 1.0
    if hasattr(skyarea, "geoms"):
        area_deg2 = sum(
            surface_of_skyarea(part, incl_projection=True)
            for part in skyarea.geoms
        )
    else:
        area_deg2 = surface_of_skyarea(skyarea, incl_projection=True)
    if area_deg2 is None:
        raise ValueError("could not determine the surface of skyarea")

    fraction = float(area_deg2) / FULL_SKY_DEG2
    if not 0.0 < fraction <= 1.0:
        raise ValueError(
            "skyarea must cover a positive area no larger than the full sky"
        )
    return fraction


def _observer_rate_density(
    redshift: np.ndarray,
    rate: float | Any,
    cosmology: Any,
) -> np.ndarray:
    """Return R(z) dV/dz /(1+z) over the full sky."""

    source_rate = rate(redshift) if callable(rate) else float(rate)
    source_rate = np.broadcast_to(
        np.asarray(source_rate, dtype=float), redshift.shape
    )
    if np.any(source_rate < 0):
        raise ValueError("volumetric rates must be non-negative")

    differential_volume = cosmology.differential_comoving_volume(redshift)
    full_sky_volume = differential_volume.to_value("Gpc3 / sr") * (4.0 * np.pi)
    return source_rate * full_sky_volume / (1.0 + redshift)


def observer_expected_count(
    *,
    rate: float | Any,
    zmin: float,
    zmax: float,
    nyears: float,
    sky_fraction: float = 1.0,
    cosmology: Any = Planck18,
    grid_size: int = 2048,
) -> float:
    """Return the observer-frame Poisson mean for a source-frame rate."""

    if not 0.0 <= zmin < zmax:
        raise ValueError("require 0 <= zmin < zmax")
    if nyears <= 0:
        raise ValueError("nyears must be positive")
    if not 0.0 < sky_fraction <= 1.0:
        raise ValueError("sky_fraction must be in the interval (0, 1]")
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")

    redshift_grid = np.linspace(float(zmin), float(zmax), int(grid_size))
    density = _observer_rate_density(redshift_grid, rate, cosmology)
    # np.trapz supports the package's declared NumPy >=1.23 compatibility range.
    rate_per_observer_year = float(np.trapz(density, redshift_grid))
    return rate_per_observer_year * float(nyears) * float(sky_fraction)


def _observer_redshift_cdf(
    *,
    rate: float | Any,
    zmin: float,
    zmax: float,
    cosmology: Any,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the inverse-sampling grid for one observer-frame population."""

    redshift_grid = np.linspace(float(zmin), float(zmax), int(grid_size))
    density = _observer_rate_density(redshift_grid, rate, cosmology)
    dz = np.diff(redshift_grid)
    shell_weights = 0.5 * (density[1:] + density[:-1]) * dz
    cumulative = np.concatenate([[0.0], np.cumsum(shell_weights)])
    if cumulative[-1] <= 0:
        raise MissingRateError(
            "the observer-frame redshift density integrates to zero"
        )
    cumulative /= cumulative[-1]
    return redshift_grid, cumulative


def draw_observer_redshift(
    *,
    size: int,
    rate: float | Any,
    zmin: float,
    zmax: float,
    cosmology: Any = Planck18,
    grid_size: int = 2048,
    rng: Any = None,
) -> np.ndarray:
    """Draw redshifts from R(z) dV/dz /(1+z)."""

    if size < 0:
        raise ValueError("size must be non-negative")
    if size == 0:
        return np.asarray([], dtype=float)
    if not 0.0 <= zmin < zmax:
        raise ValueError("require 0 <= zmin < zmax")

    rng = np.random.default_rng(rng)
    # Count integration and inverse-CDF sampling use the same observer-frame
    # density, including the cosmological time-dilation factor.
    redshift_grid, cumulative = _observer_redshift_cdf(
        rate=rate,
        zmin=zmin,
        zmax=zmax,
        cosmology=cosmology,
        grid_size=grid_size,
    )
    return np.interp(rng.random(size), cumulative, redshift_grid)


def _draw_poisson_count(expected_count: float, rng: Any = None) -> int:
    """Draw one non-negative event count from a Poisson expectation value."""

    expected_count = float(expected_count)
    if not np.isfinite(expected_count) or expected_count < 0:
        raise ValueError("expected_count must be finite and non-negative")
    return int(np.random.default_rng(rng).poisson(expected_count))


__all__ = [
    "draw_observer_redshift",
    "observer_expected_count",
]
