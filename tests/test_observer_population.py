"""Tests for observer-frame population helpers and synthetic surveys."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from astropy import units as u

from warptemplate.observer_population import (
    _draw_poisson_count,
    _normalize_time_window,
    draw_observer_redshift,
    observer_expected_count,
)


class ConstantDifferentialCosmology:
    """Minimal cosmology with constant differential comoving volume."""

    def differential_comoving_volume(self, redshift):
        """Return one Gpc cubed per steradian at every redshift."""

        return np.ones_like(np.asarray(redshift), dtype=float) * u.Gpc**3 / u.sr


class FakeFieldIds:
    """Expose the field-id column names expected by SkySurvey DataSet."""

    names = ["fieldid"]


class FakeSurvey:
    """Minimal one-field survey for real SkySurvey integration tests."""

    def __init__(self):
        """Create one observation that covers every fake target."""

        self.fieldids = FakeFieldIds()
        self.data = pd.DataFrame(
            {
                "mjd": [60000.5],
                "band": ["bessellb"],
                "skynoise": [0.1],
                "gain": [1.0],
                "zp": [25.0],
                "fieldid": [1],
            }
        )

    def radec_to_fieldid(self, radec):
        """Map every supplied coordinate to the single survey field."""

        return pd.DataFrame({"fieldid": 1}, index=radec.index)


class ObserverPopulationTest(unittest.TestCase):
    """Verify time dilation, duration, and Poisson population behavior."""

    def test_observer_expected_count_contains_time_dilation(self):
        """The analytic constant-volume case must integrate 1/(1+z)."""

        rate = 7.0
        zmin = 0.0
        zmax = 1.0
        nyears = 2.5
        sky_fraction = 0.3
        expected = (
            rate
            * 4.0
            * np.pi
            * np.log((1.0 + zmax) / (1.0 + zmin))
            * nyears
            * sky_fraction
        )
        measured = observer_expected_count(
            rate=rate,
            zmin=zmin,
            zmax=zmax,
            nyears=nyears,
            sky_fraction=sky_fraction,
            cosmology=ConstantDifferentialCosmology(),
        )
        self.assertAlmostEqual(measured, expected, places=5)

    def test_duration_and_sky_fraction_scale_expectations_linearly(self):
        """Observer duration and simulated sky area must scale lambda linearly."""

        baseline = observer_expected_count(
            rate=3.0,
            zmin=0.0,
            zmax=0.5,
            nyears=1.0,
            sky_fraction=1.0,
            cosmology=ConstantDifferentialCosmology(),
        )
        scaled = observer_expected_count(
            rate=3.0,
            zmin=0.0,
            zmax=0.5,
            nyears=2.0,
            sky_fraction=0.25,
            cosmology=ConstantDifferentialCosmology(),
        )
        self.assertAlmostEqual(scaled, 0.5 * baseline)

    def test_time_bounds_and_nyears_resolve_to_the_same_duration(self):
        """Equivalent SkySurvey time inputs must yield the same window."""

        from_nyears = _normalize_time_window(
            size=None,
            nyears=1.5,
            tstart=60000.0,
            tstop=None,
            default_tstart=56000.0,
            default_tstop=56100.0,
        )
        from_bounds = _normalize_time_window(
            size=None,
            nyears=None,
            tstart=60000.0,
            tstop=60000.0 + 1.5 * 365.25,
            default_tstart=56000.0,
            default_tstop=56100.0,
        )
        self.assertEqual(from_nyears, from_bounds)

    def test_observer_redshift_matches_time_dilated_pdf(self):
        """Large draws must follow the analytic 1/(1+z) density."""

        draws = draw_observer_redshift(
            size=100_000,
            rate=1.0,
            zmin=0.0,
            zmax=1.0,
            cosmology=ConstantDifferentialCosmology(),
            rng=1234,
        )
        expected_mean = (1.0 - np.log(2.0)) / np.log(2.0)
        self.assertAlmostEqual(float(draws.mean()), expected_mean, delta=0.005)

    def test_poisson_ensemble_has_expected_mean_and_variance(self):
        """Repeated count draws must reproduce Poisson statistics."""

        rng = np.random.default_rng(44)
        expectation = 20.0
        counts = np.asarray(
            [_draw_poisson_count(expectation, rng=rng) for _ in range(10_000)]
        )
        self.assertAlmostEqual(float(counts.mean()), expectation, delta=0.2)
        self.assertAlmostEqual(float(counts.var()), expectation, delta=0.7)


if __name__ == "__main__":
    unittest.main()
