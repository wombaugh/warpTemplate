"""Tests for dynamic colour and persistent neutral-source caching."""

from __future__ import annotations

import pickle
from pathlib import Path
import tempfile
import unittest

import extinction
import h5py
import numpy as np
import sncosmo

from warpTemplate.loaders import WarpfitTemplateLoader
from warpTemplate.models import get_model_from_warped_source
from warpTemplate.source_cache import WarpSourceCache
from warpTemplate.sources import DynamicColorWarpSource


class DynamicColorSourceTest(unittest.TestCase):
    """Verify dynamic colour numerics and independent event parameters."""

    def setUp(self):
        """Create a smooth source grid with sufficient spline support."""

        self.phase = np.linspace(-10.0, 30.0, 9)
        self.wave = np.linspace(3000.0, 9000.0, 121)
        self.flux = np.exp(-((self.phase[:, None] - 5.0) / 13.0) ** 2) * (
            1.0 + 0.15 * np.sin(self.wave[None, :] / 900.0)
        )

    def test_dynamic_colour_matches_grid_first_colour_to_required_tolerance(self):
        """Moving CCM89 after interpolation must preserve multi-band model flux."""

        test_phase = np.linspace(-9.5, 29.5, 31)
        # These observer-frame windows cover ZTF-like and LSST-like optical bands
        # without depending on remotely downloaded bandpass registrations.
        bands = [
            sncosmo.Bandpass(np.linspace(low, high, 101), np.ones(101))
            for low, high in ((3400.0, 4000.0), (4100.0, 5500.0),
                              (5600.0, 7000.0), (7500.0, 9000.0))
        ]
        for ebv in (-0.2, 0.0, 0.35):
            attenuation = extinction.ccm89(self.wave, ebv * 3.1, 3.1)
            legacy_source = sncosmo.TimeSeriesSource(
                self.phase,
                self.wave,
                extinction.apply(attenuation, self.flux),
            )
            for redshift in (0.0, 0.08):
                dynamic_model = sncosmo.Model(
                    source=DynamicColorWarpSource(
                        self.phase, self.wave, self.flux
                    )
                )
                dynamic_model.set(z=redshift, samplecorr_ebv=ebv)
                legacy_model = sncosmo.Model(source=legacy_source)
                legacy_model.set(z=redshift)
                observed_time = test_phase * (1.0 + redshift)
                for band in bands:
                    np.testing.assert_allclose(
                        dynamic_model.bandflux(band, observed_time),
                        legacy_model.bandflux(band, observed_time),
                        rtol=1e-3,
                        atol=1e-12,
                    )

    def test_models_share_spline_but_not_parameters(self):
        """Changing one event colour must not mutate its sibling or prototype."""

        source = DynamicColorWarpSource(self.phase, self.wave, self.flux)
        first = get_model_from_warped_source(source, samplecorr_ebv=0.2)
        second = get_model_from_warped_source(source, samplecorr_ebv=-0.1)
        first.set(samplecorr_ebv=0.5)
        self.assertEqual(source.get("samplecorr_ebv"), 0.0)
        self.assertEqual(second.get("samplecorr_ebv"), -0.1)
        self.assertIs(first.source._model_flux, second.source._model_flux)

    def test_redshifted_edge_band_uses_zero_flux_outside_native_support(self):
        """A partly unsupported LSST-u-like band must integrate without extrapolation."""

        source = DynamicColorWarpSource(self.phase, self.wave, self.flux)

        # Values beyond the native 3000--9000 Angstrom grid are explicitly
        # zero, while values at supported wavelengths remain untouched.
        evaluated = source._flux(
            np.array([5.0]),
            np.array([2500.0, 3000.0, 5000.0, 9500.0]),
        )
        self.assertEqual(evaluated[0, 0], 0.0)
        self.assertEqual(evaluated[0, -1], 0.0)
        self.assertGreater(evaluated[0, 1], 0.0)
        self.assertGreater(evaluated[0, 2], 0.0)

        # At z=0.06 the observer-frame lower edge maps below 3000 Angstrom,
        # reproducing the failure mode seen for the real LSST-u bandpass.
        model = sncosmo.Model(source=source)
        model.set(z=0.06)
        lsst_u_like = sncosmo.Bandpass(
            np.linspace(3105.0, 4086.0, 200),
            np.ones(200),
        )
        flux = model.bandflux(lsst_u_like, 5.0 * (1.0 + 0.06))
        self.assertTrue(np.isfinite(flux))
        self.assertGreater(flux, 0.0)

    def test_default_coverage_never_truncates_a_wider_native_grid(self):
        """Standard edge coverage must preserve unusually broad SLSN spectra."""

        wide_wave = np.linspace(1500.0, 40000.0, 121)
        wide_flux = np.ones((self.phase.size, wide_wave.size))
        source = DynamicColorWarpSource(self.phase, wide_wave, wide_flux)
        self.assertEqual(source.minwave(), 1500.0)
        self.assertEqual(source.maxwave(), 40000.0)


class WarpSourceCacheTest(unittest.TestCase):
    """Exercise atomic HDF5 builds, reads, and fingerprint invalidation."""

    def setUp(self):
        """Register a local source and write one synthetic coefficient entry."""

        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.coefficient_dir = root / "coefficients"
        self.cache_dir = root / "cache"
        self.coefficient_dir.mkdir()
        phase = np.linspace(-10.0, 20.0, 7)
        wave = np.linspace(3000.0, 8000.0, 11)
        source_flux = np.ones((phase.size, wave.size))
        self.source_name = f"warp-cache-test-{id(self)}"
        sncosmo.register(
            sncosmo.TimeSeriesSource(phase, wave, source_flux),
            name=self.source_name,
            force=True,
        )
        warp_phase = np.linspace(-8.0, 18.0, 6)
        warp_wave = np.linspace(3200.0, 7800.0, 6)
        coefficient = {
            "warpcoeff": {
                "basis": [
                    {
                        "model": self.source_name,
                        "z": 0.01,
                        "quality": "gold",
                        "draw_prob": 1.0,
                        "peakcol": 0.0,
                        "mdict": {
                            "corrmodel": {
                                "phase": warp_phase,
                                "wave": warp_wave,
                                "flux": np.ones(
                                    (warp_phase.size, warp_wave.size)
                                ),
                            }
                        },
                    }
                ]
            }
        }
        self.coefficient_path = (
            self.coefficient_dir / "warpcoeffs_v3_SN Cache.pkl"
        )
        with self.coefficient_path.open("wb") as handle:
            pickle.dump(coefficient, handle)

    def tearDown(self):
        """Release temporary cache files."""

        self.temporary.cleanup()

    def test_build_load_resume_and_invalidate(self):
        """A valid cache resumes, while changed coefficients invalidate it."""

        loader = WarpfitTemplateLoader(str(self.coefficient_dir))
        cache = WarpSourceCache(
            self.coefficient_dir, self.cache_dir, loader=loader
        )
        first = cache.build(["SN Cache"])
        self.assertTrue(first["SN Cache"]["valid"])
        cache_path = cache.fitclass_path("SN Cache")
        first_mtime = cache_path.stat().st_mtime_ns
        cache.build(["SN Cache"])
        self.assertEqual(first_mtime, cache_path.stat().st_mtime_ns)

        descriptor = loader.get_entry_probabilities("SN Cache")[0][0]
        cached_source = cache.load_source(descriptor)
        direct_source = loader.build_uncolored_source(descriptor)
        self.assertIsNotNone(cached_source)
        np.testing.assert_allclose(
            cached_source._flux(cached_source._phase, cached_source._wave),
            direct_source._flux(direct_source._phase, direct_source._wave),
            rtol=1e-12,
            atol=1e-12,
        )

        with self.coefficient_path.open("ab") as handle:
            handle.write(b"changed")
        fresh_cache = WarpSourceCache(self.coefficient_dir, self.cache_dir)
        self.assertFalse(fresh_cache.is_valid("SN Cache"))

    def test_corrupt_cache_is_a_safe_miss(self):
        """Unreadable HDF5 content must fall back instead of being trusted."""

        cache = WarpSourceCache(self.coefficient_dir, self.cache_dir)
        path = cache.fitclass_path("SN Cache")
        path.parent.mkdir(parents=True)
        path.write_bytes(b"not hdf5")
        self.assertFalse(cache.is_valid("SN Cache"))

    def test_source_fingerprint_mismatch_invalidates_partition(self):
        """A changed source fingerprint must make a partition unusable."""

        cache = WarpSourceCache(self.coefficient_dir, self.cache_dir)
        cache.build(["SN Cache"])
        with h5py.File(cache.fitclass_path("SN Cache"), "r+") as handle:
            handle.attrs["source_fingerprint"] = "changed"
        fresh_cache = WarpSourceCache(self.coefficient_dir, self.cache_dir)
        self.assertFalse(fresh_cache.is_valid("SN Cache"))

    def test_missing_base_source_leaves_no_partial_partition(self):
        """A failed source lookup must remove temporary and final cache files."""

        with self.coefficient_path.open("rb") as handle:
            coefficient = pickle.load(handle)
        coefficient["warpcoeff"]["basis"][0]["model"] = "missing-warp-source"
        with self.coefficient_path.open("wb") as handle:
            pickle.dump(coefficient, handle)
        cache = WarpSourceCache(self.coefficient_dir, self.cache_dir)
        with self.assertRaises(Exception):
            cache.build(["SN Cache"])
        path = cache.fitclass_path("SN Cache")
        self.assertFalse(path.exists())
        self.assertFalse(path.with_suffix(path.suffix + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()
