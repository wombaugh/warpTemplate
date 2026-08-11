"""Tests for Warp population-rate and magnitude-prior configuration."""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from warptemplate.population import (
    MissingRateError,
    OverlapRateError,
    discover_warp_fitclasses,
    load_warp_rate_config,
    resolve_rate,
    validate_active_fitclasses,
    validate_magabs_config,
    validate_rate_config,
)


class WarpPopulationConfigTest(unittest.TestCase):
    """Verify complete and non-overlapping population configuration."""

    @classmethod
    def setUpClass(cls):
        """Load the package configuration and repository coefficient path."""

        cls.config = load_warp_rate_config()
        cls.coefficient_dir = Path("data/warpcoeff_v3")

    def test_every_warp_fitclass_has_rate_and_magnitude_entries(self):
        """Every coefficient class must have auditable configuration entries."""

        fitclasses = discover_warp_fitclasses(self.coefficient_dir)
        validate_rate_config(self.config, available_fitclasses=fitclasses)
        validate_magabs_config(self.config, available_fitclasses=fitclasses)

    def test_fitclass_discovery_accepts_v3_and_v4_names(self):
        """Discovery must understand legacy and color-enriched coefficient names."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "warpcoeffs_v3_SN Ia.pkl").touch()
            (root / "warpcoeffs_v4_SN II_col.pkl").touch()
            (root / "unrelated.pkl").touch()
            self.assertEqual(
                discover_warp_fitclasses(root),
                ["SN II", "SN Ia"],
            )

    def test_missing_direct_rates_remain_visible(self):
        """Missing rates must fail unless explicitly allowed for auditing."""

        with self.assertRaises(MissingRateError):
            validate_active_fitclasses(
                ["SN Ia-pec"],
                self.config,
                allow_missing_rates=False,
            )
        active = validate_active_fitclasses(
            ["SN Ia-pec"],
            self.config,
            allow_missing_rates=True,
        )
        self.assertEqual(active, ["SN Ia-pec"])

    def test_aggregate_and_child_cannot_be_combined(self):
        """Overlapping aggregate and child classes must be rejected."""

        with self.assertRaises(OverlapRateError):
            validate_active_fitclasses(["SN CC (a)", "SN Ib"], self.config)
        with self.assertRaises(OverlapRateError):
            validate_active_fitclasses(["SN Ibc", "SN Ic"], self.config)

    def test_class_exclusion_does_not_renormalize_remaining_rates(self):
        """Removing Ia must not alter the resolved SLSN volumetric rate."""

        validate_active_fitclasses(["SN Ia-91bg", "SLSN-I"], self.config)
        with_ia = resolve_rate("SLSN-I", self.config).rate_gpc3_yr
        validate_active_fitclasses(["SLSN-I"], self.config)
        without_ia = resolve_rate("SLSN-I", self.config).rate_gpc3_yr
        self.assertEqual(with_ia, without_ia)


if __name__ == "__main__":
    unittest.main()
