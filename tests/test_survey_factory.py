"""Tests for local ZTF, LSST, and synthetic combined survey loading."""

from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
import pandas as pd

from coefficient_library import COEFFICIENT_DIR, requires_coefficients
from warptemplate.batch_simulation import WarpSampleSpec, WarpSimulationRunner
from warptemplate.observer_population import (
    _draw_poisson_count,
    observer_expected_count,
)
from warptemplate.survey_factory import (
    CombinedSurvey,
    ObservedSkyArea,
    SurveyConfig,
    SurveyFactory,
    WarpPolygonSurvey,
    _adopted_observed_region,
    _cylindrical_equal_area_deg2,
    _observed_instrument_regions,
    _periodic_polygon_union,
    _proposal_pixel_ids,
    _ztf_survey,
)


class _SingleTarget:
    """Expose one fixed sncosmo model through the SkySurvey target protocol."""

    def __init__(self, ra: float, dec: float):
        """Store the target coordinate and a negligible-flux spectral model."""

        import sncosmo

        self.data = pd.DataFrame({"ra": [ra], "dec": [dec]})
        phases = np.asarray([-10.0, -3.0, 3.0, 10.0])
        wavelengths = np.asarray([2500.0, 5000.0, 8000.0, 12_000.0])
        flux = np.full((len(phases), len(wavelengths)), 1.0e-20)
        source = sncosmo.TimeSeriesSource(phases, wavelengths, flux)
        self.model = sncosmo.Model(source=source)
        self.model.set(z=0.0, t0=60_000.0)

    def get_target_template(
        self,
        index: int,
        *,
        as_model: bool = True,
        set_magabs: bool = True,
    ):
        """Return the single prepared model expected by SkySurvey."""

        return self.model


class SurveyFactoryTest(unittest.TestCase):
    """Validate normalization, selection, geometry, and combined noise behavior."""

    def setUp(self):
        """Create compact ZTF Parquet and LSST SQLite observing fixtures."""

        from ztffields.fields import Fields

        self.temporary = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temporary.name)
        (self.data_root / "ztf_data" / "logs").mkdir(parents=True)
        (self.data_root / "lsst_data").mkdir()

        geometry = Fields.get_field_geometry(level="quadrant").loc[
            (375, 16), "geometry"
        ]
        representative = geometry.representative_point()
        self.overlap_ra = float(representative.x)
        self.overlap_dec = float(representative.y)

        ztf = pd.DataFrame(
            {
                "expMJD": np.asarray(
                    [58_000.0, 58_000.25, 58_001.0], dtype="float32"
                ),
                "filter": ["ztfg", "ztfr", "ztfi"],
                "fieldID": np.asarray([375, 375, 375], dtype="uint16"),
                "fieldRA": np.asarray(
                    [np.radians(self.overlap_ra)] * 3, dtype="float32"
                ),
                "fieldDec": np.asarray(
                    [np.radians(self.overlap_dec)] * 3, dtype="float32"
                ),
                "rcid": np.asarray([16, 16, 16], dtype="uint8"),
                "maglimcat": np.asarray([20.0, 21.0, 19.0], dtype="float32"),
                "zp": np.asarray([25.0, 25.0, 25.0], dtype="float32"),
                "gain": np.asarray([6.2, 6.2, 6.2], dtype="float32"),
                "expid": np.asarray([1, 2, 3], dtype="int64"),
                "infobits": np.asarray([0, 0, 8], dtype="uint64"),
            }
        )
        ztf.to_parquet(
            self.data_root
            / "ztf_data"
            / "logs"
            / "ztf_obsfile_maglimcat.parquet",
            index=False,
        )

        database = (
            self.data_root / "lsst_data" / "baseline_v5.0.0_10yrs.db"
        )
        with sqlite3.connect(database) as connection:
            connection.execute(
                "CREATE TABLE observations ("
                "observationId INTEGER, fieldRA REAL, fieldDec REAL, "
                "observationStartMJD REAL, fiveSigmaDepth REAL, filter TEXT, "
                "night INTEGER, scheduler_note TEXT, target_id INTEGER)"
            )
            connection.executemany(
                "INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        11,
                        self.overlap_ra,
                        self.overlap_dec,
                        61_000.0,
                        24.0,
                        "g",
                        0,
                        "WFD",
                        1,
                    ),
                    (
                        12,
                        self.overlap_ra,
                        self.overlap_dec,
                        61_000.25,
                        25.0,
                        "r",
                        0,
                        "WFD",
                        1,
                    ),
                    (
                        13,
                        self.overlap_ra + 30.0,
                        self.overlap_dec,
                        61_001.0,
                        23.0,
                        "i",
                        1,
                        "WFD",
                        2,
                    ),
                ],
            )

    def tearDown(self):
        """Remove temporary survey fixtures."""

        self.temporary.cleanup()

    def test_config_rejects_invalid_names_modes_and_unknown_options(self):
        """Configuration errors must fail before any large data file is read."""

        with self.assertRaisesRegex(ValueError, "survey name"):
            SurveyConfig.from_options("unknown", data_root=self.data_root)
        with self.assertRaisesRegex(ValueError, "time_mode"):
            SurveyConfig.from_options(
                "ztf",
                {"time_mode": "shift-somehow"},
                data_root=self.data_root,
            )
        with self.assertRaisesRegex(ValueError, "unknown survey_options"):
            SurveyConfig.from_options(
                "ztf",
                {"not_an_option": True},
                data_root=self.data_root,
            )
        with self.assertRaisesRegex(ValueError, "backend must be pandas"):
            SurveyConfig.from_options(
                "ztf",
                {"backend": "polars"},
                data_root=self.data_root,
            )

    def test_ztf_relative_loading_preserves_cadence_and_instrument_noise(self):
        """ZTF radians, bands, quadrant IDs, and row-wise noise must normalize."""

        survey = SurveyFactory(self.data_root).create(
            "ztf",
            tstart=60_000.0,
            tstop=60_001.0,
            options={"nside": 64},
        )
        np.testing.assert_allclose(
            survey.data["mjd"].to_numpy(),
            [60_000.0, 60_000.25, 60_001.0],
        )
        np.testing.assert_allclose(
            survey.data["skynoise"].to_numpy(),
            np.power(10.0, -0.4 * (np.asarray([20.0, 21.0, 19.0]) - 25.0))
            / 5.0,
        )
        self.assertEqual(survey.fieldids.names, ["fieldid", "rcid"])
        self.assertEqual(
            survey.warp_provenance["sources"]["ztf"]["coordinate_input_unit"],
            "radian",
        )
        self.assertLess(survey.simulation_skyarea.area_deg2, 2.0)
        ra, dec = survey.simulation_skyarea.draw_radec(
            size=4096,
            rng=np.random.default_rng(17),
        )
        matches = survey.radec_to_fieldid(
            pd.DataFrame({"ra": ra, "dec": dec}),
            observed_fields=True,
        )
        self.assertEqual(matches.index.nunique(), 4096)

    def test_relative_realizations_span_each_available_archive(self):
        """Planned windows must be bounded, distinct, and manifest-ready."""

        factory = SurveyFactory(self.data_root)
        realizations = factory.plan_relative_realizations(
            "combined",
            tstart=60_000.0,
            tstop=60_000.25,
            count=3,
            options={"time_mode": "relative", "nside": 64},
        )

        self.assertEqual(
            [item.realization_id for item in realizations],
            ["r000", "r001", "r002"],
        )
        self.assertEqual(
            realizations[0].source_mjd_ranges["ztf"],
            (58_000.0, 58_000.25),
        )
        self.assertEqual(
            realizations[-1].source_mjd_ranges["ztf"],
            (58_000.75, 58_001.0),
        )
        self.assertEqual(
            realizations[0].source_mjd_ranges["lsst"],
            (61_000.0, 61_000.25),
        )
        self.assertEqual(
            realizations[-1].source_mjd_ranges["lsst"],
            (61_000.75, 61_001.0),
        )
        self.assertEqual(
            realizations[1].to_dict()["survey_options"][
                "ztf_source_mjd_start"
            ],
            58_000.375,
        )

    def test_relative_realizations_reject_overlapping_default_windows(self):
        """Independent ensemble windows must fit unless overlap is requested."""

        with self.assertRaisesRegex(ValueError, "non-overlapping"):
            SurveyFactory(self.data_root).plan_relative_realizations(
                "ztf",
                tstart=60_000.0,
                tstop=60_000.4,
                count=3,
            )

    def test_ztf_observed_pairs_never_expand_to_multiindex_cross_pairs(self):
        """Observed ZTF keys must remain exact pairs in fields, matches, and masks."""

        import healpy as hp
        from ztffields.fields import Fields

        data = pd.DataFrame(
            {
                "mjd": [1.0, 2.0],
                "band": ["ztfg", "ztfr"],
                "skynoise": [1.0, 1.0],
                "gain": [1.0, 1.0],
                "zp": [25.0, 25.0],
                "fieldid": [375, 1],
                "rcid": [0, 60],
            }
        )
        survey = _ztf_survey(data)
        self.assertEqual(
            survey.get_fields(observed=True).index.tolist(),
            [(375, 0), (1, 60)],
        )

        cross_geometry = Fields.get_field_geometry(level="quadrant").loc[
            (375, 60), "geometry"
        ]
        cross_point = cross_geometry.representative_point()
        cross_radec = pd.DataFrame(
            {"ra": [cross_point.x], "dec": [cross_point.y]},
            index=pd.Index([41], name="target"),
        )
        matches = survey.radec_to_fieldid(cross_radec)
        self.assertTrue(matches.empty)
        self.assertEqual(matches.columns.tolist(), ["fieldid", "rcid"])

        regions, _ = _observed_instrument_regions(survey, label="ztf")
        pixels, _ = _proposal_pixel_ids(regions, nside=256)
        cross_pixel = hp.ang2pix(
            256,
            np.radians(90.0 - cross_point.y),
            np.radians(cross_point.x % 360.0),
        )
        self.assertNotIn(int(cross_pixel), set(map(int, pixels)))

    def test_near_pole_ztf_quadrant_has_tight_complete_proposal(self):
        """A real polar ZTF quadrant must draw from a non-catastrophic mask."""

        import healpy as hp

        data = pd.DataFrame(
            {
                "mjd": [1.0],
                "band": ["ztfg"],
                "skynoise": [1.0],
                "gain": [1.0],
                "zp": [25.0],
                "fieldid": [1894],
                "rcid": [40],
            }
        )
        survey = _ztf_survey(data)
        regions, _ = _observed_instrument_regions(survey, label="ztf")
        adopted = _adopted_observed_region(regions, label="ztf")
        area_deg2 = _cylindrical_equal_area_deg2(adopted)
        proposal, _ = _proposal_pixel_ids(regions, nside=64)
        proposal_area_deg2 = len(proposal) * hp.nside2pixarea(
            64,
            degrees=True,
        )
        self.assertAlmostEqual(area_deg2, 0.745, delta=0.02)
        self.assertLess(proposal_area_deg2, 100.0)

        # Exhaust every relevant nside=64 pixel center around the north pole
        # and verify that exact matcher support is contained in the proposal.
        candidate_pixels = hp.query_disc(
            64,
            hp.ang2vec(0.0, 0.0),
            np.radians(3.0),
            inclusive=True,
        )
        theta, phi = hp.pix2ang(64, candidate_pixels)
        candidate_radec = pd.DataFrame(
            {
                "ra": np.degrees(phi),
                "dec": 90.0 - np.degrees(theta),
            }
        )
        matches = survey.radec_to_fieldid(candidate_radec)
        matched_pixels = candidate_pixels[
            matches.index.unique().to_numpy(dtype=int)
        ]
        self.assertGreater(len(matched_pixels), 0)
        self.assertEqual(len(np.setdiff1d(matched_pixels, proposal)), 0)

        skyarea = ObservedSkyArea(
            survey,
            proposal,
            nside=64,
            label="ztf",
            area_deg2=area_deg2,
        )
        ra, dec = skyarea.draw_radec(
            size=512,
            rng=np.random.default_rng(1),
        )
        drawn_matches = survey.radec_to_fieldid(
            pd.DataFrame({"ra": ra, "dec": dec})
        )
        self.assertEqual(drawn_matches.index.nunique(), 512)

    def test_ztf_filter_and_quality_selection_happen_during_scan(self):
        """Requested filter and clean-bit constraints must reduce selected rows."""

        survey = SurveyFactory(self.data_root).create(
            "ztf",
            tstart=60_000.0,
            tstop=60_001.0,
            options={
                "ztf_filters": ["g"],
                "ztf_clean_only": True,
                "nside": 64,
            },
        )
        self.assertEqual(len(survey.data), 1)
        self.assertEqual(survey.data["band"].iloc[0], "ztfg")

    def test_lsst_sql_loading_supports_filters_and_degree_coordinates(self):
        """LSST selections must execute in SQL and retain the v5 degree unit."""

        survey = SurveyFactory(self.data_root).create(
            "lsst",
            tstart=60_000.0,
            tstop=60_001.0,
            options={
                "lsst_filters": ["g", "r"],
                "lsst_sql_where": "night = 0",
                "nside": 64,
            },
        )
        self.assertEqual(survey.data["band"].tolist(), ["lsstg", "lsstr"])
        np.testing.assert_allclose(survey.data["mjd"], [60_000.0, 60_000.25])
        self.assertEqual(
            survey.warp_provenance["sources"]["lsst"]["coordinate_input_unit"],
            "degree",
        )
        self.assertEqual(survey.fieldids.names, ["fieldid"])
        geometry_provenance = survey.warp_provenance["footprint_geometry"]
        self.assertEqual(
            geometry_provenance["field_geometry_method"],
            "warp_canonical_periodic_radec_v2",
        )
        self.assertEqual(
            geometry_provenance["periodic_match_ra_offsets_deg"],
            [-360.0, 0.0, 360.0],
        )
        self.assertEqual(geometry_provenance["polar_cap_count"], 0)

    def test_matcher_and_draws_are_periodic_across_the_ra_seam(self):
        """Targets on both RA branches must match and draw from one seam field."""

        database = (
            self.data_root / "lsst_data" / "baseline_v5.0.0_10yrs.db"
        )
        with sqlite3.connect(database) as connection:
            connection.execute(
                "INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (98, 0.0, 0.0, 61_000.5, 24.5, "g", 0, "SEAM", 98),
            )
        survey = SurveyFactory(self.data_root).create(
            "lsst",
            tstart=60_000.0,
            tstop=60_001.0,
            options={
                "lsst_sql_where": "observationId = 98",
                "nside": 128,
            },
        )
        seam_targets = pd.DataFrame(
            {"ra": [359.5, 0.5], "dec": [0.0, 0.0]},
            index=pd.Index([101, 202], name="target"),
        )
        matches = survey.radec_to_fieldid(seam_targets)
        self.assertEqual(matches.index.tolist(), [101, 202])
        self.assertEqual(matches["fieldid"].nunique(), 1)

        ra, dec = survey.simulation_skyarea.draw_radec(
            size=512,
            rng=np.random.default_rng(123),
        )
        self.assertTrue((ra < 1.0).any())
        self.assertTrue((ra > 359.0).any())
        drawn_matches = survey.radec_to_fieldid(
            pd.DataFrame({"ra": ra, "dec": dec})
        )
        self.assertEqual(drawn_matches.index.nunique(), 512)

    def test_proposal_mask_includes_polygon_edge_pixels_without_centers(self):
        """Inclusive triangle queries must retain every pixel touched at an edge."""

        import geopandas as gpd
        import healpy as hp
        from shapely.geometry import box

        polygon = box(10.0, 0.01, 10.25, 0.04)
        fields = gpd.GeoDataFrame(
            {"instrument": ["lsst"]},
            geometry=[polygon],
            index=pd.Index([7], name="fieldid"),
        )
        data = pd.DataFrame(
            {
                "mjd": [1.0],
                "band": ["lsstg"],
                "skynoise": [1.0],
                "gain": [1.0],
                "zp": [25.0],
                "fieldid": [7],
            }
        )
        survey = WarpPolygonSurvey.create(data, fields)
        regions, _ = _observed_instrument_regions(survey, label="lsst")
        proposal, metadata = _proposal_pixel_ids(regions, nside=256)
        coordinates = np.asarray(polygon.exterior.coords)
        vertex_pixels = hp.ang2pix(
            256,
            np.radians(90.0 - coordinates[:, 1]),
            np.radians(coordinates[:, 0] % 360.0),
        )
        theta, phi = hp.pix2ang(256, proposal)
        center_ra = np.degrees(phi)
        center_dec = 90.0 - np.degrees(theta)
        centers_inside = (
            (center_ra >= polygon.bounds[0])
            & (center_ra <= polygon.bounds[2])
            & (center_dec >= polygon.bounds[1])
            & (center_dec <= polygon.bounds[3])
        )
        self.assertFalse(centers_inside.any())
        self.assertTrue(set(map(int, vertex_pixels)).issubset(set(proposal)))
        self.assertGreater(len(np.unique(vertex_pixels)), 1)
        self.assertGreater(
            metadata["rasterization"]["lsst"]["triangle_count"],
            0,
        )

    def test_high_latitude_planar_polygon_is_conservative_at_every_nside(self):
        """Cap-guarded rasterization must retain planar latitude-edge interiors."""

        import healpy as hp
        from shapely.geometry import box

        region = _periodic_polygon_union([box(10.0, 60.0, 30.0, 70.0)])
        ra_grid, dec_grid = np.meshgrid(
            np.linspace(10.0, 30.0, 161),
            np.linspace(60.0, 70.0, 81),
        )
        for nside in (64, 256, 1024):
            proposal, statistics = _proposal_pixel_ids(
                {"lsst": region},
                nside=nside,
            )
            interior_pixels = hp.ang2pix(
                nside,
                np.radians(90.0 - dec_grid.ravel()),
                np.radians(ra_grid.ravel()),
            )
            missing = np.setdiff1d(
                np.unique(interior_pixels),
                proposal,
                assume_unique=True,
            )
            self.assertEqual(
                len(missing),
                0,
                msg=f"nside={nside} omitted planar polygon pixels",
            )
            self.assertGreater(
                statistics["rasterization"]["lsst"][
                    "bounding_cap_guard_components"
                ],
                0,
            )

    def test_polar_lsst_field_uses_one_canonical_cap_everywhere(self):
        """Polar matching, area, and proposal must share the densified cap."""

        import healpy as hp

        database = (
            self.data_root / "lsst_data" / "baseline_v5.0.0_10yrs.db"
        )
        with sqlite3.connect(database) as connection:
            connection.execute(
                "INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    99,
                    140.98318,
                    -89.67761,
                    61_000.5,
                    24.5,
                    "g",
                    0,
                    "POLAR",
                    99,
                ),
            )
        survey = SurveyFactory(self.data_root).create(
            "lsst",
            tstart=60_000.0,
            tstop=60_001.0,
            options={
                "lsst_sql_where": "observationId = 99",
                "nside": 512,
            },
        )
        canonical = survey.get_fields(observed=True).geometry.iloc[0]
        self.assertTrue(canonical.is_valid)
        self.assertAlmostEqual(canonical.bounds[1], -90.0)
        self.assertEqual(
            survey.warp_provenance["footprint_geometry"]["polar_cap_count"],
            1,
        )

        regions, _ = _observed_instrument_regions(survey, label="lsst")
        adopted = _adopted_observed_region(regions, label="lsst")
        area_deg2 = _cylindrical_equal_area_deg2(adopted)
        self.assertAlmostEqual(area_deg2, 9.574, delta=0.03)
        proposal = survey.simulation_skyarea.proposal_pixel_ids

        candidate_pixels = hp.query_disc(
            512,
            hp.ang2vec(np.pi, 0.0),
            np.radians(3.0),
            inclusive=True,
        )
        theta, phi = hp.pix2ang(512, candidate_pixels)
        candidate_radec = pd.DataFrame(
            {
                "ra": np.degrees(phi),
                "dec": 90.0 - np.degrees(theta),
            }
        )
        matches = survey.radec_to_fieldid(candidate_radec)
        matched_pixels = candidate_pixels[
            matches.index.unique().to_numpy(dtype=int)
        ]
        self.assertGreater(len(matched_pixels), 650)
        self.assertEqual(
            len(np.setdiff1d(matched_pixels, proposal)),
            0,
        )
        matched_area = len(matched_pixels) * hp.nside2pixarea(
            512,
            degrees=True,
        )
        self.assertAlmostEqual(matched_area, area_deg2, delta=0.25)

        skyarea = ObservedSkyArea(
            survey,
            proposal,
            nside=512,
            label="lsst",
            area_deg2=area_deg2,
        )
        ra, dec = skyarea.draw_radec(
            size=64,
            rng=np.random.default_rng(911),
        )
        drawn_matches = survey.radec_to_fieldid(
            pd.DataFrame({"ra": ra, "dec": dec})
        )
        self.assertEqual(drawn_matches.index.nunique(), 64)

    def test_absolute_mode_preserves_original_mjd(self):
        """Absolute time mode must select and retain the source timestamps."""

        survey = SurveyFactory(self.data_root).create(
            "ztf",
            tstart=58_000.0,
            tstop=58_000.25,
            options={"time_mode": "absolute", "nside": 64},
        )
        np.testing.assert_allclose(survey.data["mjd"], [58_000.0, 58_000.25])
        np.testing.assert_allclose(
            survey.data["source_mjd"],
            survey.data["mjd"],
        )

    def test_narrow_combined_overlap_is_proposed_and_exactly_validated(self):
        """A joint sliver without a pixel center or field centroid must remain usable."""

        import geopandas as gpd
        import healpy as hp
        from shapely.geometry import Point, box

        ztf_polygon = box(10.00, 0.00, 10.06, 0.06)
        lsst_polygon = box(10.04, 0.00, 10.10, 0.06)
        fields = gpd.GeoDataFrame(
            {"instrument": ["ztf", "lsst"]},
            geometry=[ztf_polygon, lsst_polygon],
            index=pd.Index([1, 2], name="fieldid"),
        )
        data = pd.DataFrame(
            {
                "mjd": [1.0, 2.0],
                "band": ["ztfg", "lsstg"],
                "skynoise": [1.0, 1.0],
                "gain": [1.0, 1.0],
                "zp": [25.0, 25.0],
                "fieldid": [1, 2],
            }
        )
        survey = CombinedSurvey(
            WarpPolygonSurvey.create(data, fields),
            fields["instrument"].to_dict(),
        )
        regions, _ = _observed_instrument_regions(survey, label="combined")
        adopted = _adopted_observed_region(regions, label="combined")
        pixels, metadata = _proposal_pixel_ids(regions, nside=256)
        area_deg2 = _cylindrical_equal_area_deg2(adopted)
        self.assertGreater(area_deg2, 0.0)
        self.assertGreater(len(pixels), 0)
        self.assertGreaterEqual(
            min(metadata["instrument_pixel_counts"].values()),
            len(pixels),
        )
        theta, phi = hp.pix2ang(256, pixels)
        self.assertFalse(
            any(
                adopted.covers(Point(ra, dec))
                for ra, dec in zip(
                    np.degrees(phi),
                    90.0 - np.degrees(theta),
                )
            )
        )
        representatives = fields.geometry.representative_point()
        representative_matches = survey.radec_to_fieldid(
            pd.DataFrame(
                {
                    "ra": representatives.x.to_numpy(),
                    "dec": representatives.y.to_numpy(),
                }
            )
        )
        self.assertTrue(representative_matches.empty)

        skyarea = ObservedSkyArea(
            survey,
            pixels,
            nside=256,
            label="combined",
            area_deg2=area_deg2,
        )
        empty_ra, empty_dec = skyarea.draw_radec(
            size=0,
            rng=np.random.default_rng(11),
        )
        self.assertEqual(len(empty_ra), 0)
        self.assertEqual(len(empty_dec), 0)
        ra, dec = skyarea.draw_radec(
            size=16,
            rng=np.random.default_rng(12),
        )
        matches = survey.radec_to_fieldid(
            pd.DataFrame({"ra": ra, "dec": dec})
        )
        self.assertEqual(matches.index.nunique(), 16)
        self.assertTrue(
            matches.groupby(level=0)["fieldid"].nunique().eq(2).all()
        )

    def test_combined_survey_requires_exact_overlap_and_retains_both_errors(self):
        """One joint light curve must contain both instruments with distinct errors."""

        from skysurvey import DataSet

        survey = SurveyFactory(self.data_root).create(
            "combined",
            tstart=60_000.0,
            tstop=60_001.0,
            options={"nside": 64},
        )
        inside = pd.DataFrame(
            {"ra": [self.overlap_ra], "dec": [self.overlap_dec]}
        )
        outside = pd.DataFrame(
            {"ra": [self.overlap_ra + 90.0], "dec": [self.overlap_dec]}
        )
        self.assertEqual(len(survey.radec_to_fieldid(inside)), 2)
        self.assertTrue(survey.radec_to_fieldid(outside).empty)
        self.assertTrue(survey.warp_provenance["synthetic"])

        dataset = DataSet.from_targets_and_survey(
            _SingleTarget(self.overlap_ra, self.overlap_dec),
            survey,
            incl_error=False,
            phase_range=None,
        )
        observations = dataset.data.reset_index()
        self.assertTrue(observations["band"].str.startswith("ztf").any())
        self.assertTrue(observations["band"].str.startswith("lsst").any())
        ztf_error = observations.loc[
            observations["band"].eq("ztfg"), "fluxerr"
        ].iloc[0]
        lsst_error = observations.loc[
            observations["band"].eq("lsstg"), "fluxerr"
        ].iloc[0]
        self.assertNotEqual(float(ztf_error), float(lsst_error))

    def test_periodic_equal_area_handles_union_intersection_and_wraparound(self):
        """CEA area must agree with analytic longitude-latitude rectangles."""

        import shapely
        from shapely.geometry import Polygon, box

        rectangle = _periodic_polygon_union([box(10.0, -10.0, 20.0, 10.0)])
        expected_rectangle = (
            np.radians(10.0)
            * (np.sin(np.radians(10.0)) - np.sin(np.radians(-10.0)))
            * np.square(180.0 / np.pi)
        )
        self.assertAlmostEqual(
            _cylindrical_equal_area_deg2(rectangle),
            expected_rectangle,
            places=8,
        )

        union = _periodic_polygon_union(
            [box(10.0, -10.0, 20.0, 10.0), box(20.0, -10.0, 30.0, 10.0)]
        )
        self.assertAlmostEqual(
            _cylindrical_equal_area_deg2(union),
            2.0 * expected_rectangle,
            places=8,
        )
        overlap = shapely.intersection(
            _periodic_polygon_union([box(10.0, -10.0, 25.0, 10.0)]),
            _periodic_polygon_union([box(20.0, -10.0, 30.0, 10.0)]),
        )
        self.assertAlmostEqual(
            _cylindrical_equal_area_deg2(overlap),
            0.5 * expected_rectangle,
            places=8,
        )

        wrapped = Polygon(
            [(359.0, -5.0), (1.0, -5.0), (1.0, 5.0), (359.0, 5.0)]
        )
        wrapped_region = _periodic_polygon_union([wrapped])
        expected_wrapped = (
            np.radians(2.0)
            * (np.sin(np.radians(5.0)) - np.sin(np.radians(-5.0)))
            * np.square(180.0 / np.pi)
        )
        self.assertAlmostEqual(
            _cylindrical_equal_area_deg2(wrapped_region),
            expected_wrapped,
            places=8,
        )

        wide_region = _periodic_polygon_union(
            [box(10.0, -5.0, 300.0, 5.0)]
        )
        expected_wide = (
            np.radians(290.0)
            * (np.sin(np.radians(5.0)) - np.sin(np.radians(-5.0)))
            * np.square(180.0 / np.pi)
        )
        self.assertAlmostEqual(
            _cylindrical_equal_area_deg2(wide_region),
            expected_wide,
            places=8,
        )

    def test_area_and_poisson_count_are_invariant_to_proposal_nside(self):
        """Changing proposal resolution must not alter rate normalization."""

        low_resolution = SurveyFactory(self.data_root).create(
            "ztf",
            tstart=60_000.0,
            tstop=60_001.0,
            options={"nside": 32},
        )
        high_resolution = SurveyFactory(self.data_root).create(
            "ztf",
            tstart=60_000.0,
            tstop=60_001.0,
            options={"nside": 128},
        )
        self.assertAlmostEqual(
            low_resolution.simulation_skyarea.area_deg2,
            high_resolution.simulation_skyarea.area_deg2,
            places=12,
        )
        self.assertIs(
            low_resolution.simulation_skyarea.fieldids,
            low_resolution.simulation_skyarea.proposal_pixel_ids,
        )
        self.assertEqual(
            low_resolution.warp_provenance["footprint_geometry"]["area_method"],
            "periodic_ra_polygon_union_or_intersection_then_"
            "cylindrical_equal_area_lambda_sin_phi",
        )

        full_sky_deg2 = 4.0 * np.pi * np.square(180.0 / np.pi)
        low_expectation = observer_expected_count(
            rate=10_000.0,
            zmin=0.0,
            zmax=0.05,
            nyears=1.0,
            sky_fraction=(
                low_resolution.simulation_skyarea.area_deg2 / full_sky_deg2
            ),
        )
        high_expectation = observer_expected_count(
            rate=10_000.0,
            zmin=0.0,
            zmax=0.05,
            nyears=1.0,
            sky_fraction=(
                high_resolution.simulation_skyarea.area_deg2 / full_sky_deg2
            ),
        )
        self.assertEqual(low_expectation, high_expectation)
        self.assertEqual(
            _draw_poisson_count(low_expectation, rng=987),
            _draw_poisson_count(high_expectation, rng=987),
        )

    def test_missing_source_and_out_of_range_windows_are_actionable(self):
        """Missing files and impossible source windows must raise clear errors."""

        with self.assertRaises(FileNotFoundError):
            SurveyFactory(self.data_root).create(
                "ztf",
                tstart=60_000.0,
                tstop=60_001.0,
                options={"ztf_path": self.data_root / "missing.parquet"},
            )
        with self.assertRaisesRegex(ValueError, "outside available range"):
            SurveyFactory(self.data_root).create(
                "ztf",
                tstart=60_000.0,
                tstop=60_100.0,
            )

    @requires_coefficients("SN IIP")
    def test_runner_end_to_end_for_every_automatic_survey_mode(self):
        """The runner must automatically load and simulate all three survey modes."""

        coefficients = COEFFICIENT_DIR

        common_options = {
            "ztf_path": (
                self.data_root
                / "ztf_data"
                / "logs"
                / "ztf_obsfile_maglimcat.parquet"
            ),
            "lsst_path": (
                self.data_root
                / "lsst_data"
                / "baseline_v5.0.0_10yrs.db"
            ),
            "nside": 64,
        }
        with tempfile.TemporaryDirectory() as output:
            runner = WarpSimulationRunner(
                coefficients,
                source_cache_dir=Path(output) / "source-cache",
            )
            manifests = {}
            for index, name in enumerate(("ztf", "lsst", "combined")):
                spec = WarpSampleSpec(
                    run_name=f"automatic_{name}",
                    active_fitclasses=["SN IIP"],
                    size=1,
                    zmax=0.01,
                    tstart=60_000.0,
                    tstop=60_001.0,
                    class_sampling="balanced",
                    redshift_sampling="uniform",
                    color_mode=None,
                    survey_name=name,
                    survey_options=common_options,
                    batch_size=1,
                    incl_error=False,
                    seed=100 + index,
                )
                manifests[name] = runner.run(spec, output)

        for name, manifest in manifests.items():
            self.assertEqual(manifest["survey"]["name"], name)
            self.assertGreater(manifest["observation_rows"], 0)
        self.assertTrue(manifests["combined"]["survey"]["synthetic"])


if __name__ == "__main__":
    unittest.main()
