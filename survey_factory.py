"""Load local ZTF and LSST cadences into SkySurvey-compatible surveys."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


_ZTF_COLUMNS = [
    "expMJD",
    "filter",
    "fieldID",
    "fieldRA",
    "fieldDec",
    "rcid",
    "maglimcat",
    "zp",
    "gain",
    "expid",
    "infobits",
]
_LSST_REQUIRED_COLUMNS = {
    "observationId",
    "fieldRA",
    "fieldDec",
    "observationStartMJD",
    "fiveSigmaDepth",
}
_LSST_FIELD_OFFSET = 1_000_000
_AREA_EDGE_STEP_DEG = 0.1
_MAX_TRIANGULATION_VERTICES = 128
_ORDINARY_SEAM_WINDOW_DEG = 30.0


def _json_safe(value: Any) -> Any:
    """Convert nested configuration values into deterministic JSON-safe values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _default_data_root() -> Path:
    """Return the configured or repository-local survey data directory."""

    configured = os.environ.get("WARP_TEMPLATE_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parent.parent / "data"


def _file_provenance(path: Path) -> dict[str, Any]:
    """Return inexpensive source-file identity information for manifests."""

    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _normalize_filter_names(
    values: Optional[Sequence[str]],
    *,
    prefix: str,
) -> Optional[tuple[str, ...]]:
    """Normalize short or instrument-prefixed filter names."""

    if values is None:
        return None
    normalized = []
    for value in values:
        name = str(value).lower()
        normalized.append(name if name.startswith(prefix) else f"{prefix}{name}")
    return tuple(dict.fromkeys(normalized))


def _coordinates_to_degrees(
    ra: pd.Series,
    dec: pd.Series,
    *,
    unit: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Convert radian coordinates when their finite range identifies that unit."""

    ra_values = pd.to_numeric(ra, errors="coerce").to_numpy(dtype=float)
    dec_values = pd.to_numeric(dec, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(ra_values) & np.isfinite(dec_values)
    if not finite.any():
        return ra_values, dec_values, "unknown"

    # Current ZTF logs store radians, while the LSST v5 database stores degrees.
    if unit not in {None, "radian", "degree"}:
        raise ValueError("coordinate unit must be radian, degree, or None")
    radians = unit == "radian" or (
        unit is None
        and np.nanmax(np.abs(ra_values[finite])) <= 2.0 * np.pi + 0.05
        and np.nanmax(np.abs(dec_values[finite])) <= np.pi / 2.0 + 0.05
    )
    if radians:
        return np.degrees(ra_values), np.degrees(dec_values), "radian"
    return ra_values, dec_values, "degree"


def _coordinate_unit_from_bounds(
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
) -> str:
    """Infer coordinate units from unfiltered source-wide extrema."""

    if (
        max(abs(float(ra_min)), abs(float(ra_max))) <= 2.0 * np.pi + 0.05
        and max(abs(float(dec_min)), abs(float(dec_max))) <= np.pi / 2.0 + 0.05
    ):
        return "radian"
    return "degree"


def _skynoise_from_maglimit(
    maglimit: pd.Series | np.ndarray,
    zeropoint: pd.Series | np.ndarray | float,
) -> np.ndarray:
    """Convert a five-sigma limiting magnitude into background flux noise."""

    maglimit_values = np.asarray(maglimit, dtype=float)
    zeropoint_values = np.asarray(zeropoint, dtype=float)
    return np.power(10.0, -0.4 * (maglimit_values - zeropoint_values)) / 5.0


@dataclass(frozen=True)
class SurveyConfig:
    """Validated configuration for one local pandas-backed survey import."""

    name: str
    time_mode: str = "relative"
    ztf_path: Optional[Path] = None
    lsst_path: Optional[Path] = None
    ztf_filters: Optional[tuple[str, ...]] = None
    lsst_filters: Optional[tuple[str, ...]] = None
    ztf_source_mjd_start: Optional[float] = None
    lsst_source_mjd_start: Optional[float] = None
    ztf_clean_only: bool = False
    lsst_sql_where: Optional[str] = None
    lsst_zp: float = 30.0
    lsst_gain: float = 1.0
    nside: int = 64
    backend: str = "pandas"
    # This controls Arrow scan chunks, not target simulation batches.
    parquet_batch_size: int = 262_144

    def __post_init__(self) -> None:
        """Reject unsupported names, modes, and unsafe loader settings."""

        if self.name not in {"ztf", "lsst", "combined"}:
            raise ValueError("survey name must be 'ztf', 'lsst', or 'combined'")
        if self.time_mode not in {"relative", "absolute"}:
            raise ValueError("survey time_mode must be 'relative' or 'absolute'")
        if self.nside <= 0 or self.nside & (self.nside - 1):
            raise ValueError("survey nside must be a positive power of two")
        if self.backend != "pandas":
            raise ValueError("survey backend must be pandas")
        if self.parquet_batch_size <= 0:
            raise ValueError("parquet_batch_size must be positive")
        if self.lsst_gain <= 0:
            raise ValueError("lsst_gain must be positive")
        if self.lsst_sql_where and ";" in self.lsst_sql_where:
            raise ValueError("lsst_sql_where must not contain SQL statement separators")

    @classmethod
    def from_options(
        cls,
        name: str,
        options: Optional[Mapping[str, Any]] = None,
        *,
        data_root: Optional[str | Path] = None,
    ) -> "SurveyConfig":
        """Build a config from JSON-style options and repository-local defaults."""

        options = dict(options or {})
        known = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = sorted(set(options) - (known - {"name"}))
        if unknown:
            raise ValueError(f"unknown survey_options: {', '.join(unknown)}")

        root = (
            Path(data_root).expanduser().resolve()
            if data_root is not None
            else _default_data_root()
        )
        options.setdefault(
            "ztf_path",
            root / "ztf_data" / "logs" / "ztf_obsfile_maglimcat.parquet",
        )
        options.setdefault(
            "lsst_path",
            root / "lsst_data" / "baseline_v5.0.0_10yrs.db",
        )
        for key in ("ztf_path", "lsst_path"):
            if options[key] is not None:
                options[key] = Path(options[key]).expanduser().resolve()
        options["ztf_filters"] = _normalize_filter_names(
            options.get("ztf_filters"),
            prefix="ztf",
        )
        lsst_filters = options.get("lsst_filters")
        if lsst_filters is not None:
            options["lsst_filters"] = tuple(
                str(value).lower().removeprefix("lsst") for value in lsst_filters
            )
        return cls(name=str(name).lower(), **options)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe configuration representation."""

        return _json_safe(asdict(self))


@dataclass(frozen=True)
class SurveyRealizationSpec:
    """Describe one bounded cadence window in a survey ensemble."""

    realization_id: str
    index: int
    count: int
    survey_name: str
    requested_mjd_range: tuple[float, float]
    source_mjd_ranges: Mapping[str, tuple[float, float]]
    survey_options: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation for ensemble manifests."""

        return _json_safe(asdict(self))


def _observed_field_index(survey: Any) -> pd.Index:
    """Return the exact field-key rows present in the observing cadence."""

    names = list(survey.fieldids.names)
    observed = survey.data[names].drop_duplicates(ignore_index=True)
    if len(names) == 1:
        return pd.Index(observed[names[0]].to_numpy(), name=names[0])
    return pd.MultiIndex.from_frame(observed, names=names)


def _filter_matches_to_observed_fields(
    matches: pd.DataFrame,
    observed_index: pd.Index,
) -> pd.DataFrame:
    """Remove polygon matches whose complete field key was never observed."""

    if matches.empty:
        return matches
    names = list(observed_index.names)
    if len(names) == 1:
        keep = matches[names[0]].isin(observed_index)
    else:
        match_index = pd.MultiIndex.from_frame(matches[names], names=names)
        keep = match_index.isin(observed_index)
    return matches.loc[np.asarray(keep, dtype=bool)]


def _empty_field_matches(survey: Any, radec: Any) -> pd.DataFrame:
    """Return a typed empty matcher result compatible with SkySurvey."""

    names = list(survey.fieldids.names)
    result = pd.DataFrame(
        {
            name: pd.Series(dtype=survey.data[name].dtype)
            for name in names
        }
    )
    if isinstance(radec, pd.DataFrame):
        result.index = radec.index[:0].rename("index_radec")
    else:
        result.index = pd.RangeIndex(0, name="index_radec")
    return result


def _radec_as_frame(radec: Any) -> pd.DataFrame:
    """Normalize supported coordinate inputs while preserving target indices."""

    if isinstance(radec, pd.DataFrame):
        missing = {"ra", "dec"} - set(radec.columns)
        if missing:
            raise ValueError("radec DataFrame must contain ra and dec columns")
        return radec.copy()
    values = np.asarray(radec)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"shape of radec must be (N, 2), {values.shape} given")
    return pd.DataFrame(values, columns=["ra", "dec"])


def _periodic_radec_copies(
    radec: Any,
) -> tuple[pd.DataFrame, pd.Index, int]:
    """Create unique-index coordinate copies at RA, RA-360, and RA+360."""

    frame = _radec_as_frame(radec)
    source_index = frame.index.copy()
    size = len(frame)
    copies = []
    for offset in (0.0, -360.0, 360.0):
        shifted = frame.copy()
        shifted["ra"] = shifted["ra"].to_numpy(dtype=float) + offset
        copies.append(shifted)
    return pd.concat(copies, ignore_index=True), source_index, size


def _polar_cap_from_unwrapped_ring(
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    *,
    pole: float,
) -> Any:
    """Close a once-winding projected boundary against its enclosed pole."""

    from shapely.geometry import Polygon

    if longitudes[-1] < longitudes[0]:
        longitudes = longitudes[::-1]
        latitudes = latitudes[::-1]
    midpoint = 0.5 * (longitudes[0] + longitudes[-1])
    longitudes = longitudes - np.floor(midpoint / 360.0) * 360.0
    boundary = np.column_stack([longitudes, latitudes])
    coordinates = np.vstack(
        [
            boundary,
            [longitudes[-1], pole],
            [longitudes[0], pole],
            boundary[0],
        ]
    )
    return Polygon(coordinates)


def _canonicalize_ordinary_seam_polygon(polygon: Any) -> Any:
    """Unwrap a small 0/360 seam crossing without changing wide sky regions."""

    from shapely.geometry import Polygon

    min_ra, min_dec, max_ra, max_dec = polygon.bounds
    polar_cap = (
        (min_dec <= -90.0 + 1.0e-10 or max_dec >= 90.0 - 1.0e-10)
        and max_ra - min_ra >= 360.0 - 1.0e-8
    )
    ordinary_seam = (
        not polar_cap
        and min_ra >= 0.0
        and max_ra <= 360.0
        and min_ra < _ORDINARY_SEAM_WINDOW_DEG
        and max_ra > 360.0 - _ORDINARY_SEAM_WINDOW_DEG
    )
    if not ordinary_seam:
        return polygon

    exterior = np.asarray(polygon.exterior.coords, dtype=float)
    longitude = np.degrees(np.unwrap(np.radians(exterior[:, 0])))
    if np.ptp(longitude) > _ORDINARY_SEAM_WINDOW_DEG:
        return polygon
    midpoint = 0.5 * (np.min(longitude) + np.max(longitude))
    longitude -= np.floor(midpoint / 360.0) * 360.0
    exterior = np.column_stack([longitude, exterior[:, 1]])
    holes = []
    exterior_center = float(np.mean(longitude))
    for interior in polygon.interiors:
        hole = np.asarray(interior.coords, dtype=float)
        hole_longitude = np.degrees(
            np.unwrap(np.radians(hole[:, 0]))
        )
        hole_longitude -= (
            np.round(
                (np.mean(hole_longitude) - exterior_center) / 360.0
            )
            * 360.0
        )
        holes.append(np.column_stack([hole_longitude, hole[:, 1]]))
    candidate = Polygon(exterior, holes)
    return candidate if candidate.is_valid else polygon


def _canonicalize_invalid_field_geometry(geometry: Any) -> Any:
    """Repair an invalid field once, recognizing projected polar windings."""

    import shapely

    repaired = []
    for polygon in _iter_polygons(geometry):
        coordinates = np.asarray(polygon.exterior.coords, dtype=float)
        unwrapped = np.degrees(
            np.unwrap(np.radians(coordinates[:, 0]))
        )
        winding = (unwrapped[-1] - unwrapped[0]) / 360.0
        if np.isclose(abs(winding), 1.0, atol=1.0e-6):
            pole = 90.0 if np.mean(coordinates[:, 1]) > 0 else -90.0
            repaired.append(
                _polar_cap_from_unwrapped_ring(
                    unwrapped,
                    coordinates[:, 1],
                    pole=pole,
                )
            )
        else:
            repaired.append(shapely.make_valid(polygon))
    if not repaired:
        return shapely.make_valid(geometry)
    return shapely.union_all(repaired)


def _canonicalize_survey_fields(survey: Any) -> None:
    """Replace invalid field geometries once so every consumer sees one shape."""

    import shapely

    fields = survey.fields
    geometries = fields.geometry.to_numpy().copy()
    invalid = ~shapely.is_valid(geometries)
    bounds = shapely.bounds(geometries)
    ordinary_seam = (
        (bounds[:, 0] >= 0.0)
        & (bounds[:, 2] <= 360.0)
        & (bounds[:, 0] < _ORDINARY_SEAM_WINDOW_DEG)
        & (bounds[:, 2] > 360.0 - _ORDINARY_SEAM_WINDOW_DEG)
    )
    selected = invalid | ordinary_seam
    if not selected.any():
        return
    for index in np.flatnonzero(selected):
        geometry = geometries[index]
        if invalid[index]:
            geometry = _canonicalize_invalid_field_geometry(geometry)
        if geometry.geom_type == "Polygon":
            geometry = _canonicalize_ordinary_seam_polygon(geometry)
        geometries[index] = geometry
    canonical = fields.copy()
    canonical.geometry = geometries
    if not shapely.is_valid(canonical.geometry.to_numpy()).all():
        raise ValueError("could not canonicalize invalid survey field geometry")
    survey._fields = canonical


class ObservedFieldSurvey:
    """Delegate to SkySurvey while enforcing exact observed field keys."""

    def __init__(self, survey: Any):
        """Cache the exact observed index without modifying SkySurvey internals."""

        self._survey = survey
        _canonicalize_survey_fields(self._survey)
        self._observed_field_index = _observed_field_index(survey)

    def __getattr__(self, name: str) -> Any:
        """Delegate ordinary survey attributes to the wrapped survey."""

        return getattr(self._survey, name)

    def get_fields(self, observed: bool = True) -> Any:
        """Return all fields or only the exact keys represented in ``data``."""

        if not observed:
            return self.fields.copy()
        selected = self.fields.reindex(self._observed_field_index)
        if selected.geometry.isna().any():
            missing = selected.index[selected.geometry.isna()].tolist()
            raise KeyError(f"observed survey fields are missing geometry: {missing[:5]}")
        return selected.copy()

    def radec_to_fieldid(
        self,
        radec: Any,
        observed_fields: bool = False,
    ) -> pd.DataFrame:
        """Match polygons with SkySurvey, then retain exact observed keys."""

        # SkySurvey 0.31 expands MultiIndex levels as a Cartesian product in
        # get_fields(observed=True), so filter complete keys after assignment.
        # Its planar spatial join is not RA-periodic, hence the three unchanged
        # SkySurvey assignments on coordinate copies around the seam.
        if len(radec) == 0:
            return _empty_field_matches(self, radec)
        periodic_radec, source_index, source_size = _periodic_radec_copies(radec)
        try:
            matches = self._survey.radec_to_fieldid(
                periodic_radec,
                observed_fields=False,
            )
        except ValueError as error:
            if "at least one array to stack" not in str(error):
                raise
            return _empty_field_matches(self, radec)
        matches = _filter_matches_to_observed_fields(
            matches,
            self._observed_field_index,
        )
        if matches.empty:
            return _empty_field_matches(self, radec)

        field_names = list(self._observed_field_index.names)
        positions = matches.index.to_numpy(dtype=np.int64) % source_size
        matches = matches.copy()
        matches["_warp_target_position"] = positions
        matches = matches.drop_duplicates(
            subset=["_warp_target_position", *field_names],
            keep="first",
        ).sort_values("_warp_target_position", kind="stable")
        restored_positions = matches.pop("_warp_target_position").to_numpy(
            dtype=np.int64
        )
        matches.index = source_index.take(restored_positions)
        if not isinstance(matches.index, pd.MultiIndex):
            matches.index.name = "index_radec"
        return matches


class WarpPolygonSurvey:
    """Construct an exact-key SkySurvey PolygonSurvey with Warp provenance."""

    @staticmethod
    def create(
        data: pd.DataFrame,
        fields: Any,
        *,
        footprint: Any = None,
    ) -> Any:
        """Return a wrapped PolygonSurvey with an optional overall footprint."""

        from skysurvey.survey.polygon import PolygonSurvey

        survey = PolygonSurvey(data=data, fields=fields)
        survey._footprint = footprint
        return ObservedFieldSurvey(survey)


class CombinedSurvey(ObservedFieldSurvey):
    """Wrap a polygon survey and require every target to match both instruments."""

    def __init__(self, survey: Any, field_instruments: Mapping[int, str]):
        """Store the shared survey and its encoded field-to-instrument mapping."""

        base_survey = (
            survey._survey if isinstance(survey, ObservedFieldSurvey) else survey
        )
        super().__init__(base_survey)
        self._field_instruments = dict(field_instruments)

    def _matched_fields(self, radec: Any) -> pd.DataFrame:
        """Return exact observed polygon matches before joint qualification."""

        return super().radec_to_fieldid(radec)

    def radec_to_fieldid(
        self,
        radec: Any,
        observed_fields: bool = False,
    ) -> pd.DataFrame:
        """Return fields only for coordinates observed by both ZTF and LSST."""

        matches = self._matched_fields(radec)
        if matches.empty:
            return matches
        instruments = matches["fieldid"].map(self._field_instruments)
        qualified = (
            instruments.groupby(level=0)
            .nunique()
            .loc[lambda values: values >= 2]
            .index
        )
        matches = matches.loc[matches.index.isin(qualified)]
        return matches


class ObservedSkyArea:
    """Draw deterministic sky coordinates from an observed HEALPix mask."""

    def __init__(
        self,
        survey: Any,
        fieldids: Sequence[int],
        *,
        nside: int,
        label: str,
        area_deg2: Optional[float] = None,
    ):
        """Store a conservative proposal mask and exact survey validator."""

        import healpy as hp

        unique = np.unique(np.asarray(fieldids, dtype=np.int64))
        if not len(unique):
            raise ValueError(f"{label} has no observed sky overlap at nside={nside}")
        self.survey = survey
        self.fieldids = unique
        self.proposal_pixel_ids = self.fieldids
        self.nside = int(nside)
        self.label = str(label)
        pixel_area = float(len(unique) * hp.nside2pixarea(nside, degrees=True))
        if area_deg2 is None:
            self.area_deg2 = pixel_area
        else:
            resolved_area = float(area_deg2)
            if not np.isfinite(resolved_area) or resolved_area <= 0:
                raise ValueError("area_deg2 must be finite and positive")
            self.area_deg2 = resolved_area

    def __repr__(self) -> str:
        """Return a concise auditable footprint description."""

        return (
            f"ObservedSkyArea(label={self.label!r}, nside={self.nside}, "
            f"pixels={len(self.fieldids)}, area_deg2={self.area_deg2:.3f})"
        )

    def draw_radec(
        self,
        *,
        size: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw positions from mask pixels and retain exact survey matches."""

        import healpy as hp

        if size < 0:
            raise ValueError("size must be non-negative")
        if size == 0:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        accepted: list[np.ndarray] = []
        accepted_rows = 0
        proposal_area_deg2 = float(
            len(self.proposal_pixel_ids)
            * hp.nside2pixarea(self.nside, degrees=True)
        )
        geometric_acceptance = np.clip(
            self.area_deg2 / proposal_area_deg2,
            1.0e-8,
            1.0,
        )

        # Bound candidate work from the geometric fill fraction rather than
        # from an arbitrary iteration count. The generous factors include the
        # rejection from each HEALPix pixel's enclosing spherical cap.
        absolute_work_limit = max(2_000_000, int(size) * 5_000)
        estimated_work = int(
            np.ceil(float(size) / float(geometric_acceptance) * 32.0)
        )
        work_budget = min(
            absolute_work_limit,
            max(250_000, estimated_work),
        )
        candidate_work = 0
        while accepted_rows < size:
            remaining = size - accepted_rows
            remaining_budget = work_budget - candidate_work
            if remaining_budget <= 0:
                raise RuntimeError(
                    f"could not draw {size} exact positions from {self.label}; "
                    f"accepted {accepted_rows} after {candidate_work} "
                    f"candidates (estimated geometric acceptance "
                    f"{geometric_acceptance:.3g})"
                )
            draw_size = min(
                max(4096, remaining * 8),
                remaining_budget,
            )
            chosen_pixels = rng.choice(
                self.proposal_pixel_ids,
                size=draw_size,
                replace=True,
            )
            theta_center, phi_center = hp.pix2ang(
                self.nside,
                chosen_pixels,
            )
            dec_center = np.pi / 2.0 - theta_center
            radius = hp.max_pixrad(self.nside)
            angular_distance = np.arccos(
                1.0 - rng.random(draw_size) * (1.0 - np.cos(radius))
            )
            bearing = rng.uniform(0.0, 2.0 * np.pi, size=draw_size)

            # Sample uniformly in each pixel's enclosing spherical cap, then
            # reject cap points that fall into a neighboring HEALPix pixel.
            sin_dec = (
                np.sin(dec_center) * np.cos(angular_distance)
                + np.cos(dec_center)
                * np.sin(angular_distance)
                * np.cos(bearing)
            )
            dec = np.arcsin(np.clip(sin_dec, -1.0, 1.0))
            delta_ra = np.arctan2(
                np.sin(bearing) * np.sin(angular_distance) * np.cos(dec_center),
                np.cos(angular_distance)
                - np.sin(dec_center) * np.sin(dec),
            )
            ra = (phi_center + delta_ra) % (2.0 * np.pi)
            actual_pixels = hp.ang2pix(
                self.nside,
                np.pi / 2.0 - dec,
                ra,
            )
            in_pixel = actual_pixels == chosen_pixels
            candidates = pd.DataFrame(
                {
                    "ra": np.degrees(ra[in_pixel]),
                    "dec": np.degrees(dec[in_pixel]),
                }
            )
            if not candidates.empty:
                try:
                    matches = self.survey.radec_to_fieldid(
                        candidates,
                        observed_fields=True,
                    )
                except TypeError:
                    matches = self.survey.radec_to_fieldid(candidates)
                valid = matches.index.unique().to_numpy(dtype=int)
                if len(valid):
                    accepted.append(candidates.iloc[valid].to_numpy(dtype=float))
                    accepted_rows += len(valid)
            candidate_work += draw_size

            # Once exact matches exist, use their observed end-to-end rate to
            # extend an initially pessimistic budget, while keeping a hard cap
            # for inconsistent or empty matcher geometries.
            if accepted_rows and accepted_rows < size:
                observed_acceptance = accepted_rows / float(candidate_work)
                adaptive_work = int(
                    np.ceil(
                        candidate_work
                        + (size - accepted_rows)
                        / observed_acceptance
                        * 16.0
                    )
                )
                work_budget = min(
                    absolute_work_limit,
                    max(work_budget, adaptive_work),
                )

        result = np.concatenate(accepted, axis=0)[:size]
        return result[:, 0], result[:, 1]


def _source_window(
    *,
    full_range: tuple[float, float],
    requested_range: tuple[float, float],
    configured_start: Optional[float],
    time_mode: str,
    label: str,
) -> tuple[float, float, float]:
    """Resolve source selection bounds and the later time shift."""

    target_start, target_stop = requested_range
    duration = target_stop - target_start
    if time_mode == "absolute":
        source_start, source_stop = target_start, target_stop
        shift = 0.0
    else:
        source_start = (
            float(configured_start)
            if configured_start is not None
            else float(full_range[0])
        )
        source_stop = source_start + duration
        shift = target_start - source_start

    if source_start < full_range[0] or source_stop > full_range[1]:
        raise ValueError(
            f"{label} source window [{source_start}, {source_stop}] lies outside "
            f"available range [{full_range[0]}, {full_range[1]}]"
        )
    return float(source_start), float(source_stop), float(shift)


def _parquet_column_range(path: Path, column: str) -> tuple[float, float]:
    """Read a numeric Parquet column range from row-group statistics."""

    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    column_index = parquet.schema_arrow.get_field_index(column)
    minima: list[float] = []
    maxima: list[float] = []
    for index in range(parquet.num_row_groups):
        statistics = parquet.metadata.row_group(index).column(column_index).statistics
        if statistics is not None and statistics.has_min_max:
            minima.append(float(statistics.min))
            maxima.append(float(statistics.max))
    if not minima:
        raise ValueError(f"{path} has no Parquet statistics for {column}")
    return min(minima), max(maxima)


def _load_ztf(
    config: SurveyConfig,
    requested_range: tuple[float, float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and normalize a selected ZTF cadence directly from Parquet."""

    import pyarrow.dataset as ds

    path = config.ztf_path
    if path is None or not path.exists():
        raise FileNotFoundError(f"ZTF observing log not found: {path}")
    full_range = _parquet_column_range(path, "expMJD")
    source_start, source_stop, shift = _source_window(
        full_range=full_range,
        requested_range=requested_range,
        configured_start=config.ztf_source_mjd_start,
        time_mode=config.time_mode,
        label="ZTF",
    )

    dataset = ds.dataset(path, format="parquet")
    expression = (
        (ds.field("expMJD") >= source_start)
        & (ds.field("expMJD") <= source_stop)
    )
    if config.ztf_filters:
        expression &= ds.field("filter").isin(config.ztf_filters)
    if config.ztf_clean_only:
        expression &= ds.field("infobits") == 0
    table = dataset.scanner(
        columns=_ZTF_COLUMNS,
        filter=expression,
        batch_size=config.parquet_batch_size,
        use_threads=True,
    ).to_table()
    raw = table.to_pandas()
    if raw.empty:
        raise ValueError("the selected ZTF source window contains no observations")

    ra, dec, coordinate_unit = _coordinates_to_degrees(
        raw["fieldRA"],
        raw["fieldDec"],
    )
    normalized = pd.DataFrame(
        {
            "mjd": raw["expMJD"].to_numpy(dtype=float) + shift,
            "band": raw["filter"].astype(str).str.lower(),
            "skynoise": _skynoise_from_maglimit(raw["maglimcat"], raw["zp"]),
            "gain": pd.to_numeric(raw["gain"], errors="coerce"),
            "zp": pd.to_numeric(raw["zp"], errors="coerce"),
            "fieldid": pd.to_numeric(raw["fieldID"], errors="coerce"),
            "rcid": pd.to_numeric(raw["rcid"], errors="coerce"),
            "ra": ra,
            "dec": dec,
            "instrument": "ztf",
            "source_observation_id": pd.to_numeric(
                raw["expid"], errors="coerce"
            ),
            "limiting_magnitude": pd.to_numeric(
                raw["maglimcat"], errors="coerce"
            ),
            "infobits": pd.to_numeric(raw["infobits"], errors="coerce"),
            "source_mjd": pd.to_numeric(raw["expMJD"], errors="coerce"),
        }
    )
    finite_columns = [
        "mjd",
        "skynoise",
        "gain",
        "zp",
        "fieldid",
        "rcid",
        "ra",
        "dec",
    ]
    valid = np.isfinite(normalized[finite_columns]).all(axis=1)
    valid &= normalized["band"].isin({"ztfg", "ztfr", "ztfi"})
    normalized = normalized.loc[valid].copy()
    normalized[["fieldid", "rcid"]] = normalized[
        ["fieldid", "rcid"]
    ].astype(np.int32)
    normalized = normalized.sort_values("mjd", kind="stable").reset_index(drop=True)
    if normalized.empty:
        raise ValueError("ZTF normalization removed every selected observation")

    provenance = {
        **_file_provenance(path),
        "rows": int(len(normalized)),
        "coordinate_input_unit": coordinate_unit,
        "available_mjd_range": list(map(float, full_range)),
        "selected_source_mjd_range": [
            float(normalized["source_mjd"].min()),
            float(normalized["source_mjd"].max()),
        ],
        "transformed_mjd_range": [
            float(normalized["mjd"].min()),
            float(normalized["mjd"].max()),
        ],
        "filters": sorted(normalized["band"].unique().tolist()),
        "clean_only": bool(config.ztf_clean_only),
    }
    return normalized, provenance


def _sqlite_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    """Return the columns exposed by one SQLite table."""

    return {
        str(row[1])
        for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()
    }


def _lsst_observation_range(path: Path) -> tuple[float, float]:
    """Read the complete LSST OpSim MJD range without loading observations."""

    if not path.exists():
        raise FileNotFoundError(f"LSST OpSim database not found: {path}")
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT MIN(observationStartMJD), MAX(observationStartMJD) "
            "FROM observations"
        ).fetchone()
    if row is None or row[0] is None or row[1] is None:
        raise ValueError("LSST observations table is empty")
    return float(row[0]), float(row[1])


def _load_lsst(
    config: SurveyConfig,
    requested_range: tuple[float, float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and normalize a selected LSST OpSim cadence through SQL."""

    path = config.lsst_path
    if path is None or not path.exists():
        raise FileNotFoundError(f"LSST OpSim database not found: {path}")
    with sqlite3.connect(path) as connection:
        columns = _sqlite_columns(connection, "observations")
        missing = sorted(_LSST_REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(
                f"LSST observations table is missing columns: {', '.join(missing)}"
            )
        filter_column = "filter" if "filter" in columns else "band"
        if filter_column not in columns:
            raise ValueError("LSST observations table has neither filter nor band")
        full_row = connection.execute(
            "SELECT MIN(observationStartMJD), MAX(observationStartMJD), "
            "MIN(fieldRA), MAX(fieldRA), MIN(fieldDec), MAX(fieldDec) "
            "FROM observations"
        ).fetchone()
        if full_row is None or full_row[0] is None:
            raise ValueError("LSST observations table is empty")
        full_range = (float(full_row[0]), float(full_row[1]))
        source_coordinate_unit = _coordinate_unit_from_bounds(
            full_row[2],
            full_row[3],
            full_row[4],
            full_row[5],
        )
        source_start, source_stop, shift = _source_window(
            full_range=full_range,
            requested_range=requested_range,
            configured_start=config.lsst_source_mjd_start,
            time_mode=config.time_mode,
            label="LSST",
        )

        conditions = [
            "observationStartMJD >= ?",
            "observationStartMJD <= ?",
        ]
        parameters: list[Any] = [source_start, source_stop]
        if config.lsst_filters:
            placeholders = ", ".join("?" for _ in config.lsst_filters)
            conditions.append(f'LOWER("{filter_column}") IN ({placeholders})')
            parameters.extend(config.lsst_filters)
        if config.lsst_sql_where:
            conditions.append(f"({config.lsst_sql_where})")
        optional_columns = [
            name
            for name in ("night", "scheduler_note", "target_id")
            if name in columns
        ]
        selected_columns = [
            "observationId",
            "fieldRA",
            "fieldDec",
            "observationStartMJD",
            "fiveSigmaDepth",
            f'"{filter_column}" AS source_filter',
            *optional_columns,
        ]
        query = (
            f"SELECT {', '.join(selected_columns)} FROM observations "
            f"WHERE {' AND '.join(conditions)} "
            "ORDER BY observationStartMJD"
        )
        raw = pd.read_sql_query(query, connection, params=parameters)
    if raw.empty:
        raise ValueError("the selected LSST source window contains no observations")

    ra, dec, coordinate_unit = _coordinates_to_degrees(
        raw["fieldRA"],
        raw["fieldDec"],
        unit=source_coordinate_unit,
    )
    bands = raw["source_filter"].astype(str).str.lower().str.removeprefix("lsst")
    normalized = pd.DataFrame(
        {
            "mjd": raw["observationStartMJD"].to_numpy(dtype=float) + shift,
            "band": "lsst" + bands,
            "skynoise": _skynoise_from_maglimit(
                raw["fiveSigmaDepth"],
                config.lsst_zp,
            ),
            "gain": float(config.lsst_gain),
            "zp": float(config.lsst_zp),
            "ra": ra,
            "dec": dec,
            "instrument": "lsst",
            "source_observation_id": pd.to_numeric(
                raw["observationId"], errors="coerce"
            ),
            "limiting_magnitude": pd.to_numeric(
                raw["fiveSigmaDepth"], errors="coerce"
            ),
            "source_mjd": pd.to_numeric(
                raw["observationStartMJD"], errors="coerce"
            ),
        }
    )
    for optional in ("night", "scheduler_note", "target_id"):
        if optional in raw:
            normalized[optional] = raw[optional].to_numpy()
    finite_columns = [
        "mjd",
        "skynoise",
        "gain",
        "zp",
        "ra",
        "dec",
        "source_observation_id",
    ]
    valid = np.isfinite(normalized[finite_columns]).all(axis=1)
    valid &= normalized["band"].isin(
        {"lsstu", "lsstg", "lsstr", "lssti", "lsstz", "lssty"}
    )
    normalized = normalized.loc[valid].copy()
    normalized = normalized.sort_values("mjd", kind="stable").reset_index(drop=True)
    if normalized.empty:
        raise ValueError("LSST normalization removed every selected observation")

    provenance = {
        **_file_provenance(path),
        "rows": int(len(normalized)),
        "coordinate_input_unit": coordinate_unit,
        "available_mjd_range": list(map(float, full_range)),
        "selected_source_mjd_range": [
            float(normalized["source_mjd"].min()),
            float(normalized["source_mjd"].max()),
        ],
        "transformed_mjd_range": [
            float(normalized["mjd"].min()),
            float(normalized["mjd"].max()),
        ],
        "filters": sorted(normalized["band"].unique().tolist()),
        "sql_where": config.lsst_sql_where,
    }
    return normalized, provenance


def _canonicalize_projected_lsst_field(
    geometry: Any,
    *,
    pointing_dec: float,
) -> Any:
    """Convert one densely projected LSST boundary to a valid RA branch."""

    import shapely
    from shapely.geometry import Polygon

    coordinates = np.asarray(geometry.exterior.coords, dtype=float)
    longitudes = np.degrees(
        np.unwrap(np.radians(coordinates[:, 0]))
    )
    winding = (longitudes[-1] - longitudes[0]) / 360.0
    if np.isclose(abs(winding), 1.0, atol=1.0e-6):
        canonical = _polar_cap_from_unwrapped_ring(
            longitudes,
            coordinates[:, 1],
            pole=90.0 if pointing_dec > 0 else -90.0,
        )
    else:
        midpoint = 0.5 * (np.min(longitudes) + np.max(longitudes))
        longitudes -= np.floor(midpoint / 360.0) * 360.0
        canonical = Polygon(
            np.column_stack([longitudes, coordinates[:, 1]])
        )
    if not canonical.is_valid:
        canonical = shapely.make_valid(canonical)
    return canonical


def _lsst_polygon_survey(data: pd.DataFrame) -> tuple[Any, Any]:
    """Build canonical LSST pointing polygons without expanding observations."""

    import geopandas as gpd
    from skysurvey.survey.lsst import get_lsst_footprint
    from skysurvey.tools.projection import project_to_radec

    pointings = pd.MultiIndex.from_frame(data[["ra", "dec"]])
    codes, unique_pointings = pd.factorize(pointings, sort=False)
    survey_data = data.copy()
    survey_data["fieldid"] = codes.astype(np.int32)
    unique_frame = unique_pointings.to_frame(index=False)
    unique_frame.columns = ["ra", "dec"]
    footprint = get_lsst_footprint()
    pointing_ra = unique_frame["ra"].to_numpy()
    pointing_dec = unique_frame["dec"].to_numpy()
    if len(unique_frame) == 1:
        # SkySurvey 0.31 squeezes the pointing axis for a singleton input.
        projected = project_to_radec(
            footprint,
            np.repeat(pointing_ra, 2),
            np.repeat(pointing_dec, 2),
        )[:1]
    else:
        projected = project_to_radec(
            footprint,
            pointing_ra,
            pointing_dec,
        )

    # Only fields whose projected boundary winds around a pole need the
    # denser local outline. Ordinary LSST fields retain the compact 12-corner
    # representation, which matters for large OpSim imports.
    polar_indices = []
    for index, geometry in enumerate(projected):
        coordinates = np.asarray(geometry.exterior.coords, dtype=float)
        unwrapped = np.degrees(
            np.unwrap(np.radians(coordinates[:, 0]))
        )
        if np.isclose(
            abs((unwrapped[-1] - unwrapped[0]) / 360.0),
            1.0,
            atol=1.0e-6,
        ):
            polar_indices.append(index)
    if polar_indices:
        from shapely.geometry import Polygon

        dense_footprint = Polygon(
            _densify_ring(
                footprint.exterior.coords,
                max_step_deg=_AREA_EDGE_STEP_DEG,
            )
        )
        polar_ra = pointing_ra[polar_indices]
        polar_dec = pointing_dec[polar_indices]
        if len(polar_indices) == 1:
            dense_projected = project_to_radec(
                dense_footprint,
                np.repeat(polar_ra, 2),
                np.repeat(polar_dec, 2),
            )[:1]
        else:
            dense_projected = project_to_radec(
                dense_footprint,
                polar_ra,
                polar_dec,
            )
        for index, geometry in zip(polar_indices, dense_projected):
            projected[index] = geometry
    geometries = [
        _canonicalize_projected_lsst_field(
            geometry,
            pointing_dec=float(declination),
        )
        for geometry, declination in zip(projected, pointing_dec)
    ]
    fields = gpd.GeoDataFrame(
        {"instrument": "lsst"},
        geometry=geometries,
        index=pd.Index(np.arange(len(unique_frame)), name="fieldid"),
    )
    return WarpPolygonSurvey.create(
        survey_data,
        fields,
        footprint=footprint,
    ), fields


def _ztf_survey(data: pd.DataFrame) -> Any:
    """Build the native exact-quadrant ZTF survey."""

    from skysurvey.survey.ztf import ZTF

    return ObservedFieldSurvey(ZTF(data=data, level="quadrant"))


def _combined_survey(
    ztf_data: pd.DataFrame,
    lsst_data: pd.DataFrame,
) -> CombinedSurvey:
    """Combine exact ZTF quadrants and LSST pointing polygons."""

    import geopandas as gpd
    from ztffields.fields import Fields

    ztf_native_fields = Fields.get_field_geometry(level="quadrant")
    observed_pairs = pd.MultiIndex.from_frame(
        ztf_data[["fieldid", "rcid"]].drop_duplicates()
    )
    ztf_fields = ztf_native_fields.loc[observed_pairs].copy()
    ztf_codes = (
        ztf_fields.index.get_level_values("fieldid").to_numpy(dtype=np.int64) * 64
        + ztf_fields.index.get_level_values("rcid").to_numpy(dtype=np.int64)
    )
    ztf_fields = gpd.GeoDataFrame(
        {"instrument": "ztf"},
        geometry=ztf_fields.geometry.to_numpy(),
        index=pd.Index(ztf_codes, name="fieldid"),
    )
    encoded_ztf = ztf_data.copy()
    encoded_ztf["fieldid"] = (
        encoded_ztf["fieldid"].to_numpy(dtype=np.int64) * 64
        + encoded_ztf["rcid"].to_numpy(dtype=np.int64)
    )

    lsst_survey, lsst_fields = _lsst_polygon_survey(lsst_data)
    encoded_lsst = lsst_survey.data.copy()
    encoded_lsst["fieldid"] = (
        encoded_lsst["fieldid"].to_numpy(dtype=np.int64) + _LSST_FIELD_OFFSET
    )
    encoded_lsst_fields = lsst_fields.copy()
    encoded_lsst_fields.index = pd.Index(
        encoded_lsst_fields.index.to_numpy(dtype=np.int64) + _LSST_FIELD_OFFSET,
        name="fieldid",
    )

    fields = pd.concat([ztf_fields, encoded_lsst_fields])
    fields = gpd.GeoDataFrame(fields, geometry="geometry")
    data = pd.concat([encoded_ztf, encoded_lsst], ignore_index=True, sort=False)
    data = data.sort_values("mjd", kind="stable").reset_index(drop=True)
    polygon_survey = WarpPolygonSurvey.create(data, fields)
    instruments = fields["instrument"].to_dict()
    return CombinedSurvey(polygon_survey, instruments)


def _iter_polygons(geometry: Any) -> Iterable[Any]:
    """Yield every polygon contained in a Shapely geometry."""

    if geometry is None or geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
        return
    if geometry.geom_type in {"MultiPolygon", "GeometryCollection"}:
        for child in geometry.geoms:
            yield from _iter_polygons(child)


def _normalize_input_geometries(geometries: Iterable[Any]) -> np.ndarray:
    """Collect already-canonical field polygons for periodic clipping."""

    import shapely

    normalized: list[Any] = []
    for geometry in geometries:
        if geometry is None or geometry.is_empty:
            continue
        for polygon in _iter_polygons(geometry):
            normalized.append(
                _canonicalize_ordinary_seam_polygon(polygon)
            )
    polygons = np.asarray(normalized, dtype=object)
    if len(polygons) and not shapely.is_valid(polygons).all():
        raise ValueError("periodic union requires canonical valid field polygons")
    return polygons


def _periodic_polygon_union(geometries: Iterable[Any]) -> Any:
    """Union sky polygons in one RA period while preserving wraparound overlap."""

    import shapely
    from shapely.geometry import box

    polygons = _normalize_input_geometries(geometries)
    if not len(polygons):
        return shapely.GeometryCollection()

    # Work in one fundamental longitude interval. Shifted copies make fields
    # stored below zero or above 360 degrees meet their periodic counterparts.
    domain = box(0.0, -90.0, 360.0, 90.0)
    bounds = shapely.bounds(polygons)
    pieces: list[np.ndarray] = []
    for offset in (-360.0, 0.0, 360.0):
        shifted_min = bounds[:, 0] + offset
        shifted_max = bounds[:, 2] + offset
        relevant = (shifted_max >= 0.0) & (shifted_min <= 360.0)
        if not relevant.any():
            continue
        selected = polygons[relevant]
        if offset:
            selected = shapely.transform(
                selected,
                lambda coordinates, shift=offset: coordinates
                + np.asarray([shift, 0.0]),
            )
        clipped = shapely.intersection(selected, domain)
        pieces.append(clipped[~shapely.is_empty(clipped)])
    if not pieces:
        return shapely.GeometryCollection()
    return shapely.union_all(np.concatenate(pieces))


def _observed_instrument_regions(
    survey: Any,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Dissolve exact observed fields into one periodic region per instrument."""

    fields = survey.get_fields(observed=True)
    if fields is None or fields.empty:
        raise ValueError(f"{label} has no observed field geometry")
    if "instrument" in fields:
        instrument_values = fields["instrument"].astype(str)
    else:
        instrument_values = pd.Series(label, index=fields.index)

    regions: dict[str, Any] = {}
    field_counts: dict[str, int] = {}
    for instrument in sorted(instrument_values.unique()):
        selected = fields.loc[instrument_values.eq(instrument), "geometry"]
        regions[str(instrument)] = _periodic_polygon_union(selected.to_numpy())
        field_counts[str(instrument)] = int(len(selected))
    return regions, field_counts


def _adopted_observed_region(regions: Mapping[str, Any], *, label: str) -> Any:
    """Return a single-instrument union or the joint instrument intersection."""

    import shapely

    ordered = [regions[key] for key in sorted(regions)]
    if label != "combined":
        if len(ordered) != 1:
            raise ValueError(f"{label} unexpectedly contains multiple instruments")
        return ordered[0]
    if len(ordered) < 2:
        raise ValueError("combined survey requires at least two instrument regions")
    adopted = ordered[0]
    for region in ordered[1:]:
        adopted = shapely.intersection(adopted, region)
    return adopted


def _densify_ring(coordinates: Any, *, max_step_deg: float) -> np.ndarray:
    """Densify straight RA/Dec edges before the nonlinear equal-area map."""

    values = np.asarray(coordinates, dtype=float)
    dense: list[np.ndarray] = []
    for start, stop in zip(values[:-1], values[1:]):
        span = float(np.max(np.abs(stop - start)))
        steps = max(1, int(np.ceil(span / max_step_deg)))
        fractions = np.arange(steps, dtype=float) / steps
        dense.append(start + fractions[:, None] * (stop - start))
    dense.append(values[-1:])
    return np.concatenate(dense, axis=0)


def _ring_cylindrical_equal_area(coordinates: Any) -> float:
    """Return one densified ring area after (lambda, phi)->(lambda, sin(phi))."""

    dense = _densify_ring(coordinates, max_step_deg=_AREA_EDGE_STEP_DEG)
    longitude = np.radians(dense[:, 0])
    sine_latitude = np.sin(np.radians(dense[:, 1]))
    cross = longitude[:-1] * sine_latitude[1:]
    cross -= longitude[1:] * sine_latitude[:-1]
    return 0.5 * float(np.sum(cross))


def _cylindrical_equal_area_deg2(geometry: Any) -> float:
    """Measure a periodic polygon region with a cylindrical equal-area map."""

    steradians = 0.0
    for polygon in _iter_polygons(geometry):
        exterior = abs(_ring_cylindrical_equal_area(polygon.exterior.coords))
        holes = sum(
            abs(_ring_cylindrical_equal_area(interior.coords))
            for interior in polygon.interiors
        )
        steradians += exterior - holes
    return float(steradians * np.square(180.0 / np.pi))


def _geometry_vertex_count(polygon: Any) -> int:
    """Count exterior and interior vertices in one polygon."""

    return int(
        len(polygon.exterior.coords)
        + sum(len(interior.coords) for interior in polygon.interiors)
    )


def _spherical_rectangle_cap(
    bounds: Sequence[float],
) -> tuple[float, float, float]:
    """Return a cap enclosing every lon/lat point in a planar rectangle."""

    min_ra, min_dec, max_ra, max_dec = map(float, bounds)
    longitude_span = np.radians(max_ra - min_ra)
    if longitude_span >= 2.0 * np.pi:
        return 0.0, 0.0, np.pi

    center_ra = 0.5 * (min_ra + max_ra)
    center_dec = 0.5 * (min_dec + max_dec)
    center_latitude = np.radians(center_dec)
    half_longitude_span = min(0.5 * longitude_span, np.pi)
    minimum_longitude_cosine = np.cos(half_longitude_span)
    coefficient_sine = np.sin(center_latitude)
    coefficient_cosine = (
        np.cos(center_latitude) * minimum_longitude_cosine
    )

    min_latitude = np.radians(min_dec)
    max_latitude = np.radians(max_dec)
    candidate_latitudes = [min_latitude, max_latitude]
    stationary = np.arctan2(coefficient_sine, coefficient_cosine)
    for offset in range(-2, 3):
        candidate = stationary + offset * np.pi
        if min_latitude <= candidate <= max_latitude:
            candidate_latitudes.append(candidate)

    candidate_latitudes = np.asarray(candidate_latitudes, dtype=float)
    dot_products = (
        coefficient_sine * np.sin(candidate_latitudes)
        + coefficient_cosine * np.cos(candidate_latitudes)
    )
    radius = np.arccos(np.clip(np.min(dot_products), -1.0, 1.0))
    return center_ra, center_dec, float(radius)


def _bounding_cap_pixel_ids(polygon: Any, *, nside: int) -> np.ndarray:
    """Conservatively rasterize a complex polygon through a spherical cap."""

    import healpy as hp

    min_ra, min_dec, max_ra, max_dec = map(float, polygon.bounds)
    if min_dec <= -90.0 + 1.0e-10:
        radius = np.radians(max_dec + 90.0) + 1.0e-10
        return np.asarray(
            hp.query_disc(
                nside,
                hp.ang2vec(np.pi, 0.0),
                radius,
                inclusive=True,
            ),
            dtype=np.int64,
        )
    if max_dec >= 90.0 - 1.0e-10:
        radius = np.radians(90.0 - min_dec) + 1.0e-10
        return np.asarray(
            hp.query_disc(
                nside,
                hp.ang2vec(0.0, 0.0),
                radius,
                inclusive=True,
            ),
            dtype=np.int64,
        )
    center_ra, center_dec, radius = _spherical_rectangle_cap(
        (min_ra, min_dec, max_ra, max_dec)
    )
    if radius >= np.pi - 1.0e-10:
        return np.arange(hp.nside2npix(nside), dtype=np.int64)
    center = hp.ang2vec(
        np.radians(90.0 - center_dec),
        np.radians(center_ra % 360.0),
    )
    return np.asarray(
        hp.query_disc(
            nside,
            center,
            radius + 1.0e-10,
            inclusive=True,
        ),
        dtype=np.int64,
    )


def _triangulated_polygon_pixel_ids(
    polygon: Any,
    *,
    nside: int,
) -> tuple[np.ndarray, str, int]:
    """Rasterize one simple polygon with inclusive HEALPix triangle queries."""

    import healpy as hp
    from shapely.ops import triangulate

    if (
        polygon.bounds[1] <= -90.0 + 1.0e-10
        or polygon.bounds[3] >= 90.0 - 1.0e-10
    ):
        return _bounding_cap_pixel_ids(polygon, nside=nside), "bounding_cap", 0
    vertex_count = _geometry_vertex_count(polygon)
    if vertex_count > _MAX_TRIANGULATION_VERTICES:
        return _bounding_cap_pixel_ids(polygon, nside=nside), "bounding_cap", 0

    pixels: set[int] = set()
    triangles = triangulate(polygon)
    try:
        for triangle in triangles:
            coordinates = np.asarray(triangle.exterior.coords, dtype=float)[:-1]
            vectors = hp.ang2vec(
                np.radians(90.0 - coordinates[:, 1]),
                np.radians(coordinates[:, 0] % 360.0),
            )
            pixels.update(
                map(
                    int,
                    hp.query_polygon(
                        nside,
                        vectors,
                        inclusive=True,
                    ),
                )
            )
    except (RuntimeError, ValueError):
        return _bounding_cap_pixel_ids(polygon, nside=nside), "bounding_cap", 0
    if not pixels:
        return _bounding_cap_pixel_ids(polygon, nside=nside), "bounding_cap", 0

    # Planar RA/Dec edges are not great-circle arcs. Unioning a cap proven to
    # contain the complete planar bounding box closes any high-latitude gaps
    # left by HEALPix's spherical triangle interpretation.
    cap_pixels = _bounding_cap_pixel_ids(polygon, nside=nside)
    if len(cap_pixels) == hp.nside2npix(nside):
        return (
            cap_pixels,
            "triangulated_with_bounding_cap_guard",
            len(triangles),
        )
    pixels.update(map(int, cap_pixels))
    return (
        np.asarray(sorted(pixels), dtype=np.int64),
        "triangulated_with_bounding_cap_guard",
        len(triangles),
    )


def _region_proposal_pixel_ids(
    region: Any,
    *,
    nside: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Build one conservative instrument mask from its dissolved region."""

    pixels: set[int] = set()
    statistics = {
        "triangulated_components": 0,
        "triangle_count": 0,
        "bounding_cap_guard_components": 0,
        "bounding_cap_components": 0,
        "all_sky_components": 0,
    }
    import healpy as hp

    all_sky_count = hp.nside2npix(nside)
    # Resolve the most complex components first: a global conservative cap can
    # terminate the mask construction without thousands of redundant queries.
    polygons = sorted(
        _iter_polygons(region),
        key=_geometry_vertex_count,
        reverse=True,
    )
    for polygon in polygons:
        component_pixels, method, triangle_count = (
            _triangulated_polygon_pixel_ids(polygon, nside=nside)
        )
        if method.startswith("triangulated"):
            statistics["triangulated_components"] += 1
            statistics["triangle_count"] += int(triangle_count)
            if "bounding_cap_guard" in method:
                statistics["bounding_cap_guard_components"] += 1
        else:
            statistics["bounding_cap_components"] += 1
        if len(component_pixels) == all_sky_count:
            statistics["all_sky_components"] += 1
            return component_pixels, statistics
        pixels.update(map(int, component_pixels))
    return np.asarray(sorted(pixels), dtype=np.int64), statistics


def _proposal_pixel_ids(
    regions: Mapping[str, Any],
    *,
    nside: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Union per-instrument pixels and intersect them for a joint survey."""

    instrument_pixels: dict[str, np.ndarray] = {}
    rasterization: dict[str, dict[str, int]] = {}
    for instrument in sorted(regions):
        pixels, statistics = _region_proposal_pixel_ids(
            regions[instrument],
            nside=nside,
        )
        instrument_pixels[instrument] = pixels
        rasterization[instrument] = statistics

    masks = [instrument_pixels[key] for key in sorted(instrument_pixels)]
    proposal = masks[0]
    for mask in masks[1:]:
        proposal = np.intersect1d(proposal, mask, assume_unique=True)
    metadata = {
        "method": (
            "dissolved_polygon_triangulation_query_polygon_inclusive_"
            "with_conservative_bounding_cap_guard_and_fallback"
        ),
        "instrument_pixel_counts": {
            key: int(len(value)) for key, value in instrument_pixels.items()
        },
        "rasterization": rasterization,
    }
    return np.asarray(proposal, dtype=np.int64), metadata


class SurveyFactory:
    """Create normalized ZTF, LSST, or synthetic combined surveys."""

    def __init__(self, data_root: Optional[str | Path] = None):
        """Store an optional alternative root for survey fixtures or installations."""

        self.data_root = (
            None
            if data_root is None
            else Path(data_root).expanduser().resolve()
        )

    def from_spec(self, spec: Any) -> Any:
        """Build the survey selected by a WarpSampleSpec."""

        from .observer_population import _normalize_time_window

        fixed_size = spec.size is not None or spec.class_sampling == "counts"
        tstart, tstop, _ = _normalize_time_window(
            size=0 if fixed_size else None,
            nyears=spec.nyears,
            tstart=spec.tstart,
            tstop=spec.tstop,
            default_tstart=60000.0,
            default_tstop=60365.25,
        )
        return self.create(
            spec.survey_name,
            tstart=tstart,
            tstop=tstop,
            options=spec.survey_options,
        )

    def available_mjd_ranges(
        self,
        name: str,
        *,
        options: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, tuple[float, float]]:
        """Inspect complete local cadence ranges without loading survey rows."""

        config = SurveyConfig.from_options(
            name,
            options,
            data_root=self.data_root,
        )
        ranges: dict[str, tuple[float, float]] = {}
        if config.name in {"ztf", "combined"}:
            if config.ztf_path is None or not config.ztf_path.exists():
                raise FileNotFoundError(
                    f"ZTF observing log not found: {config.ztf_path}"
                )
            ranges["ztf"] = _parquet_column_range(
                config.ztf_path,
                "expMJD",
            )
        if config.name in {"lsst", "combined"}:
            if config.lsst_path is None:
                raise FileNotFoundError("LSST OpSim database path is not configured")
            ranges["lsst"] = _lsst_observation_range(config.lsst_path)
        return ranges

    def plan_relative_realizations(
        self,
        name: str,
        *,
        tstart: float,
        tstop: float,
        count: int,
        options: Optional[Mapping[str, Any]] = None,
        source_mjd_starts: Optional[Mapping[str, Sequence[float]]] = None,
        require_non_overlapping: bool = True,
    ) -> list[SurveyRealizationSpec]:
        """Distribute bounded relative windows across each cadence archive."""

        count = int(count)
        requested_range = (float(tstart), float(tstop))
        duration = requested_range[1] - requested_range[0]
        if count <= 0:
            raise ValueError("realization count must be positive")
        if duration <= 0.0:
            raise ValueError("realization tstop must be greater than tstart")

        base_options = dict(options or {})
        if str(base_options.get("time_mode", "relative")) != "relative":
            raise ValueError("survey ensembles require time_mode='relative'")
        configured_keys = {
            instrument: f"{instrument}_source_mjd_start"
            for instrument in ("ztf", "lsst")
        }
        if source_mjd_starts is not None:
            unknown = sorted(set(source_mjd_starts) - set(configured_keys))
            if unknown:
                raise ValueError(
                    "unknown source_mjd_starts instruments: "
                    + ", ".join(unknown)
                )
        ranges = self.available_mjd_ranges(name, options=base_options)
        starts_by_instrument: dict[str, np.ndarray] = {}
        for instrument, available_range in ranges.items():
            available_start, available_stop = map(float, available_range)
            latest_start = available_stop - duration
            if latest_start < available_start:
                raise ValueError(
                    f"{instrument.upper()} archive is shorter than the requested "
                    f"{duration:g}-day realization"
                )

            explicit_starts = (
                None
                if source_mjd_starts is None
                else source_mjd_starts.get(instrument)
            )
            option_key = configured_keys[instrument]
            if explicit_starts is None and option_key in base_options:
                if count != 1:
                    raise ValueError(
                        f"{option_key} is ambiguous for {count} realizations; "
                        "pass source_mjd_starts instead"
                    )
                explicit_starts = [base_options[option_key]]
            if explicit_starts is None:
                starts = np.linspace(available_start, latest_start, count)
            else:
                starts = np.asarray(list(explicit_starts), dtype=float)
                if starts.shape != (count,):
                    raise ValueError(
                        f"source_mjd_starts[{instrument!r}] must contain "
                        f"exactly {count} values"
                    )
            if not np.isfinite(starts).all():
                raise ValueError(f"{instrument.upper()} source starts must be finite")
            if np.any(starts < available_start) or np.any(starts > latest_start):
                raise ValueError(
                    f"{instrument.upper()} source starts must lie in "
                    f"[{available_start}, {latest_start}]"
                )
            if require_non_overlapping and len(starts) > 1:
                separations = np.diff(np.sort(starts))
                if np.any(separations < duration - 1.0e-9):
                    raise ValueError(
                        f"{count} non-overlapping {duration:g}-day windows do "
                        f"not fit inside the {instrument.upper()} archive; set "
                        "require_non_overlapping=False for evenly distributed "
                        "sliding windows"
                    )
            starts_by_instrument[instrument] = starts

        realizations: list[SurveyRealizationSpec] = []
        for index in range(count):
            realization_options = dict(base_options)
            source_ranges: dict[str, tuple[float, float]] = {}
            for instrument, starts in starts_by_instrument.items():
                source_start = float(starts[index])
                realization_options[configured_keys[instrument]] = source_start
                source_ranges[instrument] = (
                    source_start,
                    source_start + duration,
                )
            realizations.append(
                SurveyRealizationSpec(
                    realization_id=f"r{index:03d}",
                    index=index,
                    count=count,
                    survey_name=str(name).lower(),
                    requested_mjd_range=requested_range,
                    source_mjd_ranges=source_ranges,
                    survey_options=realization_options,
                )
            )
        return realizations

    def create(
        self,
        name: str,
        *,
        tstart: float,
        tstop: float,
        options: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """Create one selected survey and attach complete import provenance."""

        config = SurveyConfig.from_options(
            name,
            options,
            data_root=self.data_root,
        )
        requested_range = (float(tstart), float(tstop))
        sources: dict[str, dict[str, Any]] = {}
        if config.name in {"ztf", "combined"}:
            ztf_data, sources["ztf"] = _load_ztf(config, requested_range)
        else:
            ztf_data = None
        if config.name in {"lsst", "combined"}:
            lsst_data, sources["lsst"] = _load_lsst(config, requested_range)
        else:
            lsst_data = None

        if config.name == "ztf":
            survey = _ztf_survey(ztf_data)
        elif config.name == "lsst":
            survey, _ = _lsst_polygon_survey(lsst_data)
        else:
            survey = _combined_survey(ztf_data, lsst_data)

        regions, observed_field_counts = _observed_instrument_regions(
            survey,
            label=config.name,
        )
        observed_fields = survey.get_fields(observed=True)
        field_bounds = observed_fields.geometry.bounds
        polar_cap_count = int(
            (
                (
                    field_bounds["miny"].le(-90.0 + 1.0e-10)
                    | field_bounds["maxy"].ge(90.0 - 1.0e-10)
                )
                & (field_bounds["maxx"] - field_bounds["minx"]).ge(
                    360.0 - 1.0e-8
                )
            ).sum()
        )
        adopted_region = _adopted_observed_region(regions, label=config.name)
        area_deg2 = _cylindrical_equal_area_deg2(adopted_region)
        if not np.isfinite(area_deg2) or area_deg2 <= 0:
            raise ValueError(f"{config.name} has no geometric observed sky overlap")
        pixel_ids, proposal_metadata = _proposal_pixel_ids(
            regions,
            nside=config.nside,
        )
        skyarea = ObservedSkyArea(
            survey,
            pixel_ids,
            nside=config.nside,
            label=config.name,
            area_deg2=area_deg2,
        )
        import healpy as hp

        healpix_pixel_area = float(
            hp.nside2pixarea(config.nside, degrees=True)
        )
        provenance = {
            "name": config.name,
            "class": f"{survey.__class__.__module__}.{survey.__class__.__name__}",
            "synthetic": config.name == "combined",
            "time_mode": config.time_mode,
            "requested_mjd_range": list(requested_range),
            "observed_mjd_range": [
                float(survey.data["mjd"].min()),
                float(survey.data["mjd"].max()),
            ],
            "rows": int(len(survey.data)),
            "bands": sorted(survey.data["band"].unique().tolist()),
            "nside": int(config.nside),
            "backend": config.backend,
            "sampling_footprint": repr(skyarea),
            "footprint_geometry": {
                "area_deg2": float(area_deg2),
                "area_method": (
                    "periodic_ra_polygon_union_or_intersection_then_"
                    "cylindrical_equal_area_lambda_sin_phi"
                ),
                "edge_max_step_deg": float(_AREA_EDGE_STEP_DEG),
                "observed_field_counts": observed_field_counts,
                "field_geometry_method": (
                    "warp_canonical_periodic_radec_v2"
                ),
                "periodic_match_ra_offsets_deg": [
                    -360.0,
                    0.0,
                    360.0,
                ],
                "polar_cap_count": polar_cap_count,
                "lsst_polar_projection_edge_step_deg": float(
                    _AREA_EDGE_STEP_DEG
                ),
                "combined_operation": (
                    "instrument_intersection"
                    if config.name == "combined"
                    else "single_instrument_union"
                ),
            },
            "proposal_mask": {
                **proposal_metadata,
                "nside": int(config.nside),
                "pixel_count": int(len(pixel_ids)),
                "healpix_pixel_area_deg2": healpix_pixel_area,
                "total_pixel_area_deg2": float(
                    len(pixel_ids) * healpix_pixel_area
                ),
                "bounding_cap_method": (
                    "analytic_spherical_lon_lat_rectangle_extrema"
                ),
                "exact_draw_validation": True,
            },
            "parquet_scan_batch_size": int(config.parquet_batch_size),
            "sources": sources,
            "config": config.to_dict(),
            "description": (
                "Synthetic joint ZTF+LSST cadence; not a real joint-survey forecast."
                if config.name == "combined"
                else "Local observing cadence normalized for SkySurvey."
            ),
        }
        survey.simulation_skyarea = skyarea
        survey.warp_provenance = json.loads(json.dumps(_json_safe(provenance)))
        return survey


__all__ = [
    "CombinedSurvey",
    "ObservedFieldSurvey",
    "ObservedSkyArea",
    "SurveyConfig",
    "SurveyFactory",
    "SurveyRealizationSpec",
]
