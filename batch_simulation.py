"""Resource-bounded WarpTemplate simulation through unmodified SkySurvey."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import blake2b
from itertools import islice
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import time
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import exponnorm

from .loaders import WarpfitTemplateLoader, WarpTemplateDescriptor
from .source_cache import WarpSourceCache
from .population import (
    _draw_magabs_with_provenance,
    load_warp_rate_config,
    resolve_rate,
    validate_active_fitclasses,
)
from .observer_population import (
    _normalize_time_window,
    _observer_redshift_cdf,
    _sky_fraction,
)

SIMULATION_SCHEMA_VERSION = 6


def allocate_group_balanced_counts(
    total_size: int,
    class_groups: Mapping[str, Sequence[str]],
    shard_sizes: Optional[Sequence[int]] = None,
) -> list[dict[str, int]]:
    """Balance parent classes, then divide each quota across its raw members."""

    total_size = int(total_size)
    if total_size < 0:
        raise ValueError("total_size must be non-negative")
    if not class_groups:
        raise ValueError("class_groups must contain at least one parent class")

    normalized_groups: list[tuple[str, tuple[str, ...]]] = []
    raw_classes: list[str] = []
    for parent, members in class_groups.items():
        if isinstance(members, str):
            raise TypeError("Each class group must contain a sequence of raw class names")
        normalized_members = tuple(str(member).strip() for member in members)
        if not normalized_members or any(not member for member in normalized_members):
            raise ValueError(f"Class group {parent!r} has no valid raw members")
        normalized_groups.append((str(parent), normalized_members))
        raw_classes.extend(normalized_members)
    if len(set(raw_classes)) != len(raw_classes):
        raise ValueError("A raw class may belong to only one balanced parent class")

    shards = [total_size] if shard_sizes is None else [int(size) for size in shard_sizes]
    if any(size < 0 for size in shards):
        raise ValueError("shard_sizes must be non-negative")
    if sum(shards) != total_size:
        raise ValueError("shard_sizes must sum to total_size")

    def cyclic_count(start: int, stop: int, residue: int, period: int) -> int:
        """Count integers in [start, stop) with one modular residue."""

        first = start + ((residue - start) % period)
        return 0 if first >= stop else 1 + (stop - 1 - first) // period

    # A single global cyclic allocation keeps parent totals within one object.
    # Slicing that same sequence gives each realization a locally balanced shard
    # without letting per-realization rounding accumulate into a global bias.
    allocations: list[dict[str, int]] = []
    shard_start = 0
    parent_period = len(normalized_groups)
    for shard_size in shards:
        shard_stop = shard_start + shard_size
        allocation = {raw_class: 0 for raw_class in raw_classes}
        for parent_index, (_, members) in enumerate(normalized_groups):
            member_start = cyclic_count(0, shard_start, parent_index, parent_period)
            member_stop = cyclic_count(0, shard_stop, parent_index, parent_period)
            for member_index, member in enumerate(members):
                allocation[member] = cyclic_count(
                    member_start,
                    member_stop,
                    member_index,
                    len(members),
                )
        if sum(allocation.values()) != shard_size:
            raise RuntimeError("Grouped class allocation did not preserve shard size")
        allocations.append(allocation)
        shard_start = shard_stop
    return allocations


def _stable_seed(seed: int, *parts: Any) -> int:
    """Derive a deterministic NumPy seed without Python's randomized hash."""

    payload = "|".join([str(seed), *(str(part) for part in parts)]).encode()
    return int.from_bytes(blake2b(payload, digest_size=8).digest(), "little")


def _package_version(name: str) -> Optional[str]:
    """Return an installed distribution version when available."""

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _dataframe_fingerprint(frame: pd.DataFrame) -> str:
    """Hash a table's schema, index, order, and values for resume checks."""

    hasher = blake2b(digest_size=16)
    hasher.update(repr(tuple(frame.columns)).encode())
    hasher.update(repr(tuple(map(str, frame.dtypes))).encode())
    hasher.update(repr((frame.index.name, str(frame.index.dtype))).encode())
    try:
        row_hashes = pd.util.hash_pandas_object(
            frame,
            index=True,
            categorize=True,
        ).to_numpy(dtype=np.uint64, copy=False)
    except TypeError:
        # Explicit user surveys may contain uncommon object columns. Their
        # repr is slower but still gives the fallback provenance real content.
        normalized = frame.apply(lambda column: column.map(repr))
        row_hashes = pd.util.hash_pandas_object(
            normalized,
            index=True,
            categorize=True,
        ).to_numpy(dtype=np.uint64, copy=False)
    hasher.update(row_hashes.tobytes())
    return hasher.hexdigest()


def _mapping_fingerprint(value: Mapping[str, Any]) -> str:
    """Hash one nested configuration after deterministic JSON conversion."""

    payload = json.dumps(
        WarpSampleSpec._json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return blake2b(payload, digest_size=16).hexdigest()


@dataclass(frozen=True)
class WarpSampleSpec:
    """Configuration for one reproducible WarpTemplate training sample."""

    run_name: str
    active_fitclasses: Sequence[str]
    size: Optional[int] = None
    nyears: Optional[float] = None
    zmin: float = 0.0
    zmax: float = 0.08
    tstart: float = 60000.0
    tstop: Optional[float] = None
    skyarea: Any = None
    class_sampling: str = "volumetric"
    class_weights: Optional[Mapping[str, float]] = None
    class_counts: Optional[Mapping[str, int]] = None
    redshift_sampling: str = "volumetric"
    redshift_bins: Optional[Sequence[float]] = None
    color_mode: Optional[str] = "draw"
    target_peak_color: Optional[float] = None
    min_fit_quality: Optional[str] = "bronze"
    survey_name: str = "ztf"
    survey_options: Optional[Mapping[str, Any]] = None
    survey_realization_id: Optional[str] = None
    batch_size: int = 10_000
    max_sources_per_batch: int = 512
    phase_range: Optional[Sequence[float]] = (-50.0, 200.0)
    incl_error: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        """Reject ambiguous or inconsistent sample definitions."""

        if not self.run_name or "/" in self.run_name:
            raise ValueError("run_name must be non-empty and must not contain '/'")
        if not self.active_fitclasses:
            raise ValueError("active_fitclasses must not be empty")
        if self.size is not None and self.size < 0:
            raise ValueError("size must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_sources_per_batch <= 0:
            raise ValueError("max_sources_per_batch must be positive")
        if not 0 <= self.zmin < self.zmax:
            raise ValueError("redshift bounds must satisfy 0 <= zmin < zmax")
        if self.class_sampling not in {"volumetric", "balanced", "weighted", "counts"}:
            raise ValueError("unsupported class_sampling mode")
        if self.redshift_sampling not in {"volumetric", "uniform", "binned"}:
            raise ValueError("unsupported redshift_sampling mode")
        if self.class_sampling == "weighted" and not self.class_weights:
            raise ValueError("class_weights are required for weighted sampling")
        if self.class_sampling == "counts" and not self.class_counts:
            raise ValueError("class_counts are required for counts sampling")
        if self.redshift_sampling == "binned":
            bins = np.asarray(self.redshift_bins, dtype=float)
            if bins.ndim != 1 or len(bins) < 2 or np.any(np.diff(bins) <= 0):
                raise ValueError("redshift_bins must be a strictly increasing sequence")
            if bins[0] < self.zmin or bins[-1] > self.zmax:
                raise ValueError("redshift_bins must lie inside [zmin, zmax]")
        if self.survey_options is not None and not isinstance(
            self.survey_options, Mapping
        ):
            raise TypeError("survey_options must be a mapping or None")
        if self.survey_realization_id is not None:
            realization_id = str(self.survey_realization_id)
            if not realization_id or "/" in realization_id:
                raise ValueError(
                    "survey_realization_id must be non-empty and contain no '/'"
                )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation for the run manifest."""

        data = asdict(self)
        data["active_fitclasses"] = list(self.active_fitclasses)
        data["redshift_bins"] = (
            list(self.redshift_bins) if self.redshift_bins is not None else None
        )
        data["phase_range"] = list(self.phase_range) if self.phase_range is not None else None
        data["skyarea"] = None if self.skyarea is None else repr(self.skyarea)
        data["survey_options"] = self._json_safe(data["survey_options"])
        return data

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Convert nested sample options into stable manifest values."""

        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {
                str(key): WarpSampleSpec._json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [WarpSampleSpec._json_safe(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value


class _WarpTargetSampler:
    """Stream compact SkySurvey target rows without constructing spectral models."""

    def __init__(
        self,
        warpcoeffs_dir: str | Path,
        *,
        rate_config: Optional[str | Path | Mapping[str, Any]] = None,
        loader: Optional[WarpfitTemplateLoader] = None,
        cosmology: Any = None,
    ):
        """Initialize the sampler and its shared coefficient loader."""

        self.warpcoeffs_dir = Path(warpcoeffs_dir)
        self.rate_config = load_warp_rate_config(rate_config)
        self.loader = loader or WarpfitTemplateLoader(str(self.warpcoeffs_dir))
        if cosmology is None:
            from astropy.cosmology import Planck18

            cosmology = Planck18
        self.cosmology = cosmology

    def _class_counts(self, spec: WarpSampleSpec) -> dict[str, int]:
        """Resolve fixed or rate-normalized target counts for every class."""

        fitclasses = validate_active_fitclasses(spec.active_fitclasses, self.rate_config)
        rates = np.asarray(
            [float(resolve_rate(name, self.rate_config).rate_gpc3_yr) for name in fitclasses]
        )
        if spec.class_sampling == "counts":
            counts = {name: int(spec.class_counts.get(name, 0)) for name in fitclasses}
            if any(value < 0 for value in counts.values()):
                raise ValueError("class_counts must be non-negative")
            if spec.size is not None and sum(counts.values()) != spec.size:
                raise ValueError("class_counts must sum to size when size is provided")
            return counts

        if spec.size is None:
            if spec.class_sampling != "volumetric":
                raise ValueError("size is required unless class_sampling='volumetric'")
            from .observer_population import observer_expected_count

            _, _, nyears = _normalize_time_window(
                size=None,
                nyears=spec.nyears,
                tstart=spec.tstart,
                tstop=spec.tstop,
                default_tstart=60000.0,
                default_tstop=60365.25,
            )
            counts = {}
            for name, rate in zip(fitclasses, rates):
                expectation = observer_expected_count(
                    rate=rate,
                    zmin=spec.zmin,
                    zmax=spec.zmax,
                    nyears=nyears,
                    sky_fraction=_sky_fraction(spec.skyarea, None),
                    cosmology=self.cosmology,
                )
                rng = np.random.default_rng(_stable_seed(spec.seed, name, "count"))
                counts[name] = int(rng.poisson(expectation))
            return counts

        if spec.class_sampling == "balanced":
            quotient, remainder = divmod(int(spec.size), len(fitclasses))
            allocation = np.full(len(fitclasses), quotient, dtype=int)
            order = np.random.default_rng(
                _stable_seed(spec.seed, "allocation")
            ).permutation(len(fitclasses))
            allocation[order[:remainder]] += 1
            return dict(zip(fitclasses, allocation))

        if spec.class_sampling == "weighted":
            weights = np.asarray(
                [spec.class_weights.get(name, 0.0) for name in fitclasses],
                dtype=float,
            )
        else:
            weights = rates
        if np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("class sampling weights must be non-negative with positive sum")
        rng = np.random.default_rng(_stable_seed(spec.seed, "allocation"))
        allocation = rng.multinomial(int(spec.size), weights / weights.sum())
        return dict(zip(fitclasses, allocation.astype(int)))

    def _draw_redshifts(
        self,
        spec: WarpSampleSpec,
        fitclass: str,
        size: int,
        start: int,
        rng: np.random.Generator,
        volumetric_cdf: Optional[tuple[np.ndarray, np.ndarray]] = None,
    ) -> np.ndarray:
        """Draw one redshift chunk without making its values batch-size dependent."""

        if spec.redshift_sampling == "volumetric":
            if volumetric_cdf is None:
                raise RuntimeError("volumetric redshift CDF was not initialized")
            redshift_grid, cumulative = volumetric_cdf
            return np.interp(rng.random(size), cumulative, redshift_grid)
        if spec.redshift_sampling == "uniform":
            return rng.uniform(spec.zmin, spec.zmax, size=size)

        bins = np.asarray(spec.redshift_bins, dtype=float)
        n_bins = len(bins) - 1
        bin_order = np.random.default_rng(
            _stable_seed(spec.seed, fitclass, "redshift-bin-order")
        ).permutation(n_bins)
        bin_index = bin_order[np.arange(start, start + size) % n_bins]
        return rng.uniform(bins[bin_index], bins[bin_index + 1])

    @staticmethod
    def _iter_skyarea_positions(
        skyarea: Any,
        rng: np.random.Generator,
        block_size: int = 4096,
    ) -> Iterator[np.ndarray]:
        """Yield positions from fixed blocks independent of simulation batches."""

        from skysurvey.tools.utils import random_radec

        while True:
            if hasattr(skyarea, "draw_radec"):
                drawn = skyarea.draw_radec(size=block_size, rng=rng)
            else:
                drawn = random_radec(size=block_size, skyarea=skyarea, rng=rng)
            block = np.asarray(drawn).T
            yield from block

    def iter_batches(
        self,
        spec: WarpSampleSpec,
        *,
        retain_coefficient_cache: bool = False,
    ) -> Iterator[pd.DataFrame]:
        """Yield fitclass-local, entry-grouped batches without spectral models.

        A batch ends at the current fitclass boundary, the target-row limit, or
        the selected-source-key limit. Large entries may span several batches.
        Sequential survey realizations may retain immutable coefficient data.
        """

        counts = self._class_counts(spec)
        tstart, tstop, _ = _normalize_time_window(
            size=(
                0
                if spec.size is not None or spec.class_sampling == "counts"
                else None
            ),
            nyears=spec.nyears,
            tstart=spec.tstart,
            tstop=spec.tstop,
            default_tstart=60000.0,
            default_tstop=60365.25,
        )
        for fitclass in spec.active_fitclasses:
            total = int(counts.get(fitclass, 0))
            if not total:
                continue

            entries = self.loader.get_entry_probabilities(
                fitclass, min_fit_quality=spec.min_fit_quality
            )
            if not entries:
                raise ValueError(f"no eligible Warp entries for {fitclass}")
            probabilities = np.asarray([item[1] for item in entries], dtype=float)
            probabilities /= probabilities.sum()
            template_rng = np.random.default_rng(
                _stable_seed(spec.seed, fitclass, "template-counts")
            )
            entry_counts = template_rng.multinomial(total, probabilities)

            color_mode = self.loader._normalize_color_mode(spec.color_mode)
            model_colors = self.loader.get_model_colors(fitclass)
            if color_mode is not None:
                model_colors = self.loader._validate_model_colors(model_colors)
                color_poly = np.poly1d(model_colors["ebv_corr_func"])
                color_distribution = exponnorm(
                    float(model_colors["K"]),
                    loc=float(model_colors["loc"]),
                    scale=float(model_colors["scale"]),
                )
                color_rng = np.random.default_rng(
                    _stable_seed(spec.seed, fitclass, "color")
                )
            else:
                color_poly = None
                color_distribution = None
                color_rng = None
            redshift_rng = np.random.default_rng(
                _stable_seed(spec.seed, fitclass, "redshift")
            )
            magnitude_rng = np.random.default_rng(
                _stable_seed(spec.seed, fitclass, "magnitude")
            )
            ra_rng = np.random.default_rng(_stable_seed(spec.seed, fitclass, "ra"))
            dec_rng = np.random.default_rng(_stable_seed(spec.seed, fitclass, "dec"))
            time_rng = np.random.default_rng(_stable_seed(spec.seed, fitclass, "time"))
            rate = resolve_rate(fitclass, self.rate_config)
            if spec.redshift_sampling == "volumetric":
                volumetric_cdf = _observer_redshift_cdf(
                    rate=float(rate.rate_gpc3_yr),
                    zmin=spec.zmin,
                    zmax=spec.zmax,
                    cosmology=self.cosmology,
                    grid_size=2048,
                )
            else:
                volumetric_cdf = None

            # Full-sky coordinates use independent streams. Explicit geometries
            # use fixed-size rejection-sampling blocks, so operational batch
            # boundaries do not alter positions or create class-sized arrays.
            if spec.skyarea is None or spec.skyarea == "full":
                position_iter = None
            else:
                position_rng = np.random.default_rng(
                    _stable_seed(spec.seed, fitclass, "position")
                )
                position_iter = self._iter_skyarea_positions(
                    spec.skyarea, position_rng
                )

            frames: list[pd.DataFrame] = []
            batch_rows = 0
            batch_keys: set[str] = set()
            event_index = 0
            for (
                (base_descriptor, _),
                entry_total,
                entry_sampling_probability,
            ) in zip(entries, entry_counts, probabilities):
                remaining = int(entry_total)
                if not remaining:
                    continue
                template_key = base_descriptor.template_key
                while remaining:
                    if frames and (
                        batch_rows >= spec.batch_size
                        or (
                            template_key not in batch_keys
                            and len(batch_keys) >= spec.max_sources_per_batch
                        )
                    ):
                        yield pd.concat(frames, ignore_index=True)
                        frames, batch_keys, batch_rows = [], set(), 0

                    expected = min(remaining, spec.batch_size - batch_rows)
                    redshift = self._draw_redshifts(
                        spec,
                        fitclass,
                        expected,
                        event_index,
                        redshift_rng,
                        volumetric_cdf,
                    )
                    magnitude_draws = [
                        _draw_magabs_with_provenance(
                            fitclass, magnitude_rng, self.rate_config, {}
                        )
                        for _ in range(expected)
                    ]
                    if position_iter is None:
                        ra = ra_rng.uniform(0.0, 360.0, size=expected)
                        dec = np.degrees(
                            np.arcsin(dec_rng.uniform(-1.0, 1.0, size=expected))
                        )
                        radec = np.column_stack([ra, dec])
                    else:
                        radec = np.asarray(list(islice(position_iter, expected)))

                    if color_mode == "draw":
                        target_colors = np.asarray(
                            color_distribution.rvs(
                                size=expected, random_state=color_rng
                            ),
                            dtype=float,
                        )
                    elif color_mode == "harmonize":
                        target_colors = np.full(expected, float(model_colors["loc"]))
                    elif color_mode == "target":
                        if spec.target_peak_color is None:
                            raise ValueError(
                                "target_peak_color is required for color_mode='target'"
                            )
                        target_colors = np.full(
                            expected, float(spec.target_peak_color)
                        )
                    else:
                        target_colors = None

                    if target_colors is None:
                        corrections: Any = [None] * expected
                    else:
                        if base_descriptor.peakcol is None:
                            raise ValueError(
                                f"{template_key} has no peak colour for correction"
                            )
                        corrections = color_poly(
                            target_colors - float(base_descriptor.peakcol)
                        ).astype(float)
                    descriptor = replace(
                        base_descriptor,
                        color_mode=color_mode,
                    )
                    frame = pd.DataFrame(
                        {
                            "fitclass": descriptor.fitclass,
                            "basis_sn": descriptor.basis_sn,
                            "template_index": descriptor.template_index,
                            "template_sn": descriptor.template_sn,
                            # Keep the raw within-basis coefficient-library
                            # weight separate from the joint probability used
                            # for the class-level multinomial draw.
                            "template_prob": descriptor.template_prob,
                            "entry_sampling_probability": float(
                                entry_sampling_probability
                            ),
                            "quality": descriptor.quality,
                            "peakcol": descriptor.peakcol,
                            "target_peak_color": (
                                [None] * expected
                                if target_colors is None
                                else target_colors
                            ),
                            "samplecorr_ebv": corrections,
                            "color_mode": descriptor.color_mode,
                            "template_key": template_key,
                        }
                    )
                    frame["z"] = redshift
                    frame["t0"] = time_rng.uniform(tstart, tstop, size=expected)
                    frame[["ra", "dec"]] = radec
                    frame["magabs"] = [draw[0] for draw in magnitude_draws]
                    frame["distance_modulus"] = self.cosmology.distmod(
                        redshift
                    ).value
                    frame["magabs_band"] = [
                        draw[1].get("sncosmo_band", "bessellb")
                        for draw in magnitude_draws
                    ]
                    frame["magabs_magsys"] = [
                        draw[1].get("magsys", "ab") for draw in magnitude_draws
                    ]
                    frame["magabs_prior_fitclass"] = [
                        draw[2] for draw in magnitude_draws
                    ]
                    frame["magabs_distribution"] = [
                        draw[1].get("distribution") for draw in magnitude_draws
                    ]
                    frame["magabs_status"] = [
                        draw[1].get("status") for draw in magnitude_draws
                    ]
                    frame["magabs_source_band"] = [
                        draw[1].get("source_band") for draw in magnitude_draws
                    ]
                    frame["rate_gpc3_yr"] = rate.rate_gpc3_yr
                    frame["rate_status"] = rate.status
                    frame["rate_kind"] = rate.rate_kind
                    frame["rate_parent"] = rate.parent
                    frame["rate_fraction_of_parent"] = rate.fraction_of_parent
                    frame["object_id"] = [
                        f"{spec.run_name}:{fitclass}:{index:012d}"
                        for index in range(event_index, event_index + expected)
                    ]
                    frame["template"] = "warp-batch|" + frame["object_id"]
                    frame["sampling_class_mode"] = spec.class_sampling
                    frame["sampling_redshift_mode"] = spec.redshift_sampling
                    frames.append(frame)
                    batch_keys.add(template_key)
                    batch_rows += expected
                    event_index += expected
                    remaining -= expected

            if frames:
                yield pd.concat(frames, ignore_index=True)

            # Single runs normally release each class immediately. Ensemble
            # runs keep these small immutable dictionaries for the next survey.
            if not retain_coefficient_cache:
                self.loader.clear_fitclass_cache(fitclass)

    def draw(self, spec: WarpSampleSpec) -> pd.DataFrame:
        """Collect streamed batches for diagnostics and small table-only checks."""

        batches = list(self.iter_batches(spec))
        if not batches:
            return pd.DataFrame(
                columns=["object_id", "fitclass", "basis_sn", "template_sn"]
            )
        return pd.concat(batches, ignore_index=True)


class _WarpBatchTargets:
    """Minimal one-use target container consumed by SkySurvey for one batch."""

    def __init__(
        self,
        data: pd.DataFrame,
        loader: WarpfitTemplateLoader,
        cosmology: Any,
        source_cache: WarpSourceCache,
    ):
        """Store rows and lazily share sources only within this batch."""

        self._data = data
        self.loader = loader
        self.cosmology = cosmology
        self.source_cache = source_cache
        self.sources: dict[str, Any] = {}
        self.source_cache_load_seconds = 0.0
        self.model_build_seconds = 0.0

    def _get_source(self, descriptor: WarpTemplateDescriptor) -> Any:
        """Load one required cached source on its first use in this batch."""

        key = descriptor.template_key
        source = self.sources.get(key)
        if source is not None:
            return source
        started = time.perf_counter()
        source = self.source_cache.load_source(descriptor)
        self.source_cache_load_seconds += time.perf_counter() - started
        if source is None:
            raise RuntimeError(
                "required Warp source is absent from the validated HDF5 cache: "
                f"{descriptor.template_key}"
            )
        self.sources[key] = source
        return source

    @property
    def loaded_source_count(self) -> int:
        """Return how many distinct sources SkySurvey requested in this batch."""

        return len(self.sources)

    @property
    def data(self) -> pd.DataFrame:
        """Return the compact target table expected by SkySurvey."""

        return self._data

    @property
    def ntargets(self) -> int:
        """Return the number of targets in this batch."""

        return len(self._data)

    def get_target_template(
        self,
        index: Any,
        as_model: bool = False,
        set_magabs: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Build and configure the model requested by SkySurvey."""

        started = time.perf_counter()
        row = self.data.loc[index]
        descriptor = WarpTemplateDescriptor(
            **{
                name: row[name]
                for name in WarpTemplateDescriptor.__dataclass_fields__
            }
        )
        source = self._get_source(descriptor)
        model = self.loader.materialize_descriptor(
            descriptor, source=source
        )["model"]
        model.source.name = str(row["template"])

        parameters = {
            name: float(row[name])
            for name in ("z", "t0")
            if name in model.param_names
        }
        parameters.update(kwargs)
        if parameters:
            model.set(**parameters)
        if set_magabs:
            model.set_source_peakabsmag(
                absmag=float(row["magabs"]),
                band=str(row["magabs_band"]),
                magsys=str(row["magabs_magsys"]),
                cosmo=self.cosmology,
            )
        self.model_build_seconds += time.perf_counter() - started

        if as_model:
            return model
        from skysurvey.template import Template

        return Template.from_sncosmo(model)


class WarpSimulationRunner:
    """Run deterministic SkySurvey simulations with bounded memory and output."""

    def __init__(
        self,
        warpcoeffs_dir: str | Path,
        *,
        source_cache_dir: str | Path,
        rate_config: Optional[str | Path | Mapping[str, Any]] = None,
    ):
        """Initialize sampling and the required persistent source cache."""

        self.loader = WarpfitTemplateLoader(str(warpcoeffs_dir))
        self.sampler = _WarpTargetSampler(
            warpcoeffs_dir, rate_config=rate_config, loader=self.loader
        )
        self.source_cache = WarpSourceCache(
            warpcoeffs_dir,
            source_cache_dir,
            loader=self.loader,
        )

    def _cache_manifest(self, fitclasses: Sequence[str]) -> dict[str, Any]:
        """Return stable source-cache provenance for a run manifest."""

        return self.source_cache.describe(fitclasses)

    def clear_coefficient_cache(self) -> None:
        """Release coefficient dictionaries retained across sequential runs."""

        self.loader.clear_cache()

    def _population_config_manifest(self) -> dict[str, Any]:
        """Return the scientific-prior identity that constrains a resumed run."""

        rate_config = self.sampler.rate_config
        return {
            "fingerprint": _mapping_fingerprint(rate_config),
            "schema_version": rate_config.get("schema_version"),
            "units": WarpSampleSpec._json_safe(rate_config.get("units")),
        }

    @staticmethod
    def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
        """Write one Parquet partition and expose it only after success."""

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            frame.to_parquet(temporary, index=False)
            os.replace(temporary, path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    @staticmethod
    def _write_manifest(manifest: Mapping[str, Any], path: Path) -> None:
        """Replace the restart manifest only after the new JSON is complete."""

        temporary = path.with_suffix(".json.tmp")
        try:
            temporary.write_text(
                json.dumps(manifest, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(temporary, path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    @staticmethod
    def _observation_table(dataset: Any, truth: pd.DataFrame) -> pd.DataFrame:
        """Flatten SkySurvey observations and attach persistent object IDs."""

        observations = dataset.data.reset_index()
        target_column = "index" if "index" in observations else observations.columns[0]
        object_ids = truth.reset_index(drop=True)["object_id"]
        observations["object_id"] = observations[target_column].map(object_ids)
        return observations

    @staticmethod
    def _empty_observation_table(survey: Any) -> pd.DataFrame:
        """Return the ordinary observation schema for a zero-match batch."""

        field_names = list(survey.fieldids.names)
        survey_columns = [
            "mjd",
            "band",
            "skynoise",
            "gain",
            "zp",
            *field_names,
        ]
        observations = survey.data[survey_columns].iloc[0:0].copy()
        observations.index = pd.Index(
            [],
            dtype=survey.data.index.dtype,
            name=survey.data.index.name or "index_obs",
        )
        observations = observations.reset_index()
        observations.insert(0, "index", pd.Series(dtype=np.int64))
        observations["flux"] = pd.Series(dtype=float)
        observations["fluxerr"] = pd.Series(dtype=float)
        observations["object_id"] = pd.Series(dtype=object)
        return observations

    @staticmethod
    def _observed_field_matches(survey: Any, radec: pd.DataFrame) -> pd.DataFrame:
        """Match coordinates and filter complete keys to observed log rows."""

        field_names = list(survey.fieldids.names)
        try:
            try:
                matches = survey.radec_to_fieldid(
                    radec,
                    observed_fields=True,
                )
            except TypeError:
                matches = survey.radec_to_fieldid(radec)
        except ValueError as error:
            message = str(error).lower()
            if (
                "need at least one array to stack" in message
                or "no objects to concatenate" in message
            ):
                return pd.DataFrame(columns=field_names, index=radec.index[:0])
            raise
        if matches.empty:
            return matches

        observed = survey.data[field_names].drop_duplicates(ignore_index=True)
        if len(field_names) == 1:
            keep = matches[field_names[0]].isin(observed[field_names[0]])
        else:
            match_index = pd.MultiIndex.from_frame(matches[field_names])
            observed_index = pd.MultiIndex.from_frame(observed[field_names])
            keep = match_index.isin(observed_index)
        return matches.loc[np.asarray(keep, dtype=bool)]

    @staticmethod
    def _scatter_observations(observations: pd.DataFrame, seed: int) -> None:
        """Add reproducible per-object noise without repeated DataFrame writes."""

        flux = observations["flux"].to_numpy(copy=True)
        fluxerr = observations["fluxerr"].to_numpy()
        for object_id, positions in observations.groupby(
            "object_id", sort=False
        ).indices.items():
            rng = np.random.default_rng(_stable_seed(seed, object_id, "noise"))
            flux[positions] += rng.normal(loc=0.0, scale=fluxerr[positions])
        observations["flux"] = flux

    @staticmethod
    def _report_batch_progress(
        manifest: Mapping[str, Any],
        spec: WarpSampleSpec,
        *,
        batch_id: str,
        fitclass: str,
        status: str,
        started: float,
        new_truth_rows: int,
    ) -> None:
        """Print compact cumulative progress for an explicitly monitored run."""

        completed = [
            item
            for item in manifest.get("batches", {}).values()
            if item.get("status") == "complete"
        ]
        truth_rows = sum(int(item.get("truth_rows", 0)) for item in completed)
        observation_rows = sum(
            int(item.get("observation_rows", 0)) for item in completed
        )
        elapsed = max(time.perf_counter() - started, np.finfo(float).eps)
        percentage = 100.0 * truth_rows / max(int(spec.size), 1)
        new_rate = new_truth_rows / elapsed
        print(
            f"[Warp progress] {percentage:6.2f}% | batch {batch_id} {status} | "
            f"{fitclass} | {truth_rows:,}/{int(spec.size):,} targets | "
            f"{observation_rows:,} observations | {new_rate:,.1f} new targets/s"
        )

    def run(
        self,
        spec: WarpSampleSpec,
        output_dir: str | Path | Any,
        survey: Any = None,
        *,
        resume: bool = True,
        progress_every_batches: Optional[int] = None,
        retain_coefficient_cache: bool = False,
    ) -> dict[str, Any]:
        """Draw, simulate, persist, and optionally retain coefficient data."""

        from skysurvey import DataSet

        if progress_every_batches is not None:
            progress_every_batches = int(progress_every_batches)
            if progress_every_batches <= 0:
                raise ValueError("progress_every_batches must be positive or None")
        run_started = time.perf_counter()
        new_truth_rows = 0

        # Accept both the new run(spec, output_dir, survey=...) API and the
        # historical run(spec, survey, output_dir) positional order.
        if isinstance(output_dir, (str, os.PathLike)):
            resolved_output_dir = Path(output_dir)
            resolved_survey = survey
        elif isinstance(survey, (str, os.PathLike)):
            resolved_output_dir = Path(survey)
            resolved_survey = output_dir
        else:
            raise TypeError(
                "run expects run(spec, output_dir, survey=...) or "
                "run(spec, survey, output_dir)"
            )

        # Build missing or stale partitions before any target drawing starts.
        self.source_cache.build(spec.active_fitclasses)
        run_dir = resolved_output_dir / spec.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / "manifest.json"
        cache_manifest = self._cache_manifest(spec.active_fitclasses)
        population_config = self._population_config_manifest()
        if manifest_path.exists() and resume:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != SIMULATION_SCHEMA_VERSION:
                raise ValueError(
                    "existing manifest uses an incompatible simulation schema; "
                    "choose a new run_name"
                )
            if manifest.get("spec") != spec.to_dict():
                raise ValueError(
                    "existing manifest belongs to a different sample specification"
                )
            if manifest.get("source_cache") != cache_manifest:
                raise ValueError(
                    "source-cache provenance changed since this run started; "
                    "choose a new run_name"
                )
            if manifest.get("population_config") != population_config:
                raise ValueError(
                    "population rate or magnitude configuration changed since "
                    "this run started; choose a new run_name"
                )
        elif manifest_path.exists():
            raise FileExistsError(
                f"{run_dir} already contains a run; use resume=True or a new run_name"
            )

        if resolved_survey is None:
            from .survey_factory import SurveyFactory

            resolved_survey = SurveyFactory().from_spec(spec)
        survey_provenance = getattr(resolved_survey, "warp_provenance", None)
        if survey_provenance is None:
            survey_provenance = {
                "name": spec.survey_name,
                "class": (
                    f"{resolved_survey.__class__.__module__}."
                    f"{resolved_survey.__class__.__name__}"
                ),
                "rows": len(resolved_survey.data),
                "columns": list(map(str, resolved_survey.data.columns)),
                "field_keys": list(map(str, resolved_survey.fieldids.names)),
                "data_fingerprint": _dataframe_fingerprint(
                    resolved_survey.data
                ),
                "description": "Explicitly supplied SkySurvey-compatible survey.",
            }
            explicit_fields = getattr(resolved_survey, "fields", None)
            if isinstance(explicit_fields, pd.DataFrame):
                survey_provenance["field_geometry_fingerprint"] = (
                    _dataframe_fingerprint(explicit_fields)
                )

        if manifest_path.exists():
            if manifest.get("survey") != survey_provenance:
                raise ValueError(
                    "survey provenance changed since this run started; "
                    "choose a new run_name"
                )
            recorded_batches = manifest.get("batches", {}).values()
            if manifest.get("status") == "complete" and all(
                (run_dir / item["truth"]).exists()
                and (run_dir / item["observations"]).exists()
                for item in recorded_batches
            ):
                if progress_every_batches is not None:
                    print(
                        f"[Warp progress] 100.00% | run already complete | "
                        f"{int(manifest.get('truth_rows', 0)):,} targets | "
                        f"{int(manifest.get('observation_rows', 0)):,} observations"
                    )
                return manifest
        else:
            manifest = {
                "schema_version": SIMULATION_SCHEMA_VERSION,
                "status": "running",
                "spec": spec.to_dict(),
                "software": {
                    "python": platform.python_version(),
                    "warpTemplate": _package_version("warpTemplate"),
                    "sncosmo": _package_version("sncosmo"),
                    "skysurvey": _package_version("skysurvey"),
                    "color_engine": "dynamic-ccm89-v1",
                },
                "survey": survey_provenance,
                "source_cache": cache_manifest,
                "population_config": population_config,
                "truth_schema": {
                    "template_prob": (
                        "raw coefficient-library draw_prob within basis_sn"
                    ),
                    "entry_sampling_probability": (
                        "normalized fitclass-local probability used for the "
                        "entry-count multinomial"
                    ),
                    "distance_modulus": "cosmology.distmod(z), in magnitudes",
                    "survey_realization_id": (
                        "optional cadence-window group for leakage-safe dataset splits"
                    ),
                },
                "batching": {
                    "scope": "fitclass-local",
                    "row_limit": int(spec.batch_size),
                    "selected_source_key_limit": int(
                        spec.max_sources_per_batch
                    ),
                    "loaded_source_count": (
                        "distinct sources requested by SkySurvey after field "
                        "selection"
                    ),
                    "resume": (
                        "skip complete batches and recompute an interrupted "
                        "batch from its beginning"
                    ),
                },
                "batches": {},
            }
            self._write_manifest(manifest, manifest_path)

        sampling_skyarea = getattr(
            resolved_survey,
            "simulation_skyarea",
            None,
        )
        sampling_spec = (
            replace(spec, skyarea=sampling_skyarea)
            if spec.skyarea is None and sampling_skyarea is not None
            else spec
        )
        uses_survey_sampling_domain = (
            spec.skyarea is None and sampling_skyarea is not None
        )
        if retain_coefficient_cache:
            batch_iterator = iter(
                self.sampler.iter_batches(
                    sampling_spec,
                    retain_coefficient_cache=True,
                )
            )
        else:
            # Preserve the historical call shape for custom sampler wrappers.
            batch_iterator = iter(self.sampler.iter_batches(sampling_spec))
        batch_number = 0
        try:
            while True:
                draw_started = time.perf_counter()
                try:
                    batch = next(batch_iterator)
                except StopIteration:
                    break
                draw_seconds = time.perf_counter() - draw_started
                batch_id = f"{batch_number:08d}"
                fitclass = str(batch.iloc[0]["fitclass"])
                truth_path = (
                    run_dir
                    / "truth"
                    / f"fitclass={fitclass}"
                    / f"batch-{batch_id}.parquet"
                )
                observations_path = (
                    run_dir
                    / "observations"
                    / f"fitclass={fitclass}"
                    / f"batch-{batch_id}.parquet"
                )
                recorded = manifest["batches"].get(batch_id, {})
                if (
                    resume
                    and recorded.get("status") == "complete"
                    and truth_path.exists()
                    and observations_path.exists()
                ):
                    batch_number += 1
                    if (
                        progress_every_batches is not None
                        and batch_number % progress_every_batches == 0
                    ):
                        self._report_batch_progress(
                            manifest,
                            spec,
                            batch_id=batch_id,
                            fitclass=fitclass,
                            status="resumed",
                            started=run_started,
                            new_truth_rows=new_truth_rows,
                        )
                    continue

                targets = _WarpBatchTargets(
                    batch,
                    self.loader,
                    self.sampler.cosmology,
                    self.source_cache,
                )
                simulation_started = time.perf_counter()
                if (
                    not uses_survey_sampling_domain
                    and self._observed_field_matches(
                        resolved_survey,
                        batch[["ra", "dec"]],
                    ).empty
                ):
                    dataset = None
                else:
                    dataset = DataSet.from_targets_and_survey(
                        targets,
                        resolved_survey,
                        # SkySurvey computes noiseless flux and fluxerr. Object-level
                        # scatter below is invariant to operational batch boundaries.
                        incl_error=False,
                        phase_range=spec.phase_range,
                    )
                skysurvey_seconds = time.perf_counter() - simulation_started
                truth = targets.data.reset_index(drop=True)
                observations = (
                    self._empty_observation_table(resolved_survey)
                    if dataset is None
                    else self._observation_table(dataset, truth)
                )
                # Persist cadence identity per row so downstream splits need no
                # directory-name parsing and cannot accidentally mix seasons.
                if spec.survey_realization_id is not None:
                    truth["survey_realization_id"] = str(
                        spec.survey_realization_id
                    )
                    observations["survey_realization_id"] = str(
                        spec.survey_realization_id
                    )
                if spec.incl_error:
                    self._scatter_observations(observations, spec.seed)

                write_started = time.perf_counter()
                self._atomic_parquet(truth, truth_path)
                self._atomic_parquet(observations, observations_path)
                write_seconds = time.perf_counter() - write_started
                template_keys = list(dict.fromkeys(batch["template_key"]))
                manifest["batches"][batch_id] = {
                    "status": "complete",
                    "fitclass": fitclass,
                    "template_keys": template_keys,
                    "template_count": len(template_keys),
                    "loaded_source_count": targets.loaded_source_count,
                    "truth_rows": len(truth),
                    "observation_rows": len(observations),
                    "truth": str(truth_path.relative_to(run_dir)),
                    "observations": str(observations_path.relative_to(run_dir)),
                    "timing_seconds": {
                        "target_drawing": draw_seconds,
                        "model_building": targets.model_build_seconds,
                        "source_cache_loading": (
                            targets.source_cache_load_seconds
                        ),
                        "skysurvey_total": skysurvey_seconds,
                        "parquet_writing": write_seconds,
                    },
                }
                self._write_manifest(manifest, manifest_path)

                new_truth_rows += len(truth)
                if (
                    progress_every_batches is not None
                    and (batch_number + 1) % progress_every_batches == 0
                ):
                    self._report_batch_progress(
                        manifest,
                        spec,
                        batch_id=batch_id,
                        fitclass=fitclass,
                        status="complete",
                        started=run_started,
                        new_truth_rows=new_truth_rows,
                    )

                del observations, truth, dataset, targets, batch
                batch_number += 1
        finally:
            batch_iterator.close()
            if not retain_coefficient_cache:
                self.clear_coefficient_cache()

        manifest["status"] = "complete"
        manifest["truth_rows"] = sum(
            item["truth_rows"] for item in manifest["batches"].values()
        )
        manifest["observation_rows"] = sum(
            item["observation_rows"] for item in manifest["batches"].values()
        )
        self._write_manifest(manifest, manifest_path)
        if (
            progress_every_batches is not None
            and batch_number % progress_every_batches != 0
        ):
            last_batch_id = f"{max(batch_number - 1, 0):08d}"
            last_fitclass = str(
                manifest["batches"].get(last_batch_id, {}).get("fitclass", "-")
            )
            self._report_batch_progress(
                manifest,
                spec,
                batch_id=last_batch_id,
                fitclass=last_fitclass,
                status="complete",
                started=run_started,
                new_truth_rows=new_truth_rows,
            )
        return manifest

    def run_many(
        self,
        jobs: Iterable[WarpSampleSpec | tuple[WarpSampleSpec, Any]],
        output_dir: str | Path,
        *,
        resume: bool = True,
    ) -> list[dict[str, Any]]:
        """Run automatic specs or explicit sample/survey pairs sequentially."""

        manifests = []
        try:
            for job in jobs:
                if isinstance(job, WarpSampleSpec):
                    manifests.append(
                        self.run(
                            job,
                            output_dir,
                            resume=resume,
                            retain_coefficient_cache=True,
                        )
                    )
                else:
                    spec, survey = job
                    manifests.append(
                        self.run(
                            spec,
                            output_dir,
                            survey=survey,
                            resume=resume,
                            retain_coefficient_cache=True,
                        )
                    )
        finally:
            # Sequential jobs share immutable coefficient dictionaries, then
            # release them once the whole collection finishes or fails.
            self.clear_coefficient_cache()
        return manifests


__all__ = [
    "WarpSampleSpec",
    "WarpSimulationRunner",
]
