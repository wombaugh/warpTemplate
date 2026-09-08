# warp_templates/loaders.py
import pickle
import logging
import re
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Union, List, Dict, Mapping
import numpy as np
from scipy.stats import exponnorm
from .models import get_warpedTimeSeriesModel


_QUALITY_RANK = {
    "bronze": 0,
    "silver": 1,
    "gold": 2,
}


@dataclass(frozen=True)
class WarpTemplateDescriptor:
    """Serializable reference to one selected Warp template realization."""

    fitclass: str
    basis_sn: str
    template_index: int
    template_sn: str
    template_prob: float
    quality: Optional[str]
    peakcol: Optional[float]
    target_peak_color: Optional[float]
    samplecorr_ebv: Optional[float]
    color_mode: Optional[str]

    @property
    def template_key(self) -> str:
        """Return the stable coefficient-library key for this template entry."""

        return f"{self.fitclass}|{self.basis_sn}|{self.template_index}"

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON- and DataFrame-friendly descriptor mapping."""

        return {**asdict(self), "template_key": self.template_key}


class WarpfitTemplateLoader:
    """
    Loader and sampler for warped time-series templates stored on disk.

    This class loads precomputed warp coefficient files (typically `.pkl`)
    and constructs `sncosmo.Model` instances using
    `get_warpedTimeSeriesModel`.

    It supports:
    - Caching of loaded files
    - Filtering of supernova (SN) bases
    - Random or exhaustive sampling of templates
    - Reproducible random selection via seed

    Parameters
    ----------
    warpcoeffs_dir : str
        Directory containing warp coefficient files of the form:
            warpcoeffs_<fitclass>.pkl
    logger : logging.Logger, optional
        Logger instance. If None, a default logger is created.
    openuniverse_dir : path-like, optional
        Directory containing the external OpenUniverse SED folders. By default,
        it is resolved beside the configured coefficient directory.

    Notes
    -----
    The expected structure of each `.pkl` file is:

        {
            "warpcoeff": {
                "ZTF18xxxxx": [           # Base SN-ID
                    {
                        # Identification
                        "id": "ZTF18xxxxx",
                        "model": "v19-2006bp",    # Base Template
                        "z": 0.045,
                        # Quality and selection
                        "quality": "gold",        # gold | silver | bronze
                        "draw_prob": 0.312,       # relative sampling weight

                        # Interpolated peak color information of original data
                        "peak_gp_ztfg-ztfr": 0.127,         # native peak color g-r
                        "type": "SN IIP",

                        # Warp data for template construction
                        "mdict": {
                            "warpfit_tmin": -15.0,
                            "warpfit_tmax": 80.0,
                            # ... further keys for get_warpedTimeSeriesModel()
                        },
                        # Fit result of warped model
                        "wresult": {
                            "parameters": [...],
                            "chisq": 45.3,
                            "ndof": 38,
                            # ...
                        },

                        # Survival function of warped model
                        "sf": 0.89,
                    },
                    # ... further Templates for the same SN
                ],
                # ... further SNe
            },

            "model_colors": {                 # Peak color distribution of class (optional)
                "K": 0.908,
                "loc": 0.125,
                "scale": 0.151,
                "color1": "ztfg",
                "color2": "ztfr",
                "ebv_corr_func": [-0.0001, 0.01, 0.5]
            }
        }
    Each `mdict` must match the expected input of
    `get_warpedTimeSeriesModel`.
    """


    def __init__(
        self,
        warpcoeffs_dir: str,
        version: str = "4",
        suffix: str = "_col",
        logger: Optional[logging.Logger] = None,
        openuniverse_dir: Optional[str | Path] = None,
    ):
        """Initialize the coefficient directory, logger, and fitclass cache."""

        self.warpcoeffs_dir = warpcoeffs_dir
        self.version = str(version)
        self.suffix = str(suffix)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.openuniverse_dir = (
            Path(openuniverse_dir)
            if openuniverse_dir is not None
            else Path(warpcoeffs_dir).resolve().parent / "openuniverse_templates"
        )

        if logger is None:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(message)s"
            )
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

    # -------------------------
    # Internal: load with cache
    # -------------------------
    def _load_coeffs(self, fitclass: str) -> Mapping[str, Any]:
        """
        Load warp coefficient data for a given fit class, with caching.

        Parameters
        ----------
        fitclass : str
            Identifier used to construct filename:
                warpcoeffs_<fitclass>_col.pkl
                _col suffix indicates that the file contains color correction data (ebv_meancol_corr)
            Forward slashes are sanitized from the key.

        Returns
        -------
        dict
            Parsed contents of the pickle file with normalized structure:
            {"warpcoeff": {...}, "model_colors": {...} or None}

        Raises
        ------
        FileNotFoundError
            If the corresponding file does not exist.
        """
        key = re.sub(r"/", "", fitclass)

        if key in self._cache:
            self.logger.debug(f"Cache hit for fitclass={key}")
            return self._cache[key]

        filepath = self.coefficient_path(fitclass)
        self.logger.info(f"Loading warpcoeffs from {filepath}")

        if not filepath.exists():
            self.logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(filepath)

        with filepath.open("rb") as f:
            raw_data = pickle.load(f)

        # Normalize to unified structure
        if isinstance(raw_data, dict) and "warpcoeff" in raw_data:
            data = raw_data
        elif isinstance(raw_data, dict) and all(
            isinstance(v, list) for v in raw_data.values()
        ):
            # Legacy flat format: {sn_id: [warpfits, ...], ...}
            data = {
                "warpcoeff": raw_data,
                "model_colors": None,
            }
        else:
            raise ValueError(
                f"Unrecognized warp coefficient file structure in {filepath}. "
                "Expected dict with 'warpcoeff' key or flat {{sn_id: [entries]}} mapping."
            )

        self._cache[key] = data
        return data

    def coefficient_path(self, fitclass: str) -> Path:
        """Return the configured path, falling back to an existing v3 pickle."""

        key = re.sub(r"/", "", fitclass)
        configured = (
            Path(self.warpcoeffs_dir)
            / f"warpcoeffs_v{self.version}_{key}{self.suffix}.pkl"
        )
        legacy = Path(self.warpcoeffs_dir) / f"warpcoeffs_v3_{key}.pkl"
        if not configured.exists() and legacy.exists():
            return legacy
        return configured

    def available_fitclasses(self) -> List[str]:
        """Discover fitclasses represented by coefficient files on disk."""

        prefix = f"warpcoeffs_v{self.version}_"
        suffix = f"{self.suffix}.pkl"
        configured = sorted(
            path.name[len(prefix) : -len(suffix)]
            for path in Path(self.warpcoeffs_dir).glob(f"{prefix}*{suffix}")
        )
        if configured:
            return configured

        # Existing v3 libraries remain usable while v4 data is rolled out.
        legacy_prefix = "warpcoeffs_v3_"
        return sorted(
            path.name[len(legacy_prefix) : -len(".pkl")]
            for path in Path(self.warpcoeffs_dir).glob(f"{legacy_prefix}*.pkl")
        )

    def get_coefficient_entry(
        self, fitclass: str, basis_sn: str, template_index: int
    ) -> Mapping[str, Any]:
        """Resolve one stable basis/index reference in the coefficient library."""

        collection = self._load_coeffs(fitclass)
        try:
            return collection["warpcoeff"][basis_sn][int(template_index)]
        except (KeyError, IndexError) as error:
            raise KeyError(
                f"unknown Warp entry {fitclass}|{basis_sn}|{template_index}"
            ) from error

    def get_entry_probabilities(
        self,
        fitclass: str,
        *,
        min_fit_quality: Optional[str] = None,
    ) -> List[tuple[WarpTemplateDescriptor, float]]:
        """Return eligible entries and their hierarchical draw probabilities."""

        if min_fit_quality is not None:
            quality = min_fit_quality.lower()
            if quality not in _QUALITY_RANK:
                raise ValueError(
                    "min_fit_quality must be one of: 'gold', 'silver', 'bronze'"
                )
            threshold = _QUALITY_RANK[quality]
        else:
            threshold = -1

        collection = self._load_coeffs(fitclass)
        valid: List[tuple[str, List[tuple[int, Mapping[str, Any]]]]] = []
        for basis_sn, entries in collection["warpcoeff"].items():
            selected = [
                (index, entry)
                for index, entry in enumerate(entries)
                if _QUALITY_RANK.get(entry.get("quality"), -1) >= threshold
            ]
            if selected:
                valid.append((str(basis_sn), selected))
        if not valid:
            return []

        result: List[tuple[WarpTemplateDescriptor, float]] = []
        basis_probability = 1.0 / len(valid)
        for basis_sn, entries in valid:
            weights = np.asarray(
                [float(entry.get("draw_prob", 0.0)) for _, entry in entries],
                dtype=float,
            )
            if np.any(weights < 0):
                raise ValueError(
                    f"negative template draw probability in {fitclass}|{basis_sn}"
                )
            conditional = (
                weights / weights.sum()
                if weights.sum() > 0
                else np.full(len(entries), 1.0 / len(entries))
            )
            for (template_index, entry), probability in zip(entries, conditional):
                descriptor = WarpTemplateDescriptor(
                    fitclass=str(fitclass),
                    basis_sn=basis_sn,
                    template_index=int(template_index),
                    template_sn=str(entry.get("model")),
                    template_prob=float(entry.get("draw_prob", 0.0)),
                    quality=entry.get("quality"),
                    peakcol=entry.get("peakcol"),
                    target_peak_color=None,
                    samplecorr_ebv=None,
                    color_mode=None,
                )
                result.append((descriptor, basis_probability * float(probability)))
        return result

    def build_uncolored_source(
        self, descriptor: WarpTemplateDescriptor | Mapping[str, Any]
    ) -> Any:
        """Build the reusable colour-neutral source referenced by a descriptor."""

        if not isinstance(descriptor, WarpTemplateDescriptor):
            fields = WarpTemplateDescriptor.__dataclass_fields__
            descriptor = WarpTemplateDescriptor(
                **{key: descriptor[key] for key in fields}
            )
        entry = self.get_coefficient_entry(
            descriptor.fitclass,
            descriptor.basis_sn,
            descriptor.template_index,
        )
        corr = entry["mdict"]["corrmodel"]
        from .sources import DynamicColorWarpSource

        self._ensure_base_source(descriptor.template_sn)

        return DynamicColorWarpSource.from_warp_grid(
            corr["phase"],
            corr["wave"],
            corr["flux"],
            descriptor.template_sn,
            name=f"{descriptor.basis_sn}_{descriptor.template_sn}",
        )

    def _ensure_base_source(self, template_name: str) -> None:
        """Register a known external v4 base using the coefficient data root."""

        from .openuniverse_registry import ensure_registered

        ensure_registered(template_name, base_dir=self.openuniverse_dir)

    # -------------------------
    # Public: access and mutate loaded data
    # -------------------------
    def get_warpcoeff(self, fitclass: str) -> Dict[str, List[Dict]]:
        """Return raw warp coefficient dictionary for a class."""
        return self._load_coeffs(fitclass)["warpcoeff"]

    def get_model_colors(self, fitclass: str) -> Optional[Dict[str, Any]]:
        """Return a copy of class-level peak-colour metadata when present."""

        model_colors = self._load_coeffs(fitclass).get("model_colors")
        return dict(model_colors) if model_colors is not None else None

    def update_warpcoeff(self, fitclass: str, warpcoeff: Dict[str, List[Dict]]) -> None:
        """Update cached warp coefficients (for in-place mutation by analysis scripts)."""
        key = re.sub(r"/", "", fitclass)
        if key not in self._cache:
            self._load_coeffs(fitclass)
        self._cache[key]["warpcoeff"] = warpcoeff

    def update_model_colors(self, fitclass: str, model_colors: Dict[str, Any]) -> None:
        """Update cached model colors metadata."""
        key = re.sub(r"/", "", fitclass)
        if key not in self._cache:
            self._load_coeffs(fitclass)
        self._cache[key]["model_colors"] = model_colors

    def save_class(
        self, fitclass: str, filepath: Optional[str | Path] = None
    ) -> str:
        """Persist cached data for a class to disk.

        Parameters
        ----------
        fitclass : str
            Class identifier (forward slashes sanitized).
        filepath : str, optional
            Explicit output path. If None, uses standard naming convention
            based on version and suffix configured at initialization.

        Returns
        -------
        str
            Path to which the data was written.
        """
        key = re.sub(r"/", "", fitclass)
        if key not in self._cache:
            raise ValueError(f"No cached data for class {fitclass}")

        output_path = (
            Path(filepath)
            if filepath is not None
            else Path(self.warpcoeffs_dir)
            / f"warpcoeffs_v{self.version}_{key}{self.suffix}.pkl"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("wb") as handle:
            pickle.dump(self._cache[key], handle)

        self.logger.info("Saved warp coefficients to %s", output_path)
        return str(output_path)

    # -------------------------
    # Color mode handling
    # -------------------------
    @staticmethod
    def _normalize_color_mode(color_mode: Optional[str]) -> Optional[str]:
        """Normalize supported colour-mode spellings and reject unknown modes."""

        if color_mode is None:
            return None

        mode = color_mode.lower()
        if mode == "none":
            return None

        allowed = {"harmonize", "draw", "target"}
        if mode not in allowed:
            raise ValueError(
                f"Invalid color_mode: {color_mode}. "
                "Must be one of None, 'none', 'harmonize', 'draw', 'target'."
            )
        return mode

    @staticmethod
    def _validate_model_colors(model_colors: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Return complete class-colour metadata or raise a clear error."""

        if model_colors is None:
            raise ValueError(
                "color_mode requires 'model_colors' in the warp coefficient file"
            )

        required = ("K", "loc", "scale", "color1", "color2", "ebv_corr_func")
        missing = [key for key in required if key not in model_colors]
        if missing:
            raise ValueError(
                "Incomplete model_colors metadata. "
                f"Missing keys: {', '.join(missing)}"
            )
        return model_colors

    @staticmethod
    def _color_correction_ebv(
        *,
        warpfit: Mapping[str, Any],
        target_peak_color: float,
        color_poly: np.poly1d,
    ) -> float:
        """Map a requested peak-colour change to the internal correction."""

        if "peakcol" not in warpfit:
            raise ValueError(
                "Cannot apply color correction because the warpfit entry "
                "does not contain 'peakcol'. Run color analysis pipeline first."
            )


        color_delta = float(target_peak_color) - float(warpfit["peakcol"])
        return float(color_poly(color_delta))

    # -------------------------
    # Main template loading
    # -------------------------
    def get_templates(
        self,
        fitclass: str,
        exclude_input: Optional[list] = None,
        template_selection: Union[int, str] = 1,
        snbasis_selection: Union[int, str] = 1,
        min_fit_quality: Optional[str] = None,
        random_seed: Optional[int] = None,
        color_mode: Optional[str] = None,
        target_peak_color: Optional[float] = None,
    ) -> List[Dict]:
        """
        Load, filter, and sample warped templates as `sncosmo.Model` objects.

        This method performs three main steps:
        1. Load warp coefficient data (cached)
        2. Filter SN bases and templates
        3. Sample SN bases and templates according to selection rules

        Parameters
        ----------
        fitclass : str
            Identifier for the warp coefficient file.
        exclude_input : list of str, optional
            List of SN names to exclude. Applies to both:
            - SN bases (`ztfid`)
            - Template names (`warpfit["model"]`)
        template_selection : int or "all", optional (default=1)
            Controls how many templates to draw *per SN basis*:

            - "all" → return all available templates
            - int > 0 → weighted random sampling using `draw_prob`
            - int < 0 → uniform random sampling (ignore weights),
                        using `abs(template_selection)` samples

        snbasis_selection : int or "all", optional (default=1)
            Controls how many SN bases to select:

            - "all" → use all valid SN bases
            - int → randomly sample SN bases with replacement

        min_fit_quality : str, optional (default=None)
            Minimum fit quality tier for inclusion:
            - "gold" → only best fits, type-compatible
            - "silver" → good fits or type-compatible
            - "bronze" → all SN bases regardless

        random_seed : int, optional
            Seed for reproducible random sampling.

        color_mode : str, optional (default=None)
            - "harmonize": warp to class mean peak color
            - "draw": draw peak color from observed distribution
            - "target": warp to specified `target_peak_color`

        target_peak_color : float, optional
            Peak color to use when `color_mode="target"`.

        Returns
        -------
        list of dict
            Each element has the form:

            {
                "basis_sn": str,
                "model": sncosmo.Model,
                "template_prob": float,
                "template_sn": str,
                "quality": str,
                "peakcol": float,
                "target_peak_color": float,
                "samplecorr_ebv": float,
                "model_colors": dict,
            }

        Notes
        -----
        - Sampling is done **with replacement** (via `random.choices`)
        - If no valid SN bases or templates remain after filtering,
          an empty list is returned
        - Model construction failures are logged and skipped
        """

        exclude_input = exclude_input or []
        color_mode = self._normalize_color_mode(color_mode)

        if color_mode == "target" and target_peak_color is None:
            raise ValueError("target_peak_color must be provided when color_mode='target'")

        # -------------------------
        # Local RNG (reproducible)
        # -------------------------
        rng = random.Random(random_seed)
        np_rng = np.random.default_rng(random_seed)

        if random_seed is not None:
            self.logger.info(f"Using random seed: {random_seed}")

        template_collection = self._load_coeffs(fitclass)
        warpcoeff = template_collection["warpcoeff"]
        model_colors = template_collection.get("model_colors")

        if color_mode is not None:
            model_colors = self._validate_model_colors(model_colors)
            color_poly = np.poly1d(model_colors["ebv_corr_func"])
#            print('... initialized color_poly:', color_poly)
            color_distribution = exponnorm(
                float(model_colors["K"]),
                loc=float(model_colors["loc"]),
                scale=float(model_colors["scale"]),
            )
#            print('... initialized color_distribution with K, loc, scale =',
#                  model_colors["K"], model_colors["loc"], model_colors["scale"])
        else:
            color_poly = None
            color_distribution = None

        def iter_drawn_colors() -> Iterator[float]:
            """Yield SciPy colour draws from fixed blocks to reduce call overhead."""

            while True:
                values = color_distribution.rvs(size=4096, random_state=np_rng)
                yield from np.asarray(values, dtype=float)

        drawn_colors = iter_drawn_colors() if color_mode == "draw" else None

        # -------------------------
        # Limit to quality requirement
        # -------------------------
        if min_fit_quality is not None:
            min_fit_quality = min_fit_quality.lower()

            if min_fit_quality not in _QUALITY_RANK:
                raise ValueError(
                    f"Invalid min_fit_quality: {min_fit_quality}. "
                    "Must be one of: 'gold', 'silver', 'bronze'"
                )

            threshold = _QUALITY_RANK[min_fit_quality]

            self.logger.info(
                f"Filtering SN bases with min_fit_quality={min_fit_quality}"
            )
            filtered_collection = {}
            for sn_name, warpmodels in warpcoeff.items():
                cutmodels = [
                    wm
                    for wm in warpmodels
                    if _QUALITY_RANK.get(wm.get("quality"), -1) >= threshold
                ]

                self.logger.debug(
                    f"SN basis {sn_name}: {len(cutmodels)} templates "
                    f"after quality filtering ({len(warpmodels)} original)"
                )
                if len(cutmodels) > 0:
                    filtered_collection[sn_name] = cutmodels
            warpcoeff = filtered_collection

        # -------------------------
        # Filter SN bases
        # -------------------------
        valid_snbases = [
            sn_name for sn_name in warpcoeff.keys() if sn_name not in exclude_input
        ]
        if not valid_snbases:
            self.logger.warning("No valid SN bases after filtering")
            return []

        self.logger.info(f"{len(valid_snbases)} SN bases available after filtering")

        # -------------------------
        # Select SN bases
        # -------------------------
        if snbasis_selection == "all":
            selected_snbases = valid_snbases

        elif isinstance(snbasis_selection, int):
            selected_snbases = rng.choices(
                valid_snbases,
                k=snbasis_selection
            )

        else:
            raise ValueError("snbasis_selection must be int or 'all'")

        results = []

        # -------------------------
        # Loop SN bases
        # -------------------------
        for sn_name in selected_snbases:
            warpmodels = warpcoeff[sn_name]

            possible_templates = []

            for warpfit in warpmodels:

                template_sn = warpfit.get("model")

                if template_sn in exclude_input:
                    self.logger.debug(
                        f"Excluded template {template_sn} (basis {sn_name})"
                    )
                    continue

                if color_mode == "harmonize":
                    applied_target_peak_color = float(color_distribution.median())
                elif color_mode == "draw":
                    applied_target_peak_color = float(next(drawn_colors))
                elif color_mode == "target":
                    applied_target_peak_color = float(target_peak_color)
                else:
                    applied_target_peak_color = None
                    samplecorr_ebv = None

                if applied_target_peak_color is not None:
                    samplecorr_ebv = self._color_correction_ebv(
                        warpfit=warpfit,
                        target_peak_color=applied_target_peak_color,
                        color_poly=color_poly,
                    )
#                    print(f"Color mode {color_mode}: applied_target_peak_color={applied_target_peak_color}, samplecorr_ebv={samplecorr_ebv}")

                try:
                    self._ensure_base_source(template_sn)
                    model = get_warpedTimeSeriesModel(
                        name=f"{sn_name}_{template_sn or 'tpl'}",
                        original_template_name=template_sn,
                        warpdata=warpfit["mdict"],
                        z=float(warpfit["z"]),
                        original_template_version=None,
                        samplecorr_ebv=samplecorr_ebv,
                        samplecorr_rv=3.1,
                        samplecorr_bands=[
                            model_colors["color1"],
                            model_colors["color2"],
                        ] if model_colors else None,
                    )
                except Exception as e:
                    self.logger.error(
                        f"Model construction failed for {sn_name}: {e}"
                    )
                    continue

                possible_templates.append({
                    "basis_sn": sn_name,
                    "template_sn": template_sn,
                    "model": model,
                    "template_prob": warpfit["draw_prob"],
                    "quality": warpfit.get("quality"),
                    "peakcol": warpfit.get("peakcol"),
                    "peak_gp_ztfg-ztfr": warpfit.get("peak_gp_ztfg-ztfr"),
                    "peak_gp_ztfr-ztfi": warpfit.get("peak_gp_ztfr-ztfi"),
                    "target_peak_color": applied_target_peak_color,
                    "samplecorr_ebv": samplecorr_ebv,
                    "model_colors": dict(model_colors) if model_colors else None,
                })

            if not possible_templates:
                self.logger.debug(f"No valid templates for SN {sn_name}")
                continue

            # -------------------------
            # Template selection
            # -------------------------
            if template_selection == "all":
                selected_templates = possible_templates

            elif isinstance(template_selection, int):

                if template_selection > 0:
                    weights = [tpl["template_prob"] for tpl in possible_templates]

                    if sum(weights) > 0:
                        selected_templates = rng.choices(
                            possible_templates,
                            weights=weights,
                            k=template_selection
                        )
                    else:
                        selected_templates = rng.choices(
                            possible_templates,
                            k=template_selection
                        )

                else:
                    selected_templates = rng.choices(
                        possible_templates,
                        k=abs(template_selection)
                    )

            else:
                raise ValueError("template_selection must be int or 'all'")

            results.extend(selected_templates)

        self.logger.info(f"Returning {len(results)} templates")

        return results

    def get_template_descriptors(
        self,
        fitclass: str,
        exclude_input: Optional[list] = None,
        template_selection: Union[int, str] = 1,
        snbasis_selection: Union[int, str] = 1,
        min_fit_quality: None | str = None,
        random_seed: Optional[int] = None,
        color_mode: Optional[str] = None,
        target_peak_color: Optional[float] = None,
    ) -> List[WarpTemplateDescriptor]:
        """Return selected template references without constructing models."""

        return list(
            self.iter_template_descriptors(
                fitclass,
                exclude_input=exclude_input,
                template_selection=template_selection,
                snbasis_selection=snbasis_selection,
                min_fit_quality=min_fit_quality,
                random_seed=random_seed,
                color_mode=color_mode,
                target_peak_color=target_peak_color,
            )
        )

    def iter_template_descriptors(
        self,
        fitclass: str,
        exclude_input: Optional[list] = None,
        template_selection: Union[int, str] = 1,
        snbasis_selection: Union[int, str] = 1,
        min_fit_quality: None | str = None,
        random_seed: Optional[int] = None,
        color_mode: Optional[str] = None,
        target_peak_color: Optional[float] = None,
    ) -> Iterator[WarpTemplateDescriptor]:
        """Yield template references without retaining the complete draw in RAM."""

        exclude_input = exclude_input or []
        color_mode = self._normalize_color_mode(color_mode)
        if color_mode == "target" and target_peak_color is None:
            raise ValueError("target_peak_color must be provided when color_mode='target'")

        rng = random.Random(random_seed)
        np_rng = np.random.default_rng(random_seed)
        collection = self._load_coeffs(fitclass)
        warpcoeff = collection["warpcoeff"]
        model_colors = collection.get("model_colors")
        if color_mode is not None:
            model_colors = self._validate_model_colors(model_colors)
            color_poly = np.poly1d(model_colors["ebv_corr_func"])
            color_distribution = exponnorm(
                float(model_colors["K"]),
                loc=float(model_colors["loc"]),
                scale=float(model_colors["scale"]),
            )
        else:
            color_poly = None
            color_distribution = None

        def iter_drawn_colors() -> Iterator[float]:
            """Yield SciPy colour draws from fixed blocks to reduce call overhead."""

            while True:
                values = color_distribution.rvs(size=4096, random_state=np_rng)
                yield from np.asarray(values, dtype=float)

        drawn_colors = iter_drawn_colors() if color_mode == "draw" else None

        if min_fit_quality is not None:
            min_fit_quality = min_fit_quality.lower()
            if min_fit_quality not in _QUALITY_RANK:
                raise ValueError(
                    "min_fit_quality must be one of: 'gold', 'silver', 'bronze'"
                )
            threshold = _QUALITY_RANK[min_fit_quality]
        else:
            threshold = -1

        # Retain original list indices so descriptors remain stable after filtering.
        valid = {
            basis: [
                (index, entry)
                for index, entry in enumerate(entries)
                if _QUALITY_RANK.get(entry.get("quality"), -1) >= threshold
                and entry.get("model") not in exclude_input
            ]
            for basis, entries in warpcoeff.items()
            if basis not in exclude_input
        }
        valid = {basis: entries for basis, entries in valid.items() if entries}
        if not valid:
            return []

        bases = list(valid)
        if snbasis_selection == "all":
            selected_bases = bases
        elif isinstance(snbasis_selection, int) and snbasis_selection >= 0:
            # Draw one basis at a time so million-object samples do not first
            # allocate a second, equally long Python list of basis strings.
            selected_bases = (
                rng.choices(bases, k=1)[0] for _ in range(snbasis_selection)
            )
        else:
            raise ValueError("snbasis_selection must be a non-negative int or 'all'")

        for basis in selected_bases:
            candidates = valid[basis]
            if template_selection == "all":
                selected = candidates
            elif isinstance(template_selection, int):
                count = abs(template_selection)
                weights = (
                    [float(entry.get("draw_prob", 0.0)) for _, entry in candidates]
                    if template_selection > 0
                    else None
                )
                selected = rng.choices(
                    candidates,
                    weights=weights if weights and sum(weights) > 0 else None,
                    k=count,
                )
            else:
                raise ValueError("template_selection must be int or 'all'")

            for entry_index, entry in selected:
                if color_mode == "harmonize":
                    drawn_color = float(color_distribution.median())
                elif color_mode == "draw":
                    drawn_color = float(next(drawn_colors))
                elif color_mode == "target":
                    drawn_color = float(target_peak_color)
                else:
                    drawn_color = None
                correction = (
                    self._color_correction_ebv(
                        warpfit=entry,
                        target_peak_color=drawn_color,
                        color_poly=color_poly,
                    )
                    if drawn_color is not None
                    else None
                )
                yield WarpTemplateDescriptor(
                    fitclass=str(fitclass),
                    basis_sn=str(basis),
                    template_index=int(entry_index),
                    template_sn=str(entry.get("model")),
                    template_prob=float(entry.get("draw_prob", 0.0)),
                    quality=entry.get("quality"),
                    peakcol=entry.get("peakcol"),
                    target_peak_color=drawn_color,
                    samplecorr_ebv=correction,
                    color_mode=color_mode,
                )

    def materialize_descriptor(
        self,
        descriptor: WarpTemplateDescriptor | Mapping[str, Any],
        *,
        source: Any = None,
    ) -> Dict[str, Any]:
        """Construct an event model, optionally sharing a prepared source."""

        if not isinstance(descriptor, WarpTemplateDescriptor):
            fields = WarpTemplateDescriptor.__dataclass_fields__
            descriptor = WarpTemplateDescriptor(
                **{key: descriptor[key] for key in fields}
            )
        collection = self._load_coeffs(descriptor.fitclass)
        entry = collection["warpcoeff"][descriptor.basis_sn][descriptor.template_index]
        model_colors = collection.get("model_colors")
        if source is None:
            self._ensure_base_source(descriptor.template_sn)
            model = get_warpedTimeSeriesModel(
                name=f"{descriptor.basis_sn}_{descriptor.template_sn}",
                original_template_name=descriptor.template_sn,
                warpdata=entry["mdict"],
                z=float(entry["z"]),
                original_template_version=None,
                samplecorr_ebv=descriptor.samplecorr_ebv,
                samplecorr_rv=3.1,
                samplecorr_bands=[model_colors["color1"], model_colors["color2"]]
                if model_colors
                else None,
            )
        else:
            from .models import get_model_from_warped_source

            model = get_model_from_warped_source(
                source,
                z=float(entry["z"]),
                samplecorr_ebv=descriptor.samplecorr_ebv,
            )
        return {
            "basis_sn": descriptor.basis_sn,
            "template_sn": descriptor.template_sn,
            "model": model,
            "template_prob": descriptor.template_prob,
            "quality": descriptor.quality,
            "peakcol": descriptor.peakcol,
            "target_peak_color": descriptor.target_peak_color,
            "samplecorr_ebv": descriptor.samplecorr_ebv,
            "model_colors": dict(model_colors) if model_colors else None,
        }

    def clear_fitclass_cache(self, fitclass: str) -> None:
        """Remove one coefficient file from the in-memory cache."""

        self._cache.pop(re.sub(r"/", "", fitclass), None)

    # -------------------------
    # Optional: cache control
    # -------------------------
    def clear_cache(self):
        """
        Clear the internal cache of loaded warp coefficient files.
        """
        self.logger.info("Clearing warpcoeff cache")
        self._cache.clear()
