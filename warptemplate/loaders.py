# warp_templates/loaders.py
import os
import pickle
import logging
import re
import random
from typing import Any, Optional, Union, List, Dict, Mapping
import numpy as np
from scipy.stats import johnsonsu
from scipy.stats import exponnorm
from .models import get_warpedTimeSeriesModel


_QUALITY_RANK = {
    "bronze": 0,
    "silver": 1,
    "gold": 2,
}

# -----------------------------------------------------------------------------
# Random draws from the fitted offset+scatter color model ("offsetfit_draw")
# -----------------------------------------------------------------------------
# Mirrors the AV_DISTRIBUTIONS convention from fit_offset_extinction_v2.py:
# both families are SCALE families (mean/spread set entirely by `scale`),
# parametrized in A_V (mag) -- matching what that fit reports as
# fit_result.av_dist / fit_result.av_scale, so those can be passed straight
# through as dist / av_scale below.
#
# NOTE: despite the historical name, this has nothing to do with sncosmo's
# CCM89Dust / actual Milky Way extinction -- it draws a random A_V-like
# value and converts it to a *peak-color* shift via dcolor_dav (the
# color-change-per-unit-A_V slope fit_offset_extinction_v2.py already
# computed for this band pair, via Fitzpatrick99 at the bands' effective
# wavelengths), then adds the fitted delta_c baseline. The result is a
# target peak color, fed into the same `_color_correction_ebv`/`color_poly`
# pathway as "harmonize"/"draw"/"target" -- it is a variant of color_mode
# "draw" using the class's fitted offset+scatter model instead of its raw
# native-color distribution, not a separate dust-extinction mechanism. If
# genuine Milky Way dust is ever wanted, that's `use_mw_dust`/`mwebv=` on
# get_warpedTimeSeriesModel instead, which this does not touch.

OFFSETFIT_AV_DISTRIBUTIONS = {
    'exponential': lambda rng, scale: rng.exponential(scale=scale),
    'halfnormal':  lambda rng, scale: abs(rng.normal(loc=0.0, scale=scale)),
}


def draw_offsetfit_peak_color(dist: str, av_scale: float, dcolor_dav: float,
                               color_zeropoint: float,
                               rng: Optional[np.random.Generator] = None) -> float:
    """Draw a single target peak color from a class's fitted offset+scatter
    color model (color_mode="offsetfit_draw").

    `dist` / `av_scale` should match whatever fit_offset_extinction_v2.py
    reported (av_dist / av_scale there) for this class's color pair, and
    are in A_V-like units (mag). `dcolor_dav` is that same fit's
    color-change-per-unit-A_V slope for this band pair (already encodes
    whatever R_V the fit assumed -- there is no separate R_V here), so the
    draw is converted directly to a color-space shift: no E(B-V) or
    CCM89Dust involved anywhere in this path.
    """
    if dist not in OFFSETFIT_AV_DISTRIBUTIONS:
        raise ValueError(f"dist must be one of {list(OFFSETFIT_AV_DISTRIBUTIONS)}, got {dist!r}")
    if rng is None:
        rng = np.random.default_rng()
    av_draw = OFFSETFIT_AV_DISTRIBUTIONS[dist](rng, av_scale)
    return float(av_draw) * dcolor_dav + color_zeropoint




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
                        
                        # Quality & Selection
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

                        # Phase coverage of the data used for the fit (relative to fitted peak)
                        "fit_phase_min": -12.0,
                        "fit_phase_max": 75.0,               
                    },
                    # ... further Templates for the same SN
                ],
                # ... further SNe
            },
            
            "model_colors": {
                'color1': 'ztfg',
                'color2': 'ztfr',
                'emg': {'K': np.float64(0.47560474926094615),
                'loc': np.float64(-0.027584443383387745),
                'scale': np.float64(0.2777751017084096),
                'type': 'exponnorm'},
                'linear_corr': {'type': 'LinearDust',
                'coeffs': [np.float64(-4.038045179953404), np.float64(0.0)],
                'lambda_0': 6250.0,
                'dc_da_mean': -0.25281153049389066,
                'dc_da_std': 0.011757804164962037},
                'offset_extinction_corr': {'av_dist': 'exponential',
                'delta_c_bounds': [-1.0, 1.0],
                'av_scale_bounds': [0.0, 2.0],
                'colors': {'ztfg-ztfr': {'color': 'ztfg-ztfr',
                    'delta_c': -0.244782259238491,
                    'av_scale': 0.8789663791903275,
                    'av_dist': 'exponential',
                    'av_mean': 0.8789663791903275,
                    'dcolor_dav': 0.39562545069212507,
                    'ks_stat': 0.018193146072606803,
                    'ks_pvalue': 0.9616656456300157,
                    'baseline_ks': 0.26953854234527697,
                    'baseline_pvalue': 1.741684525084477e-48,
                    'success': True,
                    'nfev': 61,
                    'obs_mean': 0.2520259749348755,
                    'obs_std': 0.4805574505658401,
                    'model_mean': 0.24460515803144633,
                    'model_std': 0.4328944722712748,
                    'n_sn': 748}}}}
                            }
        }
    Each `mdict` must match the expected input of
    `get_warpedTimeSeriesModel`.
    """


    def __init__(
        self,
        warpcoeffs_dir: str,
        version: str = "5",
        suffix: str = "_col",
        logger: Optional[logging.Logger] = None
    ):
        self.warpcoeffs_dir = warpcoeffs_dir
        self.version = version
        self.suffix = suffix
        self._cache: Dict[str, Dict[str, Any]] = {}

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
            Identifier used to construct filename.
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

        filepath = os.path.join(
            self.warpcoeffs_dir,
            f"warpcoeffs_v{self.version}_{key}{self.suffix}.pkl"
        )

        self.logger.info(f"Loading warpcoeffs from {filepath}")

        if not os.path.exists(filepath):
            self.logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(filepath)

        with open(filepath, "rb") as f:
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

    # -------------------------
    # Public: access and mutate loaded data
    # -------------------------
    def get_warpcoeff(self, fitclass: str) -> Dict[str, List[Dict]]:
        """Return raw warp coefficient dictionary for a class."""
        return self._load_coeffs(fitclass)["warpcoeff"]

    def get_model_colors(self, fitclass: str) -> Optional[Dict[str, Any]]:
        """Return class-level peak-color distribution metadata, if present."""
        model_colors = self._load_coeffs(fitclass).get("model_colors")
        return dict(model_colors) if model_colors is not None else None

    def update_warpcoeff(self, fitclass: str, warpcoeff: Dict[str, List[Dict]]) -> None:
        """Update cached warp coefficients (for in-place mutation by analysis scripts)."""
        key = re.sub(r"/", "", fitclass)
        if key not in self._cache:
            self._load_coeffs(fitclass)
        self._cache[key]["warpcoeff"] = warpcoeff

    def update_model_colors(self, fitclass: str, model_colors: Dict[str, Any]) -> None:
        """Update cached model colors metadata, merging with any existing entry.

        If a "model_colors" entry already exists for this fitclass, `model_colors`
        is merged into it rather than replacing it outright: existing keys not
        present in the new dict are kept, new keys are added, and colliding keys
        are merged recursively if both sides are dicts (so e.g. updating one
        color pair's stats doesn't wipe out another color pair's stats sitting
        next to it), otherwise the new value wins.
        """
        key = re.sub(r"/", "", fitclass)
        if key not in self._cache:
            self._load_coeffs(fitclass)

        existing = self._cache[key].get("model_colors")
        if isinstance(existing, dict) and isinstance(model_colors, dict):
            self._cache[key]["model_colors"] = self._deep_merge_dicts(existing, model_colors)
        else:
            # Nothing there yet (or one side isn't a dict) -- nothing to preserve.
            self._cache[key]["model_colors"] = model_colors

    @staticmethod
    def _deep_merge_dicts(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge `update` into a copy of `base` (neither input is mutated).

        Values in `update` win on key collisions, except where both the existing
        and new value are themselves dicts -- those are merged recursively
        instead of the new one replacing the old one wholesale.
        """
        merged = dict(base)
        for k, v in update.items():
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = WarpfitTemplateLoader._deep_merge_dicts(merged[k], v)
            else:
                merged[k] = v
        return merged
    
    def save_class(self, fitclass: str, filepath: Optional[str] = None) -> str:
        """
        Persist cached data for a class to disk.

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

        if filepath is None:
            filepath = os.path.join(
                self.warpcoeffs_dir,
                f"warpcoeffs_v{self.version}_{key}{self.suffix}.pkl"
            )

        with open(filepath, "wb") as f:
            pickle.dump(self._cache[key], f)

        self.logger.info(f"Saved warp coefficients to {filepath}")
        return filepath

    # -------------------------
    # Color mode handling
    # -------------------------
    @staticmethod
    def _normalize_color_mode(color_mode: Optional[str]) -> Optional[str]:
        if color_mode is None:
            return None

        mode = color_mode.lower()
        if mode == "none":
            return None

        allowed = {"harmonize", "draw", "target", "offsetfit_draw"}
        if mode not in allowed:
            raise ValueError(
                f"Invalid color_mode: {color_mode}. "
                "Must be one of None, 'none', 'harmonize', 'draw', 'target', 'offsetfit_draw'."
            )
        return mode

    @staticmethod
    def _validate_model_colors(model_colors: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
        if model_colors is None:
            raise ValueError(
                "color_mode requires 'model_colors' in the warp coefficient file"
            )

        required = ("color1", "color2", "linear_corr")
        missing = [key for key in required if key not in model_colors]


        # Two color definitions possible: 'exponnorm' or 'johnsonsu'
        if "emg" in model_colors:
            required = ("K", "loc", "scale")
            missing.extend([key for key in required if key not in model_colors['emg']])
        elif "johnsonsu" in model_colors:
            required = ("gamma", "delta", "loc", "scale")
            missing.extend([key for key in required if key not in model_colors['johnsonsu']])
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
        phase_buffer: Optional[float] = None,
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

        phase_buffer : float, optional
            Limit template phases to the range of the original data used for the fit, plus/minus this buffer (in days).

        random_seed : int, optional
            Seed for reproducible random sampling.

        color_mode : str, optional (default=None)
            - "harmonize": warp to class median peak color
            - "draw": draw peak color from observed distribution
            - "target": warp to specified `target_peak_color`
            - "offsetfit_draw": draw a target peak color from the class's
                fitted offset+scatter color model (the delta_c/av_scale fit
                from fit_offset_extinction_v2.py), i.e. the class's fitted
                color baseline plus a random draw from its fitted A_V-like
                scatter distribution, converted to a color shift via
                dcolor_dav. A variant of "draw" using that fitted model
                instead of the raw native-color distribution -- it does not
                apply any actual Milky Way dust effect (see
                draw_offsetfit_peak_color).

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
#        print(model_colors)

        if color_mode is not None:
            model_colors = self._validate_model_colors(model_colors)
            color_poly = np.poly1d(model_colors["linear_corr"]["coeffs"])
            color_pivot = model_colors["linear_corr"]["lambda_0"]
            if "emg" in model_colors:
                color_distribution = exponnorm(
                    float(model_colors['emg']['K']),
                    float(model_colors['emg']['loc']),
                    float(model_colors['emg']['scale'])
                )
                print('... initialized color_distribution with K, loc, scale =',
                      model_colors['emg']["K"], model_colors['emg']["loc"], model_colors['emg']["scale"])
            elif "johnsonsu" in model_colors:
                color_distribution = johnsonsu(
                    float(model_colors['johnsonsu']["gamma"]),
                    float(model_colors['johnsonsu']["delta"]),
                    loc=float(model_colors['johnsonsu']["loc"]), 
                    scale=float(model_colors['johnsonsu']["scale"]),
                )
                print('... initialized color_distribution with gamma, delta, loc, scale =',
                      model_colors['johnsonsu']["gamma"], model_colors['johnsonsu']["delta"], model_colors['johnsonsu']["loc"], model_colors['johnsonsu']["scale"])
        else:
            color_poly = None
            color_distribution = None
            color_pivot = None

        if color_mode == "offsetfit_draw":
            # Here we need to grab additional parameters for the fitted
            # offset+scatter color model. Brute force first - assume these
            # to be present in the pickle.
            # Todo: gracefully check whether these exist, possibly as part of model color validate
            colname = '{}-{}'.format(model_colors['color1'], model_colors['color2'])
            if not colname in model_colors['offset_extinction_corr']['colors']:
                raise ValueError(
                                    "Color mismatch between whats available from offset_extinction corr and linear corretion."
                                )
            offsetfit_kwargs = {
                'dist': model_colors['offset_extinction_corr']['av_dist'], 
                'av_scale': model_colors['offset_extinction_corr']['colors'][colname]['av_scale'], 
                'dcolor_dav': model_colors['offset_extinction_corr']['colors'][colname]['dcolor_dav'], 
                'color_zeropoint':  float(color_distribution.median()) + model_colors['offset_extinction_corr']['colors'][colname]['delta_c'], 
                'rng': np_rng,
            }
            print('... initialized offsetfit_draw color model from:', offsetfit_kwargs)





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
                elif color_mode == "offsetfit_draw":
                    applied_target_peak_color = draw_offsetfit_peak_color(**offsetfit_kwargs)
                elif color_mode == "draw":
                    applied_target_peak_color = float(
                        color_distribution.rvs(random_state=np_rng)
                    )
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


                # Potential phase limits to apply
                if phase_buffer is not None:
                    fit_phase_min = float(warpfit.get("fit_phase_min", -np.inf)) - phase_buffer
                    fit_phase_max = float(warpfit.get("fit_phase_max", np.inf)) + phase_buffer
                    phase_lim = (fit_phase_min, fit_phase_max)
                else:
                    phase_lim = None

                try:
#                    print(model_colors)
                    model = get_warpedTimeSeriesModel(
                        name=f"{sn_name}_{template_sn or 'tpl'}",
                        original_template_name=template_sn,
                        warpdata=warpfit["mdict"],
                        z=float(warpfit["z"]),
                        original_template_version=None,
                        sample_color_amplitude=samplecorr_ebv,
                        sample_color_pivot=color_pivot,
                        phase_lim=phase_lim,
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

    # -------------------------
    # Optional: cache control
    # -------------------------
    def clear_cache(self):
        """
        Clear the internal cache of loaded warp coefficient files.
        """
        self.logger.info("Clearing warpcoeff cache")
        self._cache.clear()
        