# warp_templates/loaders.py
import os
import pickle
import logging
import re
import random
from typing import Any, Optional, Union, List, Dict, Mapping
import numpy as np
from scipy.stats import exponnorm
from .models import get_warpedTimeSeriesModel


_QUALITY_RANK = {
    "bronze": 0,
    "silver": 1,
    "gold": 2,
}


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

        snbasisname[List[Dict]] where each inner Dict corresponds to fit information for a template:
        {
            "warpcoeff": {
                "ZTF18xxxxx": [           # Base SN-ID 
                    {
                        # Identifikation
                        "id": "ZTF18xxxxx",
                        "model": "v19-2006bp",    # Base Template
                        "z": 0.045,
                        
                        # Qualität & Auswahl
                        "quality": "gold",        # gold | silver | bronze
                        "draw_prob": 0.312,       # relative sampling weight for this template (relative to other templates for the same SN basis)
                        
                        # Interpolated peak color information of original data
                        "peak_gp_ztfg-ztfr": 0.127,         # native Peak-Farbe g-r
                        "type": "SN IIP",         # 
                        
                        
                        # Warp data for template conustrction
                        "mdict": {
                            "warpfit_tmin": -15.0,
                            "warpfit_tmax": 80.0,
                            # ... weitere Keys für get_warpedTimeSeriesModel()
                        },
                        
                        # Fit-Ergebnis des gewarpten Modells
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
        logger: Optional[logging.Logger] = None
    ):
        self.warpcoeffs_dir = warpcoeffs_dir
        self.version = version
        self.suffix = suffix
        self._cache: Dict[str, list] = {}

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
        version: str 
            Version string used in filename:
                warpcoeffs_v<version>_<fitclass>_col.pkl
        suffix: str
            Optional suffix for filename (default: "_col")

        Returns
        -------
        dict
            Parsed contents of the pickle file.

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

        # For now keeping handle to old pickle format. 
        if "warpcoeff" in raw_data:
            # Neue Struktur: bereits korrekt
            data = raw_data
        elif isinstance(raw_data, dict) and all(
            isinstance(v, list) for v in raw_data.values()
        ):
            # Alte Struktur: {sn_id: [warpfits, ...], ...}
            data = {
                "warpcoeff": raw_data,
                "model_colors": None,
            }
        else:
            raise ValueError(
                f"Unrecognized warp coefficient file structure in {filepath}. "
                "Expected dict with 'warpcoeff' key or flat {sn_id: [entries]} mapping."
            )

        self._cache[key] = data
        return data

    @staticmethod
    def _normalize_color_mode(color_mode: Optional[str]) -> Optional[str]:
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
        if "peakcol" not in warpfit:
            raise ValueError(
                "Cannot apply color correction because the warpfit entry "
                "does not contain 'peakcol'"
            )

        color_delta = float(target_peak_color) - float(warpfit["peakcol"])
        return float(color_poly(color_delta))

    def get_model_colors(self, fitclass: str) -> Optional[Dict[str, Any]]:
        """Return class-level peak-color distribution metadata, if present."""
        model_colors = self._load_coeffs(fitclass).get("model_colors")
        return dict(model_colors) if model_colors is not None else None

    def get_templates(
        self,
        fitclass: str,
        exclude_input: Optional[list] = None,
        template_selection: Union[int, str] = 1,
        snbasis_selection: Union[int, str] = 1,
        min_fit_quality: None | str = None,
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
            Can require a minimum fit quality for SN bases to be included. Valid values:
            - "gold" → only include SN bases with good fits and compatible with original template type
            - "silver" → include SN bases with `good_fit=True` or `good_fit=False` but compatible with original template type
            - "bronze" → include all SN bases regardless of fit quality

        random_seed : int, optional
            Seed for reproducible random sampling.

        color_mode: str, optional (default=None)
            If "harmonize", apply a color warping to ensure the color at peak matches the ZTF sample mean.
            If "draw", draw a peak color from the distribution of observed colors in the ZTF sample and apply as a warping correction.
            If "target", apply a color warping toward `target_peak_color`.

        target_peak_color : float, optional
            Peak color to use when `color_mode="target"`.

        Returns
        -------
        list of dict
            Each element has the form:

            {
                "basis_sn": str,              # SN basis name
                "model": sncosmo.Model,      # constructed warped model
                "template_prob": float,      # original sampling weight
                "template_sn": str,          # base sncosmo template name
                "quality": str,              # fit quality label
                "peakcol": float,            # stored native peak color
                "target_peak_color": float,  # target color used for correction, if any
                "samplecorr_ebv": float,     # E(B-V)-like warp correction, if any
                "model_colors": dict,        # class color distribution metadata
            }

        Notes
        -----
        - Sampling is done **with replacement** (via `random.choices`)
        - If no valid SN bases or templates remain after filtering,
          an empty list is returned
        - Model construction failures are logged and skipped

        Raises
        ------
        ValueError
            If `template_selection` or `snbasis_selection` are invalid.
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
            color_distribution = exponnorm(
                float(model_colors["K"]),
                loc=float(model_colors["loc"]),
                scale=float(model_colors["scale"]),
            )
        else:
            color_poly = None
            color_distribution = None

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
                    applied_target_peak_color = float(model_colors["loc"])
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

                try:
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

    # -------------------------
    # Optional: cache control
    # -------------------------
    def clear_cache(self):
        """
        Clear the internal cache of loaded warp coefficient files.
        """
        self.logger.info("Clearing warpcoeff cache")
        self._cache.clear()
