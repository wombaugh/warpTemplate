# warp_templates/loaders.py
import os
import pickle
import logging
import re
import random
from typing import Optional, Union, List, Dict
import numpy as np
from scipy.stats import exponnorm
from .models import get_warpedTimeSeriesModel

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

        List[Dict] where each dict corresponds to a "basis SN":

        [
            {
                "ztfid": str,              # SN identifier
                "good_fit": bool,          # quality flag
                "model_colors": dict    # Fit parameters of the original template fit (e.g. for color harmonization)
                    {
                        'K': 0.9079106174117336,
                        'loc': 0.1250977286061127,
                        'scale': 0.1514897618587996,
                        'color1': 'ztfg',
                        'color2': 'ztfr',
                        'ebv_corr_func': [-0.0001, 0.01, 0.5]  # polynomial coefficients to map from drawn color to E(B-V) correction
                    },
                "warpmodels": [
                    {
                        "model": str,      # template name
                        "z": float,        # redshift
                        "draw_prob": float,# sampling weight
                        "mdict": dict      # warpdata (see below)
                    },
                    ...
                ]
            },
            ...
        ]

    Each `mdict` must match the expected input of
    `get_warpedTimeSeriesModel`.
    """

    def __init__(
        self,
        warpcoeffs_dir: str,
        logger: Optional[logging.Logger] = None
    ):
        self.warpcoeffs_dir = warpcoeffs_dir
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
    def _load_coeffs(self, fitclass: str) -> list:
        """
        Load warp coefficient data for a given fit class, with caching.

        Parameters
        ----------
        fitclass : str
            Identifier used to construct filename:
                warpcoeffs_<fitclass>_col.pkl
                _col suffix indicates that the file contains color correction data (ebv_meancol_corr)

        Returns
        -------
        list
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
            f"warpcoeffs_{key}_distinfo.pkl"
        )

        self.logger.info(f"Loading warpcoeffs from {filepath}")

        if not os.path.exists(filepath):
            self.logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(filepath)

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self._cache[key] = data
        return data

    def get_templates(
        self,
        fitclass: str,
        exclude_input: Optional[list] = None,
        template_selection: Union[int, str] = 1,
        snbasis_selection: Union[int, str] = 1,
        require_good_templatefit: bool = False,
        random_seed: Optional[int] = None,
        color_mode: Optional[str] = None,
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

        require_good_templatefit : bool, optional (default=False)
            If True, only include SN bases with `good_fit=True`.

        random_seed : int, optional
            Seed for reproducible random sampling.

        color_mode: str, optional (default="none")
            If "harmonize", apply a color warping to ensure the color at peak matches the ZTF sample mean.
            If "draw", draw a peak color from the distribution of observed colors in the ZTF sample and apply as a warping correction.

        Returns
        -------
        list of dict
            Each element has the form:

            {
                "basis_sn": str,              # SN basis name
                "model": sncosmo.Model,      # constructed warped model
                "template_prob": float       # original sampling weight
                "model_colors": dict             # Fit parameters of the original template fit (e.g. for color harmonization)
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

        # -------------------------
        # Local RNG (reproducible)
        # -------------------------
        rng = random.Random(random_seed)

        if random_seed is not None:
            self.logger.info(f"Using random seed: {random_seed}")

        template_collection = self._load_coeffs(fitclass)

        # -------------------------
        # Filter SN bases
        # -------------------------
        valid_snbases = []
        for sndict in template_collection:

            sn_name = sndict.get("ztfid")

            if sn_name in exclude_input:
                self.logger.debug(f"Excluded SN basis: {sn_name}")
                continue

            if require_good_templatefit and not sndict.get("good_fit", False):
                self.logger.debug(f"Rejected (bad fit): {sn_name}")
                continue

            valid_snbases.append(sndict)

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
        for sndict in selected_snbases:
            sn_name = sndict["ztfid"]

            # Potentially load generator for color values
            if color_mode == "draw":
                if 'model_colors' not in sndict:
                    self.logger.warning(
                        f"Cannot draw model peak color for {sn_name} - missing 'model_colors'"
                    )
                    continue
                K, loc, scale = sndict['model_colors']['K'], sndict['model_colors']['loc'], sndict['model_colors']['scale']
                # Create an instance of the EMG distribution with the given parameters
                col_generator = exponnorm(K, loc, scale)
                # Polynomial coefficients to map from the color drawn from sample distribution to the E(B-V) to apply
                col_poly = np.poly1d(sndict['model_colors']['ebv_corr_func'])


            possible_templates = []

            for warpfit in sndict["warpmodels"]:

                template_sn = warpfit.get("model")

                if template_sn in exclude_input:
                    self.logger.debug(
                        f"Excluded template {template_sn} (basis {sn_name})"
                    )
                    continue

                if color_mode == "harmonize":
                    # Harmonize colors by applying a warping correction to match the sample mean color at peak
                    # Check that we have the necessary data to apply color warping
                    if 'model_colors' not in sndict:
                        self.logger.warning(
                            f"Cannot harmonize colors for {template_sn} (basis {sn_name}) - missing model_colors"
                        )
                        continue
                    # This is the MW-like color such that if applied to this template, the color at peak would match the sample mean
                    samplecorr_ebv = sndict['model_colors']['loc']
                elif color_mode == "draw":

                    # Draw a random color corresponding to the original sample
                    drawn_color = col_generator.rvs(random_state=rng.randint(0,1000))
                    # Convert the drawn color to an E(B-V) warping correction using the provided polynomial
                    samplecorr_ebv = col_poly(drawn_color)
                else:
                    samplecorr_ebv = None

                try:
                    model = get_warpedTimeSeriesModel(
                        name=f"{sn_name}_{template_sn or 'tpl'}",
                        original_template_name=template_sn,
                        warpdata=warpfit["mdict"],
                        z=float(warpfit["z"]),
                        original_template_version=None,
                        samplecorr_ebv=samplecorr_ebv,
                        samplecorr_rv=3.1,
                        samplecorr_bands=['ztfg', 'ztfr'],
                    )
                except Exception as e:
                    self.logger.error(
                        f"Model construction failed for {sn_name}: {e}"
                    )
                    continue

                possible_templates.append({
                    "basis_sn": sn_name,
                    "model": model,
                    "template_prob": warpfit["draw_prob"],
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

                    selected_templates = rng.choices(
                        possible_templates,
                        weights=weights,
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
        