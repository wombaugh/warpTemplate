#from __future__ import annotations

# warp_templates/models.py
import sncosmo
from .sources import WarpedTimeSeriesSource


from typing import Optional, Mapping, Any
import numpy as np
import sncosmo
import warnings


def get_warpedTimeSeriesModel(
    name: str,
    original_template_name: str,
    warpdata: Mapping[str, Any],
    z: Optional[float] = None,
    hostr_v: Optional[float] = 3.1,
    mwebv: Optional[float] = None,
    mwr_v: float = 3.1,
    original_template_version: Optional[str] = None,
    version: Optional[str] = None,
    use_host_dust: bool = True,
    use_mw_dust: bool = False,
    ebv_meancol_corr=0,
    ebv_meancol_rv=3.1,
) -> sncosmo.Model:
    """
    Create a `sncosmo.Model` using a warped TimeSeriesSource with optional
    host galaxy and Milky Way dust extinction.

    Parameters
    ----------
    name : str
        Name assigned to the warped source.
    original_template_name : str
        Name of the base `sncosmo` spectral time series template.
    warpdata : Mapping[str, Any]
        Dictionary containing warp correction data. Expected structure:

            {
                "corrmodel": {
                    "phase": array-like,   # shape (N,)
                    "wave": array-like,    # shape (M,)
                    "flux": array-like     # shape (N, M)
                }
            }

    z : float, optional
        Redshift of the model.
    hostr_v : float, optional
        Host galaxy dust parameter R_V (only used if `use_host_dust=True`).
    mwebv : float, optional
        Milky Way E(B−V). Required if `use_mw_dust=True`.
    mwr_v : float, optional (default=3.1)
        Milky Way R_V value.
    original_template_version : str, optional
        Version of the base template.
    version : str, optional
        Version label for the warped source.
    use_host_dust : bool, optional (default=True)
        Whether to include host galaxy dust (rest frame).
    use_mw_dust : bool, optional (default=False)
        Whether to include Milky Way dust (observer frame).
    ebv_meancol_corr : float, optional (default=0)
        If non-zero, apply a color warping to ensure the color at peak matches the sample mean. 
    ebv_meancol_rv : float, optional (default=3.1)

    Returns
    -------
    sncosmo.Model
        A model combining:
        - Warped time series source
        - Optional host dust (rest frame)
        - Optional Milky Way dust (observer frame)

    Notes
    -----
    - Host dust is applied in the **rest frame**
    - Milky Way dust is applied in the **observer frame**
    """

    # ---- Extract and validate warp data ----
    try:
        corr = warpdata["corrmodel"]
        phase = np.asarray(corr["phase"], dtype=float)
        wave = np.asarray(corr["wave"], dtype=float)
        flux = np.asarray(corr["flux"], dtype=float)
    except KeyError as e:
        raise KeyError(
            f"Missing required warpdata key: {e}. "
            "Expected structure: warpdata['corrmodel']['phase'|'wave'|'flux']"
        ) from e

    if flux.shape != (phase.size, wave.size):
        raise ValueError(
            f"Inconsistent warp data shapes: "
            f"flux.shape={flux.shape}, expected ({phase.size}, {wave.size})"
        )

    # ---- Create warped source ----
    warped_source = WarpedTimeSeriesSource(
        phase=phase,
        wave=wave,
        flux=flux,
        original_template_name=original_template_name,
        original_template_version=original_template_version,
        time_spline_degree=3,
        name=name,
        version=version,
    )

    # ---- Configure dust effects ----
    effects = []
    effect_names = []
    effect_frames = []

    # Host galaxy dust (rest frame)
    if use_host_dust:
        host_dust = sncosmo.CCM89Dust()
        effects.append(host_dust)
        effect_names.append("host")
        effect_frames.append("rest")

    # Milky Way dust (observer frame)
    if use_mw_dust:
        if mwebv is None:
            raise ValueError("mwebv must be provided if use_mw_dust=True")

        mw_dust = sncosmo.CCM89Dust()
        effects.append(mw_dust)
        effect_names.append("mw")
        effect_frames.append("obs")

    # Apply mean color warping if requested
    if ebv_meancol_corr != 0:
        ccm_colcorr = sncosmo.CCM89Dust()
        effects.append(ccm_colcorr)
        effect_names.append("colcorr")
        effect_frames.append("rest")

    # ---- Build model ----
    if effects:
        model = sncosmo.Model(
            source=warped_source,
            effects=effects,
            effect_names=effect_names,
            effect_frames=effect_frames,
        )
    else:
        model = sncosmo.Model(source=warped_source)

    # ---- Set parameters ----
    if z is not None:
        model.set(z=z)

    if use_host_dust and hostr_v is not None:
        model.set(hostr_v=hostr_v)
    elif not use_host_dust and hostr_v is not None:
        warnings.warn("hostr_v ignored because use_host_dust=False")

    if use_mw_dust:
        model.set(mwebv=mwebv)
        model.set(mwr_v=mwr_v)

    if ebv_meancol_corr != 0:
        model.set(colcorrebv=ebv_meancol_corr)
        model.set(colcorrr_v=ebv_meancol_rv)

    return model