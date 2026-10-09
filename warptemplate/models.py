#from __future__ import annotations

# warp_templates/models.py
import sncosmo
from .sources import WarpedTimeSeriesSource


from typing import Optional, Mapping, Any
import numpy as np
import sncosmo
import warnings




# -----------------------------------------------------------------------------
# Fitted color-offset ("delta_c") tilt effect
# -----------------------------------------------------------------------------
# fit_offset_extinction_v2.py's delta_c is a flat additive shift in ONE
# specific color (band1 - band2). A genuinely gray (achromatic) flux
# rescaling would cancel exactly in color and could never reproduce a
# nonzero delta_c -- so *some* wavelength dependence between the two
# reference bands is unavoidable. This effect reproduces delta_c as a
# linear-in-wavelength magnitude tilt, split symmetrically (+delta_c/2 at
# band1's effective wavelength, -delta_c/2 at band2's), held flat outside
# that range. That symmetric split is a MODELING CHOICE, not a measurement
# -- the fit only constrains the difference between the two bands, not how
# it's distributed between them. If you later use these templates for
# absolute-magnitude work (not just colors), that choice matters; for
# colors alone it's exact by construction.

_EFFECTIVE_WAVELENGTH_CACHE: dict = {}


def _band_effective_wavelength(band: str) -> float:
    """Transmission-weighted mean wavelength (Angstrom) of a bandpass
    registered with sncosmo. Small, deliberately duplicated copy of the
    identical helper in fit_offset_extinction_v2.py -- kept local here
    rather than importing an analysis script from this core package."""
    if band not in _EFFECTIVE_WAVELENGTH_CACHE:
        bp = sncosmo.get_bandpass(band)
        w_eff = np.trapezoid(bp.wave * bp.trans, bp.wave) / np.trapezoid(bp.trans, bp.wave)
        _EFFECTIVE_WAVELENGTH_CACHE[band] = float(w_eff)
    return _EFFECTIVE_WAVELENGTH_CACHE[band]


class ColorTiltEffect(sncosmo.PropagationEffect):
    """Multiplicative flux effect reproducing a fitted color offset delta_c.

    Anchored at the effective wavelengths of the two bands delta_c was fit
    against (in the same band1/band2 order used to build the color key,
    e.g. "ztfg-ztfr" -> band1='ztfg', band2='ztfr'), so the sign of delta_c
    is preserved correctly regardless of which of the two happens to be
    bluer. Linearly interpolated (and flat-extrapolated) in between via
    np.interp.
    """
    _param_names = ['delta_c']
    param_names_latex = ['\\Delta c']
    _minwave = 1000.
    _maxwave = 33333.33
    
    def __init__(self, wave_band1: float, wave_band2: float):
        wave_anchors = np.array([float(wave_band1), float(wave_band2)])
        sign_coeffs = np.array([0.5, -0.5])  # coefficient of delta_c at each anchor
        order = np.argsort(wave_anchors)
        self._wave_sorted = wave_anchors[order]
        self._sign_sorted = sign_coeffs[order]
        self._parameters = np.array([0.0])

    def propagate(self, wave, flux, phase=None):
        delta_c = self._parameters[0]
        if delta_c == 0:
            return flux
        dmag = np.interp(wave, self._wave_sorted, self._sign_sorted * delta_c)
        flux_factor = 10.0 ** (-0.4 * dmag)
        return flux * flux_factor


def _resolve_color_tilt(delta_c: Optional[float], delta_c_band1: Optional[str],
                        delta_c_band2: Optional[str]) -> Optional["ColorTiltEffect"]:
    """Build a ColorTiltEffect for the given delta_c, or None if delta_c is
    unset/zero. Raises ValueError if delta_c is set without both bands."""
    if delta_c is None or delta_c == 0:
        return None
    if delta_c_band1 is None or delta_c_band2 is None:
        raise ValueError(
            "delta_c_band1 and delta_c_band2 (the band pair the fitted delta_c "
            "refers to, e.g. 'ztfg'/'ztfr' for a color key of \"ztfg-ztfr\") "
            "are required whenever delta_c is set."
        )
    w1 = _band_effective_wavelength(delta_c_band1)
    w2 = _band_effective_wavelength(delta_c_band2)
    return ColorTiltEffect(w1, w2)


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
    phase_lim: Optional[tuple[float, float]] = None,
    use_host_dust: bool = True,
    use_mw_dust: bool = False,
    sample_color_amplitude =None,
    sample_color_pivot=6250,
    samplecorr_bands=None,
    delta_c: Optional[float] = None,
    delta_c_band1: Optional[str] = None,
    delta_c_band2: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
) -> sncosmo.Model | None:
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
        Milky Way E(B-V), fixed value. Required if `use_mw_dust=True` and
        `mwebv_dist` is not set. Mutually exclusive with `mwebv_dist`.
    mwr_v : float, optional (default=3.1)
        Milky Way R_V value. Also used to convert `mwebv_av_scale` (A_V) to
        E(B-V) when drawing -- keep consistent with whatever R_V your fit
        assumed.
    original_template_version : str, optional
        Version of the base template.
    version : str, optional
        Version label for the warped source.
    phase_lim: tuple of float, optional
        Optional (min_phase, max_phase) limits to apply to the warped source.        
    use_host_dust : bool, optional (default=True)
        Whether to include host galaxy dust (rest frame).
    use_mw_dust : bool, optional (default=False)
        Whether to include Milky Way dust (observer frame).
    sample_color_amplitude : float, optional
        Linearlized color correction to apply.
    sample_color_pivot : float, optional (default=6250)
        Pivot wavelength to use with sample_color_amplitude
    samplecorr_bands : list of str, optional
        List of band names to use for calculating the color correction. Not active??
    delta_c : float, optional
        A fitted global color offset (mag, e.g. fit_result.delta_c from
        fit_offset_extinction_v2.py), applied as a color-space tilt anchored
        at `delta_c_band1`/`delta_c_band2`'s effective wavelengths -- see
        ColorTiltEffect. Unlike `mwebv_dist`, this is applied as a FIXED
        shift to every generated template (the fit only returns a point
        estimate for delta_c, not a distribution), alongside whatever
        random mwebv draw is also requested. If you want delta_c to carry
        its own draw-to-draw scatter too, that needs a fitted uncertainty/
        scale for it, which the current fit script doesn't produce -- ask
        if you'd like that added.
    delta_c_band1, delta_c_band2 : str, optional
        The band pair `delta_c` was fit against (in that order, e.g.
        'ztfg'/'ztfr' for a color key of "ztfg-ztfr"). Required whenever
        `delta_c` is set.
    rng : numpy.random.Generator, optional
        Shared random generator for `mwebv_dist` draws. Pass the SAME
        generator (seeded once) across repeated calls when generating many
        templates, so draws are independent of each other but reproducible
        run-to-run; if omitted, each call seeds its own generator from
        system entropy (draws still independent, but not reproducible
        across runs).

    Returns
    -------
    sncosmo.Model
        A model combining:
        - Warped time series source
        - Optional host dust (rest frame)
        - Optional Milky Way dust (observer frame), fixed or randomly drawn
        - Optional fitted color-offset tilt (observer frame)

    Notes
    -----
    - Host dust is applied in the **rest frame**
    - Milky Way dust is applied in the **observer frame**
    - The color-offset tilt is applied in the **observer frame** (delta_c
      was fit against observed-frame colors at observed redshifts)
    """

    # ---- Extract and validate warp data ----
    try:
        corr = warpdata["corrmodel"]
        phase = np.asarray(corr["phase"], dtype=float)
        wave = np.asarray(corr["wave"], dtype=float)
        flux = np.asarray(corr["flux"], dtype=float)
    except KeyError as e:
        if warpdata['success'] is False:
            print('... warpfit failed, not creating model')
            return None
        raise KeyError(
            f"Missing required warpdata key: {e}. "
            "Expected structure: warpdata['corrmodel']['phase'|'wave'|'flux']"
        ) from e
    # Check sufficient sample width for spline interpolation
    if phase.size < 4 or wave.size < 4:
        raise ValueError(
            f"Insufficient warp data for spline interpolation: "
            f"phase.size={phase.size}, wave.size={wave.size} (need >=4 each)"
        )

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
        phase_lim=phase_lim,
        original_template_name=original_template_name,
        original_template_version=original_template_version,
        time_spline_degree=3,
        warp_reddening_a=sample_color_amplitude,
        warp_reddening_pivot=sample_color_pivot,
        name=name,
        version=version,
    )

    # ---- Configure dust / reddening / extinction effects ----
    effects = []
    effect_names = []
    effect_frames = []

    # Host galaxy dust (rest frame)
    if use_host_dust:
        host_dust = sncosmo.CCM89Dust()
        effects.append(host_dust)
        effect_names.append("host")
        effect_frames.append("rest")

    # Milky Way dust (observer frame) -- fixed value or None
    if use_mw_dust:
        mw_dust = sncosmo.CCM89Dust()
        effects.append(mw_dust)
        effect_names.append("mw")
        effect_frames.append("obs")

    # Fitted color-offset tilt (observer frame), applied as a fixed shift
    # alongside whatever mwebv draw is also happening above.
    color_tilt = _resolve_color_tilt(delta_c, delta_c_band1, delta_c_band2)
    if color_tilt is not None:
        effects.append(color_tilt)
        effect_names.append("colortilt")
        effect_frames.append("obs")

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

    if use_mw_dust:
        model.set(mwebv=mwebv)
        model.set(mwr_v=mwr_v)

    if color_tilt is not None:
        model.set(colortiltdelta_c=delta_c)

    return model

