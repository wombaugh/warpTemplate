# warp_templates/corrections.py
import re
import numpy as np
import sncosmo
from astropy.table import Table
from typing import Optional
from scipy.interpolate import RectBivariateSpline as Spline2d
from scipy.interpolate import make_smoothing_spline
from sncosmo.fitting import DataQualityError

from typing import TypedDict, Dict, List, Optional
import numpy as np
import numpy.typing as npt


# ============================================================
# TypedDict definitions
# ============================================================

class CorrDataPerBand(TypedDict):
    wave: float
    phase: List[float]
    frac: List[float]
    err: List[float]
    tphase: npt.NDArray[np.float64]
    tcorr: npt.NDArray[np.float64]
    terr: npt.NDArray[np.float64]


class CorrModelDict(TypedDict):
    phase: npt.NDArray[np.float64]
    wave: List[float]
    flux: npt.NDArray[np.float64]


class TemplateCorrectionResult(TypedDict, total=False):
    # --- metadata ---
    model: str

    # --- fit parameters ---
    t0: float
    amplitude: float
    hostebv: float

    # --- fit quality ---
    success: bool
    chisq: float
    ndof: int
    chidof: float
    errors: Dict[str, float]

    # --- derived ---
    absmag: float

    # --- data statistics ---
    dps_init: int
    dps_tcut: int
    dps_fcut: int
    dps_allcut: int

    # --- light curve evaluation ---
    lceval: Dict[str, int]

    # --- correction data ---
    corrdata: Dict[str, CorrDataPerBand]

    # --- final correction surface ---
    corrmodel: CorrModelDict


# ============================================================
# Main function
# ============================================================

def get_template_correction( 
    tab: Table,
    templatename: str,
    z: float,
    max_chidof: float = 50.,
    min_bands: int = 2,
    min_point_band: int = 5,
    pull_cut: float = 10,
    rv: float = 3.1,
    max_phases: Optional[List[float]] = None,
    require_phasecoverage: bool = True,
    spline_lam: float = 0.1,
    plot_dir: Optional[str] = None,
    plot_label: str = 'ZTF',
) -> TemplateCorrectionResult:
    """
    Derive multiplicative correction coefficients for a spectral time-series template.

    This function fits a template to observed photometric data and computes
    a *phase- and wavelength-dependent correction surface*:

        correction(phase, wavelength) = observed_flux / model_flux

    The resulting correction can later be used to warp the template.

    ------------------------------------------------------------------
    🔬 Algorithm Overview
    ------------------------------------------------------------------

    1. Fit template to the light curve (with host dust)
    2. Iteratively reject:
       - Phase outliers (outside allowed phase range)
       - Flux outliers (based on pull threshold)
    3. Refit using cleaned data
    4. Validate fit quality (χ²/dof, band coverage, datapoints)
    5. Compute correction factors per band:
         corr = observed_flux / model_flux
    6. Convert band observations → rest-frame wavelength
    7. Fit smoothing splines in phase for each band
    8. Interpolate onto template phase grid
    9. Store per-band correction data for later 2D interpolation

    ------------------------------------------------------------------
    📥 Parameters
    ------------------------------------------------------------------

    tab : astropy.table.Table
        Photometric light curve with required columns:
        ['time', 'band', 'flux', 'fluxerr']

    templatename : str
        Name of the base `sncosmo` template (non-SALT only).

    z : float
        Redshift of the source.

    max_chidof : float, optional
        Maximum allowed χ²/dof for accepting the fit.

    min_bands : int, optional
        Minimum number of bands with sufficient data.

    min_point_band : int, optional
        Minimum number of points per band to consider it valid.

    pull_cut : float, optional
        Threshold for rejecting flux outliers:
            |(data - model) / error| < pull_cut

    rv : float, optional
        Host galaxy dust R_V parameter.

    max_phases : list[float], optional
        Allowed phase range [min, max] in rest-frame days.
        Default: [-20, 100]

    require_phasecoverage : bool, optional
        If True:
            Only use phases directly covered by data
        If False:
            Extend correction to edges using buffers → correction → 1

    spline_lam : float, optional
        Smoothing parameter for spline interpolation.

    plot_dir : str, optional
        If provided, diagnostic plots are generated.

    plot_label : str, optional
        Label used in plots.

    ------------------------------------------------------------------
    📤 Returns
    ------------------------------------------------------------------

    dict
        Dictionary (`mdict`) containing:

        Core fit results:
        -----------------
        success : bool
            Whether a valid correction was derived
        chisq : float
        ndof : int
        chidof : float
        errors : dict
            Fit parameter uncertainties
        t0, amplitude, hostebv : float
            Best-fit parameters
        absmag : float
            Peak absolute magnitude

        Data filtering stats:
        ---------------------
        dps_init : int
            Initial number of datapoints
        dps_tcut : int
            After phase cuts
        dps_fcut : int
            After flux cuts
        dps_allcut : int
            Final used datapoints

        Light curve coverage:
        ---------------------
        lceval : dict
            {band: number of valid points}

        Correction data:
        ----------------
        corrdata : dict
            Per-band correction information:

            {
                band: {
                    'wave'   : float      # rest-frame wavelength
                    'phase'  : list       # observed phases
                    'frac'   : list       # correction values
                    'err'    : list       # uncertainties
                    'tphase' : ndarray    # template phases
                    'tcorr'  : ndarray    # interpolated correction
                    'terr'   : ndarray    # interpolated uncertainty
                }
            }

    ------------------------------------------------------------------
    ⚠️ Important Notes
    ------------------------------------------------------------------

    - SALT templates are explicitly not supported.
    - The correction is computed **per band**, then later expected to be
      combined into a 2D phase–wavelength surface.
    - If `require_phasecoverage=False`, artificial boundary points are added
      to enforce:
            correction → 1 at phase edges
    - Uses internal template phase grid (`m.source._phase`), which relies on
      sncosmo internals.
    - Failures return early with `success=False`.

    ------------------------------------------------------------------
    🚫 Failure Conditions
    ------------------------------------------------------------------

    The function returns early with `success=False` if:

    - Fit fails (exceptions)
    - ndof < 1
    - χ²/dof > max_chidof
    - Insufficient band coverage
    - Too few points per band

    ------------------------------------------------------------------
    """
    
    if max_phases is None:
        max_phases = [-20., 100.]

    # -------------------------
    # Initialize model
    # -------------------------
    if re.search('salt', templatename):
        raise ValueError('SALT templates not incorporated')

    dust = sncosmo.CCM89Dust()
    m = sncosmo.Model(
        source=templatename,
        effects=[dust],
        effect_names=['host'],
        effect_frames=['rest']
    )
    m.set(hostr_v=rv)
    m.set(z=z)

    fitprop = ['t0', 'amplitude', 'hostebv']

    # -------------------------
    # Init result dict
    # -------------------------
    mdict: TemplateCorrectionResult = {
        'model': templatename,
        'dps_init': len(tab)
    }

    try:
        # Initial fit
        result, fitted_model = sncosmo.fit_lc(tab, m, fitprop)

        mdict.update({
            result['param_names'][k]: result['parameters'][k]
            for k in range(len(result['parameters']))
        })

        # Phase filtering
        phases = (tab['time'] - mdict['t0']) / (1 + z)
        iGood = (
            (max_phases[0] < phases) & (phases < max_phases[1]) &
            (phases > m.mintime()) &
            (phases < m.maxtime())
        )

        # Refit
        result, fitted_model = sncosmo.fit_lc(
            tab[iGood], fitted_model, fitprop
        )

        # Flux filtering
        pulls = (
            tab['flux'] -
            fitted_model.bandflux(tab['band'], tab['time'], zp=25, zpsys='ab')
        ) / tab['fluxerr']

        iNorm = np.abs(pulls) < pull_cut
        mdict['dps_fcut'] = int(np.sum(iNorm))

        iTot = iGood & iNorm

        # Final fit
        result, fitted_model = sncosmo.fit_lc(
            tab[iTot], fitted_model, fitprop
        )

        mdict.update({
            result['param_names'][k]: result['parameters'][k]
            for k in range(len(result['parameters']))
        })

        # Final stats
        phases = (tab['time'] - mdict['t0']) / (1 + z)
        iGood = (
            (max_phases[0] < phases) & (phases < max_phases[1]) &
            (phases > m.mintime()) &
            (phases < m.maxtime())
        )

        mdict['dps_tcut'] = int(np.sum(iGood))
        mdict['dps_fcut'] = int(np.sum(iNorm))
        mdict['dps_allcut'] = int(np.sum(iTot))

    except (RuntimeError, ValueError, KeyError, DataQualityError):
        mdict['success'] = False
        return mdict

    if result.ndof < 1:
        mdict['success'] = False
        return mdict

    # -------------------------
    # Fit validation
    # -------------------------
    mdict['success'] = True
    mdict.update({k: result[k] for k in ['success', 'chisq', 'ndof', 'errors']})
    mdict['chidof'] = result.chisq / result.ndof
    mdict['absmag'] = fitted_model.source_peakabsmag(
        band='bessellb', magsys='ab'
    )

    mdict['lceval'] = {
        band: int(np.sum(tab['band'][iTot] == band))
        for band in set(tab['band'][iTot])
    }

    if (len([c for c in mdict['lceval'].values() if c > min_point_band])
            < min_bands):
        mdict['success'] = False
        return mdict

    if mdict['chidof'] > max_chidof:
        mdict['success'] = False
        return mdict

    # -------------------------
    # Build correction
    # -------------------------
    startphase = max(m.mintime(), max_phases[0])
    endphase = min(m.maxtime(), max_phases[1])

    corr_frac = tab['flux'] / fitted_model.bandflux(
        tab['band'], tab['time'], zp=25, zpsys='ab'
    )


    mdict['corrdata'] = {}

    for band in set(tab['band'][iTot]):
        iBand = (tab['band'][iTot] == band)

        band_corr = list(corr_frac[iTot][iBand])
        band_err = list(tab['fluxerr'][iTot][iBand])
        band_phase = list((tab['time'][iTot][iBand] - mdict['t0']) / (1 + z))

        # Fitted model fluxex could be zero, meaning infinite correction factors. This can cause issues for interpolation.
        # Cut phases with infinnite correction factors. This can happen if the fitted model is very faint at some phases, which can occur for example in the late-time tails of light curves.
        finite_mask = np.isfinite(band_corr)
        band_corr = list(np.array(band_corr)[finite_mask])
        band_err = list(np.array(band_err)[finite_mask])
        band_phase = list(np.array(band_phase)[finite_mask])

#        print(f"Band {band}: {len(band_phase)} valid points after filtering")
#        print("  Phases:", band_phase)
#        print("  Corrections:", band_corr)
#        print("  Errors:", band_err)

        bandfunc = sncosmo.get_bandpass(band)
        rest_wave = bandfunc.wave_eff / (1 + z)

        if not require_phasecoverage:
            buffercadence = 3.

            if min(band_phase) < startphase:
                # This can happen if the data extends beyond the template phase range. In this case, we do not add pre phases, since they would be outside the template range.
                pre_phases = []
            else:
                pre_phases = np.arange(startphase, min(band_phase), buffercadence)
                if len(pre_phases) == 0:
                    pre_phases = [startphase]

            if max(band_phase) > endphase:
                # This can happen if the data extends beyond the template phase range. In this case, we do not add post phases, since they would be outside the template range.
                post_phases = []
            else:
                post_phases = np.arange(max(band_phase) + buffercadence, endphase, buffercadence)
                if len(post_phases) == 0:
                    post_phases = [endphase]

            pre_err = np.linspace(0.01, band_err[0], len(pre_phases))
            post_err = np.linspace(band_err[-1], 0.01, len(post_phases))


            band_phase = [*pre_phases, *band_phase, *post_phases]
            band_corr = [1] * len(pre_phases) + band_corr + [1] * len(post_phases)
            band_err = list(pre_err) + band_err + list(post_err)

        if len(band_phase) < 5:
            continue

        spl = make_smoothing_spline(
            band_phase, band_corr,
            w=1.0 / np.array(band_err) ** 2,
            lam=spline_lam
        )
        dspl = make_smoothing_spline(
            band_phase, band_err,
            w=1.0 / np.array(band_err) ** 2,
            lam=spline_lam
        )

        tphase = m.source._phase[
            (m.source._phase >= startphase) &
            (m.source._phase <= endphase)
        ]

        finterp = spl(tphase)

        mdict['corrdata'][band] = {
            'wave': float(rest_wave),
            'phase': band_phase,
            'frac': band_corr,
            'err': band_err,
            'tphase': tphase,
            'tcorr': finterp,
            'terr': dspl(tphase)
        }

    # -------------------------
    # Build 2D correction model
    # -------------------------
    wave = [m.minwave()]
    phase: npt.NDArray[np.float64] = np.array([])
    flux: List[npt.NDArray[np.float64]] = []

    for band in ['ztfg', 'ztfr', 'ztfi']:
        if band not in mdict['corrdata']:
            continue

        wave.append(mdict['corrdata'][band]['wave'])

        if len(flux) == 0:
            phase = mdict['corrdata'][band]['tphase']
            flux = [np.ones(len(phase))]
            flux.append(mdict['corrdata'][band]['tcorr'])
        else:
            flux.append(mdict['corrdata'][band]['tcorr'])

    wave.append(m.maxwave())
    flux.append(np.ones(len(phase)))

    flux2d = np.array(flux).transpose()

    mdict['corrmodel'] = {
        'phase': phase,
        'wave': wave,
        'flux': flux2d
    }

    return mdict