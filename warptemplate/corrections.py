# warp_templates/corrections.py
import re
import numpy as np
import sncosmo
import warnings
from astropy.table import Table
from typing import Optional
from scipy.interpolate import RectBivariateSpline as Spline2d
from scipy.interpolate import make_smoothing_spline
from sncosmo.fitting import DataQualityError

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
from sklearn.gaussian_process.kernels import Matern, RBF, RationalQuadratic
from sklearn.exceptions import ConvergenceWarning
        

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
# Core spline or GP interpolation
# ============================================================


def old3_get_spline_interp(phase, corr, dcorr, outphase, lam=1.0):
    """
    Smoothing spline interpolation of corrections and their uncertainties.

    Parameters
    ----------
    phase : array-like, shape (n,)
        Input phase points where corrections are defined.
    corr : array-like, shape (n,)
        Correction values at input phases.
    dcorr : array-like, shape (n,)
        Uncertainties on corrections (1-sigma).
    outphase : array-like, shape (m,)
        Output phases for interpolation.
    lam : float, optional
        Smoothing parameter. Larger values yield smoother splines.
        Default is 1.0.

    Returns
    -------
    corr_interp : ndarray, shape (m,)
        Interpolated corrections at outphase.
    dcorr_interp : ndarray, shape (m,)
        Interpolated uncertainties at outphase. Note: this is an ad hoc
        propagation via independent spline of the raw uncertainties,
        not a rigorous statistical error estimate.
    """
    phase = np.asarray(phase)
    corr = np.asarray(corr)
    dcorr = np.asarray(dcorr)
    outphase = np.asarray(outphase)

    if not (len(phase) == len(corr) == len(dcorr)):
        raise ValueError("phase, corr, dcorr must have equal length")

    # Avoid division by zero; minimum uncertainty floor
    weights = 1.0 / np.clip(dcorr, 1e-10, None) ** 2

    spl = make_smoothing_spline(phase, corr, w=weights, lam=lam)

    # Ad hoc uncertainty propagation: spline the raw errors themselves
    dspl = make_smoothing_spline(phase, dcorr, w=weights, lam=lam)

    return spl(outphase), dspl(outphase)






def get_spline_interp(phase, corr, dcorr, outphase, lam=3.0, lam_end=None,
                      lam_power=1.0, gap_scale=2.0):
    """
    Smoothing spline with phase-dependent smoothing strength.

    Parameters
    ----------
    phase : array-like, shape (n,)
        Input phase points where corrections are defined.
    corr : array-like, shape (n,)
        Correction values at input phases.
    dcorr : array-like, shape (n,)
        Uncertainties on corrections (1-sigma).
    outphase : array-like, shape (m,)
        Output phases for interpolation.
    lam : float, optional
        Smoothing parameter at phase minimum. Default 1.0.
    lam_end : float or None, optional
        Smoothing parameter at phase maximum. If None, uses lam (uniform).
        Values > lam yield stronger smoothing at late phases.
    lam_power : float, optional
        Power-law index for λ(phase) interpolation. 
        1 = linear ramp (default); >1 = sharper transition; <1 = gradual.
    gap_scale : float, optional
        Additional gap-adaptive scaling (0 = off, from previous version).

    Returns
    -------
    corr_interp : ndarray, shape (m,)
        Interpolated corrections at outphase.
    dcorr_interp : ndarray, shape (m,)
        Interpolated uncertainties (ad hoc propagation).
    """
    phase = np.asarray(phase, dtype=float)
    corr = np.asarray(corr, dtype=float)
    dcorr = np.asarray(dcorr, dtype=float)
    outphase = np.asarray(outphase, dtype=float)

    if not (len(phase) == len(corr) == len(dcorr)):
        raise ValueError("phase, corr, dcorr must have equal length")

    n = len(phase)
    if n < 3:
        raise ValueError("Need at least 3 points for smoothing spline")

    # Sort to ensure ordered phases
    sorter = np.argsort(phase)
    phase_s = phase[sorter]
    corr_s = corr[sorter]
    dcorr_s = dcorr[sorter]

    # Base weights from uncertainties
    weights = 1.0 / np.clip(dcorr_s, 1e-10, None) ** 2

    # --- Phase-dependent λ profile ---
    if lam_end is None or lam_end == lam:
        # Uniform λ: scalar, no phase dependence
        local_lam = np.full(n, lam)
    else:
        # Normalized phase coordinate [0, 1]
        p_min, p_max = phase_s[0], phase_s[-1]
        if p_max - p_min < 1e-12:
            local_lam = np.full(n, lam)
        else:
            u = (phase_s - p_min) / (p_max - p_min)
            # Power-law ramp: λ(u) = lam + (lam_end - lam) * u^lam_power
            local_lam = lam + (lam_end - lam) * np.clip(u, 0, 1) ** lam_power

    # --- Optional gap scaling (multiplicative) ---
    if gap_scale > 0:
        dphase = np.diff(phase_s)
        dphase_median = np.median(dphase)
        if dphase_median > 1e-12:
            gap_ratio = np.clip(dphase / dphase_median, 1.0, 100.0)
            gap_lam = gap_ratio ** gap_scale
            # Assign to points (maximum of adjacent intervals)
            local_lam[:-1] *= np.maximum(1.0, gap_lam)
            local_lam[1:] *= np.maximum(1.0, gap_lam)
            local_lam[0] *= gap_lam[0]
            local_lam[-1] *= gap_lam[-1]

    # --- Approximate scalar λ with weight compensation ---
    lam_eff = np.median(local_lam)
    # Scale weights inversely with sqrt(local λ) to emulate stronger smoothing
    # where λ is large: less weight on data → smoother fit
    weight_scale = np.sqrt(lam / np.clip(local_lam, lam * 0.01, None))
    weights_eff = weights * weight_scale

    spl = make_smoothing_spline(phase_s, corr_s, w=weights_eff, lam=lam_eff)
    dspl = make_smoothing_spline(phase_s, dcorr_s, w=weights_eff, lam=lam_eff)

    return spl(outphase), dspl(outphase)



def get_gp_interp(phase, corr, dcorr, outphase,
                   length_scale=1.0, length_scale_bounds=(1e-2, 1e3),
                   nu=0.5, normalize_y=False, n_restarts_optimizer=10):
    """
    Gaussian Process interpolation of corrections and their uncertainties,
    using a Matern kernel for locally faithful (less over-smoothed) fits.

    Parameters
    ----------
    phase : array-like, shape (n,)
        Input phase points where corrections are defined.
    corr : array-like, shape (n,)
        Correction values at input phases.
    dcorr : array-like, shape (n,)
        Uncertainties on corrections (1-sigma), used as heteroscedastic
        observation noise (variance = dcorr**2) at each input point. Points
        with small dcorr are effectively pinned — the GP will pass very
        close to them rather than smoothing over them.
    outphase : array-like, shape (m,)
        Output phases for interpolation.
    length_scale : float, optional
        Initial Matern length scale. Larger = smoother/more correlated
        across phase. Optimized via marginal likelihood by default.
    length_scale_bounds : tuple or "fixed", optional
        Bounds for length-scale optimization. Pass "fixed" to disable
        optimization and use `length_scale` as given.
    nu : float, optional
        Matern smoothness parameter. Lower nu (e.g. 0.5, 1.5) gives a
        rougher, more locally-responsive function that tracks data more
        faithfully; higher nu (e.g. 2.5) approaches RBF-like smoothness.
        Default 1.5 favors fidelity to data over aggressive smoothing.
        Common values: 0.5 (exponential, not mean-square differentiable),
        1.5 (once differentiable), 2.5 (twice differentiable).
    normalize_y : bool, optional
        Whether to internally center/scale corr before fitting.
    n_restarts_optimizer : int, optional
        Number of restarts for the kernel hyperparameter optimizer.

    Returns
    -------
    corr_interp : ndarray, shape (m,)
        Interpolated (posterior mean) corrections at outphase.
    dcorr_interp : ndarray, shape (m,)
        Posterior standard deviation at outphase — shrinks toward the
        input dcorr near well-sampled points, grows in gaps/at edges.
    """
    phase = np.asarray(phase, dtype=float)
    corr = np.asarray(corr, dtype=float)
    dcorr = np.asarray(dcorr, dtype=float)
    outphase = np.asarray(outphase, dtype=float)

    if not (len(phase) == len(corr) == len(dcorr)):
        raise ValueError("phase, corr, dcorr must have equal length")

    X = phase.reshape(-1, 1)
    Xout = outphase.reshape(-1, 1)

    # Heteroscedastic observation noise variance per point; floor to avoid
    # a near-singular kernel matrix for very confident (tiny dcorr) points.
    alpha = np.clip(dcorr, 1e-10, None) ** 2

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=length_scale,
        length_scale_bounds=length_scale_bounds,
        nu=nu,
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=normalize_y,
        n_restarts_optimizer=n_restarts_optimizer,
    )
    # Some lightcurves prefer too large lengthscale - mute warnings for now
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(X, corr)



    corr_interp, dcorr_interp = gp.predict(Xout, return_std=True)

    return corr_interp, dcorr_interp

def old_get_gp_interp(phase, corr, dcorr, outphase, lam=None,
                  length_scale_max=None,  # override bound
                  kernel_type='matern32'):      # or 'matern52', 'matern32', 'rq'
    """
    Gaussian process interpolation of corrections with native uncertainty.

    Replaces spline smoothing with a GP regression using an RBF kernel
    plus heteroscedastic noise. The `lam` parameter is ignored (kept for
    API consistency with get_spline_interp).

    Parameters
    ----------
    phase : array-like, shape (n,)
        Input phase points where corrections are defined.
    corr : array-like, shape (n,)
        Correction values at input phases.
    dcorr : array-like, shape (n,)
        Uncertainties on corrections (1-sigma). Used as noise standard
        deviations, not weights.
    outphase : array-like, shape (m,)
        Output phases for interpolation.
    lam : any, optional
        Ignored. Present for call compatibility with spline variant.
    length_scale_max: Prevent length scale to hit ceiling
    kernel_type: Which kernel 

    Returns
    -------
    corr_interp : ndarray, shape (m,)
        GP mean predictions at outphase.
    dcorr_interp : ndarray, shape (m,)
        GP standard deviation (predictive uncertainty) at outphase.
        Rigorous: combines epistemic (data scarcity) and aleatoric
        (measurement noise) uncertainty.
    """
    phase = np.asarray(phase)
    corr = np.asarray(corr)
    dcorr = np.asarray(dcorr)
    outphase = np.asarray(outphase)

    if not (len(phase) == len(corr) == len(dcorr)):
        raise ValueError("phase, corr, dcorr must have equal length")

    if len(phase) < 2:
        raise ValueError("At least 2 data points required for GP fit")

    ptp = np.ptp(phase)
    ls_max = length_scale_max if length_scale_max is not None else ptp * 10
    ls_min = np.min(np.diff(np.sort(phase)))

    # Define kernel
    if kernel_type == 'rbf':
        k_smooth = RBF(length_scale=ptp/4, length_scale_bounds=(ls_min, ls_max))
    elif kernel_type == 'matern52':
        from sklearn.gaussian_process.kernels import Matern
        k_smooth = Matern(length_scale=ptp/4, nu=2.5, length_scale_bounds=(ls_min, ls_max))
    elif kernel_type == 'matern32':
        from sklearn.gaussian_process.kernels import Matern
        k_smooth = Matern(length_scale=ptp/4, nu=1.5, length_scale_bounds=(ls_min, ls_max))
    elif kernel_type == 'rq':
        from sklearn.gaussian_process.kernels import RationalQuadratic
        k_smooth = RationalQuadratic(length_scale=ptp/4, alpha=1.0, 
                                     length_scale_bounds=(ls_min, ls_max),
                                     alpha_bounds=(1e-5, 1e3))
    
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * k_smooth + \
             WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e2))

    # Define regressor
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=dcorr ** 2,  # heteroscedastic noise variance
        n_restarts_optimizer=5,
        normalize_y=True,  # centre and scale data for stability
        random_state=42
    )

    X = phase.reshape(-1, 1)
    X_out = outphase.reshape(-1, 1)

    gp.fit(X, corr)

    corr_mean, corr_std = gp.predict(X_out, return_std=True)

    return corr_mean, corr_std


def old2_get_gp_interp(phase, corr, dcorr, outphase, lam=None,
        kernel_type='matern32'):
    """
    GP with noise fixed to observed uncertainties, signal hyperparameters free.
    Avoids noise-signal degeneracy that causes bound hits.
    """
    phase = np.asarray(phase)
    corr = np.asarray(corr)
    dcorr = np.asarray(dcorr)
    outphase = np.asarray(outphase)
    
    # Noise floor prevents singular matrices
    #dcorr = np.clip(dcorr, np.median(dcorr) * 0.05, None)


    # Dynamic start/limits
    #ls_start = np.ptp(phase)/4
    #ls_min = np.min(np.diff(np.sort(phase)))
    #ls_max = ptp * 10
    # Self defined 
    ls_start, ls_min, ls_max = 3, 0.1, 50
    
    from sklearn.gaussian_process.kernels import Matern, RBF, RationalQuadratic
    
    if kernel_type == 'matern52':
        k_smooth = Matern(length_scale=ls_start, nu=2.5,
                         length_scale_bounds=(ls_min, ls_max))
    elif kernel_type == 'matern32':
        k_smooth = Matern(length_scale=ls_start, nu=1.5,
                         length_scale_bounds=(ls_min, ls_max))
    elif kernel_type == 'rbf':
        k_smooth = RBF(length_scale=ls_start,
                      length_scale_bounds=(ls_min, ls_max))
    
    # No WhiteKernel—noise is fixed via alpha
#    amp_prior = (1e-3, 1e3)
    amp_prior = (0.1, 1e3)
    kernel = ConstantKernel(1.0, amp_prior) * k_smooth
    
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=dcorr ** 2,  # FIXED heteroscedastic noise
        n_restarts_optimizer=10,
        normalize_y=True,
        random_state=42
    )
    
    gp.fit(phase.reshape(-1, 1), corr)
    
    return gp.predict(outphase.reshape(-1, 1), return_std=True)


# ============================================================
# Main function
# ============================================================

def get_template_correction( 
    tab: Table,
    templatename: str,
    z: float,
    fit_host_dust: bool = True,
    max_chidof: float = 50.,
    min_bands: int = 2,
    min_point_band: int = 5,
    pull_cut: float = 999,
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

    fit_host_dust: bool (default True)
        Fit a CCM like host dust component with the template.

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

    if fit_host_dust:
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
    else:
        m = sncosmo.Model(
            source=templatename,
        )
        m.set(z=z)
        fitprop = ['t0', 'amplitude']

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

        # Filter large single outliers - is this necessary? At this stage we have not corrected the model
        # so this could be very different 
        # Flux filtering
        pulls = (
            tab['flux'] -
            fitted_model.bandflux(tab['band'], tab['time'], zp=25, zpsys='ab')
        ) / tab['fluxerr']

        iNorm = np.abs(pulls) < pull_cut
        # Template can ignore first detection by finding peak later, causing first elements to get offset.
        # Ensure first limits are true (have not found outliers there so far)
        if iNorm[0]==False:
            print('... ensuring first points not pull cut')
            first_true = np.argmax(iNorm) if np.any(iNorm) else len(iNorm)
            iNorm[:first_true] = True
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
        mdict['warpfit_tmin'] = tab['time'][iTot].min()
        mdict['warpfit_tmax'] = tab['time'][iTot].max()


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

    # If fit with host dust, remove host dust to compute correction factors relative to dust-free model
    if fit_host_dust and mdict.get('hostebv', 0) != 0:
        # dustfree_model = fitted_model.copy()  # do we need this?
        fitted_model.set(hostebv=0.0)
        model_flux = fitted_model.bandflux(
            tab['band'], tab['time'], zp=25, zpsys='ab'
        )
    else:
        model_flux = fitted_model.bandflux(
            tab['band'], tab['time'], zp=25, zpsys='ab'
        )

    corr_frac = tab['flux'] / model_flux
    err_frac = tab['fluxerr'] / tab['flux']

    mdict['corrdata'] = {}

    for band in set(tab['band'][iTot]):

        iBand = (tab['band'][iTot] == band)

        band_corr = list(corr_frac[iTot][iBand])
        band_err = list(err_frac[iTot][iBand])
        band_phase = list((tab['time'][iTot][iBand] - mdict['t0']) / (1 + z))

        

        # Fitted model fluxex could be zero, meaning infinite correction factors. This can cause issues for interpolation.
        # Cut phases with infinnite correction factors. This can happen if the fitted model is very faint at some phases, which can occur for example in the late-time tails of light curves.
        finite_mask = np.isfinite(band_corr)
        band_corr = list(np.array(band_corr)[finite_mask])
        band_err = list(np.array(band_err)[finite_mask])
        band_phase = list(np.array(band_phase)[finite_mask])

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

        tphase = m.source._phase[
            (m.source._phase >= startphase) &
            (m.source._phase <= endphase)
        ]

        # Interpolation methodology
#        finterp, dfinterp = get_spline_interp(band_phase, band_corr, band_err, tphase, lam=spline_lam )
        finterp, dfinterp = get_gp_interp(band_phase, band_corr, band_err, tphase )

        mdict['corrdata'][band] = {
            'wave': float(rest_wave),
            'phase': band_phase,
            'frac': band_corr,
            'err': band_err,
            'tphase': tphase,
            'tcorr': finterp,
            'terr': dfinterp,
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