"""
Multi-band peak flux estimator for sparse, irregularly-sampled astronomical light curves.

Algorithm:
  1. Sigma-clipping with MAD-based scatter estimate to reject outliers
  2. Peak estimation via one of three methods:
     - 'gp'   : Matérn-3/2 Gaussian Process interpolation (recommended for sparse data)
     - 'twa'  : time-weighted (Gaussian-kernel) smoothing, then argmax
     - 'top5' : median of the top 5% of the time-weighted smoothed curve

Dependencies: numpy, scipy, matplotlib
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.optimize import minimize_scalar


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class PeakResult:
    """Result for a single band."""
    band: str
    peak_flux: float
    peak_flux_err: float        # GP posterior std at peak; 0 for non-GP methods
    peak_time: float            # time of peak on interpolation grid
    n_clipped: int              # number of points rejected by sigma-clipping
    mask: np.ndarray            # boolean mask: True = kept, False = clipped
    n_before_peak: int = 0           # surviving points with t < peak_time
    n_after_peak: int = 0            # surviving points with t > peak_time
    n_eff_before_peak: float = 0.0   # kernel-weighted effective count before peak
    n_eff_after_peak: float = 0.0    # kernel-weighted effective count after peak
    gp_time_grid: np.ndarray | None = field(default=None, repr=False)
    gp_mean: np.ndarray | None = field(default=None, repr=False)
    gp_std: np.ndarray | None = field(default=None, repr=False)

    def __str__(self) -> str:
        unc = f" ± {self.peak_flux_err:.4f}" if self.peak_flux_err > 0 else ""
        return (
            f"Band {self.band}: peak = {self.peak_flux:.4f}{unc}  "
            f"at t = {self.peak_time:.2f}  "
            f"({self.n_before_peak} pts before, {self.n_after_peak} pts after peak  "
            f"[{self.n_eff_before_peak:.1f} / {self.n_eff_after_peak:.1f} eff])  "
            f"({self.n_clipped} point(s) clipped)"
        )


# ---------------------------------------------------------------------------
# Extract colors
# ---------------------------------------------------------------------------

def get_peak_colors(
    results: dict[str, PeakResult],
    prefix: str = "",
    colors: dict[str, str] = [
        ["ztfg","ztfr"], ["ztfr","ztfi"],
    ],
    min_eff_points: int = 1,
) -> dict:
    """
    Go through the results and extract the specified colors (magnitude differences).

    Returns
    -------
    dict : color name → (color value, color error, time difference)
    """

    colout = {}
    for col in colors:
        if col[0] in results and col[1] in results:
            colout[prefix+"{}-{}".format(col[0],col[1])] = -2.5 * np.log10( 
                results[col[0]].peak_flux / results[col[1]].peak_flux 
                )
            colout[prefix+"{}-{}_err".format(col[0],col[1])] = 1.085736 * np.sqrt(
                (results[col[0]].peak_flux_err / results[col[0]].peak_flux)**2 +
                (results[col[1]].peak_flux_err / results[col[1]].peak_flux)**2
            )
            colout[prefix+"{}-{}_dt".format(col[0],col[1])] = results[col[0]].peak_time - results[col[1]].peak_time
            colout[prefix+"{}_n_eff_before_peak".format( col[0] )] = results[col[0]].n_eff_before_peak
            colout[prefix+"{}_n_eff_after_peak".format( col[0] )] =results[col[0]].n_eff_after_peak 
            colout[prefix+"{}_n_clipped".format( col[0] )] =  results[col[0]].n_clipped 
            colout[prefix+"{}_n_eff_before_peak".format( col[1] )] = results[col[1]].n_eff_before_peak
            colout[prefix+"{}_n_eff_after_peak".format( col[1] )] =results[col[1]].n_eff_after_peak 
            colout[prefix+"{}_n_clipped".format( col[1] )] =  results[col[1]].n_clipped
            colout[prefix+"{}_n_count".format( col[0] )] =  sum( results[col[0]].mask )
            colout[prefix+"{}_n_count".format( col[1] )] =  sum( results[col[1]].mask )
            if (
                results[col[0]].n_eff_before_peak < min_eff_points or 
                results[col[0]].n_eff_after_peak < min_eff_points or 
                results[col[1]].n_eff_before_peak < min_eff_points or
                results[col[1]].n_eff_after_peak < min_eff_points
            ):
                colout[prefix+"{}-{}_warning".format(col[0],col[1])] = True

    return colout




# ---------------------------------------------------------------------------
# Sigma-clipping
# ---------------------------------------------------------------------------

def _sigma_clip(
    flux: np.ndarray,
    flux_err: np.ndarray,
    n_sigma: float = 3.0,
    n_iter: int = 3,
) -> np.ndarray:
    """
    Iterative sigma-clipping using the median and MAD as robust estimators.

    The rejection threshold per point is max(n_sigma * sigma_MAD, flux_err),
    so the measurement uncertainty informs clipping near the detection limit.

    Returns
    -------
    mask : bool array, True = keep
    """
    mask = np.ones(len(flux), dtype=bool)

    for _ in range(n_iter):
        good = flux[mask]
        if len(good) < 3:
            break
        median = np.median(good)
        mad = np.median(np.abs(good - median))
        sigma_mad = 1.4826 * mad          # consistent estimator of std for Gaussian
        threshold = np.maximum(n_sigma * sigma_mad, flux_err)
        mask = mask & (np.abs(flux - median) < threshold)

    return mask


# ---------------------------------------------------------------------------
# Matérn-3/2 GP
# ---------------------------------------------------------------------------

def _matern32(r: np.ndarray, length_scale: float, amplitude: float) -> np.ndarray:
    """Matérn-3/2 covariance: k(r) = amp² (1 + √3|r|/l) exp(−√3|r|/l)."""
    x = np.sqrt(3.0) * np.abs(r) / length_scale
    return amplitude ** 2 * (1.0 + x) * np.exp(-x)


def _build_K(
    t1: np.ndarray,
    t2: np.ndarray,
    length_scale: float,
    amplitude: float,
    noise_var: np.ndarray | None = None,
) -> np.ndarray:
    r = t1[:, None] - t2[None, :]          # (n, m)
    K = _matern32(r, length_scale, amplitude)
    if noise_var is not None:
        K += np.diag(noise_var + 1e-9)     # jitter for numerical stability
    return K


def _gp_posterior(
    t_obs: np.ndarray,
    f_obs: np.ndarray,
    e_obs: np.ndarray,
    t_grid: np.ndarray,
    length_scale: float,
    amplitude: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact GP posterior mean and std on t_grid given (t_obs, f_obs ± e_obs).

    The amplitude is estimated as the standard deviation of f_obs if not given.
    """
    if amplitude is None:
        amplitude = float(np.std(f_obs)) or 1.0

    noise_var = e_obs ** 2
    K_obs = _build_K(t_obs, t_obs, length_scale, amplitude, noise_var)

    try:
        cf = cho_factor(K_obs, lower=True)
    except LinAlgError:
        # Fallback: add more jitter
        K_obs += np.eye(len(t_obs)) * 1e-6
        cf = cho_factor(K_obs, lower=True)

    alpha = cho_solve(cf, f_obs)                        # K^{-1} y

    K_cross = _build_K(t_grid, t_obs, length_scale, amplitude)   # (m, n)
    mean = K_cross @ alpha

    # Posterior variance: k** - k*^T K^{-1} k*
    v = cho_solve(cf, K_cross.T)                        # K^{-1} k*  (n, m)
    K_self = _matern32(np.zeros(len(t_grid)), length_scale, amplitude)
    var = K_self - np.einsum("ij,ij->j", K_cross.T, v)
    std = np.sqrt(np.maximum(var, 0.0))

    return mean, std


# ---------------------------------------------------------------------------
# Time-weighted (Gaussian kernel) smoothing
# ---------------------------------------------------------------------------

def _time_weighted_smooth(
    t_grid: np.ndarray,
    t_obs: np.ndarray,
    f_obs: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """
    Nadaraya-Watson estimator with a Gaussian kernel in time.

    Unlike index-based smoothing, this correctly handles irregular cadence
    and large gaps — observations far away in time contribute very little.
    """
    r = t_grid[:, None] - t_obs[None, :]               # (m, n)
    weights = np.exp(-0.5 * (r / bandwidth) ** 2)      # (m, n)
    weight_sum = weights.sum(axis=1)
    # Avoid division by zero at t_grid points far from all observations
    with np.errstate(invalid="ignore"):
        smooth = np.where(weight_sum > 1e-12, (weights @ f_obs) / weight_sum, np.nan)
    return smooth


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def estimate_peak_flux(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    band: str = "unknown",
    *,
    method: Literal["gp", "twa", "top5"] = "gp",
    n_sigma: float = 3.0,
    n_clip_iter: int = 3,
    length_scale: float | None = None,
    n_grid: int = 500,
    gp_amplitude: float | None = None,
) -> PeakResult:
    """
    Estimate the peak flux in a single photometric band.

    Parameters
    ----------
    time, flux, flux_err : array_like
        Observed epochs, fluxes, and 1-sigma uncertainties. Need not be sorted.
    band : str
        Band label (for bookkeeping only).
    method : {'gp', 'twa', 'top5'}
        Peak estimation method.
        - 'gp'   Gaussian Process interpolation with Matérn-3/2 kernel.
                 Best for sparse, irregularly-sampled data. Returns a
                 posterior uncertainty on the peak.
        - 'twa'  Time-weighted (Nadaraya-Watson) smoothing, then argmax.
                 Faster and parameter-free beyond `length_scale`.
        - 'top5' As 'twa', but the peak is the median of the top 5% of
                 smoothed values — more conservative for broad light curves.
    n_sigma : float
        Sigma-clipping rejection threshold (default 3.0).
    n_clip_iter : int
        Number of sigma-clipping iterations (default 3).
    length_scale : float or None
        Characteristic time scale in the same units as `time`.
        - For 'gp': the GP length-scale (sets correlation range).
        - For 'twa'/'top5': the Gaussian kernel bandwidth.
        If None, defaults to 1/5 of the observed time baseline. Set this
        to roughly the expected rise/fall timescale of the source.
    n_grid : int
        Number of points in the dense interpolation grid (default 500).
    gp_amplitude : float or None
        GP signal amplitude. If None, estimated as std(flux_good).

    Returns
    -------
    PeakResult
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)

    if not (len(time) == len(flux) == len(flux_err)):
        raise ValueError("time, flux, and flux_err must have the same length.")

    # Sort by time
    order = np.argsort(time)
    time, flux, flux_err = time[order], flux[order], flux_err[order]

    # Default length-scale
    t_baseline = time[-1] - time[0]
    if length_scale is None:
        length_scale = t_baseline / 5.0
    if length_scale <= 0:
        raise ValueError("length_scale must be positive.")

    # --- Step 1: Sigma-clipping ---
    mask = _sigma_clip(flux, flux_err, n_sigma=n_sigma, n_iter=n_clip_iter)
    n_clipped = int((~mask).sum())

    t_good = time[mask]
    f_good = flux[mask]
    e_good = flux_err[mask]

    if len(t_good) < 2:
        warnings.warn(
            f"Band {band}: fewer than 2 points survived clipping. "
            "Returning NaN peak.", RuntimeWarning
        )
        return PeakResult(
            band=band, peak_flux=np.nan, peak_flux_err=np.nan,
            peak_time=np.nan, n_clipped=n_clipped, mask=mask,
        )

    # Dense time grid spanning observations
    t_grid = np.linspace(t_good[0], t_good[-1], n_grid)

    # --- Step 2: Peak estimation ---
    gp_mean = gp_std = None

    if method == "gp":
        gp_mean, gp_std = _gp_posterior(
            t_good, f_good, e_good, t_grid,
            length_scale=length_scale,
            amplitude=gp_amplitude,
        )
        peak_idx = int(np.nanargmax(gp_mean))
        peak_flux = float(gp_mean[peak_idx])
        peak_flux_err = float(gp_std[peak_idx])
        peak_time = float(t_grid[peak_idx])

    elif method in ("twa", "top5"):
        smooth = _time_weighted_smooth(t_grid, t_good, f_good, bandwidth=length_scale)
        gp_mean = smooth          # expose smoothed curve via same field
        gp_std = np.zeros_like(smooth)

        if method == "twa":
            peak_idx = int(np.nanargmax(smooth))
            peak_flux = float(smooth[peak_idx])
            peak_time = float(t_grid[peak_idx])
        else:  # top5
            valid = smooth[~np.isnan(smooth)]
            cutoff = np.percentile(valid, 95)
            top_vals = smooth[smooth >= cutoff]
            peak_flux = float(np.median(top_vals))
            peak_idx = int(np.nanargmax(smooth))
            peak_time = float(t_grid[peak_idx])

        peak_flux_err = 0.0

    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'gp', 'twa', or 'top5'.")

    # --- Step 3: Count surviving observations before/after peak ---
    # Raw counts: any point on the correct side of the peak
    n_before_peak = int(np.sum(t_good < peak_time))
    n_after_peak  = int(np.sum(t_good > peak_time))

    # Effective counts: weight each point by the same kernel used for smoothing,
    # evaluated at its distance from peak_time.  This ensures n_eff is consistent
    # with how much each point actually influenced the peak estimate:
    #   - gp    -> Matern-3/2  (matches GP covariance)
    #   - twa / top5 -> Gaussian  (matches Nadaraya-Watson kernel)
    # A point >3 length-scales away contributes ~0 in either kernel.
    dt = t_good - peak_time
    if method == "gp":
        kernel_weights = _matern32(dt, length_scale, amplitude=1.0)
    else:  # twa / top5 use a Gaussian kernel
        kernel_weights = np.exp(-0.5 * (dt / length_scale) ** 2)
    n_eff_before_peak = float(np.sum(kernel_weights[t_good < peak_time]))
    n_eff_after_peak  = float(np.sum(kernel_weights[t_good > peak_time]))

    return PeakResult(
        band=band,
        peak_flux=peak_flux,
        peak_flux_err=peak_flux_err,
        peak_time=peak_time,
        n_clipped=n_clipped,
        mask=mask,
        n_before_peak=n_before_peak,
        n_after_peak=n_after_peak,
        n_eff_before_peak=n_eff_before_peak,
        n_eff_after_peak=n_eff_after_peak,
        gp_time_grid=t_grid,
        gp_mean=gp_mean,
        gp_std=gp_std,
    )


def estimate_peak_flux_multiband(
    bands: dict[str, dict],
    **kwargs,
) -> dict[str, PeakResult]:
    """
    Convenience wrapper: run `estimate_peak_flux` for each band.

    Parameters
    ----------
    bands : dict
        Keys are band names. Values are dicts with keys
        'time', 'flux', 'flux_err' (array-like).
    **kwargs
        Forwarded to `estimate_peak_flux` (same for all bands).

    Returns
    -------
    dict mapping band name → PeakResult

    Example
    -------
    >>> results = estimate_peak_flux_multiband({
    ...     'g': {'time': t_g, 'flux': f_g, 'flux_err': e_g},
    ...     'r': {'time': t_r, 'flux': f_r, 'flux_err': e_r},
    ... }, method='gp', length_scale=8.0)
    >>> for r in results.values():
    ...     print(r)
    """
    return {
        band: estimate_peak_flux(
            data["time"], data["flux"], data["flux_err"],
            band=band, **kwargs,
        )
        for band, data in bands.items()
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_peak_result(
    result: PeakResult,
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    *,
    ax=None,
    color: str | None = None,
    time_unit: str = "days",
    flux_unit: str = "flux",
    show_clipped: bool = True,
    show_uncertainty: bool = True,
    title: str | None = None,
):
    """
    Plot a single-band light curve together with its interpolation and peak.

    Parameters
    ----------
    result : PeakResult
        Output of `estimate_peak_flux`.
    time, flux, flux_err : array_like
        The *original* (unsorted, unclipped) observations passed to the estimator.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Created automatically if None.
    color : str, optional
        Base colour for the band. Defaults to the first matplotlib colour cycle entry.
    time_unit, flux_unit : str
        Axis-label units.
    show_clipped : bool
        If True, plot rejected outliers as red crosses (default True).
    show_uncertainty : bool
        If True, shade the ±1σ GP posterior uncertainty (default True).
        Has no effect for 'twa'/'top5' results (uncertainty is zero).
    title : str, optional
        Axes title. Defaults to "Band <name>".

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)

    # Re-sort to match internal order used by the estimator
    order = np.argsort(time)
    time, flux, flux_err = time[order], flux[order], flux_err[order]

    mask = result.mask
    col = color or ax._get_lines.get_next_color()

    # --- Kept observations ---
    ax.errorbar(
        time[mask], flux[mask], yerr=flux_err[mask],
        fmt="o", color=col, markersize=5, linewidth=1,
        capsize=2, label="Observations", zorder=4,
    )

    # --- Clipped outliers ---
    if show_clipped and (~mask).any():
        ax.errorbar(
            time[~mask], flux[~mask], yerr=flux_err[~mask],
            fmt="x", color="#E24B4A", markersize=7, linewidth=1.5,
            capsize=2, label="Clipped outliers", zorder=5,
        )

    # --- Interpolated curve ---
    if result.gp_time_grid is not None and result.gp_mean is not None:
        t_grid = result.gp_time_grid
        mean = result.gp_mean

        ax.plot(t_grid, mean, color=col, linewidth=1.8,
                label="GP / smoothed mean", zorder=3)

        if show_uncertainty and result.gp_std is not None:
            std = result.gp_std
            if np.any(std > 0):
                ax.fill_between(
                    t_grid, mean - std, mean + std,
                    color=col, alpha=0.18, label="±1σ uncertainty", zorder=2,
                )

    # --- Peak marker ---
    if not np.isnan(result.peak_time):
        unc_str = f" ± {result.peak_flux_err:.3f}" if result.peak_flux_err > 0 else ""
        peak_label = (
            f"Peak = {result.peak_flux:.3f}{unc_str}\n"
            f"t = {result.peak_time:.2f} {time_unit}"
        )
        ax.axvline(result.peak_time, color=col, linewidth=1, linestyle="--",
                   alpha=0.6, zorder=2)
        ax.scatter(
            [result.peak_time], [result.peak_flux],
            marker="*", s=220, color="#FAC775", edgecolors="#BA7517",
            linewidths=1.2, label=peak_label, zorder=6,
        )

    ax.set_xlabel(f"Time ({time_unit})")
    ax.set_ylabel(f"Flux ({flux_unit})")
    ax.set_title(title if title is not None else f"Band {result.band}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25, linewidth=0.5)

    return ax


def plot_peak_results_multiband(
    results: dict[str, PeakResult],
    bands_data: dict[str, dict],
    *,
    colors: dict[str, str] | None = None,
    time_unit: str = "days",
    flux_unit: str = "flux",
    show_clipped: bool = True,
    show_uncertainty: bool = True,
    ncols: int = 2,
    figsize_per_panel: tuple[float, float] = (7, 3.5),
    suptitle: str | None = None,
    savefig: str | None = None,
):
    """
    Plot all bands in a multi-panel figure.

    Parameters
    ----------
    results : dict[str, PeakResult]
        Output of `estimate_peak_flux_multiband`.
    bands_data : dict[str, dict]
        The original data dict passed to `estimate_peak_flux_multiband`.
        Each value must have keys 'time', 'flux', 'flux_err'.
    colors : dict[str, str], optional
        Map band name → matplotlib colour string.
        Unspecified bands get automatic colours.
    time_unit, flux_unit : str
        Axis-label units, forwarded to `plot_peak_result`.
    show_clipped, show_uncertainty : bool
        Forwarded to `plot_peak_result`.
    ncols : int
        Number of columns in the panel grid (default 2).
    figsize_per_panel : (width, height)
        Size of each individual panel in inches.
    suptitle : str, optional
        Overall figure title.
    savefig : str, optional
        If given, save the figure to this path (e.g. 'peaks.pdf').

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    band_names = list(results.keys())
    n = len(band_names)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0.45, wspace=0.3)

    # Default colour palette — one per band
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = colors or {}

    ax_list = axes.flat
    for idx, band in enumerate(band_names):
        ax = next(ax_list)
        col = colors.get(band, default_colors[idx % len(default_colors)])
        data = bands_data[band]
        plot_peak_result(
            results[band],
            data["time"], data["flux"], data["flux_err"],
            ax=ax, color=col,
            time_unit=time_unit, flux_unit=flux_unit,
            show_clipped=show_clipped,
            show_uncertainty=show_uncertainty,
        )

    # Hide any unused panels
    for ax in list(ax_list):
        ax.set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=1.01)

    if savefig:
        fig.savefig(savefig, bbox_inches="tight", dpi=150)

    return fig, axes.tolist()


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    def _make_band(n=20, peak_t=30.0, amp=1.2, width=8.0, seed=0):
        rng_b = np.random.default_rng(seed)
        t = np.sort(rng_b.uniform(1, 59, n))
        signal = amp * np.exp(-0.5 * ((t - peak_t) / width) ** 2) + 0.06
        err = rng_b.uniform(0.03, 0.07, n)
        noise = rng_b.normal(0, err)
        flux = signal + noise
        # inject ~6% outliers
        n_out = max(1, int(n * 0.06))
        out_idx = rng_b.choice(n, n_out, replace=False)
        flux[out_idx] += amp * rng_b.choice([-1.8, 1.8], n_out) * rng_b.uniform(0.8, 1.2, n_out)
        return t, flux, err

    bands = {}
    for i, name in enumerate(["g", "r", "i", "z"]):
        t, f, e = _make_band(n=18, peak_t=25 + i * 3, amp=1.0 + i * 0.2, seed=i)
        bands[name] = {"time": t, "flux": f, "flux_err": e}

    print("=== GP interpolation ===")
    results = estimate_peak_flux_multiband(bands, method="gp", length_scale=8.0)
    for r in results.values():
        print(r)

    print("\n=== Time-weighted smoothing ===")
    results_twa = estimate_peak_flux_multiband(bands, method="twa", length_scale=8.0)
    for r in results_twa.values():
        print(r)

    # --- Plot ---
    fig, _ = plot_peak_results_multiband(
        results, bands,
        suptitle="Multi-band light curves — GP interpolation",
        time_unit="days", flux_unit="counts",
        savefig="peak_flux_demo.png",
    )
    import matplotlib.pyplot as plt
    plt.show()

    

