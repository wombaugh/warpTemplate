#!/usr/bin/env python
"""Fit color offset + Milky Way extinction to align harmonized templates with observations.

Extends the redshifted comparison script: instead of evaluating raw/harmonized/drawn
templates separately, this script takes the harmonized distribution as baseline and
fits two nuisance parameters per color:
    - delta_c:  global additive color offset (mag)
    - av_scale: the SCALE of an A_V distribution (mag) -- see "v2 changes" below

The fit is performed by minimizing the KS statistic between observed and
extinction-shifted model distributions, evaluated at the observed redshifts.

------------------------------------------------------------------------------
v2 changes (in response to: not every template should share one A_V, it's
slow, and the delta_c initial guess should be blue-biased)
------------------------------------------------------------------------------

1. PER-DRAW A_V, NOT ONE SHARED VALUE.
   Every simulated (template, z) point now gets its OWN independently-drawn
   A_V ~ AV_DISTRIBUTIONS[av_dist](av_scale) instead of the same fixed a_v
   applied to every template. Physically, each hypothetical SN has its own
   line-of-sight reddening, so the model should show *scatter* from that,
   not just a rigid shift -- this also lets the fit reproduce a broadened
   observed distribution, which a single shared A_V structurally cannot.
   Default family is 'exponential' (1 free parameter, non-negative,
   standard choice for line-of-sight dust in the SN literature); 'halfnormal'
   is also available via --av-dist.

2. MUCH FASTER, by removing repeated sncosmo calls.
   The original script called sncosmo's Model.bandmag() *inside* the
   objective function, so every one of the ~600-1800 evaluations across the
   6-point multi-start Nelder-Mead runs (plus another 1600 evaluations for
   the diagnostic KS-landscape panel!) re-evaluated every template at every
   redshift from scratch. That's the source of the slowness.

   Here, each template's UN-shifted ("intrinsic") color at each redshift is
   computed exactly ONCE per color (a single call to the already-imported
   generate_redshifted_colors -- the only step that touches sncosmo).
   The extinction color-shift is linear in A_V at a fixed effective
   wavelength (see the bandpass-wavelength fix below), so for *any* trial
   (delta_c, av_scale) the model color is just:

       model_color = intrinsic_color + delta_c + dcolor_dav * (av_scale * base_draws)

   where `base_draws` are FIXED standard-scale random variates, pre-drawn
   ONCE and rescaled by av_scale on every evaluation (both AV_DISTRIBUTIONS
   families are scale families, so this reproduces the target distribution
   exactly for any av_scale while keeping the objective a smooth,
   reproducible function of the fit parameters -- no fresh sampling noise
   per evaluation, which matters for a derivative-free optimizer). This
   turns every objective evaluation into a few numpy ops, so the coarse
   grid search + Nelder-Mead polish (and the diagnostic KS-landscape panel)
   are now effectively free.

   The old ad hoc 6-point multi-start (with bounds that didn't even match
   some of the initial points) is replaced by a coarse grid scan -> local
   polish, which is both more robust and, thanks to the above, still fast.

3. BLUE-BIASED delta_c INITIAL GUESS.
   Instead of starting the optimizer at delta_c=0 (or an arbitrary point),
   the initial guess is:
       delta_c_init = min(0, median(obs) - median(intrinsic))
   i.e. shift toward blue by whatever the raw median mismatch already
   suggests, but never redward -- any redward residual is left for the A_V
   component to explain, consistent with treating "extra dust" as the
   physical mechanism for reddening and delta_c as a smaller, separate
   color-calibration term. This point is also seeded into the grid search
   so it can't be missed by an unlucky grid spacing.

4. BUG FIX: bandpass wavelengths.
   extinction_color_offset() previously looked up wavelengths in a
   hardcoded FILTER_PIVOT_WAVELENGTH dict that does not contain 'ztfg',
   'ztfr', or 'ztfi' -- the actual bands used throughout this pipeline's
   color keys (e.g. "ztfg-ztfr"). Since dict.get() returned None for those,
   the function silently fell back to returning 0.0, meaning A_V had no
   effect at all on the fit for real ZTF colors. Replaced with an effective
   wavelength computed directly from sncosmo's registered bandpass
   (transmission-weighted mean wavelength), which works for any band
   sncosmo knows about rather than a hand-picked list.
------------------------------------------------------------------------------
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sncosmo
import extinction
from scipy import optimize, stats

from warptemplate import WarpfitTemplateLoader, register_all

# Re-use the infrastructure from the comparison script
from compare_data_template_colors_v6 import (
    get_class_name, resolve_constituent_narrow_classes,
    load_observed_sn_data_multi,
    generate_redshifted_colors,
    _publication_rcparams,
)


# -----------------------------------------------------------------------------
# Extinction model: Fitzpatrick99 with R_V = 3.1
# -----------------------------------------------------------------------------
#
# v2 BUG FIX: the original hand-rolled polynomial approximation of F99 here
# had an inverted wavelength dependence -- numerically checked against the
# `extinction` package (the same well-tested library sncosmo's own dust
# classes are built on internally) at the ZTF g/r effective wavelengths:
#
#     reference (extinction.fitzpatrick99): A(g)-A(r) = +0.410 per unit A_V
#     old hand-rolled polynomial:           A(g)-A(r) = -0.170 per unit A_V
#
# i.e. the old code had reddening making colors *bluer*, backwards. This
# had no visible effect before because of the OTHER bug (pivot wavelengths
# missing 'ztfg'/'ztfr' -> dcolor_dav was always 0 anyway), but would have
# actively fit the wrong sign the moment that was fixed. Replaced with a
# direct call to `extinction.fitzpatrick99` (confirmed numerically linear in
# a_v above, which is what lets dcolor_dav be computed once and reused).

F99_RV = 3.1


# --- bandpass-driven effective wavelength, not a hardcoded dict ------------
# v2 BUG FIX (separate from the sign bug above): the old FILTER_PIVOT_WAVELENGTH
# dict didn't contain 'ztfg'/'ztfr'/'ztfi' -- the actual bands used throughout
# this pipeline's color keys (e.g. "ztfg-ztfr") -- so it silently fell back
# and returned 0.0, meaning A_V had NO effect at all on the fit for real ZTF
# colors. This works for any band sncosmo knows about instead.

_EFFECTIVE_WAVELENGTH_CACHE: dict[str, float] = {}


def band_effective_wavelength(band: str) -> float:
    """Transmission-weighted mean wavelength (Angstrom) of a bandpass
    registered with sncosmo. Cached since it's called repeatedly for the
    same handful of bands."""
    if band not in _EFFECTIVE_WAVELENGTH_CACHE:
        bp = sncosmo.get_bandpass(band)
        w_eff = np.trapezoid(bp.wave * bp.trans, bp.wave) / np.trapezoid(bp.trans, bp.wave)
        _EFFECTIVE_WAVELENGTH_CACHE[band] = float(w_eff)
    return _EFFECTIVE_WAVELENGTH_CACHE[band]


def extinction_color_offset(band1: str, band2: str, a_v: float, r_v: float = F99_RV) -> float:
    """Color change (band1-band2) from Fitzpatrick99 extinction at A_V,
    evaluated at each band's sncosmo-derived effective wavelength. Linear in
    a_v (confirmed numerically) -- this is what lets the fit below skip
    re-touching sncosmo for every trial value."""
    if a_v == 0:
        return 0.0
    wave = np.array([band_effective_wavelength(band1), band_effective_wavelength(band2)])
    a_lambda = extinction.fitzpatrick99(wave, a_v, r_v=r_v)
    return float(a_lambda[0] - a_lambda[1])



# -----------------------------------------------------------------------------
# v2: per-draw A_V distribution (replaces the single shared a_v)
# -----------------------------------------------------------------------------
#
# Both families below are SCALE families: sampling a fixed standard-scale
# variate once and multiplying by av_scale reproduces the target
# distribution exactly for any av_scale. That's what lets fit evaluations
# reuse one fixed `base_draws` array (see simulate_model_colors) instead of
# redrawing -- keeping the objective smooth and reproducible in av_scale.

AV_DISTRIBUTIONS = {
    'exponential': (
        lambda rng, n: rng.exponential(scale=1.0, size=n),
        lambda scale: scale,                          # mean of Exponential(scale)
    ),
    'halfnormal': (
        lambda rng, n: np.abs(rng.normal(loc=0.0, scale=1.0, size=n)),
        lambda scale: scale * np.sqrt(2.0 / np.pi),   # mean of HalfNormal(scale)
    ),
}


def make_base_draws(av_dist: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if av_dist not in AV_DISTRIBUTIONS:
        raise ValueError(f"av_dist must be one of {list(AV_DISTRIBUTIONS)}, got {av_dist!r}")
    sampler, _ = AV_DISTRIBUTIONS[av_dist]
    return sampler(rng, n)


def av_distribution_mean(av_dist: str, scale: float) -> float:
    _, mean_fn = AV_DISTRIBUTIONS[av_dist]
    return float(mean_fn(scale))


def simulate_model_colors(intrinsic_color: np.ndarray, dcolor_dav: float,
                          delta_c: float, av_scale: float,
                          base_draws: np.ndarray) -> np.ndarray:
    """Vectorized model color generation -- no sncosmo calls.

    intrinsic_color : cached un-shifted (delta_c=0, A_V=0) color for every
        (template, z) combination, from generate_redshifted_colors(), called
        exactly once per color in fit_offset_extinction().
    dcolor_dav : color change per unit A_V for this band pair (constant,
        see extinction_color_offset()).
    base_draws : fixed standard-scale variates, one per intrinsic-color
        point -- rescaled by av_scale to realize AV_DISTRIBUTIONS[av_dist].
    """
    av_i = av_scale * base_draws
    return intrinsic_color + delta_c + dcolor_dav * av_i


# -----------------------------------------------------------------------------
# Objective function: KS distance between observed and model distributions
# -----------------------------------------------------------------------------

@dataclass
class FitResult:
    """Container for offset+extinction fit results."""
    delta_c: float          # Best-fit global color offset (mag)
    av_scale: float         # Best-fit scale of the A_V distribution (mag)
    av_dist: str            # Which family av_scale parametrizes
    av_mean: float          # Mean A_V implied by av_scale, for interpretability
    ks_stat: float          # KS statistic at best fit
    ks_pvalue: float        # KS p-value at best fit
    success: bool           # Optimizer convergence flag
    nfev: int                # Number of function evaluations (local polish step)
    obs_mean: float          # Observed distribution mean
    obs_std: float           # Observed distribution std
    model_mean: float        # Best-fit model distribution mean
    model_std: float         # Best-fit model distribution std
    baseline_ks: float       # KS for unshifted harmonized (delta_c=0, av_scale=0)
    baseline_pvalue: float   # Baseline p-value


@dataclass
class FitContext:
    """Cached, reusable pieces from a fit -- passed to the plotting functions
    so they never need to touch sncosmo again."""
    obs_color: np.ndarray
    obs_z: np.ndarray
    intrinsic_color: np.ndarray
    intrinsic_z: np.ndarray
    dcolor_dav: float
    base_draws: np.ndarray
    av_dist: str


def _ks_objective(params: np.ndarray, obs_color: np.ndarray,
                  intrinsic_color: np.ndarray, dcolor_dav: float,
                  base_draws: np.ndarray) -> float:
    """Scalar objective: KS statistic between obs and (delta_c, av_scale)-shifted model."""
    delta_c, av_scale = params
    if av_scale < 0:
        return 1.0 + abs(av_scale)  # keep the optimizer out of unphysical territory
    model_color = simulate_model_colors(intrinsic_color, dcolor_dav, delta_c, av_scale, base_draws)
    ks_stat, _ = stats.ks_2samp(obs_color, model_color)
    return ks_stat


def _coarse_grid_search(obj_fn, delta_c_bounds, av_scale_bounds, n_grid,
                        extra_delta_c: float | None = None):
    """Cheap (now that obj_fn is vectorized numpy, not sncosmo) grid scan to
    get a robust starting point for the local polish. `extra_delta_c`, if
    given, is folded into the grid so a computed initial guess (e.g. the
    blue-biased delta_c_init) can't be missed by grid spacing."""
    delta_cs = np.linspace(*delta_c_bounds, n_grid)
    if extra_delta_c is not None:
        delta_cs = np.unique(np.append(delta_cs, np.clip(extra_delta_c, *delta_c_bounds)))
    av_scales = np.linspace(*av_scale_bounds, n_grid)

    best_val = np.inf
    best_dc, best_av = delta_cs[0], av_scales[0]
    for dc in delta_cs:
        for av in av_scales:
            val = obj_fn(np.array([dc, av]))
            if val < best_val:
                best_val, best_dc, best_av = val, dc, av
    return best_dc, best_av, best_val


def fit_offset_extinction(obs_data: list[dict], templates: list[dict],
                          color_key: str, z_values: np.ndarray,
                          rest_phase: float = 0,
                          n_draw_per_z: int | None = None,
                          random_seed: int = 42,
                          av_dist: str = 'exponential',
                          delta_c_bounds: tuple[float, float] = (-1.0, 1.0),
                          av_scale_bounds: tuple[float, float] = (0.0, 1.0),
                          n_grid: int = 25,
                          verbose: bool = False) -> tuple[FitResult, FitContext] | tuple[None, None]:
    """Fit delta_c and the A_V-distribution scale to minimize KS distance
    between observed and model color distributions.

    sncosmo is touched exactly once here (inside generate_redshifted_colors,
    to get each template's intrinsic color at each redshift). Every
    subsequent evaluation -- grid scan, local polish, and later the
    diagnostic plots -- reuses that cache via simulate_model_colors(), which
    is why this is now fast.
    """
    try:
        band1, band2 = color_key.split('-')
    except ValueError:
        return None, None

    obs_color_all = np.array([s['colors'].get(color_key, np.nan) for s in obs_data])
    obs_z_all = np.array([s['z'] for s in obs_data])
    valid = np.isfinite(obs_color_all)
    obs_color = obs_color_all[valid]
    obs_z = obs_z_all[valid]

    if len(obs_color) < 5 or len(templates) == 0:
        return None, None

    # The ONE sncosmo-touching step for this whole fit.
    intrinsic = generate_redshifted_colors(
        templates, band1, band2, z_values,
        rest_phase=rest_phase, n_draw_per_z=n_draw_per_z, random_seed=random_seed,
    )
    if len(intrinsic) == 0:
        return None, None
    intrinsic_color = intrinsic['color']
    intrinsic_z = intrinsic['z']

    dcolor_dav = extinction_color_offset(band1, band2, 1.0)  # slope, linear in A_V

    rng = np.random.default_rng(random_seed)
    base_draws = make_base_draws(av_dist, len(intrinsic_color), rng)

    baseline_ks, baseline_p = stats.ks_2samp(obs_color, intrinsic_color)

    # v2: blue-biased initial guess for delta_c -- see module docstring point 3.
    delta_c_init = min(0.0, float(np.median(obs_color) - np.median(intrinsic_color)))
    if verbose:
        print(f"    delta_c initial guess (blue-biased): {delta_c_init:+.4f}")

    def obj_fn(p):
        return _ks_objective(p, obs_color, intrinsic_color, dcolor_dav, base_draws)

    grid_dc, grid_av, grid_val = _coarse_grid_search(
        obj_fn, delta_c_bounds, av_scale_bounds, n_grid, extra_delta_c=delta_c_init,
    )
    if verbose:
        print(f"    grid best: delta_c={grid_dc:+.4f}, av_scale={grid_av:.4f}, KS={grid_val:.4f}")

    result = optimize.minimize(
        obj_fn, x0=np.array([grid_dc, grid_av]),
        method='Nelder-Mead',
        options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-5},
        bounds=[delta_c_bounds, av_scale_bounds],
    )

    delta_c, av_scale = result.x
    av_scale = max(av_scale, 0.0)
    model_color = simulate_model_colors(intrinsic_color, dcolor_dav, delta_c, av_scale, base_draws)
    final_ks, final_p = stats.ks_2samp(obs_color, model_color)

    fit_result = FitResult(
        delta_c=float(delta_c),
        av_scale=float(av_scale),
        av_dist=av_dist,
        av_mean=av_distribution_mean(av_dist, float(av_scale)),
        ks_stat=float(final_ks),
        ks_pvalue=float(final_p),
        success=bool(result.success),
        nfev=int(result.nfev),
        obs_mean=float(np.mean(obs_color)),
        obs_std=float(np.std(obs_color)),
        model_mean=float(np.mean(model_color)),
        model_std=float(np.std(model_color)),
        baseline_ks=float(baseline_ks),
        baseline_pvalue=float(baseline_p),
    )
    ctx = FitContext(
        obs_color=obs_color, obs_z=obs_z,
        intrinsic_color=intrinsic_color, intrinsic_z=intrinsic_z,
        dcolor_dav=dcolor_dav, base_draws=base_draws, av_dist=av_dist,
    )
    return fit_result, ctx


# -----------------------------------------------------------------------------
# Plotting: fit diagnostics
# -----------------------------------------------------------------------------

def plot_fit_diagnostics(fit_result: FitResult, ctx: FitContext,
                         color_key: str, class_name: str, outdir: Path,
                         delta_c_bounds: tuple[float, float] = (-1.0, 1.0),
                         av_scale_bounds: tuple[float, float] = (0.0, 1.0)) -> Path:
    """Four-panel diagnostic: PDF comparison, KS landscape, Q-Q, z-binned residuals.

    Everything here is derived from the cached FitContext -- no sncosmo
    calls, so the once-expensive 40x40 KS-landscape grid is now cheap.
    """
    obs_color, obs_z = ctx.obs_color, ctx.obs_z
    baseline_color = ctx.intrinsic_color
    model_color = simulate_model_colors(
        ctx.intrinsic_color, ctx.dcolor_dav, fit_result.delta_c, fit_result.av_scale, ctx.base_draws
    )

    label_fit = (f'Fit ({fit_result.av_dist}: '
                 f'\u0394c={fit_result.delta_c:+.3f}, \u27e8A_V\u27e9={fit_result.av_mean:.3f})')
    models = {'Harmonized (baseline)': baseline_color, label_fit: model_color}
    colors = {'Harmonized (baseline)': '#4C72B0', label_fit: '#55A868'}

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # Panel 1: histogram + KDE comparison
    ax = axes[0, 0]
    bins = np.linspace(
        min(obs_color.min(), min(m.min() for m in models.values())),
        max(obs_color.max(), max(m.max() for m in models.values())),
        50
    )
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    x_grid = np.linspace(bins[0], bins[-1], 400)

    from scipy.stats import gaussian_kde

    hist_obs, _ = np.histogram(obs_color, bins=bins, density=True)
    ax.fill_between(bin_centers, hist_obs, step='mid', alpha=0.25, color='0.2')
    ax.step(bin_centers, hist_obs, where='mid', color='0.2', lw=2, label=f'Observed (N={len(obs_color)})')
    kde_obs = gaussian_kde(obs_color)
    ax.plot(x_grid, kde_obs(x_grid), '0.2', lw=1.5, ls='-')

    for label, mod_c in models.items():
        c = colors[label]
        hist_mod, _ = np.histogram(mod_c, bins=bins, density=True)
        ax.step(bin_centers, hist_mod, where='mid', color=c, lw=2, label=label)
        kde_mod = gaussian_kde(mod_c)
        ax.plot(x_grid, kde_mod(x_grid), color=c, lw=1.5, ls='--')

    ax.set_xlabel(f'{color_key} (mag)')
    ax.set_ylabel('Probability density')
    ax.set_title(f'{class_name}: {color_key}')
    ax.legend(loc='best', fontsize=8)

    # Panel 2: parameter space with objective surface (cheap now -- pure numpy)
    ax = axes[0, 1]
    dc_grid = np.linspace(*delta_c_bounds, 40)
    av_grid = np.linspace(*av_scale_bounds, 40)
    DC, AV = np.meshgrid(dc_grid, av_grid)
    KS = np.empty_like(DC)
    for i in range(DC.shape[0]):
        for j in range(DC.shape[1]):
            KS[i, j] = _ks_objective(
                np.array([DC[i, j], AV[i, j]]),
                obs_color, ctx.intrinsic_color, ctx.dcolor_dav, ctx.base_draws,
            )

    im = ax.contourf(DC, AV, KS, levels=20, cmap='YlOrRd')
    ax.plot(fit_result.delta_c, fit_result.av_scale, 'w+', markersize=12, mew=2, label='Best fit')
    ax.set_xlabel('\u0394c (mag)')
    ax.set_ylabel(f'av_scale ({fit_result.av_dist}, mag)')
    ax.set_title('KS statistic landscape')
    plt.colorbar(im, ax=ax, label='KS')
    ax.legend()

    # Panel 3: Q-Q plot of observed vs best-fit model
    ax = axes[1, 0]
    q = np.linspace(0.01, 0.99, 100)
    obs_q = np.quantile(obs_color, q)
    mod_q = np.quantile(model_color, q)
    ax.plot(mod_q, obs_q, 'o', markersize=3, alpha=0.6)
    lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lim, lim, 'k--', lw=1, alpha=0.5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f'Model quantiles ({color_key})')
    ax.set_ylabel(f'Observed quantiles ({color_key})')
    ax.set_title('Q-Q: observed vs. best-fit model')
    ax.text(0.05, 0.95, f'KS = {fit_result.ks_stat:.4f}\np = {fit_result.ks_pvalue:.3g}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel 4: redshift-binned residuals
    ax = axes[1, 1]
    z_edges = np.quantile(obs_z, np.linspace(0, 1, 5))
    z_edges[0] -= 0.001
    z_edges[-1] += 0.001

    for label, mod_c in models.items():
        mod_z = ctx.intrinsic_z
        deltas, z_centers = [], []
        for i in range(len(z_edges) - 1):
            z_lo, z_hi = z_edges[i], z_edges[i + 1]
            obs_mask = (obs_z >= z_lo) & (obs_z < z_hi)
            mod_mask = (mod_z >= z_lo) & (mod_z < z_hi)
            if np.sum(obs_mask) < 3 or np.sum(mod_mask) < 3:
                continue
            deltas.append(np.mean(mod_c[mod_mask]) - np.mean(obs_color[obs_mask]))
            z_centers.append(0.5 * (z_lo + z_hi))
        ax.plot(z_centers, deltas, 'o-', label=label, color=colors.get(label))

    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('Redshift z')
    ax.set_ylabel('\u0394\u27e8color\u27e9 (model \u2212 obs)')
    ax.set_title('Mean offset by redshift bin')
    ax.legend(fontsize=8)

    plt.tight_layout()

    safe_name = class_name.replace('/', '')
    safe_color = color_key.replace('/', '-')
    outpath = outdir / f"fit_diag_{safe_name}_{safe_color}.pdf"
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return outpath


def plot_publication_fit_comparison(fit_result: FitResult, ctx: FitContext,
                                    color_key: str, class_name: str, outdir: Path,
                                    figsize: tuple[float, float] = (5.5, 4.2)) -> Path:
    """Clean publication figure: observed vs. harmonized vs. best-fit."""
    obs_color = ctx.obs_color
    baseline_color = ctx.intrinsic_color
    model_color = simulate_model_colors(
        ctx.intrinsic_color, ctx.dcolor_dav, fit_result.delta_c, fit_result.av_scale, ctx.base_draws
    )
    label_fit = f'\u0394c={fit_result.delta_c:+.2f}, \u27e8A_V\u27e9={fit_result.av_mean:.2f}'

    models = {'Harmonized': baseline_color, label_fit: model_color}
    mod_colors = {'Harmonized': '#4C72B0', label_fit: '#55A868'}

    distributions = [obs_color] + list(models.values())
    bounds = [np.percentile(d, (1, 99)) for d in distributions]
    c_lo = min(b[0] for b in bounds)
    c_hi = max(b[1] for b in bounds)
    c_pad = 0.06 * (c_hi - c_lo)
    bins = np.linspace(c_lo - c_pad, c_hi + c_pad, 40)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    x_grid = np.linspace(bins[0], bins[-1], 400)

    from scipy.stats import gaussian_kde

    with plt.rc_context(_publication_rcparams()):
        fig, ax = plt.subplots(figsize=figsize)

        hist_obs, _ = np.histogram(obs_color, bins=bins, density=True)
        ax.fill_between(bin_centers, hist_obs, step='mid', color='0.15', alpha=0.12)
        ax.step(bin_centers, hist_obs, where='mid', color='0.15', lw=1.8,
                label=f'Observed (N={len(obs_color)})')
        kde_obs = gaussian_kde(obs_color)
        ax.plot(x_grid, kde_obs(x_grid), '0.15', lw=1.3)

        for label, mod_c in models.items():
            c = mod_colors[label]
            hist_mod, _ = np.histogram(mod_c, bins=bins, density=True)
            ax.fill_between(bin_centers, hist_mod, step='mid', color=c, alpha=0.10)
            ax.step(bin_centers, hist_mod, where='mid', color=c, lw=1.8, label=label)
            kde_mod = gaussian_kde(mod_c)
            ax.plot(x_grid, kde_mod(x_grid), color=c, lw=1.3, ls='--')

        ax.text(0.97, 0.97, f'KS: {fit_result.baseline_ks:.3f} \u2192 {fit_result.ks_stat:.3f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='whitesmoke', alpha=0.9))

        ax.set_xlabel(f'{color_key} (mag)')
        ax.set_ylabel('Probability density')
        ax.set_xlim(bins[0], bins[-1])
        ax.set_ylim(bottom=0)
        ax.set_title(class_name, loc='left', style='italic')
        ax.legend(loc='upper right', handlelength=1.6)

        fig.tight_layout()

        safe_name = class_name.replace('/', '')
        safe_color = color_key.replace('/', '-')
        outpath = outdir / f"pub_fit_{safe_name}_{safe_color}.pdf"
        fig.savefig(outpath, bbox_inches='tight')
        plt.close(fig)

    return outpath


# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------

def analyze_class_offset_extinction(class_name: str, args: argparse.Namespace) -> dict | None:
    """Fit offset+extinction for all colors in a single class."""

    print(f"\n{'='*60}")
    print(f"Offset+extinction fit: {class_name}")
    print(f"{'='*60}")

    constituent_classes = resolve_constituent_narrow_classes(class_name)
    print(f"Constituent narrow classes: {constituent_classes}")

    sn_data = load_observed_sn_data_multi(
        constituent_classes, args.fit_json_pattern, args.version,
        use_z_limits=args.use_z_limits, peak_good_only=args.peak_good_only,
        skip_missing=args.skip_missing,
    )

    if len(sn_data) == 0:
        print(f"WARNING: No SN data found for {class_name}")
        return None

    print(f"Loaded {len(sn_data)} SNe with redshifts z \u2208 "
          f"[{min(s['z'] for s in sn_data):.3f}, {max(s['z'] for s in sn_data):.3f}]")

    unique_z = np.array(sorted({float(s['z']) for s in sn_data}), dtype=np.float64)

    warploader = WarpfitTemplateLoader(str(args.warpdir), version=args.version, suffix=args.suffix)
    try:
        templates = warploader.get_templates(
            fitclass=class_name, exclude_input=[],
            template_selection=args.template_selection,
            snbasis_selection=args.snbasis_selection,
            random_seed=args.random_seed,
            color_mode='harmonize',
        )
        print(f"  Harmonized templates: {len(templates)}")
    except Exception as e:
        print(f"  FAILED to load harmonized templates: {e}")
        return None

    available_colors = set()
    for s in sn_data:
        available_colors.update(s['colors'].keys())
    print(f"Available colors: {', '.join(sorted(available_colors))}")

    delta_c_bounds = tuple(args.delta_c_bounds)
    av_scale_bounds = tuple(args.av_scale_bounds)

    results = []
    for color_key in sorted(available_colors):
        if '-' not in color_key:
            continue

        print(f"\n  Fitting {color_key}...")

        fit_result, ctx = fit_offset_extinction(
            sn_data, templates, color_key, unique_z,
            rest_phase=args.phase,
            n_draw_per_z=args.n_draw_per_z,
            random_seed=args.random_seed + hash(color_key) % 10000,
            av_dist=args.av_dist,
            delta_c_bounds=delta_c_bounds,
            av_scale_bounds=av_scale_bounds,
            n_grid=args.n_grid,
            verbose=args.verbose,
        )

        if fit_result is None:
            print("    Fit failed")
            continue

        print(f"    Best fit: \u0394c = {fit_result.delta_c:+.4f}, "
              f"av_scale = {fit_result.av_scale:.4f} ({fit_result.av_dist}, "
              f"\u27e8A_V\u27e9 = {fit_result.av_mean:.4f})")
        print(f"    KS: {fit_result.baseline_ks:.4f} \u2192 {fit_result.ks_stat:.4f} "
              f"(p = {fit_result.ks_pvalue:.3g})")

        diag_path = plot_fit_diagnostics(
            fit_result, ctx, color_key, class_name, args.outdir,
            delta_c_bounds=delta_c_bounds, av_scale_bounds=av_scale_bounds,
        )
        print(f"    Diagnostics: {diag_path}")

        pub_path = plot_publication_fit_comparison(fit_result, ctx, color_key, class_name, args.outdir)
        print(f"    Publication: {pub_path}")

        results.append({
            'color': color_key,
            'fit_result': fit_result,
            'diag_path': str(diag_path),
            'pub_path': str(pub_path),
        })

    return {
        'class_name': class_name,
        'n_sn': len(sn_data),
        'z_range': [min(s['z'] for s in sn_data), max(s['z'] for s in sn_data)],
        'color_results': results,
    }


def run_offset_extinction_fit(args: argparse.Namespace) -> list[dict]:
    """Execute offset+extinction fitting pipeline."""
    register_all()

    class_name = get_class_name(args.category, args.cid)
    print(f"Processing class: {class_name}")

    results = []
    try:
        res = analyze_class_offset_extinction(class_name, args)
        if res is not None:
            results.append(res)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        if not args.continue_on_error:
            raise

    if results:
        rows = []
        for r in results:
            for cr in r['color_results']:
                fr = cr['fit_result']
                rows.append({
                    'class_name': r['class_name'],
                    'color': cr['color'],
                    'n_sn': r['n_sn'],
                    'delta_c': fr.delta_c,
                    'av_scale': fr.av_scale,
                    'av_dist': fr.av_dist,
                    'av_mean': fr.av_mean,
                    'ks_stat': fr.ks_stat,
                    'ks_pvalue': fr.ks_pvalue,
                    'baseline_ks': fr.baseline_ks,
                    'baseline_pvalue': fr.baseline_pvalue,
                    'obs_mean': fr.obs_mean,
                    'obs_std': fr.obs_std,
                    'model_mean': fr.model_mean,
                    'model_std': fr.model_std,
                    'success': fr.success,
                    'diag_path': cr['diag_path'],
                    'pub_path': cr['pub_path'],
                })

        stats_df = pd.DataFrame(rows)
        stats_path = args.outdir / "offset_extinction_fits.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"\nFit results: {stats_path}")

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for row in rows:
            improvement = row['baseline_ks'] - row['ks_stat']
            print(f"{row['color']:15s}: \u0394c={row['delta_c']:+.3f}, "
                  f"\u27e8A_V\u27e9={row['av_mean']:.3f} ({row['av_dist']}), "
                  f"KS {row['baseline_ks']:.3f} \u2192 {row['ks_stat']:.3f} "
                  f"(\u0394={improvement:+.3f})")

    return results


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit color offset + a per-draw MW extinction distribution to align harmonized templates with observations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g_class = parser.add_argument_group("class selection")
    g_class.add_argument("-c", "--category", choices=["n", "e", "w", "a"], default="n")
    g_class.add_argument("--cid", type=int, default=11)

    parser.add_argument("--warpdir", type=Path,
                        default=Path("/Users/jnordin/data/models/sncosmo/warpmod/v4"))
    parser.add_argument("--outdir", type=Path, default=Path("."))
    parser.add_argument("--fit-json-pattern", type=str,
                        default="/Users/jnordin/data/models/sncosmo/btsfitsv{version}_{class_name}.json")

    parser.add_argument("--template-selection", default='all')
    parser.add_argument("--snbasis-selection", default="all")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--version", default="4")
    parser.add_argument("--suffix", default="_col")

    parser.add_argument("--phase", type=float, default=0,
                        help="Rest-frame phase relative to peak (days)")
    parser.add_argument("--n-draw-per-z", type=int, default=None,
                        help="Subsample templates per redshift for the one-time intrinsic "
                             "evaluation (default: use all templates)")

    g_fit = parser.add_argument_group("offset + extinction fit")
    g_fit.add_argument("--av-dist", choices=list(AV_DISTRIBUTIONS), default="exponential",
                       help="Family for the per-draw A_V distribution")
    g_fit.add_argument("--delta-c-bounds", type=float, nargs=2, default=[-1.0, 1.0])
    g_fit.add_argument("--av-scale-bounds", type=float, nargs=2, default=[0.0, 1.0])
    g_fit.add_argument("--n-grid", type=int, default=25,
                       help="Coarse grid resolution per axis before the local polish")

    parser.add_argument("--use-z-limits", action="store_true", default=True)
    parser.add_argument("--no-z-limits", dest="use_z_limits", action="store_false")
    parser.add_argument("--peak-good-only", action="store_true", default=True)
    parser.add_argument("--include-bad-peak", dest="peak_good_only", action="store_false")

    parser.add_argument("--skip-missing", action="store_true", default=True)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    try:
        results = run_offset_extinction_fit(args)
        print(f"\nCompleted: {len(results)} classes")
        return 0
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
