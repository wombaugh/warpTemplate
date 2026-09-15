#!/usr/bin/env python
"""Analyze warp template color distributions and generate color-corrected warp coefficients.

Fits Johnson SU distributions to template peak colors,
derives linear color correction, and stores updated warp coefficient files.
"""

import argparse
import os
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sncosmo
from sncosmo import PropagationEffect
from scipy.optimize import curve_fit, minimize_scalar
from scipy.stats import johnsonsu

from warptemplate import WarpfitTemplateLoader, add_warpclasses, register_all


# -----------------------------------------------------------------------------
# LinearDust effect
# -----------------------------------------------------------------------------

class LinearDust(PropagationEffect):
    """
    Linear extinction: A(lambda) = a * (lambda / lambda_0 - 1)
    
    Color excess scales linearly with parameter 'a'.
    lambda_0 chosen near effective wavelength of g+r/2 ≈ 6250 Å
    for ZTF color corrections.
    """
    _param_names = ['a']
    param_names_latex = ['a']
    
    def __init__(self, lambda_0=6250., min_wave=1000., max_wave=25000.):
        self._minwave = min_wave
        self._maxwave = max_wave
        self._lambda_0 = float(lambda_0)
        self._parameters = np.array([0.])
    
    def propagate(self, wave, flux):
        a = self._parameters[0]
        # Linear extinction: (more nonlinear)
        extinction = 10.**(-0.4 * a * (wave / self._lambda_0 - 1.))
        # Fully linear 
        # extinction =  1+a * (wave / lambda_0 - 1.)
        return flux * extinction


# -----------------------------------------------------------------------------
# Class lists by category
# -----------------------------------------------------------------------------

N_CLASSES = [
    'SN IIP', 'SN Ia-91T', 'SN IIn', 'SN Ib/c', 'SN Ibn', 'SN Ia-pec', 'SLSN-I',
    'SN Ic', 'SN Ic-BL', 'SN II', 'SLSN-II', 'SN Iax', 'SN Ia-91bg', 'SN Ia-CSM',
    'SN Ia-SC', 'SN Ib', 'SN IIb', 'TDE',
]
E_CLASSES = ['SN Ib/c (e)', 'SLSN (e)']
W_CLASSES = ['SLSN (w)', 'SN II (w)', 'SN Ib/c (w)', 'SN Ia (w)', 'SN Ia-pec (w)']
A_CLASSES = ['SN Ia (a)', 'SN CC (a)']

CLASS_MAP = {
    'n': N_CLASSES,
    'e': E_CLASSES,
    'w': W_CLASSES,
    'a': A_CLASSES,
}


def get_class_name(category: str, cid: int) -> str:
    """Resolve class name from category and index."""
    if category not in CLASS_MAP:
        raise ValueError(f"Category must be one of {list(CLASS_MAP.keys())}, got '{category}'")
    class_list = CLASS_MAP[category]
    if not (0 <= cid < len(class_list)):
        raise ValueError(f"Class index {cid} out of range for category '{category}' (0-{len(class_list)-1})")
    return class_list[cid]


# -----------------------------------------------------------------------------
# Filtering utilities
# -----------------------------------------------------------------------------

def iqr_filter(data: np.ndarray, k: float = 1.5) -> np.ndarray:
    """Apply IQR-based outlier rejection."""
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return data[(data >= lower) & (data <= upper)]


def central_mask(data: np.ndarray, lower_pct: float = 5, upper_pct: float = 95) -> np.ndarray:
    """Retain central percentile range."""
    lo, hi = np.percentile(data, [lower_pct, upper_pct])
    return data[(data >= lo) & (data <= hi)]


def apply_filter_pipeline(data: np.ndarray, class_name: str) -> np.ndarray:
    """Apply adaptive filtering based on sample size and class-specific rules."""
    d = data[np.isfinite(data)]
    if len(d) < 5:
        return d

    d = iqr_filter(d, k=3.0)
    if len(d) < 5:
        return d

    core = central_mask(d, 5, 95)
    if len(core) < 2:
        return d

    mu, sigma = np.mean(core), np.std(core)
    d = d[(d > mu - 5 * sigma) & (d < mu + 5 * sigma)]

    if class_name in ['SLSN-II']:
        d = iqr_filter(data[np.isfinite(data)], k=3.0)

    return d


# -----------------------------------------------------------------------------
# Fitting and storage
# -----------------------------------------------------------------------------

def fit_johnsonsu(data: np.ndarray) -> tuple[float, float, float, float]:
    """Fit Johnson SU distribution. Returns (gamma, delta, loc, scale)."""
    gamma, delta, loc, scale = johnsonsu.fit(data)
    return gamma, delta, loc, scale


def fit_johnsonsu_and_store(
    data: np.ndarray,
    model_name: str,
    col1: str,
    col2: str,
    outfile: Path,
    also_store: dict | None = None,
) -> tuple[float, float, float, float]:
    """Fit Johnson SU and append results to CSV."""
    gamma, delta, loc, scale = fit_johnsonsu(data)

    result = {
        "model": model_name,
        "gamma": gamma,
        "delta": delta,
        "loc": loc,
        "scale": scale,
        "color1": col1,
        "color2": col2,
        "n": len(data),
        "timestamp": datetime.utcnow().isoformat(),
    }
    if also_store:
        result.update(also_store)

    df = pd.DataFrame([result])
    if outfile.exists():
        df.to_csv(outfile, mode="a", header=False, index=False)
    else:
        df.to_csv(outfile, index=False)

    return gamma, delta, loc, scale


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_johnsonsu_fit(
    data: np.ndarray,
    gamma: float,
    delta: float,
    loc: float,
    scale: float,
    class_name: str,
    outdir: Path,
) -> Path:
    """Generate publication-quality Johnson SU fit plot. Returns output path."""
    n = len(data)
    bins = 5 if n < 30 else 10 if n < 120 else 20

    s, e = float(np.min(data)), float(np.max(data))
    s = min(s, loc - 0.8)
    e = max(e, loc + 1.2)

    x = np.linspace(s, e, 1000)
    pdf = johnsonsu.pdf(x, gamma, delta, loc=loc, scale=scale)

    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.2,
        "figure.dpi": 150,
    })

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(data, bins=bins, density=True, alpha=0.5, color="steelblue",
            edgecolor="black", linewidth=0.5, label="Data")
    ax.plot(x, pdf, color="darkred", lw=2.5, label="Johnson SU fit")

    ax.set_xlabel("Peak g-R (ZTF mag)")
    ax.set_ylabel("Relative Frequency")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    text = f"$\\gamma$ = {gamma:.2f}\n$\\delta$ = {delta:.2f}\n$\\mu$ = {loc:.2f}\n$\\sigma$ = {scale:.2f}"
    ax.text(0.98, 0.95, text, transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", fc="white", ec="gray"))

    plt.tight_layout()

    safe_name = class_name.replace("/", "")
    outpath = outdir / f"johnsonsu_fit_{safe_name}.png"
    plt.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.close(fig)

    return outpath


def plot_linear_correlation(
    df: pd.DataFrame,
    coeffs: np.ndarray,
    class_name: str,
    outdir: Path,
) -> Path:
    """Generate linear color correction plot. Returns output path."""
    x = df['dcolor']
    y = df['afit']

    # Class-specific masking for fit display
    mask = np.ones(len(x), dtype=bool)
    if class_name in ['SLSN-I', 'SLSN-II', 'SLSN (e)', 'SLSN (w)']:
        mask = df['natcol'] > -1
    elif class_name in ['SN Ic-BL']:
        mask = df['natcol'] < 1

    x_fit = np.linspace(x[mask].min(), x[mask].max(), 300)
    # coeffs = [slope, intercept], enforced intercept=0
    y_fit = coeffs[0] * x_fit  # + coeffs[1] if non-zero intercept

    fig, ax = plt.subplots(figsize=(7, 5))
    hb = ax.hexbin(x, y, gridsize=60, cmap='viridis', bins='log')
    plt.colorbar(hb, ax=ax, label='log(N)')

    ax.plot(x_fit, y_fit, color='red', linewidth=2, label='Linear fit')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(linestyle="--", alpha=0.3)
    ax.legend()

    ax.set_xlabel(r"$(g-R)_{draw}$ - $(g-R)_{template}$")
    ax.set_ylabel(r"Linear parameter $a$")

    plt.tight_layout()

    safe_name = class_name.replace("/", "")
    outpath = outdir / f"distcolcorr_{safe_name}.pdf"
    plt.savefig(outpath, dpi=300)
    plt.close(fig)

    return outpath


# -----------------------------------------------------------------------------
# Core computation
# -----------------------------------------------------------------------------

def color_with_linear(
    warped_model: sncosmo.Model,
    a: float,
    band1: str,
    band2: str,
    t0_1: float,
    t0_2: float,
) -> float:
    """Evaluate color of warped model with given linear parameter a."""
    warped_model.set(hosta=a)  # LinearDust parameter
    return (
        warped_model.bandmag(band1, 'ab', t0_1)
        - warped_model.bandmag(band2, 'ab', t0_2)
    )


def _get_progress_bar(iterable, desc: str, total: int | None = None, disable: bool = False):
    """Wrapper for tqdm with graceful fallback."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, total=total, disable=disable, ncols=80)
    except ImportError:
        if not disable:
            print(f"{desc} ...", end="", flush=True)
        for i, item in enumerate(iterable):
            yield item
            if not disable and (i + 1) % max(1, (total or 100) // 10) == 0:
                print(f" {i+1}/{total}", end="", flush=True)
        if not disable:
            print(" done")


def compute_linear_correlations(
    templates: list[dict],
    cols: dict[str, float],
    peakphases: dict[float],
    colband: list[str],
    gamma: float,
    delta: float,
    loc: float,
    scale: float,
    class_name: str,
    toskip: list[str],
    n_draws: int = 100,
    random_state: int = 41,
    disable_progress: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Simulate target colors and fit linear parameter 'a'. Returns (df_fits, coeffs)."""
    jsu_dist = johnsonsu(gamma, delta, loc=loc, scale=scale)

    colfits = []

    # Pre-filter valid templates and precompute per-template slope
    valid_templates = []
    for k, t in enumerate(templates):
        mod = t['model']
        modid = mod.description
        source = mod.source

        if modid in toskip:
            continue

        natcol = cols.get(modid)
        if natcol is None or not np.isfinite(natcol):
            continue

        # Setup model with LinearDust
        warped_model = sncosmo.Model(
            source=source,
            effects=[LinearDust(lambda_0=6250.)],
            effect_names=['host'],
            effect_frames=['rest']
        )
        warped_model.set(z=0)

        t0 = peakphases[modid]

        # Precompute slope: d(color)/da via finite differences
        # Verify linearity across reasonable range
        test_a = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        test_colors = []
        for ta in test_a:
            warped_model.set(hosta=ta)
            c = (
                warped_model.bandmag(colband[0], 'ab', t0)
                - warped_model.bandmag(colband[1], 'ab', t0)
            )
            test_colors.append(c)
        
        test_colors = np.array(test_colors)
        # Linear fit to verify: should be perfect
        slope_fit = np.polyfit(test_a, test_colors, 1)
        linearity_check = np.polyfit(test_a, test_colors, 2)
        
        # Quadratic coefficient should be negligible
        quad_ratio = abs(linearity_check[0]) / abs(slope_fit[0]) if slope_fit[0] != 0 else 0
        
        dc_da = slope_fit[0]  # d(color)/da
        
        # Store: (template_index, modid, natcol, model, t0, dc_da, linearity_quality)
        valid_templates.append((k, modid, natcol, warped_model, t0, dc_da, quad_ratio))

    # Outer progress bar: templates
    template_iter = _get_progress_bar(
        valid_templates,
        desc=f"Linear fits ({class_name})",
        total=len(valid_templates),
        disable=disable_progress,
    )

    for k, modid, natcol, warped_model, t0, dc_da, quad_ratio in template_iter:
        # Analytic inversion: a = (target_col - natcol) / dc_da
        target_colors = jsu_dist.rvs(size=n_draws, random_state=random_state + k)

        for target_col in target_colors:
            a_fit = (target_col - natcol) / dc_da
            
            # Validate by forward evaluation
            warped_model.set(hosta=a_fit)
            check_col = (
                warped_model.bandmag(colband[0], 'ab', t0)
                - warped_model.bandmag(colband[1], 'ab', t0)
            )
            
            residual = check_col - target_col
            
            colfits.append({
                'k': k,
                'sn': modid,
                'z': warped_model.get('z'),
                'natcol': natcol,
                'targetcol': target_col,
                'dcolor': target_col - natcol,
                'afit': a_fit,
                'dc_da': dc_da,
                'residual': residual,
                'linearity_quad_ratio': quad_ratio,
            })

    dfcol = pd.DataFrame(colfits)

    # Fit: a vs dcolor — should be perfectly linear: a = dcolor / dc_da
    # Average slope across templates, or fit global relation
    x = dfcol['dcolor']
    y = dfcol['afit']

    mask = np.ones(len(x), dtype=bool)
    if class_name in ['SLSN-I', 'SLSN-II', 'SLSN (e)', 'SLSN (w)']:
        mask = dfcol['natcol'] > -1
    elif class_name in ['SN Ic-BL']:
        mask = dfcol['natcol'] < 1

    # Linear fit with forced zero intercept (no parameter -> no color change)
    # Use weighted fit: templates with better linearity get higher weight
    weights = 1.0 / (1.0 + dfcol['linearity_quad_ratio'])
    
    # Fit: y = slope * x, intercept forced to 0
    x_m = x[mask].values
    y_m = y[mask].values
    w_m = weights[mask].values
    
    # Weighted linear regression through origin
    slope = np.sum(w_m * x_m * y_m) / np.sum(w_m * x_m * x_m)
    
    # Verify quality
    y_pred = slope * x_m
    ss_res = np.sum(w_m * (y_m - y_pred)**2)
    ss_tot = np.sum(w_m * y_m**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    print(f"Linear fit quality: R² = {r_squared:.6f}, slope = {slope:.4f}")
    print(f"  Mean |residual| = {dfcol['residual'].abs().mean():.6f} mag")
    print(f"  Max |residual| = {dfcol['residual'].abs().max():.6f} mag")

    coeffs = np.array([slope, 0.0])  # [slope, intercept]

    return dfcol, coeffs


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def run_analysis(args: argparse.Namespace) -> dict:
    """Execute full analysis pipeline. Returns metadata dict."""
    register_all()

    class_name = get_class_name(args.category, args.cid)
    print(f"Processing class: {class_name}")

    # Initialize loader
    warploader = WarpfitTemplateLoader(args.warpdir, version=args.version, suffix='')

    # Count templates by quality tier
    tcounting = {}
    for quality in ['gold', 'silver', 'bronze']:
        print(f'  counting quality: {quality}')
        templates = warploader.get_templates(
            fitclass=class_name,
            exclude_input=args.exclude_input,
            template_selection=args.template_selection,
            snbasis_selection=args.snbasis_selection,
            min_fit_quality=quality,
            random_seed=42,
        )
        tmodels = [t['model'].description for t in templates]
        tcounting[f'{quality}_nbr_models'] = len(tmodels)
        tcounting[f'{quality}_nbr_sne'] = len(set(m.split('_')[0] for m in tmodels))
        tcounting[f'{quality}_nbr_templates'] = len(set(m.split('_')[1] for m in tmodels))

    # Load final template set
    templates = warploader.get_templates(
        fitclass=class_name,
        exclude_input=args.exclude_input,
        template_selection=args.template_selection,
        snbasis_selection=args.snbasis_selection,
        min_fit_quality=args.min_fit_quality,
        random_seed=42,
    )

    # Extract peak colors
    colband = args.colband.split(',')

    cols: dict[str, float] = {}
    obscols: dict[str, float | None] = {}
    peakphases: dict[float] = {}

    for t in templates:
        modid = t['model'].description
        if modid in args.toskip:
            print(f'skipping {modid}')
            continue

        t0 = t['model'].source.peakphase('ztfg')
        peakphases[modid] = t0 

        t['model'].set(z=0)
        cols[modid] = (
            t['model'].bandmag(colband[0], 'ab', t0)
            - t['model'].bandmag(colband[1], 'ab', t0)
        )
        obscols[modid] = t.get('peak_gp_ztfg-ztfr', None)

    # Filter and fit Johnson SU
    data_values = np.array(list(cols.values()))
    mydata = apply_filter_pipeline(data_values, class_name)

    if len(cols) >= 5:
        mydata = data_values[np.isfinite(data_values)]

    # Fit and store
    fit_csv = args.outdir / args.fit_csv
    gamma, delta, loc, scale = fit_johnsonsu_and_store(
        mydata, class_name, colband[0], colband[1],
        fit_csv, also_store=tcounting,
    )
    print(f"Johnson SU fit: gamma={gamma:.3f}, delta={delta:.3f}, loc={loc:.3f}, scale={scale:.3f}")

    # Generate plots
    jsu_plot = plot_johnsonsu_fit(mydata, gamma, delta, loc, scale, class_name, args.outdir)

    # Linear color correlations
    dfcol, coeffs = compute_linear_correlations(
        templates, cols, peakphases, colband,
        gamma, delta, loc, scale, class_name, args.toskip,
        n_draws=args.n_draws,
        disable_progress=args.no_progress,
    )
    linear_plot = plot_linear_correlation(dfcol, coeffs, class_name, args.outdir)

    # Build output data structure
    model_colors = {
        'gamma': gamma,
        'delta': delta,
        'loc': loc,
        'scale': scale,
        'color1': colband[0],
        'color2': colband[1],
        'linear_corr': {
            'type': 'LinearDust',
            'coeffs': list(coeffs),  # [slope, 0.0]
            'lambda_0': 6250.,
            'dc_da_mean': float(dfcol['dc_da'].mean()),
            'dc_da_std': float(dfcol['dc_da'].std()),
        },
        # Legacy key removed; 'ebv_corr_func' no longer applicable
    }

    # Augment warp data with peak colors
    safe_class = class_name.replace("/", "")
    
    warpcoeff = warploader.get_warpcoeff(safe_class)

    for snbase, snwarplist in warpcoeff.items():
        for snwarp in snwarplist:
            mname = snwarp['id'] + '_' + snwarp['model']
            full_id = mname + '+host'
            if full_id in args.toskip:
                print(f'skip comb: {mname}')
                continue
            snwarp['peakcol'] = cols.get(full_id, np.nan)
            print(mname, cols.get(full_id, np.nan))

    # Update via public API
    warploader.update_warpcoeff(safe_class, warpcoeff)
    warploader.update_model_colors(safe_class, model_colors)

    # Persist to disk
    pkl_path = Path(warploader.save_class(safe_class))

    return {
        'class_name': class_name,
        'n_templates': len(templates),
        'n_colors': len(cols),
        'johnsonsu_params': {'gamma': gamma, 'delta': delta, 'loc': loc, 'scale': scale},
        'linear_coeffs': list(coeffs),
        'fit_csv': str(fit_csv),
        'johnsonsu_plot': str(jsu_plot),
        'linear_plot': str(linear_plot),
        'output_pkl': str(pkl_path),
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze warp template colors and generate corrected coefficients.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g_class = parser.add_argument_group("class selection")
    g_class.add_argument(
        "-c", "--category", choices=["n", "e", "w", "a"], default="n",
        help="Category: (n)arrow, (e)xtended, (w)ide, (a)ll"
    )
    g_class.add_argument(
        "--cid", type=int, default=11,
        help="Class index within category"
    )

    g_paths = parser.add_argument_group("paths")
    g_paths.add_argument(
        "--warpdir", type=Path, default=Path("/Users/jnordin/data/models/sncosmo/warpmod"),
        help="Directory containing warp model files"
    )
    g_paths.add_argument(
        "--outdir", type=Path, default=Path("."),
        help="Directory for output plots and CSV"
    )

    g_sel = parser.add_argument_group("template selection")
    g_sel.add_argument(
        "--template-selection", default="all",
        help="'all', int, or -int for template sampling"
    )
    g_sel.add_argument(
        "--snbasis-selection", default="all",
        help="'all' or int for SN basis sampling"
    )
    g_sel.add_argument(
        "--min_fit_quality", default=None,
        choices=[None, "gold", "silver", "bronze"],
        help="Minimum fit quality tier"
    )
    g_sel.add_argument(
        "--exclude-input", nargs="*", default=[],
        help="Substrings to exclude from warptemplate IDs"
    )
    g_sel.add_argument(
        "--toskip", nargs="*", default=[],
        help="Specific template IDs to skip"
    )

    g_analysis = parser.add_argument_group("analysis")
    g_analysis.add_argument(
        "--colband", default="ztfg,ztfr",
        help="Comma-separated band pair for color measurement"
    )
    g_analysis.add_argument(
        "--version", default="5",
        help="Warp model version suffix"
    )
    g_analysis.add_argument(
        "--fit-csv", default="warptemplate_v5_color_fits.csv",
        help="CSV file for accumulating Johnson SU fit results"
    )

    g_progress = parser.add_argument_group("progress")
    g_progress.add_argument(
        "--no-progress", action="store_true",
        help="Disable progress bars"
    )
    g_progress.add_argument(
        "--n-draws", type=int, default=100,
        help="Number of target color draws per template"
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    args.outdir.mkdir(parents=True, exist_ok=True)

    if not args.warpdir.exists():
        print(f"Error: warp directory not found: {args.warpdir}", file=sys.stderr)
        return 1

    results = run_analysis(args)

    print("\n--- Results ---")
    for k, v in results.items():
        print(f"{k}: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())