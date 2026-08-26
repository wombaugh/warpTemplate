#!/usr/bin/env python
"""Analyze warp template color distributions and generate color-corrected warp coefficients.

Fits exponentially modified Gaussian (EMG) distributions to template peak colors,
derives E(B-V) correction polynomials, and stores updated warp coefficient files.
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
from scipy.optimize import curve_fit, minimize_scalar
from scipy.stats import exponnorm

#import tqdm

from warptemplate import WarpfitTemplateLoader, add_warpclasses, register_all


# -----------------------------------------------------------------------------
# Class lists by category
# -----------------------------------------------------------------------------

N_CLASSES = [
    'SN IIP', 'SN Ia-91T', 'SN IIn', 'SN Ib/c', 'SN Ibn', 'SN Ia-pec', 'SLSN-I',
    'SN Ic', 'SN Ic-BL', 'SN II', 'SLSN-II', 'SN Iax', 'SN Ia-91bg', 'SN Ia-CSM',
    'SN Ia-SC', 'SN Ib', 'SN IIb',
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
    # Remove infinities
    d = data[np.isfinite(data)]
    if len(d) < 5:
        return d

    # Mild IQR filter
    d = iqr_filter(d, k=3.0)
    if len(d) < 5:
        return d

    # Central mask for core statistics
    core = central_mask(d, 5, 95)
    if len(core) < 2:
        return d

    mu, sigma = np.mean(core), np.std(core)

    # Asymmetric sigma clipping
    d = d[(d > mu - 5 * sigma) & (d < mu + 5 * sigma)]

    # Class-specific overrides
    if class_name in ['SLSN-II']:
        d = iqr_filter(data[np.isfinite(data)], k=3.0)  # One large outlier in SLSN-II

    return d


# -----------------------------------------------------------------------------
# Fitting and storage
# -----------------------------------------------------------------------------

def fit_emg(data: np.ndarray) -> tuple[float, float, float]:
    """Fit exponentially modified Gaussian. Returns (K, loc, scale)."""
    K, loc, scale = exponnorm.fit(data)
    return K, loc, scale


def fit_emg_and_store(
    data: np.ndarray,
    model_name: str,
    col1: str,
    col2: str,
    outfile: Path,
    also_store: dict | None = None,
) -> tuple[float, float, float]:
    """Fit EMG and append results to CSV."""
    K, loc, scale = fit_emg(data)

    result = {
        "model": model_name,
        "K": K,
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

    return K, loc, scale


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_emg_fit(
    data: np.ndarray,
    K: float,
    loc: float,
    scale: float,
    class_name: str,
    outdir: Path,
) -> Path:
    """Generate publication-quality EMG fit plot. Returns output path."""
    # Determine bins
    n = len(data)
    bins = 5 if n < 30 else 10 if n < 120 else 20

    # Plot range
    s, e = float(np.min(data)), float(np.max(data))
    s = min(s, loc - 0.8)
    e = max(e, loc + 1.2)

    x = np.linspace(s, e, 1000)
    pdf = exponnorm.pdf(x, K, loc=loc, scale=scale)

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
    ax.plot(x, pdf, color="darkred", lw=2.5, label="EMG fit")

    ax.set_xlabel("Peak g-R (ZTF mag)")
    ax.set_ylabel("Relative Frequency")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    text = f"K = {K:.2f}\n$\\mu$ = {loc:.2f}\n$\\sigma$ = {scale:.2f}"
    ax.text(0.98, 0.95, text, transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", fc="white", ec="gray"))

    plt.tight_layout()

    safe_name = class_name.replace("/", "")
    outpath = outdir / f"emg_fit_{safe_name}.png"
    plt.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.close(fig)

    return outpath


def plot_ebv_correlation(
    df: pd.DataFrame,
    coeffs: np.ndarray,
    class_name: str,
    outdir: Path,
) -> Path:
    """Generate E(B-V) correlation plot. Returns output path."""
    x = df['targetcol'] - df['natcol']
    y = df['ebvfit']
    z = df['natcol']

    # Class-specific masking for fit
    mask = np.ones(len(x), dtype=bool)
    if class_name in ['SLSN-I', 'SLSN-II', 'SLSN (e)', 'SLSN (w)']:
        mask = df['natcol'] > -1
    elif class_name in ['SN Ic-BL']:
        mask = df['natcol'] < 1

    x_fit = np.linspace(x[mask].min(), x[mask].max(), 300)
    poly = np.poly1d(coeffs)
    y_fit = poly(x_fit)

    fig, ax = plt.subplots(figsize=(7, 5))
    hb = ax.hexbin(x, y, gridsize=60, cmap='viridis', bins='log')
    plt.colorbar(hb, ax=ax, label='log(N)')

    ax.plot(x_fit, y_fit, color='red', linewidth=2, label='Color fit')
    ax.grid(linestyle="--", alpha=0.3)
    ax.legend()

    ax.set_xlabel(r"$(g-R)_{draw}$ - $(g-R)_{template}$")
    ax.set_ylabel("E(B-V)")

    plt.tight_layout()

    safe_name = class_name.replace("/", "")
    outpath = outdir / f"distcolcorr_{safe_name}.pdf"
    plt.savefig(outpath, dpi=300)
    plt.close(fig)

    return outpath


# -----------------------------------------------------------------------------
# Core computation
# -----------------------------------------------------------------------------

def color_with_ebv(
    warped_model: sncosmo.Model,
    ebv: float,
    rv: float,
    band1: str,
    band2: str,
    t0_1: float,
    t0_2: float,
) -> float:
    """Evaluate color of warped model with given E(B-V)."""
    warped_model.set(hostebv=ebv, hostr_v=rv)
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



def compute_ebv_correlations(
    templates: list[dict],
    cols: dict[str, float],
    peakphases: dict[float],
    colband: list[str],
    K: float,
    loc: float,
    scale: float,
    class_name: str,
    toskip: list[str],
    n_draws: int = 100,
    random_state: int = 41,
    disable_progress: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Simulate target colors and fit E(B-V) corrections. Returns (df_fits, poly_coeffs)."""
    emg_dist = exponnorm(K, loc=loc, scale=scale)

    colfits = []

    # Pre-filter valid templates and precompute fixed parameters
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

        # Setup model once per template
        warped_model = sncosmo.Model(
            source=source,
            effects=[sncosmo.CCM89Dust()],
            effect_names=['host'],
            effect_frames=['rest']
        )
        warped_model.set(z=0)

        t0 = peakphases[modid]

        valid_templates.append((k, modid, natcol, warped_model, t0))

    # Outer progress bar: templates
    template_iter = _get_progress_bar(
        valid_templates,
        desc=f"EBV fits ({class_name})",
        total=len(valid_templates),
        disable=disable_progress,
    )

    for k, modid, natcol, warped_model, t0 in template_iter:
        def fit_function(ebv: float) -> float:
            return np.abs(
                color_with_ebv(
                    warped_model, ebv, 3.1,
                    colband[0], colband[1],
                    t0, t0,
                ) - target_col
            )

        # Sample target colors from EMG distribution
        target_colors = emg_dist.rvs(size=n_draws, random_state=random_state + k)

        # Optional inner progress for many draws (disabled by default for cleanliness)
        for target_col in target_colors:
            try:
                result = minimize_scalar(fit_function, bounds=(-5.0, 5.0), method='bounded')
                if result.success:
                    colfits.append({
                        'k': k,
                        'sn': modid,
                        'z': warped_model.get('z'),
                        'natcol': natcol,
                        'targetcol': target_col,
                        'ebvfit': result.x,
                    })
                else:
                    print(f'\nfit not success {target_col:.2f} {natcol:.2f} diff={natcol-target_col:.2f}')
            except Exception as e:
                print(f'\nfit fail {target_col:.2f} {natcol:.2f} diff={natcol-target_col:.2f}: {e}')

    dfcol = pd.DataFrame(colfits)

    # Fit polynomial
    x = dfcol['targetcol'] - dfcol['natcol']
    y = dfcol['ebvfit']

    mask = np.ones(len(x), dtype=bool)
    if class_name in ['SLSN-I', 'SLSN-II', 'SLSN (e)', 'SLSN (w)']:
        mask = dfcol['natcol'] > -1
    elif class_name in ['SN Ic-BL']:
        mask = dfcol['natcol'] < 1

    coeffs = np.polyfit(x[mask], y[mask], 3)

    return dfcol, coeffs


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def run_analysis(args: argparse.Namespace) -> dict:
    """Execute full analysis pipeline. Returns metadata dict."""
    register_all()

    class_name = get_class_name(args.category, args.cid)
    print(f"Processing class: {class_name}")

    # Initialize loader (assuming no color version)
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

        # As peak phase, try to continously use peak in ztfg 
        t0 = t['model'].source.peakphase('ztfg')
        peakphases[modid] = t0 

        t['model'].set(z=0)
        cols[modid] = (
            t['model'].bandmag(colband[0], 'ab', t0)
            - t['model'].bandmag(colband[1], 'ab', t0)
        )
        obscols[modid] = t.get('peak_gp_ztfg-ztfr', None)

    # Filter and fit EMG
    data_values = np.array(list(cols.values()))
    mydata = apply_filter_pipeline(data_values, class_name)

    # Override: use unfiltered for small samples
    if len(cols) >= 5:
        mydata = data_values[np.isfinite(data_values)]

    # Fit and store
    fit_csv = args.outdir / args.fit_csv
    K, loc, scale = fit_emg_and_store(
        mydata, class_name, colband[0], colband[1],
        fit_csv, also_store=tcounting,
    )
    print(f"EMG fit: K={K:.3f}, loc={loc:.3f}, scale={scale:.3f}")

    # Generate plots
    emg_plot = plot_emg_fit(mydata, K, loc, scale, class_name, args.outdir)

    # E(B-V) correlations
    dfcol, coeffs = compute_ebv_correlations(
        templates, cols, peakphases, colband,
        K, loc, scale, class_name, args.toskip,
        n_draws=args.n_draws,
        disable_progress=args.no_progress,
    )
    ebv_plot = plot_ebv_correlation(dfcol, coeffs, class_name, args.outdir)

    # Build output data structure
    model_colors = {
        'K': K,
        'loc': loc,
        'scale': scale,
        'color1': colband[0],
        'color2': colband[1],
        'ebv_corr_func': list(coeffs),
    }

    # Augment warp data with peak colors
    safe_class = class_name.replace("/", "")
    
    # Use public API for data mutation
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
        'emg_params': {'K': K, 'loc': loc, 'scale': scale},
        'ebv_poly': list(coeffs),
        'fit_csv': str(fit_csv),
        'emg_plot': str(emg_plot),
        'ebv_plot': str(ebv_plot),
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

    # Class selection
    g_class = parser.add_argument_group("class selection")
    g_class.add_argument(
        "-c", "--category", choices=["n", "e", "w", "a"], default="n",
        help="Category: (n)arrow, (e)xtended, (w)ide, (a)ll"
    )
    g_class.add_argument(
        "--cid", type=int, default=11,
        help="Class index within category"
    )

    # Data paths
    g_paths = parser.add_argument_group("paths")
    g_paths.add_argument(
        "--warpdir", type=Path, default=Path("/Users/jnordin/data/models/sncosmo/warpmod"),
        help="Directory containing warp model files"
    )
    g_paths.add_argument(
        "--outdir", type=Path, default=Path("."),
        help="Directory for output plots and CSV"
    )

    # Template selection
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
        "--min-fit-quality", default=None,
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

    # Analysis parameters
    g_analysis = parser.add_argument_group("analysis")
    g_analysis.add_argument(
        "--colband", default="ztfg,ztfr",
        help="Comma-separated band pair for color measurement"
    )
    g_analysis.add_argument(
        "--version", default="4",
        help="Warp model version suffix"
    )
    g_analysis.add_argument(
        "--fit-csv", default="warptemplate_v4_color_fits.csv",
        help="CSV file for accumulating EMG fit results"
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

    # Ensure output directory exists
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Validate warpdir
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