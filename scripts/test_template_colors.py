#!/usr/bin/env python
"""Compare warp template color distributions across color correction modes.

Loads fitted EMG distributions, generates templates with different color modes
(raw/harmonize/draw), and produces comparison plots with statistical metrics.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import johnsonsu

from warptemplate import WarpfitTemplateLoader, add_warpclasses, register_all


# -----------------------------------------------------------------------------
# Class lists by category
# -----------------------------------------------------------------------------

N_CLASSES = [
    'SN IIP', 'SN Ia-91T', 'SN IIn', 'SN Ib/c', 'SN Ibn', 'SN Ia-pec', 'SLSN-I',
    'SN Ic', 'SN Ic-BL', 'SN II', 'SLSN-II', 'SN Iax', 'SN Ia-91bg', 'SN Ia-CSM',
    'SN Ia-SC', 'SN Ib', 'SN IIb','TDE',
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
# Data loading
# -----------------------------------------------------------------------------

def get_latest_model_result(model_name: str, infile: Path) -> dict | None:
    """Retrieve latest Johnson SU fit parameters from accumulated CSV."""
    if not infile.exists():
        return None

    df = pd.read_csv(infile)
    if df.empty:
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df_model = df[df["model"] == model_name]

    if df_model.empty:
        return None

    return df_model.sort_values("timestamp", ascending=False).iloc[0].to_dict()


# -----------------------------------------------------------------------------
# Template color extraction
# -----------------------------------------------------------------------------

def extract_peak_colors(templates: list[dict], band1: str, band2: str, 
                        phasemode: str | float = 'ztfg', 
                        zmode: None | float = 0.) -> np.ndarray:
    """Evaluate band1-band2 color at specified phase for all templates.
    
    phasemode: If str, use peak phase in this band. If float, use this phase (days relative to peak).
    zmode: If float, set all templates to this redshift before evaluating color. If none, keep native template z.

    Filters out non-finite or unphysically negative colors.
    """
    cols = []
    for t in templates:
        mod = t['model']
        oz = mod.get('z')

        # Determine phase and redshift to use
        if isinstance(phasemode, str):
            phase = mod.source.peakphase(phasemode)
        elif isinstance(phasemode, (int, float)):
            phase = float(phasemode)
        else:
            raise ValueError(f"Invalid phasemode: {phasemode}")
        if isinstance(zmode, (int, float)):
            mod.set(z=float(zmode))

        try:
            col = mod.bandmag(band1, "ab", phase) - mod.bandmag(band2, "ab", phase)
        except ValueError:
            # No filter coverage at this phase
            mod.set(z=oz)  # restore original z
            continue
        if np.isfinite(col) and col > -10:
            cols.append(col)
        else:
            print(f'nan/invalid color for {mod.description}')
        mod.set(z=oz)  # restore original z
    
    return np.array(cols)


# -----------------------------------------------------------------------------
# Distribution comparison metrics
# -----------------------------------------------------------------------------

def compute_kl_divergence(p_samples: np.ndarray, 
                          gamma: float, delta: float, loc: float, scale: float,
                          n_bins: int = 100, range_min: float | None = None,
                          range_max: float | None = None) -> float:
    """Approximate KL divergence between sample histogram and fitted Johnson SU."""
    if len(p_samples) == 0:
        return np.nan
    
    rmin = range_min if range_min is not None else p_samples.min()
    rmax = range_max if range_max is not None else p_samples.max()
    
    hist, bin_edges = np.histogram(p_samples, bins=n_bins, range=(rmin, rmax), density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]
    
    # Johnson SU PDF
    pdf_jsu = johnsonsu.pdf(bin_centers, gamma, delta, loc=loc, scale=scale)
    
    hist = np.clip(hist, 1e-12, None)
    pdf_jsu = np.clip(pdf_jsu, 1e-12, None)
    
    kl = np.sum(hist * bin_width * np.log(hist / pdf_jsu))
    return kl


def compute_anderson_darling_statistic(samples: np.ndarray, 
                                        gamma: float, delta: float, 
                                        loc: float, scale: float) -> float:
    """Crude A² approximation for Johnson SU (relative comparison only)."""
    x = np.sort(samples)
    n = len(x)
    if n == 0:
        return np.nan
    
    y = johnsonsu.cdf(x, gamma, delta, loc=loc, scale=scale)
    
    i = np.arange(1, n + 1)
    s = np.sum((2 * i - 1) / n * (np.log(y) + np.log1p(-y[::-1])))
    a2 = -n - s
    
    return a2


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def step_hist_peaknorm(data: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalized step histogram (peak = 1)."""
    counts, edges = np.histogram(data, bins=bins, density=True)
    peak = counts.max()
    if peak > 0:
        counts = counts / peak
    return edges, counts


def plot_filled_step(edges: np.ndarray, counts: np.ndarray, color: str, label: str,
                     alpha: float = 0.25, linestyle: str = 'solid', ax=None):
    """Draw filled step histogram."""
    if ax is None:
        ax = plt.gca()
    
    x = np.repeat(edges, 2)[1:-1]
    y = np.repeat(counts, 2)
    
    ax.fill_between(x, y, step='pre', alpha=alpha, color=color)
    ax.step(edges[:-1], counts, where='post', color=color, lw=2, label=label, linestyle=linestyle)


def plot_color_comparison(rawcols: np.ndarray, harmcols: np.ndarray, drawcols: np.ndarray,
                          model_colors: dict, class_name: str, outdir: Path,
                          band1: str = 'ztfg', band2: str = 'ztfr') -> Path:
    """Generate comparison plot for three color modes. Returns output path."""
    
    # Johnson SU parameters
    gamma = model_colors['gamma']
    delta = model_colors['delta']
    loc = model_colors['loc']
    scale = model_colors['scale']
    
    # Quantile-based range (Johnson SU hat schwerere tails als EMG)
    x_min = johnsonsu.ppf(0.001, gamma, delta, loc=loc, scale=scale)
    x_max = johnsonsu.ppf(0.999, gamma, delta, loc=loc, scale=scale)
    
    all_data = np.concatenate([c for c in [rawcols, harmcols, drawcols] if len(c) > 0])
    if len(all_data) > 0:
        x_min = min(x_min, all_data.min() - 0.1)
        x_max = max(x_max, all_data.max() + 0.1)
    
    x = np.linspace(x_min, x_max, 500)
    pdf = johnsonsu.pdf(x, gamma, delta, loc=loc, scale=scale)
    pdf /= pdf.max()
    
    bins = np.linspace(x_min, x_max, 40)
    
    # Compute histograms
    edges_r, counts_r = step_hist_peaknorm(rawcols, bins)
    edges_h, counts_h = step_hist_peaknorm(harmcols, bins)
    edges_d, counts_d = step_hist_peaknorm(drawcols, bins)
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(7, 5))
    
    plot_filled_step(edges_r, counts_r, '#4C72B0', 'Observed', linestyle='solid', alpha=0.3, ax=ax)
    plot_filled_step(edges_h, counts_h, '#55A868', 'Harmonized', linestyle='dashed', alpha=0.5, ax=ax)
    plot_filled_step(edges_d, counts_d, '#C44E52', 'Randomized', linestyle='dotted', alpha=0.4, ax=ax)
    
    ax.plot(x, pdf, color='black', lw=2, label='Class PDF')
    
    # Mark distribution parameters (loc = median bei symmetrischer JSU, nicht mean)
    ax.axvline(x=loc, color='black', linestyle='--', alpha=0.5, lw=1)
    ax.text(loc, 0.95, f'ξ={loc:.3f}', transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=9, color='black', alpha=0.7)
    
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlabel(f"{band1}-{band2} (mag)")
    ax.set_ylabel("Normalized density")
    
    # Statistics annotation
    stats_text = (
        f"N(raw)={len(rawcols)}\n"
        f"N(harm)={len(harmcols)}\n"
        f"N(draw)={len(drawcols)}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, ha='left', va='top',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # Safe filename
    safe_name = class_name.replace('/', '')
    outpath = outdir / f"color_distcomp_{safe_name}.pdf"
    plt.savefig(outpath, dpi=300)
    plt.close(fig)
    
    return outpath


def plot_summary_grid(results: list[dict], outdir: Path) -> Path:
    """Generate multi-panel summary across all processed classes."""
    n = len(results)
    if n == 0:
        return None
    
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)
    
    for idx, res in enumerate(results):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]
        
        mc = res['model_colors']
        gamma, delta, loc, scale = mc['gamma'], mc['delta'], mc['loc'], mc['scale']
        
        x_min = johnsonsu.ppf(0.001, gamma, delta, loc=loc, scale=scale)
        x_max = johnsonsu.ppf(0.999, gamma, delta, loc=loc, scale=scale)

        for mode, color, ls, alpha in [
            ('raw', '#4C72B0', 'solid', 0.3),
            ('harmonize', '#55A868', 'dashed', 0.5),
            ('draw', '#C44E52', 'dotted', 0.4),
        ]:
            data = res[f'{mode}_colors']
            if len(data) == 0:
                continue
            edges, counts = step_hist_peaknorm(data, np.linspace(x_min, x_max, 30))
            plot_filled_step(edges, counts, color, mode, alpha=alpha, linestyle=ls, ax=ax)
        
        x = np.linspace(x_min, x_max, 200)
        pdf = johnsonsu.pdf(x, gamma, delta, loc=loc, scale=scale)
        pdf /= pdf.max()
        ax.plot(x, pdf, 'k-', lw=1.5, label='PDF')

        ax.set_title(res['class_name'], fontsize=10)
        ax.set_xlabel('g-r (mag)', fontsize=8)
        ax.set_ylabel('norm. density', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc='upper right')
    
    # Hide unused subplots
    for idx in range(n, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    outpath = outdir / "color_distcomp_summary.pdf"
    plt.savefig(outpath, dpi=300)
    plt.close(fig)
    
    return outpath


# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------

def analyze_class(class_name: str, args: argparse.Namespace) -> dict | None:
    """Run full comparison for a single class. Returns result dict or None on failure."""
    print(f"\n{'='*50}")
    print(f"Processing: {class_name}")
    print(f"{'='*50}")
    
 # Load model colors from fit CSV
    fit_csv = args.fit_csv
    model_colors = get_latest_model_result(class_name, fit_csv)
    
    if model_colors is None:
        print(f"WARNING: No Johnson SU fit found for {class_name} in {fit_csv}")
        if not args.skip_missing:
            raise ValueError(f"Missing fit data for {class_name}")
        return None    
    # Verify required keys exist
    required_keys = ['gamma', 'delta', 'loc', 'scale']
    if not all(k in model_colors for k in required_keys):
        print(f"WARNING: Fit for {class_name} missing required keys {required_keys}")
        if not args.skip_missing:
            raise ValueError(f"Incomplete fit data for {class_name}")
        return None
    
    print(f"Loaded fit: γ={model_colors['gamma']:.3f}, δ={model_colors['delta']:.3f}, "
          f"ξ={model_colors['loc']:.3f}, λ={model_colors['scale']:.3f}")
        
    # Initialize loader with version and suffix (matches analysis pipeline)
    warploader = WarpfitTemplateLoader(
        str(args.warpdir),
        version=args.version,
        suffix=args.suffix,
    )
    
    # Verify model colors are available in the .pkl file (not just CSV)
    pkl_model_colors = warploader.get_model_colors(class_name)
    if pkl_model_colors is None:
        print(f"WARNING: No model_colors in .pkl file for {class_name}; color modes will fail")
    
    # Generate templates in three modes
    modes = {}
    for mode in ['raw', 'harmonize', 'draw']:
        color_mode = None if mode == 'raw' else mode
        
        try:
            templates = warploader.get_templates(
                fitclass=class_name,
                exclude_input=[],
                template_selection=args.template_selection,
                snbasis_selection=args.snbasis_selection,
                random_seed=args.random_seed,
                color_mode=color_mode,
#                min_fit_quality='gold',            
            )
            modes[mode] = templates
            print(f"  {mode:12s}: {len(templates)} templates")
        except ValueError as e:
            if "color_mode requires 'model_colors'" in str(e) or "missing 'peakcol'" in str(e):
                print(f"  {mode:12s}: FAILED — {e}")
                if mode != 'raw':
                    modes[mode] = []
                    continue
            raise
    
    # Extract colors
    rawcols = extract_peak_colors(modes['raw'], args.band1, args.band2, args.phasemode, args.zmode)
    harmcols = extract_peak_colors(modes['harmonize'], args.band1, args.band2, args.phasemode, args.zmode)
    drawcols = extract_peak_colors(modes['draw'], args.band1, args.band2, args.phasemode, args.zmode)
    
    print(f"  Valid colors: raw={len(rawcols)}, harm={len(harmcols)}, draw={len(drawcols)}")
    
    # Compute statistics
    gamma = model_colors['gamma']
    delta = model_colors['delta']
    loc = model_colors['loc']
    scale = model_colors['scale']

    stats = {
        'raw_mean': float(np.mean(rawcols)) if len(rawcols) > 0 else np.nan,
        'raw_std': float(np.std(rawcols)) if len(rawcols) > 0 else np.nan,
        'harm_mean': float(np.mean(harmcols)) if len(harmcols) > 0 else np.nan,
        'harm_std': float(np.std(harmcols)) if len(harmcols) > 0 else np.nan,
        'draw_mean': float(np.mean(drawcols)) if len(drawcols) > 0 else np.nan,
        'draw_std': float(np.std(drawcols)) if len(drawcols) > 0 else np.nan,
        'target_loc': float(loc),
    }
    
    # KL divergences
    for mode, cols in [('raw', rawcols), ('harm', harmcols), ('draw', drawcols)]:
        if len(cols) > 0:
            stats[f'{mode}_kl_div'] = compute_kl_divergence(cols, gamma, delta, loc, scale)
        else:
            stats[f'{mode}_kl_div'] = np.nan

    # Generate plot
    plot_path = plot_color_comparison(rawcols, harmcols, drawcols, model_colors,
                                      class_name, args.outdir, args.band1, args.band2)
    print(f"  Plot saved: {plot_path}")
    
    return {
        'class_name': class_name,
        'model_colors': {k: model_colors[k] for k in ['gamma', 'delta', 'loc', 'scale', 'color1', 'color2']},
        'raw_colors': rawcols,
        'harmonize_colors': harmcols,
        'draw_colors': drawcols,
        'stats': stats,
        'plot_path': str(plot_path),
    }


def run_comparison(args: argparse.Namespace) -> list[dict]:
    """Execute comparison for all requested classes."""

    # Also load additional sncosmo models
    register_all()

    class_name = get_class_name(args.category, args.cid)
    print(f"Processing class: {class_name}")

    # Eventually we can support multiple classes, but for now we just process one class at a time
    class_list = [class_name]
    
    print(f"Classes to process: {class_list}")
    
    results = []
    for class_name in class_list:
        try:
            res = analyze_class(class_name, args)
            if res is not None:
                results.append(res)
        except Exception as e:
            print(f"ERROR processing {class_name}: {e}")
            if not args.continue_on_error:
                raise
    
    # Summary plot
    if len(results) > 1:
        summary_path = plot_summary_grid(results, args.outdir)
        print(f"\nSummary plot: {summary_path}")
    
    # Save statistics table
    if results:
        stats_df = pd.DataFrame([r['stats'] for r in results])
        stats_df.insert(0, 'class_name', [r['class_name'] for r in results])
        stats_path = args.outdir / "color_comparison_stats.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"Statistics table: {stats_path}")
    
    return results


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def str_or_float(value):
    try:
        return float(value)
    except ValueError:
        return value
def none_or_float(value):
    if value is None or value.lower() == 'none':
        return None
    return float(value)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare warp template color distributions across correction modes.",
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
    
    # Paths
    parser.add_argument(
        "--warpdir", type=Path, default=Path("/Users/jnordin/data/models/sncosmo/warpmod/v5"),
        help="Directory containing warp coefficient files"
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("."),
        help="Output directory for plots and tables"
    )
    parser.add_argument(
        "--fit-csv", type=Path, default=Path("warptemplate_v5_color_fits.csv"),
        help="CSV file with accumulated Johnson SU fit results"  # geändert
    )
        
    # Template sampling
    parser.add_argument(
        "--template-selection", default='all',
        help="Templates per SN basis (int for weighted sampling)"
    )
    parser.add_argument(
        "--snbasis-selection", default="all",
        help="'all' or int for SN basis sampling"
    )
    parser.add_argument(
        "--random-seed", type=int, default=42,
        help="Reproducibility seed"
    )
    parser.add_argument(
        "--version", default="5",
        help="Warp model version"
    )
    parser.add_argument(
        "--suffix", default="",
        help="File suffix (e.g., '_col' for color-corrected files)"
    )
    
    # Color evaluation
    parser.add_argument(
        "--band1", default="ztfg",
        help="First band for color"
    )
    parser.add_argument(
        "--band2", default="ztfr",
        help="Second band for color"
    )
    parser.add_argument(
        "--phasemode", type=str_or_float, default='ztfg',
        help="Phase at which to evaluate color (days relative to peak): if str, use peak phase in this band; if float, use this phase"
    )

    parser.add_argument(
        "--zmode", type=none_or_float, default=0,
        help="Redshift to set for all templates before evaluating color; if None, keep native template z"
    )
    
    # Error handling
    parser.add_argument(
        "--skip-missing", action="store_true",
        help="Skip classes without fit data instead of failing"
    )
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="Continue with remaining classes if one fails"
    )
    
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    if not args.fit_csv.exists():
        print(f"Error: Fit CSV not found: {args.fit_csv}", file=sys.stderr)
        return 1
    
    try:
        results = run_comparison(args)
        print(f"\nCompleted: {len(results)} classes processed successfully")
        return 0
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())