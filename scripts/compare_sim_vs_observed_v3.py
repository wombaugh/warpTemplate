#!/usr/bin/env python
# coding: utf-8
"""
Compare simulated spectroscopic-sample peak g-r colours with observed data.

For each template colour mode (e.g. None/raw, harmonize, draw, offsetfit_draw) this
script runs the spec-sample simulation from ``simulate_spec_peak_colors.py``
(imported, must sit in the same directory) and compares the resulting
distribution of GP peak g-r colours with the observed SNe, whose colours are
read from the per-class ``btsfits{version}_{class}.json`` files written by the
sncosmo fitting script.

Only the g-r colour (``ztfg-ztfr``) is compared, since that is what the
simulation script measures.

Class selection
---------------
Prefer ``--category``/``--cid`` (like the fitting and color-analysis
scripts) over typing ``--fitclass`` directly: it validates against CLASS_MAP,
the same hand-written N/E/W/A_CLASSES lists used by the script that actually
writes the warp coefficient pickles, so it can't select a combined class
(e/w/a) that was never analysed and has no saved template file. It resolves
the class name and passes it to the simulation as ``--fitclass``. A raw
``--fitclass`` is still accepted and forwarded as before, but only warns
(doesn't fail) if it isn't in CLASS_MAP.

The observed sample uses the same name: narrow classes read one fit file,
combined classes (e/w/a) merge the fit files of their constituent narrow
classes (via the official extended/wide/all taxonomy), each with its own
redshift window. Use ``--obs-class`` to compare against a different observed
class than what's simulated.

Default ``--zmax``
-------------------
If ``--zmax`` isn't given, the simulation is run out to the highest
per-narrow-class redshift limit (``NARROW_Z_LIMITS``) among the narrow
classes making up the simulated class -- e.g. simulating "SN CC (a)" defaults
to the SLSN-I/SLSN-II limit of 0.3, not the narrower core-collapse limits,
since that combined class includes SLSN. Pass ``--zmax`` explicitly to
override.

Passing simulation options
--------------------------
Options not known to this script are forwarded to the simulation parser, so
you can use any of them here (--fitclass, --rate, --magabs-mean, --zmax,
--warpdir, --spec-loc, --seed, ...). Do not pass --color-mode/--tag/--outdir
for the simulation: they are set per mode by this script. Note the
observed-fit-file version is ``--fit-version`` (``--version`` is the warp
template version of the simulation).

Example
-------
    python compare_sim_vs_observed.py --fitclass "SN II (e)" \
        --color-modes None harmonize draw --zmax 0.08 --match-z \
        --warpdir /path/to/warpmod/v6 --outdir ./cmp

Outputs (in --outdir)
---------------------
    <tag>_<mode>_sim.csv        simulated sample per colour mode (cached)
    <tag>_comparison.png        g-r and redshift distributions, obs vs sims
    <tag>_comparison_stats.csv  mean/std/median/KS per colour mode
    <tag>_comparison.json       same statistics plus run configuration
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde

import simulate_spec_peak_colors as sim

COLOR_KEY = "ztfg-ztfr"  # observed key (from 'peak_gp_ztfg-ztfr' in the fit json)

MODE_STYLE = {  # label, colour
    "none": ("Raw", "#4C72B0"),
    "harmonize": ("Harmonized", "#55A868"),
    "draw": ("Randomized", "#C44E52"),
    "offsetfit_draw": ("Offset+Curve", "#8172B2"),
}

# -----------------------------------------------------------------------------
# Class selection
# -----------------------------------------------------------------------------
#
# N/E/W/A_CLASSES below are copied verbatim from the color-analysis script
# (the one that runs WarpfitTemplateLoader.save_class() and actually writes
# the warp coefficient pickles). They are the source of truth for which
# combined classes *exist on disk* -- E/W/A_CLASSES there are short,
# hand-written lists of only the combined classes someone has actually run
# the analysis for, NOT every combination the taxonomy could produce.
#
# WARP_MAP_EXTENDED/WIDE/ALL are the *official* narrow -> extended -> wide ->
# all taxonomy used only to answer a different question: which narrow classes
# (and therefore which observed fit-json files) feed into a given, already-
# confirmed-to-exist combined class. Composing these maps forward can suggest
# combined classes with no pickle and no consistent --cid numbering with the
# analysis script -- that mismatch was the earlier bug. Never use the
# composed set to decide which classes are selectable; only CLASS_MAP below
# does that.

N_CLASSES = [
    'SN IIP', 'SN Ia-91T', 'SN IIn', 'SN Ib/c', 'SN Ibn', 'SN Ia-pec', 'SLSN-I',
    'SN Ic', 'SN Ic-BL', 'SN II', 'SLSN-II', 'SN Iax', 'SN Ia-91bg', 'SN Ia-CSM',
    'SN Ia-SC', 'SN Ib', 'SN IIb', 'TDE',
]
E_CLASSES = ['SN Ib/c (e)', 'SLSN (e)']
W_CLASSES = ['SLSN (w)', 'SN II (w)', 'SN Ib/c (w)', 'SN Ia (w)', 'SN Ia-pec (w)']
A_CLASSES = ['SN Ia (a)', 'SN CC (a)']

CLASS_MAP = {'n': N_CLASSES, 'e': E_CLASSES, 'w': W_CLASSES, 'a': A_CLASSES}


def get_class_name(category, cid):
    """Resolve a class name from --category/--cid, exactly as the analysis script does."""
    if category not in CLASS_MAP:
        raise ValueError(f"Category must be one of {list(CLASS_MAP.keys())}, got '{category}'")
    class_list = CLASS_MAP[category]
    if not (0 <= cid < len(class_list)):
        raise ValueError(f"Class index {cid} out of range for category '{category}' "
                         f"(0-{len(class_list)-1}); classes are {class_list}")
    return class_list[cid]


WARP_MAP_EXTENDED = {
    "SN Ia-91bg": "SN Ia-91bg (e)", "SN IIn": "SN IIn (e)", "SN IIb": "SN Ib/c (e)",
    "SN Ia-CSM": "SN Ia-pec (e)", "SN Ibn": "SN Ibn (e)", "SN Ia-SC": "SN Ia-pec (e)",
    "SN Ib": "SN Ib/c (e)", "SLSN-II": "SLSN (e)", "SN Iax": "SN Ia-pec (e)",
    "SN Ia-91T": "SN Ia-91T (e)", "SLSN-I": "SLSN (e)", "SN Ic": "SN Ib/c (e)",
    "SN Ia-pec": "SN Ia-pec (e)", "SN IIP": "SN II (e)", "SN Ic-BL": "SN Ib/c (e)",
    "SN Ia": "SN Ia (e)", "SN II": "SN II (e)", "SN Ib/c": "SN Ib/c (e)", "TDE": "TDE (e)",
}
WARP_MAP_WIDE = {
    "SN II (e)": "SN II (w)", "SN Ib (e)": "SN Ib/c (w)", "SN Ibn (e)": "SN Ib/c (w)",
    "SN Ia-91T (e)": "SN Ia (w)", "SLSN (e)": "SLSN (w)", "SN IIn (e)": "SLSN (w)",
    "SN Ia-pec (e)": "SN Ia-pec (w)", "SN Ia-91bg (e)": "SN Ia-91bg (w)",
    "SN Ic (e)": "SN Ib/c (w)", "SN Ia (e)": "SN Ia (w)", "SN Ib/c (e)": "SN Ib/c (w)",
    "TDE (e)": "TDE (w)",
}
WARP_MAP_ALL = {
    "SN Ia (w)": "SN Ia (a)", "SLSN (w)": "SN CC (a)", "SN Ib/c (w)": "SN CC (a)",
    "SN Ia-pec (w)": "SN Ia (a)", "SN II (w)": "SN CC (a)",
    "SN Ia-91bg (w)": "SN Ia (a)", "TDE (w)": "TDE (a)",
}
NARROW_Z_LIMITS = {
    'SLSN-II': [0.0, 0.3], 'SLSN-I': [0.0, 0.3], 'SN Ia-91bg': [0.0, 0.055],
    'SN Ia-91T': [0.0, 0.10], 'SN Ia-CSM': [0.0, 0.10], 'SN IIn': [0.0, 0.10],
    'SN Ia-SC': [0.0, 0.10], 'SN Ia-pec': [0.01, 0.055], 'SN Iax': [0.0, 0.055],
    'TDE': [0.0, 0.3], 'SN Ibn': [0.0, 0.055], 'SN Ic-BL': [0.0, 0.055],
}
DEFAULT_Z_LIMITS = [0.0, 0.04]


def build_combined_class_map(narrow_classes):
    """Compose the extended/wide/all maps into {combined_class: [narrow, ...]}.

    Used only to resolve constituents of a class already confirmed to exist
    in CLASS_MAP -- see the module-level note above.
    """
    combined = {}
    for narrow in narrow_classes:
        ext = WARP_MAP_EXTENDED.get(narrow, narrow)
        wide = WARP_MAP_WIDE.get(ext, ext)
        allc = WARP_MAP_ALL.get(wide, wide)
        for level_class in (ext, wide, allc):
            bucket = combined.setdefault(level_class, [])
            if narrow not in bucket:
                bucket.append(narrow)
    return combined


WIDE_CLASS_MAP = build_combined_class_map(list(WARP_MAP_EXTENDED))

# Classes this script is actually willing to select -- the union of the four
# hand-written CLASS_MAP lists, i.e. only classes that exist on disk.
_VALID_CLASSES = set(N_CLASSES) | set(E_CLASSES) | set(W_CLASSES) | set(A_CLASSES)


def resolve_constituent_narrow_classes(class_name):
    """Narrow class -> itself; combined class -> its narrow constituents.

    Only resolves names present in CLASS_MAP (N/E/W/A_CLASSES); a name that
    the extended/wide/all taxonomy could theoretically produce but that has
    no entry there (no saved pickle) is rejected rather than silently
    resolved via WIDE_CLASS_MAP.
    """
    if class_name not in _VALID_CLASSES:
        raise ValueError(
            f"Unknown class '{class_name}': not in N/E/W/A_CLASSES (see CLASS_MAP). "
            f"Available: n={N_CLASSES}, e={E_CLASSES}, w={W_CLASSES}, a={A_CLASSES}"
        )
    if class_name in N_CLASSES:
        return [class_name]
    return WIDE_CLASS_MAP[class_name]


# -----------------------------------------------------------------------------
# Observed data
# -----------------------------------------------------------------------------
def load_observed_sn_data(fit_json, z_limits=None, peak_good_only=True):
    """Per-SN observed colours and redshift from one fit-result json.

    One entry per SN; colours come from the best-fitting model (lowest chi2/dof)
    that provides them. Keys: id, z, colors{'ztfg-ztfr': ...}.

    NOTE: Actually, peak_gp_{col} is determine from the raw data and should not depend on the model.
    So should be enough to look at the first model
    """
    with open(fit_json) as f:
        results = json.load(f)

    sn_data = {}
    # We loop through all model values here ... should not be needed?
    # so iterate model and then iterate sne under each ... inefficient but works?
    for model_results in results.values():
        for res in model_results:
            # success of the fit should not matter
            #if not res.get("success", False):
            #    continue
            if peak_good_only and not res.get("peak_good", False):
                continue
            z = float(res.get("z", res.get("redshift", np.nan)))
            if not np.isfinite(z):
                continue
            if z_limits is not None and not (z_limits[0] <= z <= z_limits[1]):
                continue

            chidof = res.get("chidof", res.get("chisq", np.inf) / max(res.get("ndof", 1), 1))
            sn = sn_data.setdefault(res["id"], {"id": res["id"], "z": z, "colors": {}, "_chi": {}})
            for k, v in res.items():
                if k.startswith("peak_gp_") and np.isfinite(v):
                    key = k.replace("peak_gp_", "")
                    if chidof < sn["_chi"].get(key, np.inf):
                        sn["colors"][key] = float(v)
                        sn["_chi"][key] = chidof
    return list(sn_data.values())


def load_observed_multi(class_name, args):
    """Merge per-SN records across the narrow classes making up `class_name`."""
    combined, seen = [], set()
    for narrow in resolve_constituent_narrow_classes(class_name):
        z_limits = NARROW_Z_LIMITS.get(narrow, DEFAULT_Z_LIMITS) if args.use_z_limits else None
        fit_json = Path(args.fit_json_pattern.format(
            version=args.fit_version, class_name=narrow.replace("/", "")))
        try:
            sn_list = load_observed_sn_data(fit_json, z_limits, args.peak_good_only)
        except FileNotFoundError:
            msg = f"no fit file for narrow class '{narrow}' ({fit_json})"
            if args.skip_missing:
                print(f"WARNING: {msg}; skipping.")
                continue
            raise FileNotFoundError(msg)
        new = [sn for sn in sn_list if sn["id"] not in seen]
        seen.update(sn["id"] for sn in new)
        combined.extend(new)
        print(f"  + {narrow}: {len(new)} SNe (z limits {z_limits})")
    return combined


# -----------------------------------------------------------------------------
# Simulated data (one run per colour mode)
# -----------------------------------------------------------------------------
def get_simulated_sample(mode, sim_extra, args):
    """Simulated spec sample for one colour mode; cached to CSV in --outdir."""
    csv_path = args.outdir / f"{args.tag}_{mode}_sim.csv"
    if args.reuse_sim and csv_path.exists():
        print(f"[{mode}] reusing {csv_path}")
        return pd.read_csv(csv_path, index_col=0)

    print(f"\n[{mode}] running simulation")
    sim_args = sim.parse_args(sim_extra + [
        "--color-mode", mode,
        "--tag", mode,
        "--outdir", str(args.outdir / f"sim_{mode}"),
        "--tstart", args.tstart,
        "--tstop", args.tstop,
    ])
    out, info = sim.run_simulation(sim_args)
    print(f"[{mode}] {info}")
    out.to_csv(csv_path)
    return out


# -----------------------------------------------------------------------------
# Statistics and plotting
# -----------------------------------------------------------------------------
def z_match_weights(sim_z, obs_z, n_bins):
    """Per-object weights making the simulated z distribution match the observed one."""
    edges = np.linspace(min(sim_z.min(), obs_z.min()), max(sim_z.max(), obs_z.max()), n_bins + 1)
    h_obs, _ = np.histogram(obs_z, edges)
    h_sim, _ = np.histogram(sim_z, edges)
    w_bin = np.where(h_sim > 0, (h_obs / h_obs.sum()) / np.maximum(h_sim / h_sim.sum(), 1e-12), 0.0)
    idx = np.clip(np.digitize(sim_z, edges) - 1, 0, n_bins - 1)
    w = w_bin[idx]
    return w / w.sum() if w.sum() > 0 else np.full(len(sim_z), 1.0 / len(sim_z))


def compare(obs_color, sim_color, weights, rng):
    """Summary statistics and KS test (weighted case: KS on a weighted resample)."""
    if weights is not None:
        sim_eval = rng.choice(sim_color, size=len(sim_color), replace=True, p=weights)
    else:
        sim_eval = sim_color
    ks, p = stats.ks_2samp(obs_color, sim_eval)
    return {
        "n_obs": int(len(obs_color)), "n_sim": int(len(sim_color)),
        "obs_mean": float(obs_color.mean()), "sim_mean": float(sim_eval.mean()),
        "obs_std": float(obs_color.std()), "sim_std": float(sim_eval.std()),
        "obs_median": float(np.median(obs_color)), "sim_median": float(np.median(sim_eval)),
        "delta_mean": float(sim_eval.mean() - obs_color.mean()),
        "delta_std": float(sim_eval.std() - obs_color.std()),
        "ks_stat": float(ks), "ks_pvalue": float(p),
    }


def plot_comparison(obs_color, obs_z, samples, weights, class_name, args, path):
    """Left: g-r distributions. Right: redshift distributions (obs vs sims)."""
    with plt.rc_context({"font.size": 10, "axes.linewidth": 0.9, "xtick.direction": "in",
                         "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
                         "legend.frameon": False}):
        fig, (ax, axz) = plt.subplots(1, 2, figsize=(11, 4.2))

        dists = [obs_color] + [s["gr"].values for s in samples.values()]
        lo = min(np.percentile(d, args.range_percentiles[0]) for d in dists)
        hi = max(np.percentile(d, args.range_percentiles[1]) for d in dists)
        pad = 0.06 * (hi - lo)
        bins = np.linspace(lo - pad, hi + pad, args.hist_bins + 1)
        centers = 0.5 * (bins[:-1] + bins[1:])
        grid = np.linspace(bins[0], bins[-1], 400)

        h_obs, _ = np.histogram(obs_color, bins=bins, density=True)
        ax.fill_between(centers, h_obs, step="mid", color="0.15", alpha=0.12)
        ax.step(centers, h_obs, where="mid", color="0.15", lw=1.8,
                label=f"Observed (N={len(obs_color)})", zorder=6)
        ax.plot(grid, gaussian_kde(obs_color)(grid), color="0.15", lw=1.3, zorder=6)

        for mode, df in samples.items():
            label, color = MODE_STYLE.get(mode.lower(), (mode, None))
            w = weights.get(mode)
            g = df["gr"].values
            h, _ = np.histogram(g, bins=bins, density=True, weights=w)
            ax.fill_between(centers, h, step="mid", color=color, alpha=0.10)
            ax.step(centers, h, where="mid", color=color, lw=1.8,
                    label=f"{label} (N={len(g)})")
            ax.plot(grid, gaussian_kde(g, weights=w)(grid), color=color, lw=1.3, ls="--")

        ax.set_xlabel("Peak g - r (mag)")
        ax.set_ylabel("Probability density")
        ax.set_xlim(bins[0], bins[-1])
        ax.set_ylim(bottom=0)
        ax.set_title(f"{class_name}" + (" (sims z-matched)" if args.match_z else ""),
                     loc="left", style="italic")
        ax.legend(loc="upper right", fontsize=9)

        zmax = max([obs_z.max()] + [s["z"].max() for s in samples.values() if "z" in s])
        zbins = np.linspace(0, zmax, 25)
        axz.hist(obs_z, bins=zbins, density=True, color="0.15", alpha=0.3, label="Observed")
        for mode, df in samples.items():
            if "z" not in df:
                continue
            label, color = MODE_STYLE.get(mode.lower(), (mode, None))
            axz.hist(df["z"], bins=zbins, density=True, histtype="step", lw=1.6,
                     color=color, label=label)
        axz.set_xlabel("Redshift")
        axz.set_ylabel("Probability density")
        axz.legend(fontsize=9)

        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Unrecognised options are forwarded to simulate_spec_peak_colors.py.",
    )
    g = p.add_argument_group("class selection")
    g.add_argument("-c", "--category", choices=["n", "e", "w", "a"], default=None,
                   help="Select the class via CLASS_MAP, like the analysis/fitting "
                        "scripts (n=narrow, e=extended, w=wide, a=all), instead of "
                        "typing --fitclass. Takes precedence over --fitclass.")
    g.add_argument("--cid", type=int, default=None,
                   help="Class index within --category (see CLASS_MAP for this script "
                        "with no arguments, or the analysis script, for the index)")

    g = p.add_argument_group("simulation colour modes")
    g.add_argument("--color-modes", nargs="+", default=["None", "offsetfit_draw", "draw", "harmonize"],
                   help="warptemplate color_mode values to simulate (None, harmonize, draw, offsetfit_draw)")
    g.add_argument("--reuse-sim", action="store_true",
                   help="Reuse cached <tag>_<mode>_sim.csv files in --outdir instead of re-simulating")

    g = p.add_argument_group("observed data")
    g.add_argument("--fit-json-pattern", default="/Users/jnordin/data/models/sncosmo/btsfitsv{version}_{class_name}.json")
    # Note that for redshift generation to the full observed limit, use v7. Use v6 for the volume limit.
    g.add_argument("--fit-version", default="8", help="Version string in the fit json file names")
    g.add_argument("--use-z-limits", dest="use_z_limits", action="store_true", default=True)
    g.add_argument("--no-z-limits", dest="use_z_limits", action="store_false")
    g.add_argument("--peak-good-only", dest="peak_good_only", action="store_true", default=True)
    g.add_argument("--include-bad-peak", dest="peak_good_only", action="store_false")
    g.add_argument("--skip-missing", action="store_true",
                   help="Skip narrow classes without a fit file instead of failing")
    g.add_argument("--obs-class", default=None,
                   help="Observed class name (default: the simulation --fitclass)")

    # Simulation parameters
    g.add_argument("--tstart", default="2019-01-01",
                   help="Start of draw window (date string)")
    g.add_argument("--tstop", default="2022-01-01", help="End of draw window")

    g = p.add_argument_group("comparison")
    g.add_argument("--match-z", action="store_true",
                   help="Reweight each simulated sample to the observed redshift distribution")
    g.add_argument("--z-bins", type=int, default=10, help="Bins for --match-z weights")
    g.add_argument("--hist-bins", type=int, default=30)
    g.add_argument("--range-percentiles", nargs=2, type=float, default=[1, 99])
    g.add_argument("--seed-compare", type=int, default=0, help="Seed for the weighted KS resampling")

    g = p.add_argument_group("output")
    g.add_argument("--outdir", type=Path, default=Path("./cmp"))
    g.add_argument("--tag", default="simobs")
    return p


def main(argv=None):
    args, sim_extra = build_parser().parse_known_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed_compare)

    # Class selection: --category/--cid (validated against CLASS_MAP, the
    # same hand-written lists the analysis script uses) takes precedence.
    # It's injected into sim_extra as --fitclass so the simulation gets the
    # identical, validated name; --obs-class can still override the
    # observed-data side alone.
    if args.category is not None or args.cid is not None:
        if args.category is None or args.cid is None:
            sys.exit("--category and --cid must be given together.")
        sim_class_name = get_class_name(args.category, args.cid)
        if "--fitclass" in sim_extra:
            sys.exit("Pass either --category/--cid or --fitclass, not both.")
        sim_extra = sim_extra + ["--fitclass", sim_class_name]
    else:
        base_sim = sim.parse_args(sim_extra)
        sim_class_name = base_sim.fitclass
        if sim_class_name not in _VALID_CLASSES:
            print(f"WARNING: '{sim_class_name}' is not in N/E/W/A_CLASSES (CLASS_MAP); "
                  f"it may not correspond to a saved template pickle. Available: "
                  f"n={N_CLASSES}, e={E_CLASSES}, w={W_CLASSES}, a={A_CLASSES}")

    # Default --zmax: the highest per-narrow-class redshift limit (NARROW_Z_LIMITS)
    # among the classes making up sim_class_name, so a combined class (e/w/a) is
    # simulated out to the range its most distant constituent actually needs (e.g.
    # SLSN reaching z=0.3 inside a "SN CC (a)" run). Only applied if the person
    # didn't pass --zmax themselves.
    if "--zmax" not in sim_extra:
        try:
            constituents = resolve_constituent_narrow_classes(sim_class_name)
            zmax_default = max(NARROW_Z_LIMITS.get(n, DEFAULT_Z_LIMITS)[1] for n in constituents)
            print(f"No --zmax given; using {zmax_default} (max NARROW_Z_LIMITS upper bound "
                  f"over constituents {constituents}).")
            sim_extra = sim_extra + ["--zmax", str(zmax_default)]
        except ValueError as e:
            print(f"WARNING: could not derive a default --zmax ({e}); "
                  f"falling back to the simulation script's own default.")

    class_name = args.obs_class or sim_class_name
    print(f"Class: {class_name}   colour modes: {args.color_modes}")

    # Observed
    print("\nLoading observed data")
    sn_data = load_observed_multi(class_name, args)
    obs = [(s["colors"][COLOR_KEY], s["z"]) for s in sn_data if COLOR_KEY in s["colors"]]
    if not obs:
        sys.exit(f"No observed {COLOR_KEY} colours found for {class_name}.")
    obs_color, obs_z = (np.array(x) for x in zip(*obs))
    print(f"Observed: {len(obs_color)} SNe with {COLOR_KEY}, z in [{obs_z.min():.3f}, {obs_z.max():.3f}]")

    # Simulated, one sample per colour mode
    samples, weights, rows = {}, {}, []
    for mode in args.color_modes:
        df = get_simulated_sample(mode, sim_extra, args)
        df = df[np.isfinite(df["gr"])]
        if df.empty:
            print(f"WARNING: no valid simulated colours for mode {mode}; skipping.")
            continue
        samples[mode] = df

        w = None
        if args.match_z:
            if "z" in df:
                w = z_match_weights(df["z"].values, obs_z, args.z_bins)
                print(f"[{mode}] z-matching: effective N = {1 / np.sum(w ** 2):.0f} of {len(df)}")
            else:
                print(f"WARNING: [{mode}] no 'z' column; cannot z-match.")
        weights[mode] = w

        row = {"class_name": class_name, "color_mode": mode, "z_matched": w is not None}
        row.update(compare(obs_color, df["gr"].values, w, rng))
        rows.append(row)

    if not samples:
        sys.exit("No simulated samples to compare.")

    stats_df = pd.DataFrame(rows)
    stats_path = args.outdir / f"{args.tag}_comparison_stats.csv"
    stats_df.to_csv(stats_path, index=False)

    png_path = args.outdir / f"{args.tag}_comparison.png"
    plot_comparison(obs_color, obs_z, samples, weights, class_name, args, png_path)

    json_path = args.outdir / f"{args.tag}_comparison.json"
    with open(json_path, "w") as f:
        json.dump({"class_name": class_name, "observed_color": COLOR_KEY,
                   "config": {k: str(v) for k, v in vars(args).items()},
                   "sim_args": sim_extra, "results": rows}, f, indent=2, default=str)

    print("\n" + stats_df[["color_mode", "n_sim", "sim_mean", "sim_std", "delta_mean",
                           "ks_stat", "ks_pvalue"]].to_string(index=False))
    print(f"\nObserved: N={len(obs_color)}, mean={obs_color.mean():.3f}, std={obs_color.std():.3f}")
    print(f"Wrote {stats_path}\n      {png_path}\n      {json_path}")


if __name__ == "__main__":
    main()
