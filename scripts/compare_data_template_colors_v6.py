#!/usr/bin/env python
"""Compare redshifted warp templates with observed SN colors at matching redshifts.

For each observed SN, evaluates templates at that redshift and observer-frame phase,
then compares the joint distribution of predicted vs. observed colors.

NOTE ON THIS VERSION
---------------------
The original script could only build an observed-data sample for "narrow" (n)
classes, since each of those maps 1:1 onto a single per-class fit-result JSON
file written by the sncosmo fitting script (btsfits{version}_{class}.json).
The "extended" (e), "wide" (w) and "all" (a) categories are *combinations* of
several narrow classes, so there is no single file to load for them -- you
have to load each constituent narrow class's file and merge the per-SN
records together.

The new pieces are:
  - WARP_MAP_EXTENDED / WARP_MAP_WIDE / WARP_MAP_ALL: the official narrow ->
    extended -> wide -> all class mappings, composed by _build_combined_class_map()
    into WIDE_CLASS_MAP (which narrow classes make up each combined class).
  - resolve_constituent_narrow_classes(): narrow class -> itself; combined
    class -> its list of narrow classes via WIDE_CLASS_MAP.
  - load_observed_sn_data_multi(): loads + merges the per-SN records across
    however many narrow-class files a combined class requires, applying each
    narrow class's own redshift window (NARROW_Z_LIMITS) *before* merging,
    since e.g. SLSN is fit out to z=0.3 while most other classes cut at
    z~0.07-0.10 -- a single shared z-limit for the merged sample would be
    wrong.
  - analyze_class_redshifted() now calls these instead of loading one file
    directly, but is otherwise unchanged: everything downstream (template
    evaluation, KS stats, plotting) just consumes the resulting list of
    per-SN dicts as before, whether it came from one file or several.

NOTE: 'SN Ia' and 'TDE' appear in WARP_MAP_EXTENDED but have no per-class fit
file in this pipeline -- see the warning next to _ALL_MAPPED_NARROW below.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from warptemplate import WarpfitTemplateLoader, register_all


# -----------------------------------------------------------------------------
# Class lists
# -----------------------------------------------------------------------------
#
# Narrow classes actually fit by the per-class pipeline (doc 2's `nclasses`
# list). Note 'SN Ia' (normal Ia, fit separately via salt2/salt3) and 'TDE'
# (not fit by this pipeline at all) are NOT in this list, even though both
# appear as keys in WARP_MAP_EXTENDED below -- see the warning further down.

N_CLASSES = [
    'SN IIP', 'SN Ia-91T', 'SN IIn', 'SN Ib/c', 'SN Ibn', 'SN Ia-pec', 'SLSN-I',
    'SN Ic', 'SN Ic-BL', 'SN II', 'SLSN-II', 'SN Iax', 'SN Ia-91bg', 'SN Ia-CSM',
    'SN Ia-SC', 'SN Ib', 'SN IIb',
]


# -----------------------------------------------------------------------------
# Official WARP class mappings, and machinery to compose them into a
# combined-class -> constituent-narrow-classes map.
# -----------------------------------------------------------------------------
#
# These three dicts are the source of truth, each mapping one taxonomy level
# to the next: narrow -> extended (e) -> wide (w) -> all (a). Per how they're
# defined: a class not present as a key at some level keeps its own name
# unchanged at that level (identity fallback). WARP_MAP_NARROW is a separate,
# preceding step that normalizes some raw/alias labels (e.g. "SN II-pec")
# onto one of the N_CLASSES base names above, before any of this; it isn't
# used in the extended/wide/all composition below.

WARP_MAP_NARROW = {
    "SN II-pec": "SN II",
    "SN Ib-pec": "SN Ib",
    "SN Icn": "SN Ic",
    "SN IIL": "SN II",
    "SN IIn-pec": "SN IIn",
    "SN Ic-pec": "SN Ic",
}

WARP_MAP_EXTENDED = {
    "SN Ia-91bg": "SN Ia-91bg (e)",
    "SN IIn": "SN IIn (e)",
    "SN IIb": "SN Ib/c (e)",
    "SN Ia-CSM": "SN Ia-pec (e)",
    "SN Ibn": "SN Ibn (e)",
    "SN Ia-SC": "SN Ia-pec (e)",
    "SN Ib": "SN Ib/c (e)",
    "SLSN-II": "SLSN (e)",
    "SN Iax": "SN Ia-pec (e)",
    "SN Ia-91T": "SN Ia-91T (e)",
    "SLSN-I": "SLSN (e)",
    "SN Ic": "SN Ib/c (e)",
    "SN Ia-pec": "SN Ia-pec (e)",
    "SN IIP": "SN II (e)",
    "SN Ic-BL": "SN Ib/c (e)",
    "SN Ia": "SN Ia (e)",
    "SN II": "SN II (e)",
    "SN Ib/c": "SN Ib/c (e)",
    "TDE": "TDE (e)",
}

WARP_MAP_WIDE = {
    "SN II (e)": "SN II (w)",
    "SN Ib (e)": "SN Ib/c (w)",   # unreachable via WARP_MAP_EXTENDED's actual
    "SN Ibn (e)": "SN Ib/c (w)",  # outputs -- kept only for fidelity with the
    "SN Ia-91T (e)": "SN Ia (w)",
    "SLSN (e)": "SLSN (w)",
    "SN IIn (e)": "SLSN (w)",
    "SN Ia-pec (e)": "SN Ia-pec (w)",
    "SN Ia-91bg (e)": "SN Ia-91bg (w)",
    "SN Ic (e)": "SN Ib/c (w)",   # source dict as given.
    "SN Ia (e)": "SN Ia (w)",
    "SN Ib/c (e)": "SN Ib/c (w)",
    "TDE (e)": "TDE (w)",
}

WARP_MAP_ALL = {
    "SN Ia (w)": "SN Ia (a)",
    "SLSN (w)": "SN CC (a)",
    "SN Ib/c (w)": "SN CC (a)",
    "SN Ia-pec (w)": "SN Ia (a)",
    "SN II (w)": "SN CC (a)",
    "SN Ia-91bg (w)": "SN Ia (a)",
    "TDE (w)": "TDE (a)",
}


def _build_combined_class_map(narrow_classes: list[str]) -> dict[str, list[str]]:
    """Compose WARP_MAP_EXTENDED/WIDE/ALL into {combined_class: [narrow, ...]}.

    Walks each narrow class forward through extended -> wide -> all, using
    identity (`dict.get(x, x)`) whenever a class is absent from the next map,
    i.e. it keeps its own name at that level -- matching how these maps are
    specified to behave. Returns one flat dict spanning all three combined
    levels at once, e.g.:
        {'SN Ib/c (e)': ['SN Ib/c', 'SN Ic', 'SN Ic-BL', 'SN Ib', 'SN IIb'],
         'SN Ib/c (w)': [..., 'SN Ibn'],
         'SN CC (a)':   ['SN IIP', 'SN IIn', ...], ...}
    A narrow class that keeps its own name all the way up (no entry in any
    map) simply never appears as a combined-class key -- it stays purely
    narrow, resolved by identity elsewhere.
    """
    combined: dict[str, list[str]] = {}

    for narrow in narrow_classes:
        extended = WARP_MAP_EXTENDED.get(narrow, narrow)
        wide = WARP_MAP_WIDE.get(extended, extended)
        allc = WARP_MAP_ALL.get(wide, wide)

        for level_class in (extended, wide, allc):
            bucket = combined.setdefault(level_class, [])
            if narrow not in bucket:
                bucket.append(narrow)

    return combined


# All narrow-level keys appearing anywhere in WARP_MAP_EXTENDED -- this is
# N_CLASSES (17 classes) *plus* 'SN Ia' and 'TDE', which the official map
# also assigns combined names for even though neither has a per-class fit
# file in this pipeline (doc 2's `nclasses` doesn't include them: normal Ia
# is fit separately via salt2/salt3, TDE isn't fit by this pipeline at all).
#
# *** WARNING: any combined class whose constituents include 'SN Ia' or
# *** 'TDE' -- namely 'SN Ia (e)', 'TDE (e)', 'TDE (w)', 'TDE (a)', and
# *** partially 'SN Ia (w)'/'SN Ia (a)' -- will be missing that component
# *** unless you also produce a fit-json file for it yourself. With the
# *** default --skip-missing, load_observed_sn_data_multi will just warn and
# *** carry on with whatever constituents *do* have files, which for e.g.
# *** 'SN Ia (w)' silently drops the 'SN Ia' part and leaves only the
# *** 'SN Ia-91T' contribution. Worth checking your run logs for these
# *** warnings before trusting a class that touches 'SN Ia' or 'TDE'.
_ALL_MAPPED_NARROW = list(WARP_MAP_EXTENDED.keys())

WIDE_CLASS_MAP = _build_combined_class_map(_ALL_MAPPED_NARROW)

# Combined-class name lists for each category, derived (not hand-picked) from
# the same composition -- so they can't drift out of sync with WIDE_CLASS_MAP.
# NOTE: alphabetical order, which is very likely NOT the same order as the
# previous hand-written E_CLASSES/W_CLASSES/A_CLASSES -- if you have scripts
# or notes that refer to a class by --cid index, double-check the index still
# points at the class you expect (print CLASS_MAP or use get_class_name to
# check) rather than assuming the old numbering still holds.
E_CLASSES = sorted({WARP_MAP_EXTENDED[n] for n in _ALL_MAPPED_NARROW})
W_CLASSES = sorted({WARP_MAP_WIDE.get(WARP_MAP_EXTENDED[n], WARP_MAP_EXTENDED[n])
                     for n in _ALL_MAPPED_NARROW})
A_CLASSES = sorted({WARP_MAP_ALL.get(w, w) for w in W_CLASSES})

CLASS_MAP = {
    'n': N_CLASSES,
    'e': E_CLASSES,
    'w': W_CLASSES,
    'a': A_CLASSES,
}

# Per-narrow-class redshift windows (unchanged from before). Used both for
# single-narrow-class runs and, per-constituent, when merging a combined
# class's sample.
NARROW_Z_LIMITS = {
    'SLSN-II': [0.0, 0.3], 'SLSN-I': [0.0, 0.3],
    'SN Ia-91bg': [0.01, 0.055], 'SN Ia-91T': [0.01, 0.10],
    'SN Ia-CSM': [0.01, 0.10], 'SN IIn': [0.0, 0.10],
    'SN Ia-SC': [0.01, 0.10], 'SN Ia-pec': [0.01, 0.055],
    'SN Iax': [0.0, 0.055],
}
DEFAULT_Z_LIMITS = [0.0, 0.07]


def get_class_name(category: str, cid: int) -> str:
    """Resolve class name from category and index."""
    if category not in CLASS_MAP:
        raise ValueError(f"Category must be one of {list(CLASS_MAP.keys())}, got '{category}'")
    class_list = CLASS_MAP[category]
    if not (0 <= cid < len(class_list)):
        raise ValueError(f"Class index {cid} out of range for category '{category}' (0-{len(class_list)-1})")
    return class_list[cid]


def resolve_constituent_narrow_classes(class_name: str) -> list[str]:
    """Return the narrow-class fit files that make up `class_name`.

    Narrow classes map to themselves (the single-file case). Combined
    (extended/wide/all) classes are looked up in WIDE_CLASS_MAP.
    """
    if class_name in N_CLASSES:
        return [class_name]
    if class_name in WIDE_CLASS_MAP:
        return WIDE_CLASS_MAP[class_name]
    raise ValueError(
        f"Don't know how to build a sample for '{class_name}': it isn't a "
        f"narrow class and isn't in WIDE_CLASS_MAP. Add it there first."
    )


# -----------------------------------------------------------------------------
# Data loading: observed SN properties with redshifts
# -----------------------------------------------------------------------------

def load_observed_sn_data(class_name: str, fit_json: Path,
                          z_limits: tuple | None = None,
                          peak_good_only: bool = True) -> list[dict]:
    """Extract per-SN observed colors and redshifts from fit results.

    Returns list of dicts with keys: z, color_{key}, id, peak_good, etc.
    One entry per SN (not per model fit).
    """
    with open(fit_json) as f:
        results = json.load(f)

    # Collect best fit per SN across models
    sn_data = {}

    for modelname, model_results in results.items():
        for res in model_results:
            if not res.get('success', False):
                continue
            if peak_good_only and not res.get('peak_good', False):
                continue

            z = float(res.get('z', res.get('redshift', np.nan)))
            if not np.isfinite(z):
                continue
            if z_limits is not None and not (z_limits[0] <= z <= z_limits[1]):
                continue

            snid = res['id']

            # Initialize SN entry if new
            if snid not in sn_data:
                sn_data[snid] = {
                    'id': snid,
                    'z': z,
                    'nbr_bands': res.get('nbr_bands', 0),
                    'ndet': res.get('ndet', 0),
                    'peak_good': res.get('peak_good', False),
                    'colors': {},
                    'best_chidof': np.inf,
                    'best_model': None,
                }

            # Track best-fitting model by chi2/dof
            chidof = res.get('chidof', res.get('chisq', np.inf) / max(res.get('ndof', 1), 1))
            if chidof < sn_data[snid]['best_chidof']:
                sn_data[snid]['best_chidof'] = chidof
                sn_data[snid]['best_model'] = modelname

            # Collect all available colors (may appear from multiple models)
            for k, v in res.items():
                if k.startswith('peak_gp_') and np.isfinite(v):
                    color_key = k.replace('peak_gp_', '')
                    # Keep first valid or prefer best model
                    if color_key not in sn_data[snid]['colors'] or chidof < sn_data[snid].get(f'_chidof_{color_key}', np.inf):
                        sn_data[snid]['colors'][color_key] = float(v)
                        sn_data[snid][f'_chidof_{color_key}'] = chidof

    return list(sn_data.values())


def load_observed_sn_data_multi(constituent_classes: list[str], fit_json_pattern: str,
                                 version: str, use_z_limits: bool = True,
                                 peak_good_only: bool = True,
                                 skip_missing: bool = True) -> list[dict]:
    """Load and merge per-SN observed data across several narrow-class fit files.

    This is the multi-file analogue of load_observed_sn_data: instead of
    reading one JSON file, it reads one per constituent narrow class, applies
    that narrow class's *own* redshift window (from NARROW_Z_LIMITS) before
    merging -- since e.g. SLSN goes out to z=0.3 while most other classes cut
    much lower, a single shared z-limit for the merged sample would be wrong
    -- and concatenates the resulting per-SN dicts into one sample.

    Each returned SN dict is unchanged from load_observed_sn_data except for
    an added 'source_class' key recording which narrow-class file it came
    from, so provenance survives the merge (handy for later sanity plots
    split by sub-class).
    """
    combined: list[dict] = []
    seen_ids: set = set()

    for narrow_class in constituent_classes:
        z_limits = NARROW_Z_LIMITS.get(narrow_class, DEFAULT_Z_LIMITS) if use_z_limits else None

        fit_json = Path(fit_json_pattern.format(
            version=version, class_name=narrow_class.replace('/', '')
        ))

        try:
            sn_list = load_observed_sn_data(narrow_class, fit_json, z_limits, peak_good_only)
        except FileNotFoundError as e:
            msg = f"no fit file for narrow class '{narrow_class}' ({fit_json})"
            if skip_missing:
                print(f"WARNING: {msg}; skipping.")
                continue
            raise FileNotFoundError(msg) from e

        n_dupe = 0
        for sn in sn_list:
            if sn['id'] in seen_ids:
                n_dupe += 1
                continue
            sn['source_class'] = narrow_class
            seen_ids.add(sn['id'])
            combined.append(sn)

        if n_dupe:
            print(f"WARNING: {n_dupe} SNe from '{narrow_class}' were already "
                  f"seen under another constituent class -- kept first "
                  f"occurrence only.")

        print(f"  + {narrow_class}: {len(sn_list) - n_dupe} SNe (z limits {z_limits})")

    return combined


# -----------------------------------------------------------------------------
# Redshifted template evaluation
# -----------------------------------------------------------------------------

def evaluate_template_at_redshift(template: dict, band1: str, band2: str,
                                  z: float, rest_phase: float = 0) -> float | None:
    """Evaluate template color at observed frame for given redshift.

    Parameters
    ----------
    template : dict
        Template dict with 'model' key containing sncosmo Model
    band1, band2 : str
        Observer-frame bandpasses
    z : float
        Redshift
    rest_phase : float
        Phase in rest-frame days relative to peak

    Returns
    -------
    Observed color (band1 - band2) or None if evaluation fails
    """
    mod = template['model']  # We directly use the sncosmo Model object, but should be fine?

    # Ensure float type for sncosmo C extensions
    z = float(z)
    rest_phase = float(rest_phase)

    try:
        inz = float(mod.get('z'))  # Also ensure float on retrieval
    except (TypeError, ValueError):
        inz = 0.0


    try:
        mod.set(z=z)
        # Observer-frame phase = rest-frame phase * (1+z)
        obs_phase = rest_phase * (1 + z)

        mag1 = mod.bandmag(band1, "ab", obs_phase)
        mag2 = mod.bandmag(band2, "ab", obs_phase)
        color = mag1 - mag2
        mod.set(z=inz)

        if not np.isfinite(color) or color < -20:  # Unphysical cutoff
            return None
        return float(color)

    except (ValueError, RuntimeError):
        # Filter coverage issues, extrapolation, etc.
        mod.set(z=inz)
        return None


def generate_redshifted_colors(templates: list[dict], band1: str, band2: str,
                               z_values: np.ndarray, rest_phase: float = 0,
                               n_draw_per_z: int | None = None,
                               random_seed: int | None = None) -> np.ndarray:
    """Generate colors by evaluating templates at matching redshifts.

    For each z in z_values, draws templates (with or without replacement)
    and evaluates at that redshift. Returns array of colors aligned with z_values.
    """
    if len(templates) == 0 or len(z_values) == 0:
        return np.array([])

    rng = np.random.default_rng(random_seed)

    # Determine draws: if more templates than redshifts, sample without replacement
    # per z; otherwise allow repeats or use all templates
    n_templates = len(templates)
    if n_draw_per_z is None:
        # Default: use all templates at each redshift (N_templates * N_z samples)
        n_draw_per_z = n_templates



    colors = []

    for z in z_values:
        z = float(z)
        # Draw template indices for this redshift
        if n_draw_per_z >= n_templates:
            indices = np.arange(n_templates)
        else:
            indices = rng.choice(n_templates, size=n_draw_per_z, replace=False)

        for idx in indices:
            col = evaluate_template_at_redshift(templates[idx], band1, band2, z, rest_phase)
            if col is not None:
                colors.append({
                    'z': z,
                    'color': col,
                    'template_idx': int(idx),
                })

    # Return structured array for z-alignment in analysis
    if len(colors) == 0:
        return np.array([])

    return np.array([(c['z'], c['color'], c['template_idx']) for c in colors],
                    dtype=[('z', float), ('color', float), ('template_idx', int)])


# -----------------------------------------------------------------------------
# Statistical comparison with redshift structure
# -----------------------------------------------------------------------------

def compute_redshift_binned_stats(obs_data: list[dict], model_colors: np.ndarray,
                                  color_key: str, n_z_bins: int = 4) -> dict:
    """Compare distributions within redshift bins.

    Accounts for evolution: templates and data must match per-bin.
    """
    if len(model_colors) == 0 or len(obs_data) == 0:
        return {}

    # Extract observed z and color
    obs_z = np.array([s['z'] for s in obs_data])
    obs_color = np.array([s['colors'].get(color_key, np.nan) for s in obs_data])
    valid = np.isfinite(obs_color)
    obs_z, obs_color = obs_z[valid], obs_color[valid]

    # Redshift bins
    z_edges = np.quantile(obs_z, np.linspace(0, 1, n_z_bins + 1))
    z_edges[0] -= 0.001  # Ensure coverage
    z_edges[-1] += 0.001

    bin_stats = []

    for i in range(n_z_bins):
        z_lo, z_hi = z_edges[i], z_edges[i+1]

        obs_mask = (obs_z >= z_lo) & (obs_z < z_hi)
        mod_mask = (model_colors['z'] >= z_lo) & (model_colors['z'] < z_hi)

        obs_bin = obs_color[obs_mask]
        mod_bin = model_colors['color'][mod_mask]

        if len(obs_bin) < 3 or len(mod_bin) < 3:
            continue

        ks_stat, ks_p = stats.ks_2samp(obs_bin, mod_bin)

        bin_stats.append({
            'z_bin': i,
            'z_lo': float(z_lo),
            'z_hi': float(z_hi),
            'n_obs': len(obs_bin),
            'n_mod': len(mod_bin),
            'obs_mean': float(np.mean(obs_bin)),
            'mod_mean': float(np.mean(mod_bin)),
            'obs_std': float(np.std(obs_bin)),
            'mod_std': float(np.std(mod_bin)),
            'ks_stat': float(ks_stat),
            'ks_pvalue': float(ks_p),
            'delta_mean': float(np.mean(mod_bin) - np.mean(obs_bin)),
        })

    return {'z_edges': z_edges.tolist(), 'bins': bin_stats}


def compute_global_weighted_stats(obs_data: list[dict], model_colors: np.ndarray,
                                  color_key: str) -> dict:
    """Global comparison of observed and template color distributions."""
    obs_color = np.array([s['colors'].get(color_key, np.nan) for s in obs_data])
    valid = np.isfinite(obs_color)
    obs_color = obs_color[valid]

    if len(obs_color) == 0 or len(model_colors) == 0:
        return {}

    mod_color = model_colors['color']

    # Direct comparison: templates evaluated at matching redshifts, no reweighting
    ks_stat, ks_p = stats.ks_2samp(obs_color, mod_color)

    return {
        'n_obs': len(obs_color),
        'n_mod': len(mod_color),
        'obs_mean': float(np.mean(obs_color)),
        'mod_mean': float(np.mean(mod_color)),
        'obs_std': float(np.std(obs_color)),
        'mod_std': float(np.std(mod_color)),
        'obs_median': float(np.median(obs_color)),
        'mod_median': float(np.median(mod_color)),
        'ks_stat': float(ks_stat),
        'ks_pvalue': float(ks_p),
        'delta_mean': float(np.mean(mod_color) - np.mean(obs_color)),
        'delta_std': float(np.std(mod_color) - np.std(obs_color)),
    }

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_color_diagnostics(obs_data: list[dict],
                          raw_model_colors: np.ndarray | None,
                          harm_model_colors: np.ndarray | None,
                          draw_model_colors: np.ndarray | None,
                          color_key: str, class_name: str,
                          outdir: Path, stats: dict) -> Path:
    """Generate focused histogram diagnostics with statistical annotations."""
    obs_color = np.array([s['colors'].get(color_key, np.nan) for s in obs_data])
    obs_color = obs_color[np.isfinite(obs_color)]

    mod_collections = {
        'Observed': (None, '#1a1a1a', None),  # Dark gray, no hatch
        'Raw': (raw_model_colors, '#4C72B0', None),
        'Harmonized': (harm_model_colors, '#55A868', '//'),
        'Randomized': (draw_model_colors, '#C44E52', '\\\\'),
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 8),
                            gridspec_kw={'height_ratios': [3, 1], 'width_ratios': [3, 1]})
    ax_main = axes[0, 0]
    ax_resid = axes[1, 0]
    ax_legend = axes[0, 1]
    ax_stats = axes[1, 1]

    # Common binning
    all_colors = [obs_color]
    for _, (mod_cols, _, _) in mod_collections.items():
        if mod_cols is not None and len(mod_cols) > 0:
            all_colors.append(mod_cols['color'])

    c_min = min(c.min() for c in all_colors)
    c_max = max(c.max() for c in all_colors)
    c_pad = 0.05 * (c_max - c_min)
    bins = np.linspace(c_min - c_pad, c_max + c_pad, 50)
    x_grid = np.linspace(bins[0], bins[-1], 500)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width = bins[1] - bins[0]

    # Main: filled histograms with stepped edges
    from scipy.stats import gaussian_kde

    hist_obs, _ = np.histogram(obs_color, bins=bins, density=True)

    # Plot observed as reference: filled with solid edge
    ax_main.fill_between(bin_centers, hist_obs, alpha=0.25, color='#1a1a1a', step='mid')
    ax_main.step(bin_centers, hist_obs, where='mid', color='#1a1a1a', lw=2.5, label='Observed')

    kde_obs = gaussian_kde(obs_color, bw_method='scott')
    ax_main.plot(x_grid, kde_obs(x_grid), 'k-', lw=1.5, alpha=0.7, zorder=5)

    for label, (mod_cols, color, hatch) in mod_collections.items():
        if label == 'Observed' or mod_cols is None or len(mod_cols) == 0:
            continue

        mod_c = mod_cols['color']
        hist_mod, _ = np.histogram(mod_c, bins=bins, density=True)

        # Filled histogram with transparency
        ax_main.fill_between(bin_centers, hist_mod, alpha=0.2, color=color, step='mid')

        # Distinct edge styling
        if hatch:
            # For hatched: show edge as thick line with pattern
            ax_main.bar(bin_centers, hist_mod, width=bin_width * 0.9, bottom=0,
                       color=color, alpha=0.15, edgecolor=color, linewidth=1.5,
                       hatch=hatch, label=label, zorder=3)
        else:
            ax_main.step(bin_centers, hist_mod, where='mid', color=color, lw=2.5,
                        label=label, zorder=4)

        # KDE overlay
        kde_mod = gaussian_kde(mod_c, bw_method='scott')
        ax_main.plot(x_grid, kde_mod(x_grid), color=color, lw=1.5, ls='--', alpha=0.8, zorder=5)

    ax_main.set_xlabel(f'{color_key} (mag)')
    ax_main.set_ylabel('Probability density')
    ax_main.set_title(f'{class_name}: {color_key}')
    ax_main.set_xlim(bins[0], bins[-1])

    # Residuals: KDE differences
    ax_resid.axhline(0, color='black', lw=0.5, alpha=0.5)

    for label, (mod_cols, color, hatch) in mod_collections.items():
        if label == 'Observed' or mod_cols is None or len(mod_cols) == 0:
            continue

        kde_mod = gaussian_kde(mod_cols['color'], bw_method='scott')
        resid = kde_mod(x_grid) - kde_obs(x_grid)
        ax_resid.plot(x_grid, resid, color=color, lw=2, label=label)

    ax_resid.set_xlabel(f'{color_key} (mag)')
    ax_resid.set_ylabel('ΔKDE (model − obs)')
    ax_resid.set_xlim(bins[0], bins[-1])

    # Legend panel with full entries
    ax_legend.axis('off')
    handles, labels = ax_main.get_legend_handles_labels()
    # Deduplicate and order
    seen = {}
    ordered_handles = []
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen[label] = handle
            ordered_handles.append((handle, label))
    # Reorder: Observed first, then models
    order = ['Observed', 'Raw', 'Harmonized', 'Randomized']
    ordered = []
    for o in order:
        for handle, label in ordered_handles:
            if label == o:
                ordered.append((handle, label))
                break
    ax_legend.legend([handle for handle, _ in ordered], [label for _, label in ordered],
                    loc='center', frameon=False, fontsize=10,
                    title='Distribution', title_fontsize=11)

    # Statistics panel
    ax_stats.axis('off')

    stat_lines = [f"{color_key}", f"N(obs) = {len(obs_color):d}", ""]

    for label, (mod_cols, _, _) in mod_collections.items():
        if label == 'Observed':
            data = obs_color
        elif mod_cols is None or len(mod_cols) == 0:
            continue
        else:
            data = mod_cols['color']

        stat_lines.extend([
            f"{label}:",
            f"  μ = {np.mean(data):.3f}",
            f"  σ = {np.std(data):.3f}",
            f"  med = {np.median(data):.3f}",
            f"  IQR = {np.percentile(data, 75) - np.percentile(data, 25):.3f}",
        ])

        if label != 'Observed' and 'global' in stats:
            g = stats['global']
            if 'ks_stat' in g:
                stat_lines.append(f"  KS = {g['ks_stat']:.3f} (p={g['ks_pvalue']:.3g})")
        stat_lines.append("")

    if 'z_binned' in stats and len(stats['z_binned'].get('bins', [])) > 0:
        bins_data = stats['z_binned']['bins']
        stat_lines.append("Δμ by z-bin:")
        for b in bins_data:
            stat_lines.append(f"  [{b['z_lo']:.2f},{b['z_hi']:.2f}]: {b['delta_mean']:+.3f}")

    ax_stats.text(0.05, 0.95, '\n'.join(stat_lines), transform=ax_stats.transAxes,
                 fontsize=7.5, verticalalignment='top', fontfamily='monospace',
                 linespacing=1.3,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='whitesmoke', alpha=0.8))

    plt.tight_layout()

    safe_name = class_name.replace('/', '')
    safe_color = color_key.replace('/', '-')
    outpath = outdir / f"diag_hist_{safe_name}_{safe_color}.pdf"
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return outpath


# -----------------------------------------------------------------------------
# NEW: publication-quality version of the diagnostic histogram's main panel
# -----------------------------------------------------------------------------

def _publication_rcparams() -> dict:
    """rcParams for a clean, journal-ready single-panel figure.

    Used via `with plt.rc_context(...)` so it never leaks into the other
    (diagnostic/exploratory) plots produced elsewhere in this script.
    """
    return {
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'STIXGeneral'],
        'mathtext.fontset': 'stix',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9.5,
        'axes.linewidth': 0.9,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'legend.frameon': False,
        # Store real, editable text in the PDF rather than paths/outlines --
        # standard practice for journal figures (and much smaller files).
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    }


def plot_publication_color_comparison(obs_data: list[dict],
                                      raw_model_colors: np.ndarray | None,
                                      harm_model_colors: np.ndarray | None,
                                      draw_model_colors: np.ndarray | None,
                                      color_key: str, class_name: str,
                                      outdir: Path,
                                      include_modes: tuple[str, ...] = ('Raw', 'Harmonized', 'Randomized'),
                                      figsize: tuple[float, float] = (5.5, 4.2),
                                      range_percentiles: tuple[float, float] = (1, 99)) -> Path:
    """Clean, single-panel, publication-quality color-distribution plot.

    This is a standalone version of just the main histogram panel from
    plot_color_diagnostics -- observed vs. template color distributions for
    one color, with KDE overlays -- restyled for direct use in a paper:
    serif font, inward ticks on all four sides, no crowded stats dump or
    annotation. Vector (PDF, editable text) plus a PNG for quick previewing.

    Parameters
    ----------
    include_modes : which template variants to draw, in {'Raw', 'Harmonized',
        'Randomized'}. Default is all three; drop to a subset for a cleaner
        two-distribution comparison if that's what a given figure calls for.
    range_percentiles : (low, high) percentile applied *separately* to each
        included distribution (observed, and each of the chosen model
        modes), then unioned -- so a handful of outliers/long KDE tails
        don't stretch the plot away from where most of the data actually
        is, and no single distribution (e.g. a large model draw) dominates
        the range just because it has more points than the others. Padded
        by a further 6% of the resulting span for breathing room. Widen
        (e.g. (0, 100)) if you want the full range including tails.
    """
    obs_color = np.array([s['colors'].get(color_key, np.nan) for s in obs_data])
    obs_color = obs_color[np.isfinite(obs_color)]

    all_mod_collections = {
        'Raw': (raw_model_colors, '#4C72B0'),
        'Harmonized': (harm_model_colors, '#55A868'),
        'Randomized': (draw_model_colors, '#C44E52'),
    }
    mod_collections = {k: all_mod_collections[k] for k in include_modes if k in all_mod_collections}

    distributions = [obs_color]
    for mod_cols, _ in mod_collections.values():
        if mod_cols is not None and len(mod_cols) > 0:
            distributions.append(mod_cols['color'])

    bounds = [np.percentile(d, range_percentiles) for d in distributions]
    c_lo = min(b[0] for b in bounds)
    c_hi = max(b[1] for b in bounds)
    c_pad = 0.06 * (c_hi - c_lo)
    bins = np.linspace(c_lo - c_pad, c_hi + c_pad, 40)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    x_grid = np.linspace(bins[0], bins[-1], 400)

    from scipy.stats import gaussian_kde

    with plt.rc_context(_publication_rcparams()):
        fig, ax = plt.subplots(figsize=figsize)

        # Observed: reference distribution, drawn darkest and on top
        hist_obs, _ = np.histogram(obs_color, bins=bins, density=True)
        ax.fill_between(bin_centers, hist_obs, step='mid', color='0.15', alpha=0.12, zorder=2)
        ax.step(bin_centers, hist_obs, where='mid', color='0.15', lw=1.8,
                label=f'Observed (N={len(obs_color)})', zorder=6)
        kde_obs = gaussian_kde(obs_color, bw_method='scott')
        ax.plot(x_grid, kde_obs(x_grid), color='0.15', lw=1.3, alpha=0.9, zorder=6)

        for label, (mod_cols, color) in mod_collections.items():
            if mod_cols is None or len(mod_cols) == 0:
                continue
            mod_c = mod_cols['color']
            hist_mod, _ = np.histogram(mod_c, bins=bins, density=True)
            ax.fill_between(bin_centers, hist_mod, step='mid', color=color, alpha=0.10, zorder=1)
            ax.step(bin_centers, hist_mod, where='mid', color=color, lw=1.8,
                    label=f'{label} (N={len(mod_c)})', zorder=5)
            kde_mod = gaussian_kde(mod_c, bw_method='scott')
            ax.plot(x_grid, kde_mod(x_grid), color=color, lw=1.3, ls='--', alpha=0.9, zorder=5)

        ax.set_xlabel(f'{color_key} (mag)')
        ax.set_ylabel('Probability density')
        ax.set_xlim(bins[0], bins[-1])
        ax.set_ylim(bottom=0)
        ax.set_title(class_name, loc='left', style='italic')

        ax.legend(loc='upper right', handlelength=1.6, borderaxespad=0.4)

        fig.tight_layout()

        safe_name = class_name.replace('/', '')
        safe_color = color_key.replace('/', '-')
        outpath_pdf = outdir / f"pub_hist_{safe_name}_{safe_color}.pdf"
        outpath_png = outdir / f"pub_hist_{safe_name}_{safe_color}.png"
        fig.savefig(outpath_pdf, bbox_inches='tight')
        fig.savefig(outpath_png, bbox_inches='tight', dpi=300)
        plt.close(fig)

    return outpath_pdf


def plot_redshift_color_comparison(obs_data: list[dict],
                                   raw_model_colors: np.ndarray | None,
                                   harm_model_colors: np.ndarray | None,
                                   draw_model_colors: np.ndarray | None,
                                   color_key: str, class_name: str,
                                   outdir: Path, stats: dict) -> Path:
    """Generate redshift-aware comparison plot."""

    obs_z = np.array([s['z'] for s in obs_data])
    obs_color = np.array([s['colors'].get(color_key, np.nan) for s in obs_data])
    valid = np.isfinite(obs_color)
    obs_z, obs_color = obs_z[valid], obs_color[valid]

    # Determine plot limits
    all_colors = [obs_color]
    all_z = [obs_z]
    labels = []
    colors = []

    mod_collections = {
        'raw': (raw_model_colors, '#4C72B0', 'Raw'),
        'harmonize': (harm_model_colors, '#55A868', 'Harmonized'),
        'draw': (draw_model_colors, '#C44E52', 'Randomized'),
    }

    for mode, (mod_cols, color, label) in mod_collections.items():
        if mod_cols is not None and len(mod_cols) > 0:
            all_z.append(mod_cols['z'])
            all_colors.append(mod_cols['color'])
            labels.append(label)
            colors.append(color)

    z_min, z_max = min(z.min() for z in all_z), max(z.max() for z in all_z)
    c_min, c_max = min(c.min() for c in all_colors), max(c.max() for c in all_colors)
    # Extend color range by 10%
    c_pad = 0.1 * (c_max - c_min)
    c_min -= c_pad
    c_max += c_pad

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 2, 1], width_ratios=[2, 2, 1])

    # Main: color vs redshift scatter with density
    ax_scatter = fig.add_subplot(gs[0:2, 0:2])

    # Observed data with error bars (bootstrap or measurement error if available)
    ax_scatter.scatter(obs_z, obs_color, c='black', s=30, alpha=0.6, label='Observed', zorder=5)

    # Running statistics for observed
    z_sort = np.argsort(obs_z)
    z_binned = np.array_split(obs_z[z_sort], max(len(obs_z)//20, 3))
    c_binned = np.array_split(obs_color[z_sort], max(len(obs_z)//20, 3))
    z_centers = [np.mean(zb) for zb in z_binned]
    c_means = [np.mean(cb) for cb in c_binned]
    c_stds = [np.std(cb) for cb in c_binned]
    ax_scatter.errorbar(z_centers, c_means, yerr=c_stds, fmt='o', color='black',
                       markersize=8, capsize=3, label='Observed binned', zorder=6)

    # Template predictions
    for mode, (mod_cols, color, label) in mod_collections.items():
        if mod_cols is None or len(mod_cols) == 0:
            continue

        # Subsample for visibility if too many points
        plot_idx = np.random.choice(len(mod_cols), size=min(len(mod_cols), 2000), replace=False)
        ax_scatter.scatter(mod_cols['z'][plot_idx], mod_cols['color'][plot_idx],
                          c=color, s=5, alpha=0.3, label=label)

        # Binned template statistics
        mod_sort = np.argsort(mod_cols['z'])
        mod_zb = np.array_split(mod_cols['z'][mod_sort], max(len(mod_cols)//500, 3))
        mod_cb = np.array_split(mod_cols['color'][mod_sort], max(len(mod_cols)//500, 3))
        mod_zc = [np.mean(zb) for zb in mod_zb]
        mod_cm = [np.mean(cb) for cb in mod_cb]
        mod_cs = [np.std(cb) for cb in mod_cb]
        ax_scatter.errorbar(mod_zc, mod_cm, yerr=mod_cs, fmt='s', color=color,
                           markersize=6, capsize=3, alpha=0.8)

    ax_scatter.set_xlabel('Redshift z')
    ax_scatter.set_ylabel(f'{color_key} (mag)')
    ax_scatter.set_xlim(z_min, z_max)
    ax_scatter.set_ylim(c_min, c_max)
    ax_scatter.legend(loc='best', framealpha=0.9)
    ax_scatter.set_title(f'{class_name}: observer-frame colors')

    # Marginal: color distribution (all redshifts)
    ax_color = fig.add_subplot(gs[0, 2])

    bins_c = np.linspace(c_min, c_max, 40)
    ax_color.hist(obs_color, bins=bins_c, orientation='horizontal', density=True,
                 color='black', alpha=0.5, label='Obs')

    for mode, (mod_cols, color, label) in mod_collections.items():
        if mod_cols is not None and len(mod_cols) > 0:
            ax_color.hist(mod_cols['color'], bins=bins_c, orientation='horizontal',
                         density=True, color=color, alpha=0.4,
                         histtype='step', lw=2, label=label)

    ax_color.set_ylim(c_min, c_max)
    ax_color.set_xticklabels([])
    ax_color.set_yticklabels([])
    ax_color.legend(loc='best', fontsize=8)

    # Marginal: redshift distribution
    ax_z = fig.add_subplot(gs[2, 0:2])

    bins_z = np.linspace(z_min, z_max, 30)
    ax_z.hist(obs_z, bins=bins_z, density=True, color='black', alpha=0.5, label='Observed')

    for mode, (mod_cols, color, label) in mod_collections.items():
        if mod_cols is not None and len(mod_cols) > 0:
            ax_z.hist(mod_cols['z'], bins=bins_z, density=True, color=color,
                     alpha=0.3, histtype='step', lw=2, label=label)

    ax_z.set_xlabel('Redshift z')
    ax_z.set_ylabel('Density')
    ax_z.set_xlim(z_min, z_max)
    ax_z.legend(loc='best', fontsize=8)

    # Statistics panel
    ax_stats = fig.add_subplot(gs[2, 2])
    ax_stats.axis('off')

    stat_lines = [
        f"Color: {color_key}",
        f"N(obs): {len(obs_color)}",
        "",
        "Global (z-weighted):",
    ]
    if 'global' in stats:
        g = stats['global']
        stat_lines.extend([
            f"  Δmean = {g.get('delta_mean', np.nan):.3f}",
            f"  Δstd = {g.get('delta_std', np.nan):.3f}",
            f"  KS = {g.get('ks_stat', np.nan):.3f} (p={g.get('ks_pvalue', np.nan):.3f})",
        ])

    if 'z_binned' in stats:
        stat_lines.append("")
        stat_lines.append("Redshift binned:")
        for b in stats['z_binned'].get('bins', []):
            stat_lines.append(f"  z=[{b['z_lo']:.3f},{b['z_hi']:.3f}]: "
                            f"Δμ={b['delta_mean']:.3f}, KS={b['ks_stat']:.3f}")

    ax_stats.text(0.05, 0.95, '\n'.join(stat_lines), transform=ax_stats.transAxes,
                 fontsize=8, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    safe_name = class_name.replace('/', '')
    safe_color = color_key.replace('/', '-')
    outpath = outdir / f"redshift_comp_{safe_name}_{safe_color}.pdf"
    plt.savefig(outpath, dpi=300)
    plt.close(fig)

    return outpath


# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------

def analyze_class_redshifted(class_name: str, args: argparse.Namespace) -> dict | None:
    """Run redshift-aware comparison for a single class."""
    print(f"\n{'='*60}")
    print(f"Redshifted comparison: {class_name}")
    print(f"{'='*60}")

    # Load observed SN data with redshifts. Narrow classes read a single fit
    # file directly; extended/wide/all classes merge several narrow-class
    # fit files together (see WIDE_CLASS_MAP at the top of this file).
    constituent_classes = resolve_constituent_narrow_classes(class_name)
    print(f"Constituent narrow classes: {constituent_classes}")

    sn_data = load_observed_sn_data_multi(
        constituent_classes,
        args.fit_json_pattern,
        args.version,
        use_z_limits=args.use_z_limits,
        peak_good_only=args.peak_good_only,
        skip_missing=args.skip_missing,
    )

    if len(sn_data) == 0:
        print(f"WARNING: No SN data found for {class_name}")
        if not args.skip_missing:
            raise ValueError(f"No data for {class_name}")
        return None

    print(f"Loaded {len(sn_data)} SNe with redshifts z ∈ "
          f"[{min(s['z'] for s in sn_data):.3f}, {max(s['z'] for s in sn_data):.3f}]")

    # Extract unique redshifts for template evaluation
#    unique_z = np.array(sorted(set(s['z'] for s in sn_data)))
    unique_z = np.array(sorted({float(s['z']) for s in sn_data}), dtype=np.float64)
    print(f"Unique redshifts: {len(unique_z)}")

    # Generate templates in three modes
    warploader = WarpfitTemplateLoader(
        str(args.warpdir),
        version=args.version,
        suffix=args.suffix,
    )

    mode_templates = {}
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
            )
            mode_templates[mode] = templates
            print(f"  {mode:12s}: {len(templates)} templates")
        except ValueError as e:
            if "color_mode requires 'model_colors'" in str(e):
                print(f"  {mode:12s}: FAILED — {e}")
                if mode != 'raw':
                    mode_templates[mode] = []
                    continue
            raise

    # Determine which colors are available
    available_colors = set()
    for s in sn_data:
        available_colors.update(s['colors'].keys())

    print(f"Available colors: {', '.join(sorted(available_colors))}")

    # Process each color
    results = []
    for color_key in sorted(available_colors):
        try:
            band1, band2 = color_key.split('-')
        except ValueError:
            print(f"Skipping unparseable: {color_key}")
            continue

        print(f"\nProcessing {color_key}...")

        # Evaluate templates at observed redshifts
        # For each mode, generate redshift-matched colors
        mode_colors = {}

        for mode, templates in mode_templates.items():
            if len(templates) == 0:
                mode_colors[mode] = None
                continue

            # Number of draws: enough for smooth statistics
            #n_draw = args.n_draw_per_z if args.n_draw_per_z else max(min(len(templates), 50), 10)
            n_draw = None   # As long as possible, use all templates for each redshift

            cols = generate_redshifted_colors(
                templates, band1, band2, unique_z,
                rest_phase=args.phase,
                n_draw_per_z=n_draw,
                random_seed=args.random_seed + hash(mode) % 10000,
            )
            mode_colors[mode] = cols
            print(f"  {mode}: {len(cols)} evaluations")

        # Statistics
        stats = {}

        # Global z-weighted comparison
        for mode in ['raw', 'harmonize', 'draw']:
            if mode_colors.get(mode) is not None and len(mode_colors[mode]) > 0:
                stats[f'{mode}_global'] = compute_global_weighted_stats(
                    sn_data, mode_colors[mode], color_key
                )

        # Redshift-binned comparison (use raw as primary)
        if mode_colors.get('raw') is not None:
            stats['z_binned'] = compute_redshift_binned_stats(
                sn_data, mode_colors['raw'], color_key, n_z_bins=args.n_z_bins
            )

        # Plot
        plot_path = plot_redshift_color_comparison(
            sn_data,
            mode_colors.get('raw'),
            mode_colors.get('harmonize'),
            mode_colors.get('draw'),
            color_key, class_name, args.outdir,
            {'global': stats.get('raw_global', {}), 'z_binned': stats.get('z_binned', {})}
        )
        print(f"  Plot: {plot_path}")

        # Diagnostics histogram (exploratory: full stats panel + legend)
        diag_path = plot_color_diagnostics(
            sn_data,
            mode_colors.get('raw'),
            mode_colors.get('harmonize'),
            mode_colors.get('draw'),
            color_key, class_name, args.outdir,
            {'global': stats.get('raw_global', {}), 'z_binned': stats.get('z_binned', {})}
        )
        print(f"  Diagnostics: {diag_path}")

        # Publication-quality version of the diagnostics main panel: clean,
        # single-panel, journal-ready styling with all three modes (Raw,
        # Harmonized, Randomized) by default; no KS annotation -- see
        # include_modes on the function if you want a subset instead.
        pub_path = plot_publication_color_comparison(
            sn_data,
            mode_colors.get('raw'),
            mode_colors.get('harmonize'),
            mode_colors.get('draw'),
            color_key, class_name, args.outdir,
        )
        print(f"  Publication plot: {pub_path}")

        # NOTE: this used to be two separate results.append() calls (one
        # before diag_path/pub_path existed, one after) which silently
        # duplicated every row of the output stats CSV -- fixed here to a
        # single append with all three plot paths.
        results.append({
            'color': color_key,
            'n_sn': len(sn_data),
            'stats': stats,
            'plot_path': str(plot_path),
            'diag_path': str(diag_path),
            'pub_plot_path': str(pub_path),
        })


    return {
        'class_name': class_name,
        'n_sn': len(sn_data),
        'z_range': [min(s['z'] for s in sn_data), max(s['z'] for s in sn_data)],
        'color_results': results,
    }


def run_redshifted_comparison(args: argparse.Namespace) -> list[dict]:
    """Execute redshift-aware comparison."""
    register_all()

    class_name = get_class_name(args.category, args.cid)
    print(f"Processing class: {class_name}")

    results = []
    try:
        res = analyze_class_redshifted(class_name, args)
        if res is not None:
            results.append(res)
    except Exception as e:
        print(f"ERROR: {e}")
        if not args.continue_on_error:
            raise

    # Save statistics
    if results:
        rows = []
        for r in results:
            for cr in r['color_results']:
                row = {
                    'class_name': r['class_name'],
                    'color': cr['color'],
                    'n_sn': cr['n_sn'],
                }
                for mode in ['raw', 'harmonize', 'draw']:
                    if f'{mode}_global' in cr['stats']:
                        g = cr['stats'][f'{mode}_global']
                        for k, v in g.items():
                            row[f'{mode}_{k}'] = v
                rows.append(row)

        stats_df = pd.DataFrame(rows)
        stats_path = args.outdir / "redshift_comparison_stats.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"\nStatistics: {stats_path}")

    return results


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare redshifted warp templates with observed SN colors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g_class = parser.add_argument_group("class selection")
    g_class.add_argument("-c", "--category", choices=["n", "e", "w", "a"], default="n")
    g_class.add_argument("--cid", type=int, default=11)

    # Paths
    parser.add_argument("--warpdir", type=Path,
                        default=Path("/Users/jnordin/data/models/sncosmo/warpmod/v4"))
    parser.add_argument("--outdir", type=Path, default=Path("."))
    parser.add_argument("--fit-json-pattern", type=str,
                        default="/Users/jnordin/data/models/sncosmo/btsfitsv{version}_{class_name}.json")

    # Template sampling
    parser.add_argument("--template-selection", default='all')
    parser.add_argument("--snbasis-selection", default="all")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--version", default="4")
    parser.add_argument("--suffix", default="_col")

    # Redshifted evaluation
    parser.add_argument("--phase", type=float, default=0,
                        help="Rest-frame phase relative to peak (days)")
    parser.add_argument("--n-draw-per-z", type=int, default=None,
                        help="Template draws per redshift (default: min(N_templates, 50))")
    parser.add_argument("--n-z-bins", type=int, default=4,
                        help="Number of redshift bins for binned statistics")

    # Data selection
    parser.add_argument("--use-z-limits", action="store_true", default=True)
    parser.add_argument("--no-z-limits", dest="use_z_limits", action="store_false")
    parser.add_argument("--peak-good-only", action="store_true", default=True)
    parser.add_argument("--include-bad-peak", dest="peak_good_only", action="store_false")

    # Error handling
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    try:
        results = run_redshifted_comparison(args)
        print(f"\nCompleted: {len(results)} classes")
        return 0
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
