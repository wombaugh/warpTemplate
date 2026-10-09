#!/usr/bin/env python
# coding: utf-8
"""
Simulate a spectroscopically-selected ZTF sample and measure its peak g-r colours.

Pipeline
--------
1. Load warped SN templates (warptemplate) and build a transient population
   (rate, absolute-magnitude distribution) for one template class.
2. Draw targets over a date range / redshift limit.
3. Load the ZTF observing logs (skysurvey) and rescale the sky noise by
   per-band correction factors (Rigault+2025 / Amenouche+2025).
4. Simulate light curves and apply the DR2 "good sampling" cuts:
   full, pre-peak and post-peak windows, each with a minimum number of
   filters and detections.
5. Apply a spectroscopic-completeness selection as a function of observed
   peak magnitude (logistic survival function).
6. For the selected targets, fit a GP to each light curve, measure the
   peak g-r colour, and save the colours, a histogram and a summary.

Example
-------
    python simulate_spec_peak_colors.py \
        --warpdir /path/to/warpmod/v6 --category a --cid 1 \
        --rate 7.0e4 --magabs-mean -16.75 --magabs-sigma 1.0 \
        --tstart 2018-03-01 --tstop 2020-12-21 --zmax 0.08 \
        --outdir ./out

Class selection
---------------
Prefer ``--category``/``--cid`` over typing ``--fitclass`` directly: it
resolves the class from CLASS_MAP, the same hand-written N/E/W/A_CLASSES
lists used by the color-analysis script that actually saves the warp
coefficient pickles -- so it can't select a combined class (e/w/a) that was
never analysed and has no saved template file. A raw ``--fitclass`` is still
accepted and used as-is, but only warns (doesn't fail) if it isn't in
CLASS_MAP, since the loader may support classes this script doesn't know
about.

Outputs (in --outdir, prefixed with --tag)
------------------------------------------
    <tag>_peak_colors.csv   index, magobs, pobs_spec, g-r, g-r error
    <tag>_peak_colors.png   histogram of g-r
    <tag>_summary.json      selection fractions and colour statistics
    <plotdir>/lc_<index>.png  (optional, --plot-lcs) light curve per target with
                            the GP-estimated peak time in each band, and the
                            phase of that peak relative to the true t0
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import expit

import skysurvey
from warptemplate import (
    WarpTemplatePopulation,
    estimate_peak_flux_multiband,
    get_peak_colors,
    register_all,
)
from warptemplate.loaders import WarpfitTemplateLoader

from db_efficiency import (add_db_efficiency_args, select_database_efficiency,
                           db_efficiency_info)


# Default sky-noise scaling factors (Rigault+2025 LC / Amenouche+2025)
DEFAULT_SKYNOISE_COEFS = {"ztfg": 1.23, "ztfr": 1.17, "ztfi": 1.2}

# -----------------------------------------------------------------------------
# Class selection
# -----------------------------------------------------------------------------
#
# Copied verbatim from the color-analysis script -- the one that runs
# WarpfitTemplateLoader.save_class() and actually writes the warp
# coefficient pickles. E/W/A_CLASSES are short, hand-written lists of only
# the combined classes someone has actually analysed and saved, NOT every
# combination the narrow -> extended -> wide -> all taxonomy could produce.
# A --fitclass typed by hand can silently name a combined class that was
# never saved (no pickle to load); --category/--cid below can't, since it
# only ever picks from these exact lists.

N_CLASSES = [
    'SN IIP', 'SN Ia-91T', 'SN IIn', 'SN Ib/c', 'SN Ibn', 'SN Ia-pec', 'SLSN-I',
    'SN Ic', 'SN Ic-BL', 'SN II', 'SLSN-II', 'SN Iax', 'SN Ia-91bg', 'SN Ia-CSM',
    'SN Ia-SC', 'SN Ib', 'SN IIb', 'TDE',
]
E_CLASSES = ['SN Ib/c (e)', 'SLSN (e)']
W_CLASSES = ['SLSN (w)', 'SN II (w)', 'SN Ib/c (w)', 'SN Ia (w)', 'SN Ia-pec (w)']
A_CLASSES = ['SN Ia (a)', 'SN CC (a)']

CLASS_MAP = {'n': N_CLASSES, 'e': E_CLASSES, 'w': W_CLASSES, 'a': A_CLASSES}
_VALID_CLASSES = set(N_CLASSES) | set(E_CLASSES) | set(W_CLASSES) | set(A_CLASSES)


def get_class_name(category, cid):
    """Resolve a class name from --category/--cid, exactly as the analysis script does."""
    if category not in CLASS_MAP:
        raise ValueError(f"Category must be one of {list(CLASS_MAP.keys())}, got '{category}'")
    class_list = CLASS_MAP[category]
    if not (0 <= cid < len(class_list)):
        raise ValueError(f"Class index {cid} out of range for category '{category}' "
                         f"(0-{len(class_list)-1}); classes are {class_list}")
    return class_list[cid]


# -----------------------------------------------------------------------------
# Command line
# -----------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = p.add_argument_group("selection mode")
    g.add_argument("--selection-mode", choices=["dr2", "bts"], default="bts",
                   help="'dr2': good-sampling + spec completeness on magobs (SN Ia DR2). "
                        "'bts': discovery + observed peak mag in g/r (Bright Transient Survey).")

    g = p.add_argument_group("BTS selection (--selection-mode bts)")
    g.add_argument("--bts-bands", nargs="+", default=["ztfg", "ztfr"],
                   help="Bands in which the peak magnitude is evaluated")
    g.add_argument("--bts-mag-limit", type=float, default=18.5,
                   help="Peak magnitude limit (threshold mode)")
    g.add_argument("--bts-min-snr", type=float, default=5.0,
                   help="Minimum S/N of a detection used for the observed peak")
    g.add_argument("--bts-det-range", nargs=2, type=float, default=[-30, 50],
                   help="Phase window (days) in which discovery detections must fall")
    g.add_argument("--bts-min-det", type=int, default=2,
                   help="Minimum detections (any band) in that window to be discovered")
    g.add_argument("--bts-min-abs-b", type=float, default=7.0,
                   help="Minimum |Galactic latitude| in deg (needs ra/dec; <=0 disables)")
    g.add_argument("--bts-mode", choices=["threshold", "random"], default="random",
                   help="'threshold': peak mag < --bts-mag-limit; 'random': Bernoulli with "
                        "plateau * logistic survival function of the peak mag")
    g.add_argument("--bts-loc", type=float, default=18.9,
                   help="[random] mag at which completeness = plateau/2 (fitted)")
    g.add_argument("--bts-scale", type=float, default=3.2,
                   help="[random] steepness of the completeness drop (fitted)")
    g.add_argument("--bts-plateau", type=float, default=0.999,
                   help="[random] completeness at bright magnitudes (fitted)")
    
    g = p.add_argument_group("template loading")
    g.add_argument("--warpdir", default="/Users/jnordin/data/models/sncosmo/warpmod/v8",
                   help="Directory with the warp-fit templates")
    g.add_argument("--version", type=int, default=8, help="Warp template version")
    g.add_argument("--suffix", default="", help="Warp template file suffix")

    g = p.add_argument_group("template selection")
    g.add_argument("--template-selection", default="all")
    g.add_argument("--snbasis-selection", default="all")
    g.add_argument("--color-mode", default="None",
                   help="draw, harmonize, offsetfit_draw or None")
    g.add_argument("--min-fit-quality", default="None",
                   help="gold, silver, bronze or None")
    g.add_argument("--phase-buffer", default="10",
                   help="Integer phase buffer, or None")
    g.add_argument("--exclude-input", nargs="*", default=[],
                   help="Template names to exclude")

    g = p.add_argument_group("population")
    g.add_argument("-c", "--category", choices=["n", "e", "w", "a"], default=None,
                   help="Select --fitclass via CLASS_MAP (n=narrow, e=extended, "
                        "w=wide, a=all) instead of typing it; takes precedence "
                        "over --fitclass if both are given.")
    g.add_argument("--cid", type=int, default=None,
                   help="Class index within --category (see CLASS_MAP)")
    g.add_argument("--fitclass", default="SN CC (a)", help="Template class to simulate")
    g.add_argument("--rate", type=float, default=7.0e4,
                   help="Volumetric rate (Gpc^-3 yr^-1)")
    g.add_argument("--magabs-mean", type=float, default=-16.75)
    g.add_argument("--magabs-sigma", type=float, default=1.0)
    g.add_argument("--tstart", default="2019-01-01",
                   help="Start of draw window (date string, or MJD if --size is set)")
    g.add_argument("--tstop", default="2022-01-01", help="End of draw window")
    g.add_argument("--zmax", type=float, default=0.08)
    g.add_argument("--size", type=int, default=None,
                   help="Draw a fixed number of targets instead of rate-based draw")
    g.add_argument("--seed", type=int, default=42, help="Random seed for template selection")

    g = p.add_argument_group("survey")
    g.add_argument("--skynoise-coefs", nargs=3, type=float, metavar=("G", "R", "I"),
                   default=[DEFAULT_SKYNOISE_COEFS[b] for b in ("ztfg", "ztfr", "ztfi")],
                   help="Sky-noise scaling for ztfg ztfr ztfi")

    g = p.add_argument_group("light-curve sampling cuts (DR2 'good sampling')")
    g.add_argument("--full-range", nargs=2, type=float, default=[-10, 40])
    g.add_argument("--pre-range", nargs=2, type=float, default=[-10, 0])
    g.add_argument("--post-range", nargs=2, type=float, default=[0, 40])
    g.add_argument("--min-filters", type=int, default=2,
                   help="Minimum filters in every window")
    g.add_argument("--min-det-full", type=int, default=5)
    g.add_argument("--min-det-pre", type=int, default=2)
    g.add_argument("--min-det-post", type=int, default=2)

    g = p.add_argument_group("spectroscopic selection")
    g.add_argument("--spec-loc", type=float, default=18.55,
                   help="Magnitude at which spectroscopic completeness is 50%%")
    g.add_argument("--spec-scale", type=float, default=3.8,
                   help="Steepness of the completeness drop")
    g.add_argument("--spec-threshold", type=float, default=0.5,
                   help="Keep targets with completeness above this (threshold mode)")
    g.add_argument("--spec-mode", choices=["threshold", "random"], default="random",
                   help="'threshold': deterministic cut; 'random': Bernoulli draw "
                        "with probability = completeness")

    g = p.add_argument_group("peak colour measurement")
    g.add_argument("--gp-length-scale", type=float, default=10.0)
    g.add_argument("--peakflux-iter", type=int, default=0,
                   help="Number of sigma-clipping iterations in the GP fit")
    g.add_argument("--n-sigma", type=float, default=3.0)
    g.add_argument("--min-eff-points", type=int, default=1)
    g.add_argument("--color-floor", type=float, default=-99.0,
                   help="Discard colours <= this value (NaN are always discarded)")
    g.add_argument("--hist-bins", type=int, default=30)

    g = p.add_argument_group("light-curve plots")
    g.add_argument("--plot-lcs", choices=["none", "all", "random"], default="none",
                   help="Plot light curves of the spectroscopically selected targets: "
                        "'all', a 'random' subset (see --n-plot), or 'none'")
    g.add_argument("--n-plot", type=int, default=20,
                   help="Number of light curves to plot in 'random' mode")
    g.add_argument("--plotdir", default=None,
                   help="Directory for light-curve plots (default: <outdir>/lcplots)")

    g = p.add_argument_group("output")
    g.add_argument("--outdir", default="./out")
    g.add_argument("--tag", default="specsample", help="Prefix for output files")
    g.add_argument("--no-progress", action="store_true",
                   help="Disable the light-curve simulation progress bar")

    # Add arguments regarding loss due to db inclusion 
    add_db_efficiency_args(p)
    args = p.parse_args(argv)

    if args.category is not None or args.cid is not None:
        if args.category is None or args.cid is None:
            p.error("--category and --cid must be given together.")
        args.fitclass = get_class_name(args.category, args.cid)
    elif args.fitclass not in _VALID_CLASSES:
        print(f"WARNING: --fitclass '{args.fitclass}' is not in N/E/W/A_CLASSES "
              f"(CLASS_MAP); it may not correspond to a saved template pickle. "
              f"Available: n={N_CLASSES}, e={E_CLASSES}, w={W_CLASSES}, a={A_CLASSES}")

    return args


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _none_or(value, cast=str):
    """Translate the string 'None' from the CLI to None, otherwise cast."""
    return None if str(value).lower() == "none" else cast(value)


def _maybe_float(value):
    """Return float(value) if it parses as a number (MJD), else the string."""
    try:
        return float(value)
    except ValueError:
        return value


def get_spectro_completeness(mag, loc=18.8, scale=4.5):
    """Probability of obtaining a spectrum at observed magnitude `mag`.

    Survival sigmoid: 1 - expit((mag - loc) * scale). Equals 0.5 at
    mag == loc and falls faster for larger `scale`.
    """
    mag = np.atleast_1d(mag)
    return 1 - expit((mag - loc) * scale)


def fit_peaks(lc, gp_length_scale=10.0, peakflux_iter=0, n_sigma=3.0, min_eff_points=1):
    """Fit a GP per band and derive peak colours.

    Returns (results_gp, peakcol): the per-band peak estimates (with
    `.peak_time`) and the dict of peak colours.
    """
    banddict = {
        band: {
            "time": lc[lc["band"] == band]["mjd"],
            "flux": lc[lc["band"] == band]["flux"],
            "flux_err": lc[lc["band"] == band]["fluxerr"],
        }
        for band in set(lc["band"])
    }
    results_gp = estimate_peak_flux_multiband(
        banddict, method="gp", length_scale=gp_length_scale,
        n_sigma=n_sigma, n_clip_iter=peakflux_iter,
    )
    peakcol = get_peak_colors(results_gp, prefix="gp_", min_eff_points=min_eff_points)
    return results_gp, peakcol


def get_peakgr(lc, **kwargs):
    """Peak g-r colour and its error from a simulated light curve.

    Returns (nan, nan) if the colour cannot be determined (e.g. a band is
    missing). Keyword arguments are passed on to `fit_peaks`.
    """
    _, peakcol = fit_peaks(lc, **kwargs)
    if "gp_ztfg-ztfr" not in peakcol:
        return np.nan, np.nan
    return peakcol["gp_ztfg-ztfr"], peakcol.get("gp_ztfg-ztfr_err", np.nan)


BAND_COLORS = {"ztfg": "tab:green", "ztfr": "tab:red", "ztfi": "tab:orange"}


def plot_lightcurve(lc, results_gp, path, title="", t0=np.nan, z=np.nan):
    """Plot a simulated light curve with the estimated peak time in each band.

    Dashed vertical lines mark the GP peak time per band; the legend gives its
    phase relative to the true simulated t0 (observer frame, and rest frame if
    the redshift is known). The dotted black line marks the true t0.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for band in sorted(set(lc["band"])):
        sel = lc[lc["band"] == band]
        color = BAND_COLORS.get(band)
        ax.errorbar(sel["mjd"], sel["flux"], sel["fluxerr"], fmt="o", ms=3,
                    color=color, alpha=0.8, label=band)

        res = results_gp.get(band) if hasattr(results_gp, "get") else None
        peak_time = getattr(res, "peak_time", None)
        if peak_time is not None and np.isfinite(peak_time):
            label = f"{band} peak"
            if np.isfinite(t0):
                phase = peak_time - t0
                label += f": {phase:+.1f} d"
                if np.isfinite(z):
                    label += f" ({phase / (1 + z):+.1f} d rest)"
            ax.axvline(peak_time, color=color, ls="--", lw=1.2, label=label)

    if np.isfinite(t0):
        ax.axvline(t0, color="k", ls=":", lw=1.2, label="true t0")
    ax.set_xlabel("MJD")
    ax.set_ylabel("Flux")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Pipeline steps
# -----------------------------------------------------------------------------
def build_population(args):
    """Load warp templates and draw the target population."""
    loader = WarpfitTemplateLoader(args.warpdir, version=args.version, suffix=args.suffix)

    get_templates_kwargs = {
        "exclude_input": args.exclude_input,
        "template_selection": args.template_selection,
        "snbasis_selection": args.snbasis_selection,
        "random_seed": args.seed,
        "color_mode": _none_or(args.color_mode),
        "min_fit_quality": _none_or(args.min_fit_quality),
        "phase_buffer": _none_or(args.phase_buffer, int),
    }

    pop = WarpTemplatePopulation.from_warp_loader(
        loader, fitclass=args.fitclass, rate=args.rate,
        magabs=(args.magabs_mean, args.magabs_sigma),
        get_templates_kwargs=get_templates_kwargs,
    )

    draw_kwargs = dict(zmax=args.zmax, inplace=True)
    if args.size is not None:
        draw_kwargs.update(size=args.size, tstart=_maybe_float(args.tstart),
                           tstop=_maybe_float(args.tstop))
    else:
        draw_kwargs.update(tstart=args.tstart, tstop=args.tstop)
    print('simulating with args', draw_kwargs)
    pop.draw(**draw_kwargs)
    print(f"Drew {len(pop.data)} targets.")
    return pop


def load_survey(args):
    """Load ZTF logs and rescale sky noise by band.

    The original per-band values are kept in the 'skynoise_orig' column.
    Bands without a correction factor are left unscaled.
    """
    ztf = skysurvey.ZTF.from_logs()
    coefs = dict(zip(("ztfg", "ztfr", "ztfi"), args.skynoise_coefs))
    ztf.data["skynoise_orig"] = ztf.data["skynoise"].copy()
    ztf.data["skynoise"] = (
        ztf.data["skynoise_orig"] * ztf.data["band"].map(coefs).fillna(1.0)
    )
    return ztf


def _window_ok(dset, phase_range, min_filters, min_det):
    """Boolean per-target series: enough filters and detections in a phase window."""
    data = dset.get_ndetection(phase_range=list(phase_range), per_band=True,
                               join_bandday=True)
    enough_filters = data.groupby(level=0).size() >= min_filters
    enough_det = data.groupby(level=0).sum() >= min_det
    return enough_filters & enough_det


def select_good_sampling(dset, n_drawn, args):
    """Apply full / pre-peak / post-peak sampling cuts. Returns index of passing targets."""
    ok = _window_ok(dset, args.full_range, args.min_filters, args.min_det_full)
    print(f"Pass full-range cut:  {ok.mean():.3f}")
    ok = ok & _window_ok(dset, args.pre_range, args.min_filters, args.min_det_pre)
    print(f"... and pre-peak cut: {ok.mean():.3f}")
    ok = ok & _window_ok(dset, args.post_range, args.min_filters, args.min_det_post)
    print(f"... and post-peak cut: {ok.mean():.3f}")
    print(f"Good sampling: {ok.sum()} / {n_drawn} targets ({ok.sum() / n_drawn:.3f})")
    return ok[ok].index


def select_spectroscopic(data_good, args, rng):
    """Add 'pobs_spec' and return a boolean mask of spectroscopically observed targets."""
    data_good["pobs_spec"] = get_spectro_completeness(
        data_good["magobs"], loc=args.spec_loc, scale=args.spec_scale
    )
    if args.spec_mode == "threshold":
        mask = data_good["pobs_spec"] > args.spec_threshold
    else:
        mask = pd.Series(rng.random(len(data_good)) < data_good["pobs_spec"].values,
                         index=data_good.index)
    print(f"Fraction with spec obs: {mask.mean():.3f}")
    return mask

def observed_peak_mag(lc, bands, min_snr=5.0):
    """Brightest significant detection (AB mag) in `bands`; NaN if none."""
    sel = lc[lc["band"].isin(bands)]
    sel = sel[(sel["flux"] > 0) & (sel["flux"] / sel["fluxerr"] >= min_snr)]
    if len(sel) == 0:
        return np.nan
    zp = sel["zp"] if "zp" in sel.columns else 25.0
    return float(np.min(-2.5 * np.log10(sel["flux"]) + zp))


def select_bts_discovery(dset, data, args):
    """Discovery cut: >= N detections (any band) in a window, plus |b| cut."""
    ok = _window_ok(dset, args.bts_det_range, 1, args.bts_min_det)
    index = ok[ok].index
    print(f"BTS discovered: {len(index)} / {len(data)} ({len(index) / len(data):.3f})")

    if args.bts_min_abs_b > 0 and {"ra", "dec"} <= set(data.columns):
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        sub = data.loc[index]
        b = SkyCoord(sub["ra"].values * u.deg, sub["dec"].values * u.deg).galactic.b.deg
        index = index[np.abs(b) > args.bts_min_abs_b]
        print(f"... and |b| > {args.bts_min_abs_b}: {len(index)}")
    return index


def select_bts_spectroscopic(dset, data_good, args, rng):
    """Add 'mag_peak_obs' and 'pobs_spec'; return mask of BTS-classified targets."""
    data_good["mag_peak_obs"] = [
        observed_peak_mag(dset.get_target_lightcurve(i, detection=True),
                          args.bts_bands, args.bts_min_snr)
        for i in data_good.index
    ]
    mag = data_good["mag_peak_obs"]
    if args.bts_mode == "threshold":
        data_good["pobs_spec"] = (mag < args.bts_mag_limit).astype(float)
        mask = data_good["pobs_spec"] > 0.5
    else:
        p = args.bts_plateau * get_spectro_completeness(
            mag.fillna(0.0), loc=args.bts_loc, scale=args.bts_scale)
        p = np.where(mag.isna(), 0.0, p)
        data_good["pobs_spec"] = p
        mask = pd.Series(rng.random(len(data_good)) < p, index=data_good.index)
    print(f"BTS classified fraction (of discovered): {mask.mean():.3f}")
    return mask

def choose_plot_targets(index_obs, args, rng):
    """Targets whose light curves should be plotted, per --plot-lcs / --n-plot."""
    if args.plot_lcs == "all":
        return set(index_obs)
    if args.plot_lcs == "random":
        n = min(args.n_plot, len(index_obs))
        return set(rng.choice(np.asarray(index_obs), size=n, replace=False))
    return set()


def measure_colors(dset, index_obs, args, targets, plot_targets=(), plotdir=None):
    """Peak g-r colour for each spectroscopically selected target.

    Targets in `plot_targets` also get a light-curve plot in `plotdir`, with
    the GP peak phase per band (relative to the true t0 in `targets`).
    """
    rows = []
    for index in index_obs:
        lc = dset.get_target_lightcurve(index, detection=True)
        col, dcol = np.nan, np.nan
        try:
            results_gp, peakcol = fit_peaks(
                lc, gp_length_scale=args.gp_length_scale,
                peakflux_iter=args.peakflux_iter, n_sigma=args.n_sigma,
                min_eff_points=args.min_eff_points,
            )
            if "gp_ztfg-ztfr" in peakcol:
                col = peakcol["gp_ztfg-ztfr"]
                dcol = peakcol.get("gp_ztfg-ztfr_err", np.nan)
        except Exception as exc:  # a single bad light curve should not stop the run
            warnings.warn(f"Colour measurement failed for target {index}: {exc}")
            results_gp = None
        rows.append({"index": index, "gr": col, "gr_err": dcol})

        if results_gp is not None and index in plot_targets:
            t0 = targets["t0"].get(index, np.nan) if "t0" in targets else np.nan
            z = targets["z"].get(index, np.nan) if "z" in targets else np.nan
            title = f"target {index}   g-r = {col:.2f}" if np.isfinite(col) \
                else f"target {index}   g-r unavailable"
            try:
                plot_lightcurve(lc, results_gp,
                                os.path.join(plotdir, f"lc_{index}.png"),
                                title=title, t0=t0, z=z)
            except Exception as exc:
                warnings.warn(f"Plotting failed for target {index}: {exc}")
    return pd.DataFrame(rows).set_index("index")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def run_simulation(args):
    """Run the full simulation for one parameter set (no files written except plots).

    Returns (out, info): `out` is a DataFrame indexed by target with columns
    magobs, pobs_spec, z (if available), gr, gr_err; `info` holds selection counts.
    Importable, so other scripts can loop over e.g. --color-mode values.
    """
    rng = np.random.default_rng(args.seed)
    register_all()

    pop = build_population(args)
    ztf = load_survey(args)

    dset = skysurvey.DataSet.from_targets_and_survey(
        pop, ztf, progress_bar=not args.no_progress
    )



    if args.selection_mode == "bts":
        index_good = select_bts_discovery(dset, pop.data, args)
        data_good = pop.data.loc[index_good].copy()
        mask_spec = select_bts_spectroscopic(dset, data_good, args, rng)
    else:
        index_good = select_good_sampling(dset, len(pop.data), args)
        data_good = pop.data.loc[index_good].copy()
        mask_spec = select_spectroscopic(data_good, args, rng)
    index_obs = index_good[mask_spec.values]

    # Estimate loss due to forced photometry extraction and baseline subtraction
    n_spec_selected = len(index_obs)

    if args.db_eff_params:
        print('adding lc processing inefficiency, starting from', n_spec_selected)
        data_obs = data_good.loc[index_obs].copy()
        mask_db = select_database_efficiency(data_obs, args, rng)
        data_good["pobs_db"] = data_obs["pobs_db"]
        index_obs = index_obs[mask_db.values]    
        print('... after db inefficiecny', len(index_obs))

    plot_targets = choose_plot_targets(index_obs, args, rng)
    plotdir = args.plotdir or os.path.join(args.outdir, "lcplots")
    if plot_targets:
        os.makedirs(plotdir, exist_ok=True)
        print(f"Plotting {len(plot_targets)} light curves to {plotdir}")

    colors = measure_colors(dset, index_obs, args, pop.data,
                            plot_targets=plot_targets, plotdir=plotdir)
    n_measured = len(colors)
    colors = colors[colors["gr"].notna() & (colors["gr"] > args.color_floor)]
    print(f"Valid g-r colours: {len(colors)} / {n_measured}")

#    keep = [c for c in ("magobs", "pobs_spec", "z") if c in data_good.columns]
    keep = [c for c in ("magobs", "mag_peak_obs", "pobs_spec", "z") if c in data_good.columns]
    out = data_good.loc[colors.index, keep].join(colors)
    info = {
        "n_drawn": int(len(pop.data)),
        "n_good_sampling": int(len(index_good)),
        "n_spec_selected": int(n_spec_selected),
        "n_valid_colors": int(len(colors)),
        "frac_good_sampling": float(len(index_good) / len(pop.data)),
        "frac_spec_of_good": float(mask_spec.mean()),
        "n_db_selected": int(len(index_obs)),
        "db_efficiency": db_efficiency_info(args),        
    }
    return out, info


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    out, info = run_simulation(args)
    colors = out
    csv_path = os.path.join(args.outdir, f"{args.tag}_peak_colors.csv")
    out.to_csv(csv_path)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(colors["gr"], bins=args.hist_bins)
    ax.set_xlabel("Peak g - r (mag)")
    ax.set_ylabel("Number of targets")
    ax.set_title(f"{args.fitclass}: spectroscopically selected sample")
    fig.tight_layout()
    png_path = os.path.join(args.outdir, f"{args.tag}_peak_colors.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    summary = {
        "args": vars(args),
        **info,
        "gr_mean": float(colors["gr"].mean()) if len(colors) else None,
        "gr_median": float(colors["gr"].median()) if len(colors) else None,
        "gr_std": float(colors["gr"].std()) if len(colors) else None,
    }
    json_path = os.path.join(args.outdir, f"{args.tag}_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Wrote {csv_path}\n      {png_path}\n      {json_path}")


if __name__ == "__main__":
    main()
