#!/usr/bin/env python
# coding: utf-8
"""
Fit sncosmo SALT3 to every simulated, spectroscopically-selected light curve
and study the resulting fit-parameter distributions.

This is a variation of ``simulate_spec_peak_colors.py``'s pipeline: it reuses
that script's population draw, survey, sampling cuts and spectroscopic
selection (steps 1-5 of its docstring) for one template colour mode, but
replaces the GP peak-colour measurement (step 6) with a SALT3 light-curve fit
at the target's true, known redshift -- appropriate for a spectroscopically
confirmed sample, where the redshift is fixed rather than fit. The point is
not that SALT3 is the right model for these (typically non-Ia) templates, but
to characterise how a standard SN Ia fitter behaves when applied to them:
fit success rate, chi2/dof, the SALT3 shape/colour parameters (x1, c) it
lands on, how well it recovers the true time of peak, and (since the
simulation draws each target's true absolute magnitude) any systematic bias
or scatter in the peak brightness a Tripp-style SALT3 analysis would infer.

Pipeline
--------
1-5. Same as simulate_spec_peak_colors.py (population draw, sampling cuts,
     spectroscopic selection) for a single --color-mode.
6. For each selected target: estimate the light-curve peak time per band
   with a GP (reusing simulate_spec_peak_colors.fit_peaks, for a t0 guess),
   then fit sncosmo SALT3 with the redshift fixed to the simulation's true z
   and t0/x0/x1/c left free.
7. Collect per-target fit parameters, uncertainties, chi2/dof and (since the
   truth is known) the true t0 and absolute magnitude, and save diagnostic
   plots of the fit-parameter distributions and recovered-vs-true magnitude.

Note: no Milky Way or host-galaxy dust correction is applied here (matching
simulate_spec_peak_colors.py); SALT3's own colour parameter c absorbs any
such effects, which is standard practice for this kind of fit.

Example
-------
    python simulate_salt3_fit.py --category a --cid 1 \
        --warpdir /path/to/warpmod/v6 --color-mode None \
        --zmax 0.08 --plot-fits random --n-plot-fits 20 \
        --outdir ./salt3out --tag ccsn

Outputs (in --outdir, prefixed with --tag)
-------------------------------------------
    <tag>_salt3_fits.csv          one row per attempted fit: id, z, t0, x0,
                                   x1, c (+errors), chisq, ndof, chidof,
                                   success, absmag_fit, t0_true, magabs_true
                                   (if available), delta_t0, delta_absmag
    <tag>_salt3_diagnostics.png   histograms/scatter of the fit results
    <tag>_salt3_color_check.png   (only with --measure-gp-color) SALT3 c vs.
                                   the independently GP-measured peak g-r
    <tag>_salt3_summary.json      success rate and fit-result statistics
    <plotdir>/salt3_lc_<id>.png   (optional, --plot-fits) data + SALT3 model
                                   light curve, via sncosmo.plot_lc
"""

import argparse
import json
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sncosmo
import skysurvey
from astropy.table import Table

import simulate_spec_peak_colors as sim

try:
    import compare_sim_vs_observed as cmp  # reuse its class maps for default --zmax
    _HAVE_CMP = True
except Exception:  # pragma: no cover - optional convenience only
    _HAVE_CMP = False


# -----------------------------------------------------------------------------
# Command line
# -----------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Unrecognised options are forwarded to simulate_spec_peak_colors.py "
               "(--category/--cid or --fitclass, --warpdir, --color-mode, --rate, "
               "--zmax, --spec-loc, ...).",
    )

    g = p.add_argument_group("SALT3 fit")
    g.add_argument("--x1-bounds", nargs=2, type=float, default=[-5.0, 5.0])
    g.add_argument("--c-bounds", nargs=2, type=float, default=[-0.3, 1.0])
    g.add_argument("--t0-pad", type=float, default=15.0,
                   help="t0 bounds are [data tmin - pad, data tmax + pad]")
    g.add_argument("--min-bands", type=int, default=2,
                   help="Minimum number of observed bands required to attempt a fit")
    g.add_argument("--default-zp", type=float, default=25.0,
                   help="Zero point used if the light curve has no 'zp' column")
    g.add_argument("--default-zpsys", default="ab",
                   help="Photometric system used if the light curve has no 'zpsys' column")
    g.add_argument("--sigma-int", type=float, default=0.0,
                   help="Intrinsic dispersion added in quadrature to the flux errors "
                        "before fitting (0 disables; default unit is mag, see "
                        "--sigma-int-mode). Inflates both the fit weighting and the "
                        "reported chi2/chi2-per-dof, same as sncosmo's own errors would.")
    g.add_argument("--sigma-int-mode", choices=["mag", "frac", "abs"], default="mag",
                   help="How to interpret --sigma-int: 'mag' (magnitudes, converted to "
                        "a per-point fractional flux term via ln(10)/2.5), 'frac' "
                        "(fractional flux error directly), or 'abs' (a flat additive "
                        "flux error in the light curve's own flux units)")

    g = p.add_argument_group("light-curve fit plots")
    g.add_argument("--plot-fits", choices=["none", "all", "random"], default="none",
                   help="Save a data+model light-curve plot per fitted target")
    g.add_argument("--n-plot-fits", type=int, default=20,
                   help="Number of light curves to plot in 'random' mode")
    g.add_argument("--plot-only-success", action="store_true",
                   help="Only plot targets where the fit succeeded")
    g.add_argument("--plotdir", default=None,
                   help="Directory for fit plots (default: <outdir>/salt3lcplots)")

    g = p.add_argument_group("extra diagnostics")
    g.add_argument("--measure-gp-color", action="store_true",
                   help="Also measure the GP peak g-r colour per target (as in "
                        "simulate_spec_peak_colors.py) and compare it with SALT3's c")
    g.add_argument("--chidof-max", type=float, default=10.0,
                   help="Upper clip for chi2/dof in the diagnostics histogram")

    g = p.add_argument_group("output")
    g.add_argument("--outdir", default="./salt3out")
    g.add_argument("--tag", default="salt3fit")

    return p


def resolve_sim_args(sim_extra):
    """Parse the forwarded args for simulate_spec_peak_colors.py.

    If --zmax wasn't given, default it the same way compare_sim_vs_observed.py
    does: the highest NARROW_Z_LIMITS upper bound among the narrow classes
    making up the simulated (already-resolved, --category/--cid or --fitclass)
    class. Falls back to simulate_spec_peak_colors.py's own default if that
    script isn't importable or class resolution fails.
    """
    sim_args = sim.parse_args(sim_extra)
    if "--zmax" in sim_extra or not _HAVE_CMP:
        return sim_args
    try:
        constituents = cmp.resolve_constituent_narrow_classes(sim_args.fitclass)
        zmax_default = max(cmp.NARROW_Z_LIMITS.get(n, cmp.DEFAULT_Z_LIMITS)[1] for n in constituents)
        print(f"No --zmax given; using {zmax_default} (max NARROW_Z_LIMITS upper bound "
              f"over constituents {constituents}).")
        return sim.parse_args(sim_extra + ["--zmax", str(zmax_default)])
    except ValueError as e:
        print(f"WARNING: could not derive a default --zmax ({e}); "
              f"using simulate_spec_peak_colors.py's own default.")
        return sim_args


# -----------------------------------------------------------------------------
# Light-curve conversion
# -----------------------------------------------------------------------------
_warned_missing_zp = {"zp": False, "zpsys": False}


def to_sncosmo_table(lc, default_zp, default_zpsys):
    """Convert a skysurvey target light curve to an sncosmo-ready astropy Table.

    Handles either 'mjd' or 'time' as the time column name. If 'zp'/'zpsys'
    are missing, fills them with the given defaults and warns once.
    """
    cols = list(lc.keys()) if hasattr(lc, "keys") else list(lc.colnames)
    time_col = "mjd" if "mjd" in cols else "time"
    n = len(lc)

    if "zp" in cols:
        zp = np.asarray(lc["zp"], dtype=float)
    else:
        if not _warned_missing_zp["zp"]:
            print(f"NOTE: light curves have no 'zp' column; using default_zp={default_zp} "
                  f"for all targets.")
            _warned_missing_zp["zp"] = True
        zp = np.full(n, default_zp)

    if "zpsys" in cols:
        zpsys = np.asarray(lc["zpsys"])
    else:
        if not _warned_missing_zp["zpsys"]:
            print(f"NOTE: light curves have no 'zpsys' column; using "
                  f"default_zpsys='{default_zpsys}' for all targets.")
            _warned_missing_zp["zpsys"] = True
        zpsys = np.full(n, default_zpsys, dtype=object)

    return Table({
        "time": np.asarray(lc[time_col], dtype=float),
        "band": np.asarray(lc["band"]),
        "flux": np.asarray(lc["flux"], dtype=float),
        "fluxerr": np.asarray(lc["fluxerr"], dtype=float),
        "zp": zp,
        "zpsys": zpsys,
    })


# -----------------------------------------------------------------------------
# SALT3 fitting
# -----------------------------------------------------------------------------
def add_intrinsic_dispersion(tab, sigma_int, mode="mag"):
    """Add an intrinsic-dispersion term to `tab['fluxerr']`, in quadrature.

    Returns `tab` unchanged if sigma_int <= 0. Otherwise returns a copy with
    inflated fluxerr, so both the fit weighting and sncosmo's own chi2/ndof
    reflect it (rather than only being recomputed after the fact).

    mode:
      'mag'  -- sigma_int is a magnitude dispersion, converted per point to a
                fractional flux term via ln(10)/2.5 (~0.4605 * sigma_int),
                then scaled by that point's flux -- the usual SN-cosmology
                convention for "intrinsic scatter in mag".
      'frac' -- sigma_int is already a fractional flux dispersion, scaled by
                that point's flux directly.
      'abs'  -- sigma_int is a flat additive flux error, in the light curve's
                own flux units (not scaled by flux) -- use this only if you
                know those units, since they depend on the light curve's zp.
    """
    if sigma_int <= 0:
        return tab
    tab = tab.copy()
    flux = np.asarray(tab["flux"], dtype=float)
    fluxerr = np.asarray(tab["fluxerr"], dtype=float)
    if mode == "mag":
        extra = (sigma_int * np.log(10) / 2.5) * np.abs(flux)
    elif mode == "frac":
        extra = sigma_int * np.abs(flux)
    elif mode == "abs":
        extra = np.full_like(flux, sigma_int)
    else:
        raise ValueError(f"Unknown sigma_int_mode '{mode}' (expected mag, frac, or abs)")
    tab["fluxerr"] = np.sqrt(fluxerr ** 2 + extra ** 2)
    return tab


def fit_salt3(tab, z, args, t0_guess):
    """Fit SALT3 to one light curve at fixed redshift `z`.

    If args.sigma_int > 0, an intrinsic-dispersion term is added in
    quadrature to the flux errors (see add_intrinsic_dispersion) before
    fitting, so both the best-fit weighting and the resulting chi2/ndof
    include it. Returns (result, fitted_model) from sncosmo.fit_lc, or
    (None, None) if the fit raises. `z` is set on the model and excluded
    from the free parameters, matching a spectroscopically confirmed sample.
    """
    tab = add_intrinsic_dispersion(tab, args.sigma_int, args.sigma_int_mode)
    m = sncosmo.Model(source="salt3")
    m.set(z=float(z), t0=float(t0_guess))
    bounds = {
        "t0": [float(tab["time"].min()) - args.t0_pad, float(tab["time"].max()) + args.t0_pad],
        "x1": list(args.x1_bounds),
        "c": list(args.c_bounds),
    }
    try:
        result, fitted_model = sncosmo.fit_lc(tab, m, ["t0", "x0", "x1", "c"], bounds=bounds)
        return result, fitted_model
    except Exception as exc:
        warnings.warn(f"SALT3 fit failed: {exc}")
        return None, None


def estimate_t0_guess(lc, sim_args, salt3_peakphase_ztfr):
    """GP-based t0 guess (peak time in ztfr minus SALT3's own ztfr peak phase).

    Falls back to the midpoint of the observed time range if the GP peak
    estimate is unavailable (e.g. too few points in ztfr).
    """
    try:
        results_gp, _ = sim.fit_peaks(
            lc, gp_length_scale=sim_args.gp_length_scale,
            peakflux_iter=sim_args.peakflux_iter, n_sigma=sim_args.n_sigma,
            min_eff_points=sim_args.min_eff_points,
        )
        peak_time = getattr(results_gp.get("ztfr"), "peak_time", None)
        if peak_time is not None and np.isfinite(peak_time):
            return peak_time - salt3_peakphase_ztfr, True
    except Exception:
        pass
    time_col = "mjd" if "mjd" in (lc.keys() if hasattr(lc, "keys") else lc.colnames) else "time"
    return 0.5 * (float(np.min(lc[time_col])) + float(np.max(lc[time_col]))), False


def choose_plot_targets(candidates, mode, n_plot, rng):
    """Which of `candidates` (an index-like) get a saved light-curve plot."""
    if mode == "all":
        return set(candidates)
    if mode == "random":
        n = min(n_plot, len(candidates))
        return set(rng.choice(np.asarray(candidates), size=n, replace=False))
    return set()


def plot_salt3_fit(tab, result, fitted_model, path, title=""):
    """Data + SALT3 model light curve, via sncosmo's own plot_lc."""
    fig = sncosmo.plot_lc(tab, model=fitted_model, errors=result.errors)
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=120)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main fitting loop
# -----------------------------------------------------------------------------
def run_fits(dset, index_obs, data_good, sim_args, args, rng, plotdir):
    """Fit SALT3 to every target in `index_obs`. Returns a results DataFrame."""
    m0 = sncosmo.Model(source="salt3")
    peakphase_ztfr = m0.source.peakphase("ztfr")

    plot_targets = choose_plot_targets(index_obs, args.plot_fits, args.n_plot_fits, rng)
    if plot_targets:
        os.makedirs(plotdir, exist_ok=True)
        print(f"Will plot {len(plot_targets)} light curve fits to {plotdir}")

    rows = []
    n_skipped_bands, n_t0_fallback, n_fit_error = 0, 0, 0

    for index in index_obs:
        lc = dset.get_target_lightcurve(index, detection=True)
        n_bands = len(set(np.asarray(lc["band"])))
        if n_bands < args.min_bands:
            n_skipped_bands += 1
            continue

        z = float(data_good["z"].loc[index])
        tab = to_sncosmo_table(lc, args.default_zp, args.default_zpsys)

        t0_guess, gp_ok = estimate_t0_guess(lc, sim_args, peakphase_ztfr)
        if not gp_ok:
            n_t0_fallback += 1

        row = {"id": index, "z": z, "nbr_bands": n_bands, "ndet": len(tab), "t0_guess_gp": gp_ok}

        result, fitted_model = fit_salt3(tab, z, args, t0_guess)
        if result is None:
            n_fit_error += 1
            row["success"] = False
        else:
            row["success"] = bool(result.success)
            row["chisq"] = float(result.chisq)
            row["ndof"] = int(result.ndof)
            row["chidof"] = float(result.chisq / result.ndof) if result.ndof > 0 else np.nan
            errors = result["errors"]  # dict of {param_name: uncertainty}, sncosmo Result is dict-like
            for pname in ("t0", "x0", "x1", "c"):
                idx = result["param_names"].index(pname)
                row[pname] = float(result["parameters"][idx])
                err = errors.get(pname, np.nan) if errors else np.nan
                row[f"{pname}_err"] = float(err) if err is not None else np.nan
            try:
                row["absmag_fit"] = float(fitted_model.source_peakabsmag("bessellb", "ab"))
            except Exception:
                row["absmag_fit"] = np.nan

        if "t0" in data_good.columns:
            row["t0_true"] = float(data_good["t0"].loc[index])
            if result is not None:
                row["delta_t0"] = row["t0"] - row["t0_true"]
        if "magabs" in data_good.columns:
            row["magabs_true"] = float(data_good["magabs"].loc[index])
            if result is not None and np.isfinite(row.get("absmag_fit", np.nan)):
                row["delta_absmag"] = row["absmag_fit"] - row["magabs_true"]

        if args.measure_gp_color:
            try:
                _, peakcol = sim.fit_peaks(
                    lc, gp_length_scale=sim_args.gp_length_scale,
                    peakflux_iter=sim_args.peakflux_iter, n_sigma=sim_args.n_sigma,
                    min_eff_points=sim_args.min_eff_points,
                )
                row["gp_gr"] = peakcol.get("gp_ztfg-ztfr", np.nan)
            except Exception:
                row["gp_gr"] = np.nan

        rows.append(row)

        if result is not None and index in plot_targets:
            if args.plot_only_success and not row["success"]:
                pass
            else:
                sigma_note = f"   (sigma_int={args.sigma_int} {args.sigma_int_mode})" if args.sigma_int > 0 else ""
                title = (f"target {index}   z={z:.3f}   "
                        f"chi2/dof={row.get('chidof', np.nan):.2f}{sigma_note}")
                try:
                    plot_salt3_fit(tab, result, fitted_model,
                                   os.path.join(plotdir, f"salt3_lc_{index}.png"), title=title)
                except Exception as exc:
                    warnings.warn(f"Plotting failed for target {index}: {exc}")

    print(f"Skipped (fewer than {args.min_bands} bands): {n_skipped_bands}")
    print(f"t0 guess fell back to time-range midpoint (GP peak unavailable): {n_t0_fallback}")
    print(f"Fit raised an exception: {n_fit_error}")

    return pd.DataFrame(rows).set_index("id") if rows else pd.DataFrame()


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------
def make_diagnostics_plot(df, args, path):
    """2x3 grid: chi2/dof, x1, c, delta_t0, recovered-vs-true magnitude, c-x1."""
    ok = df[df["success"] == True] if "success" in df else df  # noqa: E712
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))

    def hist(ax, data, label, clip=None):
        d = np.asarray(data, dtype=float)
        d = d[np.isfinite(d)]
        if clip is not None:
            d = d[(d >= clip[0]) & (d <= clip[1])]
        if len(d) == 0:
            ax.set_title(f"{label} (no data)")
            return
        ax.hist(d, bins=30, color="steelblue", alpha=0.8, edgecolor="black", linewidth=0.4)
        ax.set_xlabel(label)
        ax.set_ylabel("N")
        ax.set_title(f"{label}: mean={d.mean():.3f}, std={d.std():.3f}, N={len(d)}", fontsize=9)

    hist(axes[0, 0], ok.get("chidof", []), r"$\chi^2$/dof", clip=(0, args.chidof_max))
    hist(axes[0, 1], ok.get("x1", []), "x1")
    hist(axes[0, 2], ok.get("c", []), "c")
    hist(axes[1, 0], ok.get("delta_t0", []), r"$t_{0,fit} - t_{0,true}$ (days)")

    ax = axes[1, 1]
    if "delta_absmag" in ok and "z" in ok:
        d = ok.dropna(subset=["delta_absmag", "z"])
        if len(d):
            ax.scatter(d["z"], d["delta_absmag"], s=10, alpha=0.5, color="darkorange")
            ax.axhline(0, color="k", lw=0.8, ls="--")
            ax.set_xlabel("Redshift")
            ax.set_ylabel(r"$M_{B,fit} - M_{B,true}$")
            ax.set_title(f"mean={d['delta_absmag'].mean():.3f}, std={d['delta_absmag'].std():.3f}",
                        fontsize=9)
        else:
            ax.set_title("delta_absmag vs z (no data)")
    else:
        ax.set_title("delta_absmag not available")
        ax.axis("off")

    ax = axes[1, 2]
    if "x1" in ok and "c" in ok:
        d = ok.dropna(subset=["x1", "c"])
        if len(d):
            sc = ax.scatter(d["x1"], d["c"], c=d.get("chidof", None), s=10, alpha=0.6, cmap="viridis")
            if "chidof" in d:
                fig.colorbar(sc, ax=ax, label=r"$\chi^2$/dof")
            ax.set_xlabel("x1")
            ax.set_ylabel("c")
            ax.set_title("SALT3 shape-colour plane", fontsize=9)
        else:
            ax.set_title("x1 vs c (no data)")
    else:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_color_check_plot(df, path):
    """SALT3 c vs. the independently GP-measured peak g-r colour."""
    ok = df[df["success"] == True] if "success" in df else df  # noqa: E712
    d = ok.dropna(subset=["c", "gp_gr"]) if "gp_gr" in ok and "c" in ok else pd.DataFrame()

    fig, ax = plt.subplots(figsize=(5.5, 5))
    if len(d):
        ax.scatter(d["gp_gr"], d["c"], s=10, alpha=0.5, color="teal")
        lo = min(d["gp_gr"].min(), d["c"].min())
        hi = max(d["gp_gr"].max(), d["c"].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
        ax.legend()
        r = np.corrcoef(d["gp_gr"], d["c"])[0, 1]
        ax.set_title(f"Pearson r = {r:.2f}, N={len(d)}", fontsize=10)
    else:
        ax.set_title("No data")
    ax.set_xlabel("GP peak g-r (mag)")
    ax.set_ylabel("SALT3 c")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(argv=None):
    args, sim_extra = build_parser().parse_known_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    plotdir = args.plotdir or os.path.join(args.outdir, "salt3lcplots")

    sim_args = resolve_sim_args(sim_extra)
    rng = np.random.default_rng(sim_args.seed)
    sim.register_all()

    print(f"Class: {sim_args.fitclass}   colour mode: {sim_args.color_mode}")

    pop = sim.build_population(sim_args)
    ztf = sim.load_survey(sim_args)
    dset = skysurvey.DataSet.from_targets_and_survey(pop, ztf, progress_bar=not sim_args.no_progress)

    index_good = sim.select_good_sampling(dset, len(pop.data), sim_args)
    data_good = pop.data.loc[index_good].copy()

    mask_spec = sim.select_spectroscopic(data_good, sim_args, rng)
    index_obs = index_good[mask_spec.values]
    print(f"Fitting SALT3 to {len(index_obs)} spectroscopically selected targets")

    df = run_fits(dset, index_obs, data_good, sim_args, args, rng, plotdir)
    if df.empty:
        sys.exit("No SALT3 fits were attempted (check --min-bands and the selection cuts).")

    csv_path = os.path.join(args.outdir, f"{args.tag}_salt3_fits.csv")
    df.to_csv(csv_path)

    diag_path = os.path.join(args.outdir, f"{args.tag}_salt3_diagnostics.png")
    make_diagnostics_plot(df, args, diag_path)

    color_check_path = None
    if args.measure_gp_color:
        color_check_path = os.path.join(args.outdir, f"{args.tag}_salt3_color_check.png")
        make_color_check_plot(df, color_check_path)

    n_success = int(df["success"].sum()) if "success" in df else 0
    summary = {
        "class_name": sim_args.fitclass,
        "color_mode": sim_args.color_mode,
        "n_drawn": int(len(pop.data)),
        "n_spec_selected": int(len(index_obs)),
        "n_attempted": int(len(df)),
        "n_success": n_success,
        "success_rate": float(n_success / len(df)) if len(df) else None,
        "chidof_mean": float(df.loc[df["success"], "chidof"].mean()) if n_success else None,
        "chidof_median": float(df.loc[df["success"], "chidof"].median()) if n_success else None,
        "x1_mean": float(df.loc[df["success"], "x1"].mean()) if n_success else None,
        "x1_std": float(df.loc[df["success"], "x1"].std()) if n_success else None,
        "c_mean": float(df.loc[df["success"], "c"].mean()) if n_success else None,
        "c_std": float(df.loc[df["success"], "c"].std()) if n_success else None,
        "delta_t0_mean": float(df["delta_t0"].mean()) if "delta_t0" in df else None,
        "delta_t0_std": float(df["delta_t0"].std()) if "delta_t0" in df else None,
        "delta_absmag_mean": float(df["delta_absmag"].mean()) if "delta_absmag" in df else None,
        "delta_absmag_std": float(df["delta_absmag"].std()) if "delta_absmag" in df else None,
        "config": {k: str(v) for k, v in vars(args).items()},
        "sim_args": sim_extra,
    }
    json_path = os.path.join(args.outdir, f"{args.tag}_salt3_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSuccess rate: {summary['success_rate']}")
    print(f"Wrote {csv_path}\n      {diag_path}"
          + (f"\n      {color_check_path}" if color_check_path else "")
          + f"\n      {json_path}")


if __name__ == "__main__":
    main()