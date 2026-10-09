#!/usr/bin/env python
# coding: utf-8
"""
Database-efficiency selection step for the spectroscopic-sample simulation.

The efficiency is CLASS INDEPENDENT: fit it once, pooled over all classes
(run fit_efficiency_sigmoid.py without --group/--classes -> group "All classes"),
and it is applied to every simulated target whatever --fitclass is. The fit
script tests that assumption (class_independence_*.csv); check it before
trusting the pooled curve.

Applies the efficiency fitted by fit_efficiency_sigmoid.py (its
sigmoid_fit_params_<ver>.csv) to simulated targets: each target survives with
probability eff(m), evaluated at its peak magnitude. This mimics objects that
are in the input catalogue (BTS) but missing from the local database.

Usage in the simulation script
    from db_efficiency import (add_db_efficiency_args, select_database_efficiency,
                               db_efficiency_info)
    add_db_efficiency_args(p)                      # in parse_args()
    mask = select_database_efficiency(data, args, rng)   # in run_simulation()

Which magnitude: the fit used the CATALOGUE peak magnitude, so by default the
simulated 'mag_peak_obs' (observed BTS-style peak in g/r) is used when present
(--selection-mode bts), else 'magobs'. Override with --db-eff-magcol.

Several flags can be given (--db-eff-flag db_ok gate_cond); their efficiencies
are multiplied (independent draws). Default is db_ok only: gate_cond partly
overlaps with the simulation's own sampling cuts and GP colour measurement, so
adding it can double count.

The efficiency is held constant outside the magnitude range the fit was made
on (mag_min / mag_max columns), instead of extrapolating the model.

The model formulas below mirror MODELS in fit_efficiency_sigmoid.py; keep them
in sync if you add a model there.
"""

import numpy as np
import pandas as pd
from scipy.special import expit


POOLED_GROUPS = ("All classes", "Joint (all)")   # class-independent fits


def add_db_efficiency_args(parser):
    g = parser.add_argument_group("database efficiency")
    g.add_argument("--db-eff-params", default=None,
                   help="sigmoid_fit_params_<ver>.csv from fit_efficiency_sigmoid.py; "
                        "enables the database-efficiency selection")
    g.add_argument("--db-eff-flag", nargs="+", default=["db_ok"],
                   help="Efficiency flag(s) in that file to apply (default: db_ok)")
    g.add_argument("--db-eff-group", default=None,
                   help="Group name in the file. Default: the pooled, class-independent "
                        "fit ('All classes', or 'Joint (all)'); required only if the "
                        "file holds neither")
    g.add_argument("--db-eff-mode", choices=["random", "weight"], default="random",
                   help="'random': Bernoulli draw per target; 'weight': keep all "
                        "targets and only store pobs_db for use as weights")
    g.add_argument("--db-eff-magcol", default=None,
                   help="Magnitude column (default: mag_peak_obs if present, else magobs)")
    return g


def load_db_efficiency(path, flag="db_ok", group=None):
    """Return the parameter row (pd.Series) for one flag/group of the fit table."""
    df = pd.read_csv(path)
    sub = df[df["flag"] == flag]
    if sub.empty:
        raise ValueError(f"Flag '{flag}' not in {path}; available: "
                         f"{sorted(df['flag'].unique())}")
    if group is None:
        for pooled in POOLED_GROUPS:
            if pooled in set(sub["group"]):
                return sub[sub["group"] == pooled].iloc[0]
        if len(sub) == 1:
            return sub.iloc[0]
        raise ValueError(f"{path} has no pooled group ({POOLED_GROUPS}) for '{flag}' "
                         f"and holds several: {list(sub['group'])}; choose with "
                         f"--db-eff-group, or refit pooled (no --group/--classes)")
    sub = sub[sub["group"] == group]
    if sub.empty:
        raise ValueError(f"Group '{group}' not found for '{flag}' in {path}; "
                         f"available: {list(df[df['flag'] == flag]['group'])}")
    return sub.iloc[0]


def efficiency_from_row(row, mag):
    """Efficiency at magnitude(s) `mag` from a fit-table row.

    Returns (p, n_outside): n_outside counts finite magnitudes outside the
    fitted range (efficiency is held at the edge value there).
    """
    mag = np.asarray(mag, float)
    n_out = 0
    lo, hi = row.get("mag_min", np.nan), row.get("mag_max", np.nan)
    if np.isfinite(lo) and np.isfinite(hi):
        fin = np.isfinite(mag)
        n_out = int(np.sum(fin & ((mag < lo) | (mag > hi))))
        mag = np.clip(mag, lo, hi)          # NaN stays NaN
    model = row["model"]
    if model == "sigmoid":
        p = row["p_eps"] * expit(-(mag - row["p_m50"]) / row["p_s"])
    elif model == "floor":
        p = row["p_eps"] * (row["p_floorfrac"]
                            + (1 - row["p_floorfrac"]) * expit(-(mag - row["p_m50"]) / row["p_s"]))
    elif model == "logistic":
        p = expit(row["p_a"] + row["p_b"] * (mag - row["mref"]))
    elif model == "linear":
        p = np.clip(row["p_p0"] + row["p_slope"] * (mag - row["mref"]), 0, 1)
    else:
        raise ValueError(f"Unknown efficiency model '{model}'")
    return p, n_out


def select_database_efficiency(data, args, rng):
    """Add 'pobs_db' to `data` and return a boolean mask of targets that survive.

    `data` is the DataFrame of targets to be thinned (here: the spectroscopically
    selected ones). In 'random' mode each target survives with probability
    pobs_db; in 'weight' mode all survive (mask all True) and pobs_db is meant
    to be used as a weight downstream. Magnitudes that are NaN get pobs_db = 0.
    """
    magcol = args.db_eff_magcol or ("mag_peak_obs" if "mag_peak_obs" in data.columns else "magobs")
    if magcol not in data.columns:
        raise KeyError(f"Magnitude column '{magcol}' not in the simulated data "
                       f"({list(data.columns)})")
    if magcol == "magobs":
        print("NOTE: database efficiency evaluated on 'magobs' (model peak mag), "
              "not an observed catalogue-style peak mag; use --selection-mode bts "
              "for a closer match to how the fit magnitudes were defined.")
    mag = data[magcol].to_numpy(float)

    p = np.ones(len(data))
    for flag in args.db_eff_flag:
        row = load_db_efficiency(args.db_eff_params, flag, args.db_eff_group)
        pf, n_out = efficiency_from_row(row, mag)
        p *= pf
        print(f"Database efficiency [{flag}, {row['group']}, model={row['model']}]: "
              f"mean {np.nanmean(pf):.3f}; {n_out}/{len(mag)} targets outside fitted "
              f"range [{row.get('mag_min', np.nan):.2f}, {row.get('mag_max', np.nan):.2f}]; "
              f"class-independent, applied regardless of simulated class")
    n_nan = int(np.sum(~np.isfinite(p)))
    if n_nan:
        print(f"WARNING: {n_nan} targets have no finite '{magcol}'; assigned pobs_db=0.")
    p = np.nan_to_num(p, nan=0.0)
    data["pobs_db"] = p

    if args.db_eff_mode == "random":
        mask = pd.Series(rng.random(len(data)) < p, index=data.index)
    else:
        mask = pd.Series(True, index=data.index)
    print(f"Fraction surviving database efficiency ({args.db_eff_mode}): "
          f"{mask.mean():.3f}  (mean pobs_db {p.mean():.3f})")
    return mask


def db_efficiency_info(args):
    """Small JSON-able description of what was applied, for the run summary."""
    if not args.db_eff_params:
        return None
    info = {"params_file": args.db_eff_params, "mode": args.db_eff_mode,
            "magcol": args.db_eff_magcol, "flags": []}
    for flag in args.db_eff_flag:
        row = load_db_efficiency(args.db_eff_params, flag, args.db_eff_group)
        info["flags"].append({"flag": flag, "group": row["group"], "model": row["model"],
                              "mag_min": float(row.get("mag_min", np.nan)),
                              "mag_max": float(row.get("mag_max", np.nan))})
    return info

