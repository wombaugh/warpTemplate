#!/usr/bin/env python
# coding: utf-8
"""
Warp template coefficient extraction from BTS fits.

Builds on v2_I: redshift limits pre-applied, classes as in v2_0.
Parses btsfits summary files and collects warp template coefficients
for specified class combinations.
"""

import argparse
import json
import pickle
import re
import os
import warnings
from pathlib import Path
from typing import Optional, List


import numpy as np
import pandas as pd
import pymongo
import sncosmo
from astropy.table import Table
from iminuit.util import IMinuitWarning
from scipy.stats.distributions import chi2


from ampel.ztf.util.ZTFIdMapper import ZTFIdMapper
from ampel.ztf.view.ZTFFPTabulator import ZTFFPTabulator
from warptemplate import (
    TEMPLATE_CLOSE_TYPES,
    add_warpclasses,
    get_template_correction,
    get_warpedTimeSeriesModel,
    register_all,
)

# Suppress warnings
warnings.filterwarnings("ignore", category=IMinuitWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ─── Configuration defaults ──────────────────────────────────────────────────

DEFAULT_BTS_FILE = "/Users/jnordin/data/ztf/bts/bts_explorer_260601.csv"
DEFAULT_FDIR = "/Users/jnordin/data/models/sncosmo/"
DEFAULT_OUTDIR = "/Users/jnordin/data/models/sncosmo/warpmod/"

# Selection parameters
MIN_BANDS = 2
MIN_PEAK_NDOF = 3
REQUIRE_PEAK_GOOD = False
CHIDOF_MAX = 6

# Error model
ERRFLOOR = 0.0
FLUX_FRAC_DISPERSION = 0.02

# Warp selection
TEMPLATE_COUNT = 5
MIN_DRAW_PROB = 10**-99.    # Looks small, but thats how xchi2 survival function scales with this data
GOOD_WARPFIT_SF = 0.5

# Fit properties for warped model
# Note that hostebv is not included - it should be absorbed into the warp correction
FITPROP = ["t0", "amplitude"]

# Max phases for warpfit, should probably be class depentent
MAX_PHASES = {
    "SN IIP": {'n':[-20,150]},
}



# ─── Argument parsing ────────────────────────────────────────────────────────

def parse_args():
    """Parse command-line options for one v4 coefficient-construction run."""

    parser = argparse.ArgumentParser(description="Extract warp template coefficients")
    parser.add_argument(
        "--classwidth",  "--cw",
        choices=["n", "e", "w", "a"], default="n",
        help="Class width: (n)arrow, (e)xtended, (w)ide, (a)ll"
    )
    parser.add_argument(
        "--cid",  "--classid",
        type=int, default=11, help="Class index to process")
    parser.add_argument(
        '--version', '-v',
        default=os.environ.get('VERSION', 4),
        help='Version string for input and output files (default: $VERSION or 4)'
    )
    parser.add_argument(
        "--fit-host-dust", action="store_true", default=True,
        help="Fit host extinction when warping"
    )
    parser.add_argument("--no-fit-host-dust", dest="fit_host_dust", action="store_false")
    parser.add_argument("--bts-file", default=DEFAULT_BTS_FILE)
    parser.add_argument("--fdir", default=DEFAULT_FDIR)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--db-name", default="bts_ipacfp_strictbase_train_jul26")
    return parser.parse_args()


# ─── Data loading ────────────────────────────────────────────────────────────

def load_bts_data(bts_file: str):
    """Load BTS explorer data and add warp classes."""
    df_bts = pd.read_csv(bts_file)
    df_bts = add_warpclasses(df_bts, purge=True)
    return df_bts


def get_process_classes(df_bts: pd.DataFrame, category: str, class_name: str):
    """Determine narrow classes to process for given category class."""
    tmask = df_bts[f"type_{category}"] == class_name
    return list(set(df_bts.loc[tmask]["type_n"]))


def load_typefit_data(process_classes: list, fdir: str, version: int = 4):
    """Load and combine btsfits JSON files for specified narrow classes."""
    typefitdata = None

    for readclass in process_classes:
        if readclass == "SN Ia":
            print("... skipping normal SNe Ia - did not run the bts sample.")
            continue

        rootname = f"btsfitsv{version}_{readclass.replace('/', '')}.json"
        fname = Path(fdir) / rootname
        print(f"Loading {fname} ...")

        if not fname.exists():
            print(f"Warning: {fname} not found, skipping")
            continue

        with open(fname) as infile:
            newdata = json.load(infile)
            if typefitdata is None:
                typefitdata = newdata
            else:
                typefitdata = combine_typefits(typefitdata, newdata)

    return typefitdata


def combine_typefits(dictlI: dict, dictlII: dict):
    """Combine internal dicts of two lists of dicts."""
    for templatename, datalist in dictlII.items():
        dictlI.setdefault(templatename, []).extend(datalist)
    return dictlI


# ─── Fit quality assessment ──────────────────────────────────────────────────

def get_salt_cosmofit(
    fitlist: list,
    chidofmax: float = 3.0,
    truetypes: list = None,
    peakfit: bool = True,
):
    """Filter SALT fits by quality criteria."""
    if truetypes is None:
        truetypes = []

    outd = []
    keys = [
        "z", "chisq", "ndof", "peakmag", "chidof", "id", "nbr_bands", "ndet",
        "class", "peakchi", "peakdet", "earlydet", "peakbands", "peak_good",
        "presum", "postsum", "thendet", "postdet", "peak_gp_ztfg-ztfr", "peak_gp_ztfr-ztfri",
    ]

    for snfit in fitlist:
        if not snfit.get("success", False):
            continue

        snd = {k: snfit[k] for k in keys if k in snfit}
        fitp = snfit["ndet"] - snfit["ndof"]
        snd["aic"] = 2 * fitp + snd["chisq"]
        snd["peakndof"] = snfit["peakdet"] - fitp

        if peakfit:
            if snd["peakndof"] <= 0:
                continue
            snd["peakchisqdof"] = snfit["peakchi"] / snd["peakndof"]
            snd["peakaic"] = 2 * fitp + snfit["peakchi"]
            snd["chidof"] = snfit["peakchi"] / snd["peakndof"]

        snd["correct"] = snd["class"] in truetypes
        snd["goodfit"] = snd["chidof"] < chidofmax

        outd.append(snd)

    return pd.DataFrame.from_dict(outd)


def get_timeseries_goodfit(
    fitlist: list,
    chidofmax: float = 3.0,
    truetypes: list = None,
    peakfit: bool = True,
):
    """Filter time-series model fits by quality criteria."""
    if truetypes is None:
        truetypes = []

    outd = []
    keys = [
        "z", "chisq", "ndof", "peakmag", "chidof", "id", "nbr_bands", "ndet",
        "class", "peakchi", "peakdet", "earlydet", "peakbands", "peak_good",
        "presum", "postsum", "thendet", "postdet", "peak_gp_ztfg-ztfr", "peak_gp_ztfr-ztfri",
    ]

    for snfit in fitlist:
        if not snfit.get("success", False):
            continue

        snd = {k: snfit[k] for k in keys if k in snfit}
        fitp = snfit["ndet"] - snfit["ndof"]
        snd["aic"] = 2 * fitp + snd["chisq"]
        snd["peakndof"] = snfit["peakdet"] - fitp

        if peakfit:
            if snd["peakndof"] <= 0:
                continue
            snd["peakchisqdof"] = snfit["peakchi"] / snd["peakndof"]
            snd["peakaic"] = 2 * fitp + snfit["peakchi"]
            snd["chidof"] = snfit["peakchi"] / snd["peakndof"]

        snd["correct"] = snd["class"] in truetypes
        snd["goodfit"] = snd["chidof"] < chidofmax

        outd.append(snd)

    return pd.DataFrame.from_dict(outd)

# ─── Check for unphysical model variability ──────────────────────────────────────────

def evaluate_gap_physicality(model, t_start, t_end, f_start, f_end,
                             band, zp=25., zpsys='ab',
                             n_eval=50):
    """
    Compare flux at gap endpoints vs. interior for single band.
    Returns max relative flux change and location of extremum.
    """
    t_eval = np.linspace(t_start, t_end, n_eval)
    flux = model.bandflux(band, t_eval, zp=zp, zpsys=zpsys)

    modf_start, modf_end = flux[0], flux[-1]

    # Linear interpolation baseline
    f_linear = np.linspace(modf_start, modf_end, n_eval)

    # Deviation from linear trend
    deviation = flux - f_linear
    max_dev_idx = np.argmax(np.abs(deviation))
    max_dev = np.abs(deviation[max_dev_idx])

    # Relative to mean flux in gap
    rel_dev = max_dev / (np.mean(flux) + 1e-10)

    # Relative to flux evolution between gap limits
    evo_dev = max_dev / abs(f_start-f_end)

    # Sign: positive = bump, negative = dip vs. linear
    sign = np.sign(deviation[max_dev_idx])

    return {
        'max_abs_deviation': max_dev,
        'relative_deviation': rel_dev,
        'deviation_sign': sign,
        'evolution_deviation': evo_dev,
        'flux_range': (np.min(flux), np.max(flux)),
        'non_monotonic': np.any(np.diff(flux)[:-1] * np.diff(flux)[1:] < 0)
    }, t_eval, flux


def find_band_gaps(band_table, min_gap):
    """Find gaps in single band's time sampling."""

    # Sort by time to ensure ordered differencing
    sort_idx = np.argsort(band_table['time'])
    t_sorted = band_table['time'].data[sort_idx]
    f_sorted = band_table['flux'].data[sort_idx]

    dt = np.diff(t_sorted)
    gaps = [(t_sorted[i], t_sorted[i+1], f_sorted[i], f_sorted[i+1] ) for i, d in enumerate(dt) if d > min_gap]
    return gaps


def screen_model_gaps(data_table, model,
                      tmin, tmax,
                      min_gap=5,
                      rel_dev_threshold=0.5,
                      evo_dev_threshold=10,
                      bands=None):
    """
    Main entry point. Astropy table with 'time', 'band', 'flux' columns.
    Band-outer, gap-inner loop: assess each band's gaps independently.


    """
    if bands is None:
        bands = np.unique(data_table['band'])

    flagged = []

    for b in bands:
#        print('...', b)


        # Select data for this band
        mask = data_table['band'] == b
        if np.sum(mask) < 2:
            continue  # Need at least 2 points to define a gap

        gaps = find_band_gaps(data_table[mask], min_gap)


        try:
            for t_start, t_end, f_start, f_end in gaps:
                results, t_eval, flux = evaluate_gap_physicality(
                    model, t_start, t_end, f_start, f_end, b  # single band
                )
                if ( results['relative_deviation'] > rel_dev_threshold or results['evolution_deviation'] > evo_dev_threshold):
    #                print('... flagging')
                    flagged.append({
                        't_start': t_start,
                        't_end': t_end,
                        'gap_width': t_end - t_start,
                        'band': b,
                        'evolution_deviation': results['evolution_deviation'],
                        'max_rel_deviation': results['relative_deviation'],
                        'deviation_sign': results['deviation_sign'],
                        'non_monotonic': results['non_monotonic']
                    })


            # Add gap checks at start, getting flux limits from the model.
            # Only do for g and r (often very little i data), and only when we lack data close to edges.
            if b in ['ztfg','ztfr']:
                if (data_table['time'][mask][0]-tmin)>min_gap:
                    results, t_eval, flux = evaluate_gap_physicality(
                        model, tmin, data_table['time'][mask][0],
                        model.bandflux(b, tmin, zp=25, zpsys='ab'), data_table['flux'][mask][0], b  # single band
                    )
                    if ( results['relative_deviation'] > rel_dev_threshold or results['flux_range'][1]>1.5*data_table['flux'][mask][0]
                    ):
                        flagged.append({
                            't_start': tmin,
                            't_end': data_table['time'][mask][0],
                            'band': b,
                            'evolution_deviation': results['evolution_deviation'],
                            'max_rel_deviation': results['relative_deviation'],
                            'deviation_sign': results['deviation_sign'],
                            'non_monotonic': results['non_monotonic'],
                            'preflux_scale':  results['flux_range'][1] / data_table['flux'][mask][0]
                        })
        except ValueError:
            # Band not covered by model wavelength range. So nothing to flag (model wont fly though...)
            pass



    return flagged

# ─── Photometry retrieval ────────────────────────────────────────────────────

def get_ztftable_from_ampel(
    ztfid: str,
    dbhandle,
    include_sigma: float = 5.0,
    **kwargs,
):
    """
    Retrieve ZTF photometry from AMPEL database with outlier rejection.

    Parameters
    ----------
    ztfid : str
        ZTF identifier, e.g. 'ZTF18aaayemw'
    dbhandle : pymongo.database.Database
        AMPEL MongoDB handle
    include_sigma : float
        Sigma threshold for outlier rejection
    **kwargs
        Additional metadata for table

    Returns
    -------
    astropy.table.Table
        Sorted photometry table
    """
    tabulators = [ZTFFPTabulator(inclusion_sigma=include_sigma)]
    tab = get_db_table(ztfid, database=dbhandle, tabulators=tabulators)
    tab.sort("time")
    tab.meta = {"object_id": ztfid, **kwargs}
    return tab


def get_db_table(name, database, tabulators):
    """Retrieve photopoints and convert to table."""
    if isinstance(name, int):
        stock = name
    elif re.search("ZTF", name):
        stock = ZTFIdMapper.to_ampel_id(name)
    else:
        raise ValueError(f"Cannot parse {name}")

    dps = list(database.t0.find({"stock": stock}))
    ftables = [tabulator.get_flux_table(dps) for tabulator in tabulators]

    if len(ftables) > 1:
        raise NotImplementedError("Multiple tabulators not supported")
    return ftables[0]


def deredden_flux_table(table: Table, A_V: float, R_V: float = 3.1):
    """Correct photometry table for Milky Way extinction."""
    result = table.copy()
    dust = sncosmo.CCM89Dust()
    dust.set(ebv=A_V / R_V)

    corrections = {}
    for band in np.unique(result["band"]):
        bp = sncosmo.get_bandpass(band)
        trans = dust.propagate(bp.wave, np.ones_like(bp.wave))
        T_eff = np.trapezoid(trans * bp.trans, bp.wave) / np.trapezoid(bp.trans, bp.wave)
        corrections[band] = 1.0 / T_eff

    factors = np.array([corrections[b] for b in result["band"]])
    result["flux"] *= factors
    result["fluxerr"] *= factors
    return result


def truncate_after_gap(table, max_sep, time_col='time'):
    """
    Sort by time, find first gap > max_sep days, keep everything before it.
    """
    t = table[time_col]

    # Sort and track original indices
    sort_idx = np.argsort(t)
    t_sorted = t[sort_idx]

    # Find gaps
    gaps = np.diff(t_sorted)
    gap_idx = np.where(gaps > max_sep)[0]

    if len(gap_idx) == 0:
        return table  # No large gap found

    # Keep up to and including the point before the first gap
    cutoff = gap_idx[0] + 1  # +1 because np.diff reduces length by 1
    keep_idx = sort_idx[:cutoff]

#    print('... truncating after gap of {:.2f} days at day {}, keeping {} points'.format(
#        gaps[gap_idx[0]], t_sorted[gap_idx[0]]-t_sorted[0], len(keep_idx)
#    ))

    return table[np.sort(keep_idx)]  # Restore original order if desired


# ─── Probability utilities ───────────────────────────────────────────────────

def apply_floor_and_normalize(p, floor):
    """Apply floor constraint and renormalize probabilities."""
    p = np.array(p, dtype=float)
    n = len(p)

    if floor * n > 1:
        raise ValueError("Floor too large to maintain sum=1")

    result = np.full(n, floor)
    remaining = 1 - floor * n
    excess = np.maximum(p - floor, 0)

    if excess.sum() > 0:
        result += remaining * (excess / excess.sum())
    else:
        result += remaining / n

    return result


# ─── Key pruning ─────────────────────────────────────────────────────────────

KEYS_TO_CUT_ROW = [
    "Index", "chisq", "ndof", "_10", "earlydet", "peakbands",
    "presum", "postsum", "thendet", "postdet", "aic",
    "peakchisqdof", "peakaic", "goodfit", "wresult",
]

KEYS_TO_CUT_MDICT = [
    "dps_init", "t0", "amplitude", "hostebv", "hostr_v",
    "dps_fcut", "dps_tcut", "dps_allcut", "success", "chisq",
    "ndof", "errors", "chidof", "absmag", "corrdata",
]


def remove_keys(obj, keys_to_remove):
    """Recursively remove specified keys from nested dict/list structure."""
    if isinstance(obj, dict):
        return {
            k: remove_keys(v, keys_to_remove)
            for k, v in obj.items()
            if k not in keys_to_remove
        }
    elif isinstance(obj, list):
        return [remove_keys(item, keys_to_remove) for item in obj]
    else:
        return obj


# ─── Main processing ─────────────────────────────────────────────────────────

def process_single_sn(
    id_value: str,
    group: pd.DataFrame,
    df_bts: pd.DataFrame,
    db,
    fit_host_dust: bool,
    close_templates: list,
    template_count: int,
    min_draw_prob: float,
    max_chi_dof: float,
    good_warpfit_sf: float,
    max_phases: Optional[List[float]] = None,
):
    """
    Fit warped templates for single SN, returning list of successful warp fits.
    """
    av = df_bts.loc[df_bts["ZTFID"] == id_value].iloc[0]["A_V"]

    # Priority: correct+good > correct > rest, then by chidof
    ordered = (
        group.assign(
            priority=(
                ((group["correct"]) & (group["goodfit"])) * 2 +
                ((group["correct"]) & (~group["goodfit"])) * 1
            )
        )
        .sort_values(["priority", "chidof"], ascending=[False, True])
    )

    sn_warplist = []
    goodfits = 0
    chicomp = {"prechi": [], "postsf": []}

    for rowi, row in ordered.iterrows():
        row = dict( row )

        # Skip SALT models for warping
        if re.search("salt", row["model"]):
            continue

        # Retrieve and prepare photometry
        tab = get_ztftable_from_ampel(
            row["id"],
            db,
            redshift=float(row["z"]),
            include_sigma=5,
            type=row["class"],
        )
        # Remove points after first gap > 20 days
        tab =  truncate_after_gap(tab, 20)


        tab = deredden_flux_table(tab, av, R_V=3.1)
        tab["fluxerr"] = np.sqrt(
            (ERRFLOOR * np.mean(tab["flux"])) ** 2 + tab["fluxerr"] ** 2
        )
        tab["fluxerr"] = np.sqrt(
            (FLUX_FRAC_DISPERSION * tab["flux"]) ** 2 + tab["fluxerr"] ** 2
        )

        # Get warp correction
        mdict = get_template_correction(
            tab,
            row["model"],
            z=float(row["z"]),
            fit_host_dust=fit_host_dust,
            pull_cut=999,     # Disable this for now
            max_phases=max_phases,
            plot_dir=None,
#            spline_lam=1,
            require_phasecoverage=False,
        )

        # Build and fit warped model
        wm = get_warpedTimeSeriesModel(
            name=f"{row['id']}_{row['model']}",
            original_template_name=row["model"],
            warpdata=mdict,
            z=float(row["z"]),
            use_host_dust=False,      # Also when fitting with dust above, this should have been absorbed into the warp correction
            original_template_version=None,
        )
        if wm is None:
            print(f"Failed to create warped model for {row['id']} with {row['model']}")
            continue

        # We now evaluate how well the warped model match the data
        # To make the comparison fair we only use times that were used in the warpfit (i.e. the phase range used to fit the warp)
        # NOTE: This might not be the ideal choice - one could also check how well the model
        # works outside this bounds. But we have not adapted it for this...
        fitted_time_mask  = (tab["time"] >= mdict['warpfit_tmin']) & (tab["time"] <= mdict['warpfit_tmax'])

        try:
            wresult, wfitted_model = sncosmo.fit_lc(tab[fitted_time_mask], wm, FITPROP)
        except RuntimeError:
            continue

        # Phase coverage checks
        # For a real check, should really limit to the range which was used when constructing
        # fit
        t0 = wresult["parameters"][1]
        # The fitted peak needs to be within the bounds of the data used for the fit
        if not (tab["time"][fitted_time_mask].min() <= t0 <= tab["time"][fitted_time_mask].max()):
            continue
        # The first datapoint should not be way earlier than the template starting phase
        # Note that we use all data for this
        if (tab["time"].min() - t0) < (wm.source.minphase() - 3):
            continue
        if (tab["time"][fitted_time_mask].min() - t0) > 5:
            continue
        if (tab["time"][fitted_time_mask].max() - tab['time'][fitted_time_mask].min()) < 20:
            continue

        # Assess fit quality.
        row["sf"] = chi2.sf(wresult["chisq"], wresult["ndof"])
        chicomp["prechi"].append(row["chidof"])
        chicomp["postsf"].append(row["sf"])

        row["mdict"] = mdict
        row["wresult"] = wresult


        # Make inspection plot - later parameterize
        import matplotlib.pyplot as plt
        plotname = '{}_{}_{}_{}_{:.2}_{:.2}.png'.format(
                row['id'],
                row["sf"] > min_draw_prob,
                row["sf"] > good_warpfit_sf,
                row["model"],
                wresult["chisq"] / wresult["ndof"], row["sf"]
            )

        # Evaluate fit quality - three cateogiry, default bad
        fiteval = 'poor'
        fout = os.path.join('/Users/jnordin/tmp/wmod/bad',plotname)
        if screen_model_gaps(
                tab[fitted_time_mask], wfitted_model,
                mdict['warpfit_tmin'], mdict['warpfit_tmax'],
                min_gap=5, rel_dev_threshold=0.5, evo_dev_threshold=100):
            fiteval = 'var'
            fout = os.path.join('/Users/jnordin/tmp/wmod/var',plotname)
        elif row["sf"] > 10**-99 and (wresult["chisq"] / wresult["ndof"]) < 8:
            fiteval = 'good'
            fout = os.path.join('/Users/jnordin/tmp/wmod/good',plotname)

        # Finish the plot
        fig = sncosmo.plot_lc(tab, model=wfitted_model, errors=wresult.errors)
        for ax in fig.axes:
            ax.axvline(x=mdict['warpfit_tmin']-t0, color='red', linestyle='--', alpha=0.7, label='peak')
            # or multiple lines
            ax.axvline(x=mdict['warpfit_tmax']-t0, color='blue', linestyle=':', alpha=0.5)
        plt.savefig(fout)
        plt.close()

        if fiteval in ['poor', 'var']:
            print('... reject fit', fiteval, row["sf"], wresult["chisq"] / wresult["ndof"])
            continue

#        if row["sf"] < min_draw_prob or (wresult["chisq"] / wresult["ndof"])>max_chi_dof:
#            print('... fit sf below min_draw_prob, skipping ', row["sf"], wresult["chisq"] / wresult["ndof"])
#            continue
        if row["sf"] > good_warpfit_sf:
            goodfits += 1

        # Assign quality tier
        if row["sf"] > good_warpfit_sf and row["priority"] == 2:
            row["quality"] = "gold"
        elif row["sf"] > good_warpfit_sf and row["priority"] == 1:
            row["quality"] = "silver"
        else:
            row["quality"] = "bronze"

        sn_warplist.append(row)


        if goodfits >= template_count:
            print("... reached target!")
            break

    if not sn_warplist:
        print(f"No template fits for {id_value}")
        return None

    print(len(sn_warplist), len(ordered))

    # Normalize drawing probabilities
    drawprobs = apply_floor_and_normalize(
        [modfit["sf"] for modfit in sn_warplist], min_draw_prob
    )
    for k, prob in enumerate(drawprobs):
        sn_warplist[k]["draw_prob"] = prob

    return sn_warplist


def main():
    args = parse_args()

    # Load data
    df_bts = load_bts_data(args.bts_file)
    classlist = df_bts[f"type_{args.classwidth}"].unique()
    print('Available classes:', classlist   )

    class_name = classlist[args.cid]

    print(f"Target class {class_name} from category type_{args.classwidth} (index {args.cid})")

    process_classes = get_process_classes(df_bts, args.classwidth, class_name)
    print("Will combine files for narrow classes:", process_classes)

    close_templates = [
        key for key, val in TEMPLATE_CLOSE_TYPES.items() if class_name in val
    ]
    print("Close template classes:", close_templates)

    # Determine max phases for this class if specified
    if class_name in MAX_PHASES and isinstance(MAX_PHASES[class_name], dict):
        max_phases = MAX_PHASES[class_name].get(args.classwidth, None)
    else:
        max_phases = None
    if max_phases is not None:
        print('... class specific phase limit', max_phases)

    # Load fits
    typefitdata = load_typefit_data(process_classes, args.fdir, version=int(args.version))
    if typefitdata is None:
        raise RuntimeError("No fit data loaded")

    # Load OpenUniverse templates and register them with sncosmo
    register_all()

    # Filter fits by quality
    modelfits = {}
    for modelname, modeldata in typefitdata.items():
        if re.search("salt", modelname):
            df = get_salt_cosmofit(
                modeldata,
                truetypes=close_templates,
                peakfit=True,
                chidofmax=CHIDOF_MAX,
            )
        else:
            df = get_timeseries_goodfit(
                modeldata,
                truetypes=close_templates,
                peakfit=True,
                chidofmax=CHIDOF_MAX,
            )
        df.insert(0, "model", modelname)
        modelfits[modelname] = df

    dfall = pd.concat(modelfits.values(), ignore_index=True)

    print(
        f"Total pairs: {dfall.shape[0]}, "
        f"correct type: {sum(dfall['correct'])}, "
        f"good fit: {sum(dfall['goodfit'])}, "
        f"both: {sum(dfall['correct'] & dfall['goodfit'])}"
    )

    dfall = dfall.loc[dfall["goodfit"]]
    print(f"After goodfit cut: {dfall.shape[0]}")

    nbr_sn = len( dfall["id"].unique() )
    print(f"Unique SN IDs: {nbr_sn}")

    # Database connection
    client = pymongo.MongoClient()
    db = getattr(client, args.db_name)

    # Main warp fitting loop
    full_warplist = {}

    for i, (id_value, group) in enumerate(dfall.groupby("id")):
        print(i, id_value)
        result = process_single_sn(
            id_value=id_value,
            group=group,
            df_bts=df_bts,
            db=db,
            fit_host_dust=args.fit_host_dust,
            close_templates=close_templates,
            max_phases=args.max_phases if hasattr(args, 'max_phases') else None,
            template_count=TEMPLATE_COUNT,
            min_draw_prob=MIN_DRAW_PROB,
            max_chi_dof=CHIDOF_MAX,
            good_warpfit_sf=GOOD_WARPFIT_SF,
        )
        if result is not None:
            full_warplist[id_value] = result
#        if i>1:
#            break

    # Quality accounting
    accounting = []
    for sn, wdata in full_warplist.items():
        account = sum(
            3 if m["quality"] == "gold" else 2 if m["quality"] == "silver" else 1
            for m in wdata
        )
        accounting.append(account)

    # Prune and save
    full_warplist = remove_keys(full_warplist, KEYS_TO_CUT_ROW)
    full_warplist = remove_keys(full_warplist, KEYS_TO_CUT_MDICT)

    safe_name = re.sub(r"/", "", class_name)
    storefile = Path(args.outdir) / f"warpcoeffs_v{args.version}_{safe_name}.pkl"

    storefile.parent.mkdir(parents=True, exist_ok=True)
    with open(storefile, "wb") as file:
        pickle.dump(full_warplist, file)

    print(f"Saved {len(full_warplist)} out of {nbr_sn} SN warp sets to {storefile}")


if __name__ == "__main__":
    main()
