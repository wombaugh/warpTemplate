#!/usr/bin/env python
# coding: utf-8
"""
Extract per-SN warp template coefficients from stage-1 sncosmo fit summaries.
 
For one taxonomy class (narrow, extended, wide, or "all" -- see
--classwidth), this script:
 
  1. Loads every narrow class's per-model fit results (the
     btsfits{version}_*.json files produced by sample_sncosmo_from_db.py)
     that belongs under the requested class, and keeps only fits flagged
     as a "correct" template type and a "good" chi-square/dof.
  2. For each SN with at least one surviving fit, re-fetches its light
     curve, dereddens it for Milky Way extinction, and fits a *warped*
     version of each candidate template (warptemplate.get_template_correction
     + warptemplate.get_warpedTimeSeriesModel) -- a per-object correction
     surface letting one base template reproduce that specific SN's actual
     color and shape evolution, not just its overall light curve.
  3. Screens each resulting warp fit for basic physical sanity (phase
     coverage around the fitted peak, unphysical behavior across data
     gaps) and assigns it a gold/silver/bronze quality tier.
  4. Collects every SN's surviving warp fits, each with a normalized draw
     probability, into one pickle file consumed by
     warptemplate.WarpfitTemplateLoader for later template generation.
 
Data sources: standard classes read their SN list, redshifts, and
coordinates from --bts-file (via warptemplate.add_warpclasses); SLSN-I,
SLSN-II, and TDE instead read a combined literature-plus-BTS catalog
(--alt-csv). Both are merged into one working dataframe up front (see
load_combined_bts_data) so that any --classwidth/--cid selection --
including combined classes that mix standard and alt-source narrow
classes, such as 'SN CC (a)', which includes SLSN -- resolves consistently
regardless of which source a given SN's data actually comes from. Milky
Way A_V is computed the same way (via the SFD dust map, see
get_mw_extinction_av) for every object regardless of source, so
dereddening isn't split across two different conventions depending on
which catalog an SN happens to be in. Photometry itself still comes from
two separate MongoDB databases (--db-name for standard objects,
--alt-mongodb for alt-source ones), selected per-SN rather than per-run.
 
Usage:
    python templatecreation_II_class2warpset.py --classwidth w --cid 3
    python templatecreation_II_class2warpset.py --cw n --cid 8 --version 5
 
See the accompanying extract_warp_coeffs.md for full architecture notes,
a function-by-function reference, the output pickle schema, and known
caveats.
"""

import argparse
import json
import pickle
import re, os
import sys
import warnings
from pathlib import Path
from typing import Optional, List


import numpy as np
import pandas as pd
import pymongo
import seaborn as sns
import sncosmo
from astropy.cosmology import Planck13 as cosmo
from astropy.coordinates import SkyCoord
from astropy.table import Table
from iminuit.util import IMinuitWarning
from scipy.stats.distributions import chi2
import sfdmap


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
DEFAULT_ALT_CSV = "/Users/jnordin/data/ztf/dr4/dr4_slsntde_coordlist.csv"
DEFAULT_ALT_MONGODB = "bts_ipacfp_strictbase_slsntns"

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


# ─── SLSN / TDE alternate catalog (matching the first pipeline stage) ───────

ALT_SOURCE_CLASSES = {'SLSN-I', 'SLSN-II', 'TDE'}

_TDE_TYPE_VARIANTS = {
    'TDE', 'TDE-H-He', 'TDE-He', 'TDE-featureless', 'TDE-H+He', 'TDE-H+He?',
}

# Official narrow -> extended -> wide -> all class mapping (duplicated from
# the analysis scripts' compare_data_template_colors_v6.py), needed to give
# the alt-csv rows type_e/type_w/type_a without going through
# add_warpclasses(), which doesn't know about them.
WARP_MAP_EXTENDED = {
    "SN Ia-91bg": "SN Ia-91bg (e)", "SN IIn": "SN IIn (e)", "SN IIb": "SN Ib/c (e)",
    "SN Ia-CSM": "SN Ia-pec (e)", "SN Ibn": "SN Ibn (e)", "SN Ia-SC": "SN Ia-pec (e)",
    "SN Ib": "SN Ib/c (e)", "SLSN-II": "SLSN (e)", "SN Iax": "SN Ia-pec (e)",
    "SN Ia-91T": "SN Ia-91T (e)", "SLSN-I": "SLSN (e)", "SN Ic": "SN Ib/c (e)",
    "SN Ia-pec": "SN Ia-pec (e)", "SN IIP": "SN II (e)", "SN Ic-BL": "SN Ib/c (e)",
    "SN Ia": "SN Ia (e)", "SN II": "SN II (e)", "SN Ib/c": "SN Ib/c (e)",
    "TDE": "TDE (e)",
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
    "SN Ia-pec (w)": "SN Ia (a)", "SN II (w)": "SN CC (a)", "SN Ia-91bg (w)": "SN Ia (a)",
    "TDE (w)": "TDE (a)",
}


def _infer_target_class(raw_type: str, source: str):
    """Resolve one alt-csv row's (type, source) onto 'TDE', 'SLSN-I',
    'SLSN-II', or None. Direct type matches (including known TDE
    sub-classification variants) are trusted first; anything else -- an
    uninformative type ('Unknown') or a contested/stale one ('Ic') -- falls
    back to `source`, which is treated as authoritative (list membership
    wins over a specific type label). Identical to the first pipeline
    stage's version."""
    t = str(raw_type).strip()

    if t in _TDE_TYPE_VARIANTS:
        return 'TDE'
    if t == 'SLSN-I':
        return 'SLSN-I'
    if t in ('SLSN-II', 'SLSNII'):
        return 'SLSN-II'

    s = str(source).lower()
    if 'tde' in s:
        return 'TDE'
    if 'slsnii' in s or 'slsn-ii' in s:
        return 'SLSN-II'
    if 'slsn' in s:
        return 'SLSN-I'

    print(f"NOTE: could not resolve type='{t}', source='{source}'; dropping row.")
    return None


def get_mw_extinction_av(row, allow_missing=False, R_V=3.1):
    """
    Get Milky Way extinction A_V for a candidate based on coordinate
    information from the table. We assume this has been converted to
        RAdeg/Decdeg (already in deg)
    If not present, return None unless allow_missing is True, in which case return 0.0.

    Copied verbatim from the first pipeline stage so both stages use
    exactly the same convention for every object, standard or alt-source
    alike -- this is what replaces the old class-correlated split where
    standard objects used BTS's own (NED-based) A_V and alt-source objects
    defaulted to 0.0.
    """
    if not allow_missing and 'RAdeg' not in row and 'Decdeg' not in row:
        raise ValueError("Row does not contain RAdeg and Decdeg columns for Milky Way extinction lookup.")
    elif allow_missing and 'RAdeg' not in row and 'Decdeg' not in row:
        print("Row does not contain RAdeg and Decdeg columns for Milky Way extinction lookup. Returning A_V=0.0.")
        return 0.0

    return sfdmap.SFDMap().ebv(row['RAdeg'], row['Decdeg']) * R_V


def _load_alt_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    """Load and resolve the WHOLE SLSN/TDE catalog (all three classes at
    once, unlike a per-class loader -- this stage needs every alt-source
    class available up front, before it knows which narrow classes a given
    --classwidth/--cid run will touch).

    A_V/peakmag are no longer load-bearing here (see get_mw_extinction_av,
    called uniformly in process_single_sn) -- filled with NaN if absent
    purely so the sanity-check comparison print doesn't crash on a missing
    column.
    """
    print(f"Loading SLSN/TDE catalog from alternate source: {args.alt_csv}")
    df_alt = pd.read_csv(args.alt_csv, index_col=0)

    resolved = df_alt.apply(lambda row: _infer_target_class(row['type'], row['source']), axis=1)
    n_dropped = resolved.isna().sum()
    if n_dropped:
        dropped_types = sorted(df_alt.loc[resolved.isna(), 'type'].unique())
        print(f"NOTE: {n_dropped} rows in {args.alt_csv} did not resolve to "
              f"TDE/SLSN-I/SLSN-II and were dropped (raw type(s): {dropped_types}).")

    df = df_alt.loc[resolved.notna()].copy()
    df['type_n'] = resolved.loc[resolved.notna()]

    if df['redshift'].dtype == object:
        df = df[df['redshift'] != '-']
    df['redshift'] = pd.to_numeric(df['redshift'])

    if 'A_V' not in df.columns:
        df['A_V'] = np.nan
    if 'peakmag' not in df.columns:
        df['peakmag'] = np.nan

    df['type_e'] = df['type_n'].map(WARP_MAP_EXTENDED)
    df['type_w'] = df['type_e'].map(lambda e: WARP_MAP_WIDE.get(e, e))
    df['type_a'] = df['type_w'].map(lambda w: WARP_MAP_ALL.get(w, w))

    print(f"  Resolved {len(df)} rows: {df['type_n'].value_counts().to_dict()}")
    return df


# ─── Data loading ────────────────────────────────────────────────────────────

def load_bts_data(args: argparse.Namespace) -> pd.DataFrame:
    """Load standard BTS explorer data, add warp classes, and compute
    RAdeg/Decdeg from RA/Dec hour/deg strings (matching the first stage)."""
    df_bts = pd.read_csv(args.bts_file)
    df_bts = add_warpclasses(df_bts, purge=True)

    c = SkyCoord(df_bts['RA'], df_bts['Dec'], unit=("hour", "deg"))
    df_bts['RAdeg'] = c.ra.deg
    df_bts['Decdeg'] = c.dec.deg

    return df_bts


def load_combined_bts_data(args: argparse.Namespace) -> pd.DataFrame:
    """Standard BTS data plus the SLSN/TDE alternate catalog, concatenated
    into one dataframe covering every narrow class this pipeline knows
    about -- so classlist/process_classes (driven by --classwidth) work
    the same regardless of which source a narrow class's data comes from.
    """
    df_standard = load_bts_data(args)
    df_alt = _load_alt_dataframe(args)

    required_cols = ['ZTFID', 'redshift', 'RAdeg', 'Decdeg',
                      'type_n', 'type_e', 'type_w', 'type_a']
    missing_std = [c for c in required_cols if c not in df_standard.columns]
    missing_alt = [c for c in required_cols if c not in df_alt.columns]
    if missing_std or missing_alt:
        raise ValueError(
            f"Cannot combine standard/alt BTS dataframes -- missing required "
            f"columns (standard missing: {missing_std}, alt missing: {missing_alt})"
        )

    df = pd.concat([df_standard, df_alt], ignore_index=True, sort=False)
    print(f"Combined dataframe: {len(df_standard)} standard + {len(df_alt)} "
          f"alt-source rows = {len(df)} total")
    return df


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
    db_standard,
    db_alt,
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
    bts_row = df_bts.loc[df_bts["ZTFID"] == id_value].iloc[0]

    # Milky Way A_V is always computed via the SFD dust map, for every
    # object regardless of source (matching the first stage's finalized
    # approach) -- whatever A_V the source catalog itself provides is only
    # a sanity-check comparison, never used for the actual dereddening.
    av = get_mw_extinction_av(bts_row)
    if 'A_V' in bts_row and not pd.isna(bts_row['A_V']):
        print(f"NOTE: catalog A_V={bts_row['A_V']:.3f} for {id_value}, computed A_V={av:.3f}.")
        if not np.isclose(av, bts_row['A_V'], rtol=0.1):
            print(f"NOTE: computed A_V={av:.3f} differs from catalog A_V={bts_row['A_V']:.3f} "
                  f"for {id_value}. Using computed value.")

    # A single --classwidth/--cid run can mix standard and alt-source narrow
    # classes (e.g. 'SN CC (a)' includes SLSN per WARP_MAP_ALL), so which
    # database to query is decided per-SN here, from the same row already
    # used for A_V, rather than once for the whole run.
    db = db_alt if bts_row["type_n"] in ALT_SOURCE_CLASSES else db_standard

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
        try:
            wm = get_warpedTimeSeriesModel(
                name=f"{row['id']}_{row['model']}",
                original_template_name=row["model"],
                warpdata=mdict,
                z=float(row["z"]),
                use_host_dust=False,      # Also when fitting with dust above, this should have been absorbed into the warp correction
                original_template_version=None,
            )
        except ValueError as e:
            print(f"Failed to create warped model for {row['id']} with {row['model']}: {e}")
            print('... skipping and continuing')
            continue
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
        if (varflag:=screen_model_gaps(
                tab[fitted_time_mask], wfitted_model, 
                mdict['warpfit_tmin'], mdict['warpfit_tmax'],
                min_gap=5, rel_dev_threshold=0.5, evo_dev_threshold=100)):
            fiteval = 'var'
            fout = os.path.join('/Users/jnordin/tmp/wmod/var',plotname)
        elif row["sf"] > 10**-99 and (wresult["chisq"] / wresult["ndof"]) < 8:
            fiteval = 'good'
            fout = os.path.join('/Users/jnordin/tmp/wmod/good',plotname)

        # Finish the plot
        try:
            fig = sncosmo.plot_lc(tab, model=wfitted_model, errors=wresult.errors)
            for ax in fig.axes:
                ax.axvline(x=mdict['warpfit_tmin']-t0, color='red', linestyle='--', alpha=0.7, label='peak')
                # or multiple lines
                ax.axvline(x=mdict['warpfit_tmax']-t0, color='blue', linestyle=':', alpha=0.5)
            plt.savefig(fout)
            plt.close()
        except:
            print('XXXX ... failed to make plot', fout)
            print('... tab', tab)
            print('... wfitted_model', wfitted_model)
            print('... wresult', wresult)
            raise ValueError('Failed to make plot')

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


# ─── Argument parsing ────────────────────────────────────────────────────────

def parse_args():
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
        default=os.environ.get('VERSION', 5),
        help='Version string for input and output files (default: $VERSION or 5)'
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
    parser.add_argument(
        "--alt-csv", default=os.environ.get('ALT_CSV', DEFAULT_ALT_CSV),
        help="Combined SLSN/TDE catalog CSV (ZTFID, type, redshift, RAdeg, "
             "Decdeg, source), same file the first pipeline stage uses"
    )
    parser.add_argument(
        "--alt-mongodb", default=os.environ.get('ALT_MONGODB', DEFAULT_ALT_MONGODB),
        help="MongoDB database for SLSN-I, SLSN-II, and TDE photometry"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load data -- standard BTS file plus the SLSN/TDE alternate catalog,
    # combined into one dataframe covering every narrow class (see
    # load_combined_bts_data() and the module docstring).
    df_bts = load_combined_bts_data(args)
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

    # Database connections -- standard and alt-source SLSN/TDE photometry
    # live in different databases; process_single_sn() picks per-SN.
    client = pymongo.MongoClient()
    db_standard = getattr(client, args.db_name)
    db_alt = getattr(client, args.alt_mongodb)

    # Main warp fitting loop
    full_warplist = {}

    for i, (id_value, group) in enumerate(dfall.groupby("id")):
        print(i, id_value)
        result = process_single_sn(
            id_value=id_value,
            group=group,
            df_bts=df_bts,
            db_standard=db_standard,
            db_alt=db_alt,
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
