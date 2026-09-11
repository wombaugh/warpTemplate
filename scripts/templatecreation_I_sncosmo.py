#!/usr/bin/env python
# coding: utf-8

"""
Redoing of sample_sncosmo_from_db using the more consistent class definition.
Pick one non-SN Ia narrow class, fit sncosmo models, store results.

------------------------------------------------------------------------------
CHANGES: SLSN / TDE from a combined alternate catalog
------------------------------------------------------------------------------
SLSN-I, SLSN-II, and (newly added) TDE now read their SN list + redshifts
from a single combined catalog CSV (--alt-csv) spanning several source
surveys (BTS, plus dedicated TDE and SLSN literature samples: TDE_Yao23,
SLSNI_Chen23, SLSNII_Pessi25), and their photometry from a separate MongoDB
database (--alt-mongodb). Everything else is unchanged.

Class resolution (see _infer_target_class): most rows resolve directly from
the catalog's own `type` column, normalizing known TDE sub-classification
variants (TDE-H-He, TDE-He, TDE-featureless, TDE-H+He, TDE-H+He?) onto the
base 'TDE' class, and 'SLSNII' (no hyphen, used by the Pessi25 sample) onto
'SLSN-II'. Any other type value -- uninformative ('Unknown') or contested/
stale ('Ic', 2 rows within the SLSNI_Chen23 sample) -- falls back to the
`source` column instead: each literature sample is single-class by
construction, and being listed there at all is treated as authoritative
over the specific type label. Verified against every row: 85 TDE /
160 SLSN-I / 135 SLSN-II resolved (the 2 'Ic' rows now count as SLSN-I via
their SLSNI_Chen23 source), 0 dropped.

All alt-source rows -- BTS-sourced and literature-sourced (TDE_Yao23,
SLSNI_Chen23, SLSNII_Pessi25) alike -- are confirmed to live in the same
--alt-mongodb, so get_class_database() doesn't need per-`source` routing.

The catalog has no A_V or peakmag columns -- filled with 0.0 / NaN with a
printed warning. It DOES have RAdeg/Decdeg, so if you want real Milky Way
extinction instead of defaulting A_V to 0, a dust-map lookup (e.g. `sfdmap`
or `dustmaps`) could be added using those coordinates -- not done here since
it's a new dependency, not something the catalog itself specifies.

Redshift limits (zclass, near the bottom of main()): the real per-class
ranges in this catalog are TDE [0.011, 0.519], SLSN-I [0.039, 0.670],
SLSN-II [0.018, 0.4846] -- notably, the *previous* SLSN-I/SLSN-II zclass
entries here were [0.0, 0.3], which would silently exclude roughly half the
SLSN-I sample (the Chen23 literature objects push well past z=0.3) and a
meaningful chunk of SLSN-II (Pessi25 extends to z=0.48). Updated below to
comfortably cover the observed range with a little padding -- but that's a
data-availability choice, not necessarily whatever science reason motivated
the original 0.3 cutoff (e.g. GP peak-color reliability at high z); adjust
if you had a specific reason for the narrower limit.

STILL UNRESOLVED, flagged rather than guessed:
- Whether --classfile has any rows tagged Type=='TDE'. If not, the TDE
  fitting loop will only ever attempt salt2/salt3 (SN Ia templates, wrong
  for a TDE) -- a runtime warning fires if none are found, but you'll still
  need to add TDE template rows yourself.
------------------------------------------------------------------------------
"""

import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
import pymongo
import sncosmo
from datetime import datetime
from astropy.cosmology import Planck13 as cosmo
from sncosmo.fitting import DataQualityError
from ampel.ztf.util.ZTFIdMapper import ZTFIdMapper
from ampel.ztf.view.ZTFFPTabulator import ZTFFPTabulator
import sfdmap 

from warptemplate import add_warpclasses, SN_REJECT, estimate_peak_flux_multiband, get_peak_colors, register_all


# Sncosmo fitting can throw a lot of warnings, typically when skipping a band due to datapoints
# or wavelength coverage.
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
#    module="sncosmo.fitting"
)


# -----------------------------------------------------------------------------
# Classes sourced from the alternate catalog/database instead of --bts-csv/--mongodb
# -----------------------------------------------------------------------------
ALT_SOURCE_CLASSES = {'SLSN-I', 'SLSN-II', 'TDE'}

# Known TDE sub-classification labels in --alt-csv's `type` column that all
# fold into the base 'TDE' class for fitting purposes.
_TDE_TYPE_VARIANTS = {
    'TDE', 'TDE-H-He', 'TDE-He', 'TDE-featureless', 'TDE-H+He', 'TDE-H+He?',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fit sncosmo models to a specific narrow transient class.'
    )
    parser.add_argument(
        'classid',
        type=int,
        choices=range(18),
        metavar='CLASSID',
        help='Class index 0-17'
    )
    parser.add_argument(
        '--bts-csv',
        default=os.environ.get('BTS_CSV', '/Users/jnordin/data/ztf/bts/bts_explorer_260601.csv'),
        help='BTS explorer CSV path (default: $BTS_CSV or /Users/jnordin/...)'
    )
    parser.add_argument(
        '--classfile',
        default=os.environ.get('CLASSFILE', '/Users/jnordin/data/models/sncosmo/sncosmo_timeseriesmodels.csv'),
        help='SNCosmo model class file (default: $CLASSFILE or /Users/jnordin/...)'
    )
    parser.add_argument(
        '--outdir',
        default=os.environ.get('OUTDIR', '/Users/jnordin/data/models/sncosmo/'),
        help='Output directory for results and plots (default: $OUTDIR or /Users/jnordin/...)'
    )
    parser.add_argument(
        '--plotdir',
        default=os.environ.get('PLOTDIR', '/Users/jnordin/tmp/sncosmosample/'),
        help='Directory for diagnostic plots, or "none" to disable (default: $PLOTDIR or /Users/jnordin/...)'
    )
    parser.add_argument(
        '--ampel-path',
        default=os.environ.get('AMPEL_PATH', '/Users/jnordin/github/ampelFeb25'),
        help='Path to ampel repository for sys.path (default: $AMPEL_PATH or /Users/jnordin/...)'
    )
    parser.add_argument(
        '--version', '-v',
        default=os.environ.get('VERSION', 'v5'),
        help='Version string for output files and log (default: $VERSION or v5)'
    )
    parser.add_argument(
        '--mongodb',
        default=os.environ.get('MONGODB', 'bts_ipacfp_strictbase_train_jul26'),
        help='MongoDB database name (default: $MONGODB or bts_ipacfp_strictbase_train_jul26)'
    )
    parser.add_argument(
        '--alt-csv',
        default=os.environ.get('ALT_CSV', '/Users/jnordin/data/ztf/dr4/dr4_slsntde_coordlist.csv'),
        help='Combined SLSN/TDE catalog CSV (ZTFID, type, redshift, RAdeg, '
             'Decdeg, source), used for SLSN-I, SLSN-II, and TDE '
             '(default: $ALT_CSV or /Users/jnordin/...)'
    )
    parser.add_argument(
        '--alt-mongodb',
        default=os.environ.get('ALT_MONGODB', 'bts_ipacfp_strictbase_slsntns'),
        help='Alternate MongoDB database used for SLSN-I, SLSN-II, and TDE '
             '(default: $ALT_MONGODB or ztf_slsn_tde_photometry)'
    )

    return parser.parse_args()


def _infer_target_class(raw_type: str, source: str):
    """Resolve one --alt-csv row's (type, source) onto 'TDE', 'SLSN-I',
    'SLSN-II', or None if neither can be determined.

    Direct, unambiguous type matches (known TDE sub-classification variants,
    plus the hyphen-less 'SLSNII' label used by the Pessi25 sample) are
    trusted first. Anything else -- an uninformative type ('Unknown'), or a
    contested/stale one ('Ic', which appears twice within the SLSNI_Chen23
    sample) -- falls back to the source catalog instead. Each literature
    sample is single-class by construction, and its INCLUSION of an object
    is treated as authoritative over whatever a specific `type` entry says:
    being listed in SLSNI_Chen23 at all means it's counted as SLSN-I here,
    regardless of a possibly-outdated 'Ic' label.
    """
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


def load_class_dataframe(classname: str, args: argparse.Namespace) -> pd.DataFrame:
    """Build the working dataframe (ZTFID, redshift, A_V, peakmag, ...) for
    one narrow class.

    Standard classes: --bts-csv, run through add_warpclasses() -- unchanged.

    SLSN-I / SLSN-II / TDE (ALT_SOURCE_CLASSES): --alt-csv instead. This
    catalog already directly IS the class list with redshifts (spanning
    BTS plus several literature samples), so unlike the standard path it
    does NOT go through add_warpclasses() -- class is resolved per-row via
    _infer_target_class() using the catalog's own `type` + `source` columns.
    """
    if classname not in ALT_SOURCE_CLASSES:
        df_raw = pd.read_csv(args.bts_csv)
        df = add_warpclasses(df_raw, purge=True)
        df = df[(df['type_n'] == classname) & (df['redshift'] != '-')]
        df['redshift'] = pd.to_numeric(df['redshift'])

        # We already here wish to convert hour / deg coordinates to RAdeg/Decdeg
        from astropy.coordinates import SkyCoord
        c = SkyCoord(df['RA'], df['Dec'], unit=("hour", "deg"))
        df['RAdeg'] = c.ra.deg
        df['Decdeg'] = c.dec.deg
 
        return df

    print(f"Loading {classname} from alternate catalog: {args.alt_csv}")
    df_alt = pd.read_csv(args.alt_csv, index_col=0)

    resolved = df_alt.apply(lambda row: _infer_target_class(row['type'], row['source']), axis=1)
    n_dropped = resolved.isna().sum()
    if n_dropped:
        dropped_types = sorted(df_alt.loc[resolved.isna(), 'type'].unique())
        print(f"NOTE: {n_dropped} rows in {args.alt_csv} did not resolve to "
              f"TDE/SLSN-I/SLSN-II and were dropped (raw type(s): {dropped_types}).")

    df = df_alt.loc[resolved == classname].copy()
    print(f"  {len(df)} rows resolved to '{classname}' "
          f"(sources: {sorted(df['source'].unique())})")

    if df['redshift'].dtype == object:
        df = df[df['redshift'] != '-']
    df['redshift'] = pd.to_numeric(df['redshift'])

    for optional_col, fill_value in (('A_V', 0.0), ('peakmag', np.nan)):
        if optional_col not in df.columns:
            print(f"NOTE: '{optional_col}' not in {args.alt_csv}; filling with "
                  f"{fill_value} for all {classname} rows. RAdeg/Decdeg ARE "
                  f"available in this file if you'd rather look up real "
                  f"Milky Way A_V from a dust map instead of defaulting to 0.")
            df[optional_col] = fill_value

    return df


def get_class_database(classname: str, args: argparse.Namespace, client: pymongo.MongoClient):
    """Pick --mongodb or --alt-mongodb depending on the class.

    All alt-source rows (BTS-sourced and literature-sourced alike) live in
    the same --alt-mongodb, confirmed -- no per-`source` routing needed.
    """
    dbname = args.alt_mongodb if classname in ALT_SOURCE_CLASSES else args.mongodb
    print(f"Using MongoDB database '{dbname}' for class '{classname}'")
    return getattr(client, dbname)


def get_db_table(name, database, tabulators=[]):
    """For ZTF name, get photopoints and then tables."""
    if isinstance(name, int):
        print('Assuming name given as stock')
        stock = int
    elif re.search('ZTF', name):
        stock = ZTFIdMapper.to_ampel_id(name)
    else:
        print('Cannot parse', name)
        return None

    dps = [dp for dp in database.t0.find({'stock': stock})]

    ftables = []
    for tabulator in tabulators:
        ftables.append(tabulator.get_flux_table(dps))
    if len(ftables) > 1:
        print('Implement astropy table appending!')
    return ftables.pop(0)

def get_mw_extinction_av(row, allow_missing=False, R_V=3.1):
    """
    Get Milky Way extinction A_V for a candidate based on coordinate
    information from the table. We assume this has been converted to
        RAdeg/Decdeg (already in deg)
    If not present, return None unless allow_missing is True, in which case return 0.0.
    """

    if not allow_missing and 'RAdeg' not in row and 'Decdeg' not in row:
        raise ValueError("Row does not contain RAdeg and Decdeg columns for Milky Way extinction lookup.")
    elif allow_missing and 'RAdeg' not in row and 'Decdeg' not in row:
        print("Row does not contain RAdeg and Decdeg columns for Milky Way extinction lookup. Returning A_V=0.0.")
        return 0.0
    
    return sfdmap.SFDMap().ebv(row['RAdeg'], row['Decdeg']) * R_V




def deredden_flux_table(table, A_V, R_V=3.1):
    """Correct a photometry table for Milky Way extinction."""
    result = table.copy()

    dust = sncosmo.CCM89Dust()
    dust.set(ebv=A_V / R_V)

    corrections = {}
    for band in np.unique(result["band"]):
        bp = sncosmo.get_bandpass(band)
        trans = dust.propagate(bp.wave, np.ones_like(bp.wave))
        T_eff = (
            np.trapezoid(trans * bp.trans, bp.wave)
            / np.trapezoid(bp.trans, bp.wave)
        )
        corrections[band] = 1.0 / T_eff

    factors = np.array([corrections[b] for b in result["band"]])
    result["flux"] *= factors
    result["fluxerr"] *= factors

    return result


def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(v) for v in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def main():
    args = parse_args()
    classid = args.classid

    # Setup paths
    sys.path.append(args.ampel_path)

    plotdir = None if args.plotdir.lower() == 'none' else args.plotdir
    if plotdir is not None:
        os.makedirs(plotdir + 'goodpeak', exist_ok=True)
        os.makedirs(plotdir + 'badpeak', exist_ok=True)

    os.makedirs(args.outdir, exist_ok=True)

    logfile = os.path.join(args.outdir, f"{args.version}_I_btssncosmo.log")

    # Class definitions -- TDE added as the 18th class (index 17), matching
    # the CLI's pre-existing choices=range(18).
    nclasses = [
        'SLSN-II', 'SLSN-I', 'SN Ia-CSM', 'SN Iax', 'SN Ia-SC', 'SN Ia-91T',
        'SN Ic-BL', 'SN Ib', 'SN Ib/c', 'SN IIn', 'SN Ic', 'SN Ia-91bg',
        'SN IIP', 'SN Ia-pec', 'SN II', 'SN IIb', 'SN Ibn', 'TDE',
    ]

    print('doing fits for', nclasses[classid])

    # Pipeline parameters
    include_sigma = 3

#    min_tot = 6.
#    min_early = 1.
#    earlytime = 15
#    min_bands = 2

    phaserange = 60
    intdisp = 0.10
    fracdisp = True
#    peak_chicut = 100.0

    gp_length_scale = 10.0
    peakflux_iter = 0
    peak_gp_maxdiff = 15

    # Load data (standard BTS source, or the combined alt catalog for
    # SLSN/TDE -- see load_class_dataframe() and the module docstring)
    df = load_class_dataframe(nclasses[classid], args)
    print(df.shape)

    df_class = pd.read_csv(args.classfile, sep=';')

    tabulators = [ZTFFPTabulator(inclusion_sigma=include_sigma)]

    client = pymongo.MongoClient()
    db = get_class_database(nclasses[classid], args, client)

    to_reject = []
    [to_reject.extend(l) for key, l in SN_REJECT.items()]
    print('Going to veto {} sne from the initial list'.format(len(to_reject)))

    # Setup models
    models = {}

    # Load OpenUniverse templates and register them with sncosmo
    register_all()

    models['salt2'] = {
        'model': sncosmo.Model(source='salt2'),
        'fitprop': ['t0', 'x0', 'x1', 'c'],
        'bounds': None,
        'class': 'SN Ia'
    }
    peakphase = {
        'ztfg': models['salt2']['model'].source.peakphase('ztfg'),
        'ztfr': models['salt2']['model'].source.peakphase('ztfr'),
    }
    peakphase['avg'] = (peakphase['ztfg'] + peakphase['ztfr']) / 2
    models['salt2']['peakphase'] = peakphase

    models['salt3'] = {
        'model': sncosmo.Model(source='salt3'),
        'fitprop': ['t0', 'x0', 'x1', 'c'],
        'bounds': None,
        'class': 'SN Ia'
    }
    peakphase = {
        'ztfg': models['salt3']['model'].source.peakphase('ztfg'),
        'ztfr': models['salt3']['model'].source.peakphase('ztfr'),
    }
    peakphase['avg'] = (peakphase['ztfg'] + peakphase['ztfr']) / 2
    models['salt3']['peakphase'] = peakphase

    dust = sncosmo.CCM89Dust()

    tsources = {n: c for n, c in zip(df_class['Name'], df_class['Type'])}

    for tsource, sourceclass in tsources.items():
        if tsource[0] == '#':
            continue
        if tsource in ['salt2', 'salt3']:
            continue
        print('Adding', tsource)
        m = sncosmo.Model(source=tsource,
                          effects=[dust],
                          effect_names=['host'],
                          effect_frames=['rest'])
        m.set(hostr_v=3.1)
        peakphase = {
            'ztfg': m.source.peakphase('ztfg'),
            'ztfr': m.source.peakphase('ztfr'),
        }
        peakphase['avg'] = (peakphase['ztfg'] + peakphase['ztfr']) / 2
        models[tsource] = {
            'model': m,
            'fitprop': ['t0', 'amplitude', 'hostebv'],
            'bounds': {'hostebv': [-3., 5.]},
            'class': sourceclass,
            'peakphase': peakphase,
        }

    results = {modelname: [] for modelname in models.keys()}
    failkey = []

    # Main fitting loop
    for k, row in df.iterrows():
        print(k, df.shape[0], row['ZTFID'], row['redshift'])
        (name, z) = (row['ZTFID'], row['redshift'])

        if name in to_reject:
            print('{} skipped - on veto list'.format(name))
            failkey.append(1)
            continue

        tab = get_db_table(name, database=db, tabulators=tabulators)
        tab.sort('time')
        if len(tab) == 0:
            failkey.append(2)
            continue

        bands = len(set(tab['band']))
        if bands < 2:
            failkey.append(3)
            continue

        # So, instead of assuming Av in file we can derive it based on position. More consistent among different catalogs.
        Av = get_mw_extinction_av(row)
        # Compare with bts value if there
        if 'A_V' in row and not np.isnan(row['A_V']):
            print(f"NOTE: catalog A_V={row['A_V']:.3f} for {name}, computed A_V={Av:.3f}.")
            if not np.isclose(Av, row['A_V'], rtol=0.1):
                print(f"NOTE: computed A_V={Av:.3f} differs from catalog A_V={row['A_V']:.3f} for {name}. Using computed value.")

        tab = deredden_flux_table(tab, Av, R_V=3.1)

        banddict = {
            band: {
                'time': tab[tab['band'] == band]['time'],
                'flux': tab[tab['band'] == band]['flux'],
                'flux_err': tab[tab['band'] == band]['fluxerr']
            }
            for band in set(tab['band'])
        }

        results_gp = estimate_peak_flux_multiband(
            banddict, method="gp",
            length_scale=gp_length_scale,
            n_sigma=3,
            n_clip_iter=peakflux_iter,
        )
        peakcol = get_peak_colors(results_gp, prefix='gp_', min_eff_points=1)

        if 'gp_ztfg-ztfr' not in peakcol:
            print('not gp peak col')
            failkey.append(4)
            continue

        presum = peakcol.get('gp_ztfg_n_eff_before_peak', 0) + peakcol.get('gp_ztfr_n_eff_before_peak', 0)
        postsum = peakcol.get('gp_ztfg_n_eff_after_peak', 0) + peakcol.get('gp_ztfr_n_eff_after_peak', 0)
        peakr = results_gp['ztfr'].peak_time

        if presum < 2 or postsum < 2:
            print('few gp dp', name)
            failkey.append(5)
            continue

        atleastone = 0

        for modelname, modelpar in models.items():
            m = modelpar['model']
            if modelname=='snana-2007ny':
                # This one seems to often cause fit fails, skip to confirm
                continue
#            print('..', modelname)
            t0guess = peakr - modelpar['peakphase']['ztfr']
            m.set(z=z)
            m.set(t0=t0guess)
            bounds = {'t0': [tab['time'].min() - 5., tab['time'].max() + 5.]}
            if modelpar['bounds'] is not None:
                bounds.update(modelpar['bounds'])

            try:
                result, fitted_model = sncosmo.fit_lc(
                    tab, m,
                    modelpar['fitprop'],
                    bounds=bounds)
            except (RuntimeError, ValueError, KeyError, DataQualityError) as e:
                print('.. fit fail', modelname)
                results[modelname].append({
                    'id': name,
                    'nbr_bands': bands,
                    'ndet': len(tab),
                    'success': False,
                })
                continue

            if result.ndof == 0:
                results[modelname].append({
                    'id': name,
                    'nbr_bands': bands,
                    'ndet': len(tab),
                    'success': False,
                })
                continue

            mdict = {
                result['param_names'][k]: result['parameters'][k]
                for k in range(len(result['parameters']))
            }
            mdict.update({k: result[k] for k in ['success', 'chisq', 'ndof', 'errors']})
            mdict['absmag'] = fitted_model.source_peakabsmag(band='bessellb', magsys='ab')
            mdict['peakmag'] = row['peakmag']
            mdict['chidof'] = result.chisq / result.ndof
            mdict['id'] = name
            mdict['nbr_bands'] = bands
            mdict['ndet'] = len(tab)
            mdict['class'] = modelpar['class']
            mdict['presum'] = presum
            mdict['postsum'] = postsum
            mdict['t0diff'] = np.abs(fitted_model.get('t0') - t0guess)

            tstart = fitted_model.mintime()
            tend = tstart + min(fitted_model.maxtime(), tstart + phaserange)

            mdict['premod'] = sum((tab['time'] < tstart))
            mdict['postmod'] = sum((tab['time'] > tend))
            mdict['predt'] = min(tab['time']) - tstart
            mdict['postdt'] = max(tab['time']) - tend

            if mdict['premod'] > 1 or mdict['postmod'] > 1 or mdict['predt'] < -3 or mdict['postdt'] > 3:
                continue

            peaktab = tab[(tab['time'] >= tstart) & (tab['time'] <= tend)]

            try:
                fdiff = peaktab['flux'] - fitted_model.bandflux(
                    peaktab['band'], peaktab['time'],
                    zp=peaktab['zp'], zpsys=peaktab['zpsys']
                )
                if fracdisp:
                    serr = np.sqrt((peaktab['flux'] * intdisp) ** 2 + peaktab['fluxerr'] ** 2)
                else:
                    serr = np.sqrt((np.mean(peaktab['flux']) * intdisp) ** 2 + peaktab['fluxerr'] ** 2)
                chi = sum(fdiff ** 2 / serr ** 2)
                mdict['peakchi'] = chi
            except ValueError:
                mdict['peakchi'] = 1000.
                print('why fail? - prob because filter extends outside template size')
                continue

            peakpoint = peaktab[peaktab['flux'] == peaktab['flux'].max()]
            peakdiff = (
                (peakpoint['flux'] - fitted_model.bandflux(
                    peakpoint['band'], fitted_model.maxtime(),
                    zp=peakpoint['zp'], zpsys=peakpoint['zpsys']
                )) / peakpoint['flux']
            )
            if peakdiff > 1.5 or peakdiff < 0.5:
                continue

            mdict['peakdet'] = len(peaktab)
            mdict['peakbands'] = len(set(peaktab['band']))
            if len(peaktab) > 1:
                mdict['peakduration'] = max(peaktab['time']) - min(peaktab['time'])

            fitpeaktime = modelpar['peakphase']['avg'] + mdict['t0']
            verypeakdt = 3
            mdict['earlydet'] = sum(peaktab['time'] <= (fitpeaktime - verypeakdt))
            mdict['thendet'] = sum((peaktab['time'] <= (fitpeaktime + verypeakdt)) & (peaktab['time'] > (fitpeaktime - verypeakdt)))
            mdict['postdet'] = sum((peaktab['time'] <= (fitpeaktime + 5 * verypeakdt)) & (peaktab['time'] > (fitpeaktime + verypeakdt)))
            mdict['latedet'] = sum((peaktab['time'] <= (fitpeaktime + 10 * verypeakdt)) & (peaktab['time'] >= (fitpeaktime + 5 * verypeakdt)))

            verypeaktab = tab[
                (tab['time'] >= (fitpeaktime - 10 * (1 + z))) & (tab['time'] <= (fitpeaktime + 10 * (1 + z)))
            ]
            mdict['verypeakcount'] = len(verypeaktab)
            mdict['verypeakbands'] = len(set(verypeaktab['band']))

            if (mdict['t0diff'] > peak_gp_maxdiff or mdict['presum'] < 3 or mdict['postsum'] < 3
                    or mdict['earlydet'] < 2 or mdict['postdet'] < 2 or mdict['latedet'] < 2):
                mdict['peak_good'] = False
                if atleastone == 0:
                    atleastone = 1
            else:
                mdict['peak_good'] = True
                atleastone = 2

            # Also store fitted peak info. Slightly inefficient, but not sure what is saved later
            for peak_col_label in ['gp_ztfg-ztfr', 'gp_ztfr-ztfri']:
                if peak_col_label in peakcol:
                    mdict['peak_'+peak_col_label] = peakcol[peak_col_label]
            results[modelname].append(mdict)

            plotname = '{:.2}_{}_{}_{:.2}_{:.2}_{:.2}.png'.format(
                presum, name, modelname,
                mdict['peakchi'] / mdict['peakdet'], postsum, mdict['t0diff']
            )

            if plotdir is not None:
                if mdict['peak_good']:
                    fout = os.path.join(plotdir, 'goodpeak', plotname)
                else:
                    fout = os.path.join(plotdir, 'badpeak', plotname)

                fig = sncosmo.plot_lc(tab, model=fitted_model, errors=result.errors)
                plt.axvline(x=fitpeaktime - fitted_model.get('t0'))
                plt.savefig(fout)
                plt.close()

        if atleastone == 0:
            failkey.append(6)
        elif atleastone == 1:
            failkey.append(9)
        elif atleastone == 2:
            failkey.append(99)

    # Analysis and output
    zclass = {
        # Updated from the actual combined SLSN/TDE catalog's real observed
        # ranges (TDE [0.011, 0.519], SLSN-I [0.039, 0.670], SLSN-II
        # [0.018, 0.4846]) plus a little padding -- NOT the same as the
        # previous [0.0, 0.3] entries, which excluded roughly half the
        # SLSN-I sample and a meaningful chunk of SLSN-II. This is a
        # data-availability choice; adjust if [0.0, 0.3] was chosen for a
        # specific physical/quality reason rather than just "what BTS alone
        # covered".
        'SLSN-II': [0.0, 0.5],
        'SLSN-I': [0.0, 0.7],
        'SN Ia-91bg': [0.01, 0.055],
        'SN Ia-91T': [0.01, 0.10],
        'SN Ia-CSM': [0.01, 0.10],
        'SN IIn': [0.0, 0.10],
        'SN Ia-SC': [0.01, 0.10],
        'SN Ia-pec': [0.01, 0.055],
        'SN Iax': [0.0, 0.055],
        'TDE': [0.0, 0.55],
    }
    zlim = zclass.get(nclasses[classid], [0.0, 0.07])
    print('Using {} z lim.'.format('specific' if nclasses[classid] in zclass else 'default'))

    fitstore = []
    fitvolume = []
    fitgood = []
    for modname, reslist in results.items():
        for res in reslist:
            fitstore.append(res['id'])
            if not res['success'] or not res['peak_good']:
                continue
            fitgood.append(res['id'])
            if zlim[0] <= float(res['z']) <= zlim[1]:
                fitvolume.append(res['id'])

    # Plot redshift distributions
    plt.figure(1, figsize=(10, 5))
    _, bins, __ = plt.hist(df['redshift'], bins=10)
    plt.hist(df['redshift'].loc[df['ZTFID'].isin(fitstore)], bins=bins, label='Fit done')
    plt.hist(df['redshift'].loc[df['ZTFID'].isin(fitgood)], bins=bins, label='Fit good')
    plt.hist(df['redshift'].loc[df['ZTFID'].isin(fitvolume)], bins=bins, label='Fit in vol')
    plt.legend()
    zplot_path = os.path.join(args.outdir, f'zdist_{args.version}_{nclasses[classid].replace("/", "")}.png')
    plt.savefig(zplot_path)
    plt.close()

    print(nclasses[classid], zlim[0], zlim[1], df.shape[0],
          len(set(fitstore)), len(set(fitgood)), len(set(fitvolume)))

    with open(logfile, "a") as f:
        timestamp = datetime.now().isoformat(timespec="seconds")
        f.write("{} {} {} {} {} {} {} {} \n".format(
            nclasses[classid], zlim[0], zlim[1], df.shape[0],
            len(set(fitstore)), len(set(fitgood)), len(set(fitvolume)), timestamp
        ))

    fname = os.path.join(args.outdir, f'btsfits{args.version}_{nclasses[classid].replace("/", "")}.json')
    jresults = make_json_serializable(results)

    with open(fname, "w") as outfile:
        outfile.write(json.dumps(jresults, indent=2))

    print("Stored", len(jresults), "model results to", fname)


if __name__ == '__main__':
    main()
