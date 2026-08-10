#!/usr/bin/env python
# coding: utf-8

"""
Redoing of sample_sncosmo_from_db using the more consistent class definition.
Pick one non-SN Ia narrow class, fit sncosmo models, store results.
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

from warptemplate import add_warpclasses, SN_REJECT, estimate_peak_flux_multiband, get_peak_colors, register_all


# Sncosmo fitting can throw a lot of warnings, typically when skipping a band due to datapoints
# or wavelength coverage.
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
#    module="sncosmo.fitting"
)

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
        default=os.environ.get('VERSION', 'v4'),
        help='Version string for output files and log (default: $VERSION or v4)'
    )
    parser.add_argument(
        '--mongodb',
        default=os.environ.get('MONGODB', 'bts_ipacfp_strictbase_train_jul26'),
        help='MongoDB database name (default: $MONGODB or bts_ipacfp_strictbase_train_jul26)'
    )

    return parser.parse_args()


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

    # Class definitions
    nclasses = [
        'SLSN-II', 'SLSN-I', 'SN Ia-CSM', 'SN Iax', 'SN Ia-SC', 'SN Ia-91T',
        'SN Ic-BL', 'SN Ib', 'SN Ib/c', 'SN IIn', 'SN Ic', 'SN Ia-91bg',
        'SN IIP', 'SN Ia-pec', 'SN II', 'SN IIb', 'SN Ibn'
    ]

    print('doing fits for', nclasses[classid])

    # Pipeline parameters
    include_sigma = 3

    min_tot = 6.
    min_early = 1.
    earlytime = 15
    min_bands = 2

    phaserange = 60
    intdisp = 0.10
    fracdisp = True
    peak_chicut = 100.0

    gp_length_scale = 10.0
    peakflux_iter = 0
    peak_gp_maxdiff = 15

    # Load data
    df_bts_types = pd.read_csv(args.bts_csv)
    df = add_warpclasses(df_bts_types, purge=True)
    print(df.shape)

    df = df[(df['type_n'] == nclasses[classid]) & (df['redshift'] != '-')]
    df['redshift'] = pd.to_numeric(df['redshift'])
    print(df.shape)

    df_class = pd.read_csv(args.classfile, sep=';')

    tabulators = [ZTFFPTabulator(inclusion_sigma=include_sigma)]

    client = pymongo.MongoClient()
    db = getattr(client, args.mongodb)

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

        tab = deredden_flux_table(tab, row['A_V'], R_V=3.1)

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
        'SLSN-II': [0.0, 0.3],
        'SLSN-I': [0.0, 0.3],
        'SN Ia-91bg': [0.01, 0.055],
        'SN Ia-91T': [0.01, 0.10],
        'SN Ia-CSM': [0.01, 0.10],
        'SN IIn': [0.0, 0.10],
        'SN Ia-SC': [0.01, 0.10],
        'SN Ia-pec': [0.01, 0.055],
        'SN Iax': [0.0, 0.055],
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