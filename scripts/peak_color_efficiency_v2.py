#!/usr/bin/env python
# coding: utf-8

"""
Stage II: GP peak / colour measurement efficiency per narrow class.

For every class (default: all 18) this re-runs ONLY the front end of the
stage-I pipeline (photometry fetch -> MW dereddening -> GP peak estimate ->
GP peak colours -> point-count gate), WITHOUT any sncosmo fitting, and records
per object how far it got. It then plots the fraction of objects that end up
with a GP colour / pass the gate as a function of redshift and of catalogue
peak magnitude.

Stage-I functions (load_class_dataframe, get_class_database, get_db_table,
deredden_flux_table, ...) are imported from the stage-I script itself, so the
selection logic cannot drift. Point --stage1 at that file.

Per-object outcome ('status'), in the order the pipeline checks them:
    vetoed         on SN_REJECT list
    no_photometry  empty flux table
    one_band       < 2 bands
    no_gp_color    'gp_ztfg-ztfr' not in peak colours        (pipeline failkey 4)
    few_gp_points  presum < 2 or postsum < 2                 (failkey 5)
    ok             passes everything before the sncosmo loop
    error          exception while processing (see 'error' column)

Efficiencies reported:
    db_ok     : object has a non-empty flux table in any of the the local databases
                (the pipeline's 'no_photometry' check). Evaluated BEFORE the
                veto, so it is the pure database efficiency of the input list.
                color_ok / peak_ok / gate_ok below are 'total' efficiencies
                (denominator = all input objects, DB misses count as failures);
                the conditional gate|db efficiency is derived downstream.
    color_ok  : has a GP g-r colour (status in {few_gp_points, ok})
    peak_ok   : GP gave a finite r-band peak time (independent of colour)
    gate_ok   : status == 'ok'  (colour AND enough points around peak)

Notes
  * Redshift limits are NOT applied -- the point is efficiency over the full
    range. Class z-range of each sample is whatever is in the input CSVs.
  * 'peakmag' is the catalogue value. The alt (SLSN/TDE) catalogue has none,
    so those classes only get the redshift plot unless you add the column.
  * The raw GP peak-result attributes (e.g. peak flux) are stored if present,
    under gp_<band>_peak_time / gp_<band>_peak_flux.
  * Per-object results are cached to CSV; use --plot-only to re-bin/re-plot.
  * --db-only skips all GP processing and only records db_ok (fast). The
    GP-derived flags are then NaN (= undefined) and the table is written to
    efficiency_objects_<ver>_dbonly.csv; downstream scripts accept it directly
    and simply show only the database efficiency.
"""

import argparse
import importlib.util
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NCLASSES = [
    'SLSN-II', 'SLSN-I', 'SN Ia-CSM', 'SN Iax', 'SN Ia-SC', 'SN Ia-91T',
    'SN Ic-BL', 'SN Ib', 'SN Ib/c', 'SN IIn', 'SN Ic', 'SN Ia-91bg',
    'SN IIP', 'SN Ia-pec', 'SN II', 'SN IIb', 'SN Ibn', 'TDE',
]


# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
def load_stage1(path, ampel_path):
    """Import the stage-I script as a module (its main() is not run)."""
    if ampel_path:
        sys.path.append(ampel_path)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))  # for warptemplate
    spec = importlib.util.spec_from_file_location('stage1', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
def _env_list(var, default):
    val = os.environ.get(var)
    return [v.strip() for v in val.split(',') if v.strip()] if val else default

def parse_args():
    env = os.environ.get
    p = argparse.ArgumentParser(description='GP peak/colour efficiency per class.')
    p.add_argument('--stage1', default=env('STAGE1', 'templatecreation_I_sncosmo.py'),
                   help='Path to the stage-I script (default: $STAGE1)')
    p.add_argument('--classes', nargs='*', default=None,
                   help='Class names to process (default: all 18)')
    # These are the ones used for searching - should not be used for efficiency estimates
#    p.add_argument('--bts-csv', default=env('BTS_CSV', '/Users/jnordin/data/ztf/bts/bts_explorer_260601.csv'))
    p.add_argument('--bts-csv', default=env('BTS_CSV', '/Users/jnordin/data/ztf/bts/bts_explorer_241122.csv'))
    p.add_argument('--alt-csv', default=env('ALT_CSV', '/Users/jnordin/data/ztf/dr4/dr4_slsntde_coordlist.csv'))
    p.add_argument('--mongodbs', nargs='+',
                   default=_env_list('MONGODBS', [
                    'bts_ipacfp_strictbase_train_jul26',
                    'bts_ipacfp_strictbase_slsntns',
                    'dr4dr3ipac_parsnip',
                    'bts_ipacfp_strictbase',
                    'dr4bts_parsnip',
                   ]),
                   help='Ordered list of MongoDB databases to search for photometry; '
                        'first database with data wins (default: $MONGODBS, comma-separated). '
                        'Use the same order as in stages I and II.')
    p.add_argument('--ampel-path', default=env('AMPEL_PATH', '/Users/jnordin/github/ampelJul26'))
    p.add_argument('--outdir', default=env('EFF_OUTDIR', '/Users/jnordin/data/models/sncosmo/efficiency/'))
    p.add_argument('--version', '-v', default=env('VERSION', 'v8'))
    p.add_argument('--gp-length-scale', type=float, default=10.0)
    p.add_argument('--nbins', type=int, default=8, help='Bins for z / peakmag')
    p.add_argument('--min-per-bin', type=int, default=3,
                   help='Bins with fewer objects are not drawn')
    p.add_argument('--db-only', action='store_true',
                   help='Only check presence in the local database (no GP)')
    p.add_argument('--plot-only', action='store_true',
                   help='Skip processing, re-plot from cached per-object CSV')
    p.add_argument('--limit-timeframe', type=bool, default=False,
                   help='Only include objects from within the central time frame (avoiding edge photometry effects)')
    return p.parse_args()


# ----------------------------------------------------------------------------
# Per-object processing
# ----------------------------------------------------------------------------
def process_object(s1, row, databases, tabulators, sfd, to_reject, gp_length_scale,
                   db_only=False):
    """Return dict of outcome + measured quantities for one object."""
    name = row['ZTFID']
    out = {'ZTFID': name, 'redshift': row['redshift'],
           'peakmag': row.get('peakmag', np.nan),
           'status': None, 'error': '', 'photodb':'',
           'db_ok': False, 'vetoed': bool(name in to_reject),
           'color_ok': False, 'peak_ok': False, 'gate_ok': False}

    # Database lookup first, independent of the veto list
    tab, photdb = s1.get_db_table(name, databases=databases, tabulators=tabulators)
    out['db_ok'] = bool(tab is not None and len(tab) > 0)
    out['photdb'] = photdb or ''

    if db_only:
        for f in ('color_ok', 'peak_ok', 'gate_ok'):
            out[f] = np.nan   # undefined, not failed
        out['status'] = 'db_only'
        return out

    if name in to_reject:
        out['status'] = 'vetoed'
        return out

    if not out['db_ok']:
        out['status'] = 'no_photometry'
        return out
    tab.sort('time')
    out['ndet'] = len(tab)

    bands = set(tab['band'])
    out['nbr_bands'] = len(bands)
    if len(bands) < 2:
        out['status'] = 'one_band'
        return out

    Av = sfd.ebv(row['RAdeg'], row['Decdeg']) * 3.1
    tab = s1.deredden_flux_table(tab, Av, R_V=3.1)

    banddict = {
        b: {'time': tab[tab['band'] == b]['time'],
            'flux': tab[tab['band'] == b]['flux'],
            'flux_err': tab[tab['band'] == b]['fluxerr']}
        for b in bands
    }
    results_gp = s1.estimate_peak_flux_multiband(
        banddict, method='gp', length_scale=gp_length_scale,
        n_sigma=3, n_clip_iter=0)
    peakcol = s1.get_peak_colors(results_gp, prefix='gp_', min_eff_points=1)

    # Raw per-band peak results, stored defensively (attribute names vary)
    for b in ('ztfg', 'ztfr', 'ztfi'):
        res = results_gp.get(b) if hasattr(results_gp, 'get') else None
        if res is None:
            continue
        for attr in ('peak_time', 'peak_flux'):
            val = getattr(res, attr, None)
            try:
                out[f'gp_{b}_{attr}'] = float(val)
            except (TypeError, ValueError):
                pass

    pt = out.get('gp_ztfr_peak_time', np.nan)
    out['peak_ok'] = bool(np.isfinite(pt))

    for label in ('gp_ztfg-ztfr', 'gp_ztfr-ztfri'):
        if label in peakcol:
            out[label] = peakcol[label]

    if 'gp_ztfg-ztfr' not in peakcol:
        out['status'] = 'no_gp_color'
        return out
    out['color_ok'] = True

    presum = peakcol.get('gp_ztfg_n_eff_before_peak', 0) + peakcol.get('gp_ztfr_n_eff_before_peak', 0)
    postsum = peakcol.get('gp_ztfg_n_eff_after_peak', 0) + peakcol.get('gp_ztfr_n_eff_after_peak', 0)
    out['presum'], out['postsum'] = presum, postsum

    if presum < 2 or postsum < 2:
        out['status'] = 'few_gp_points'
        return out

    out['status'] = 'ok'
    out['gate_ok'] = True
    return out


def run_class(s1, classname, args, databases, tabulators, sfd, to_reject, limit_timeframe=False):
    df = s1.load_class_dataframe(classname, args)
    if limit_timeframe:
        print('Limit datafiles to the ZTF19* to ZTF22* objects. Going from',df.shape[0])
        df = df.loc[ df['ZTFID'].str.match("ZTF(19|20|21)") ]
        print('.. to',df.shape[0])
    recs = []
    n = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        print(f'[{classname}] {i + 1}/{n} {row["ZTFID"]}')
        try:
            rec = process_object(s1, row, databases, tabulators, sfd, to_reject,
                                 args.gp_length_scale, db_only=args.db_only)
        except Exception as e:  # keep going; record the failure
            rec = {'ZTFID': row['ZTFID'], 'redshift': row['redshift'],
                   'peakmag': row.get('peakmag', np.nan), 'status': 'error',
                   'error': f'{type(e).__name__}: {e}', 'photodb':'',
                   'db_ok': np.nan, 'vetoed': False,   # unknown -> excluded from db stats
                   'color_ok': False, 'peak_ok': False, 'gate_ok': False}
            print('  error:', rec['error'])
        rec['class'] = classname
        recs.append(rec)
    return pd.DataFrame(recs)


# ----------------------------------------------------------------------------
# Statistics and plotting
# ----------------------------------------------------------------------------
def wilson(k, n, z=1.0):
    """Wilson score interval (default 1 sigma) -> (frac, lo, hi)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = k / n
        den = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / den
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    # np.minimum/maximum with p guards against float rounding at p=0 or 1
    return (p, np.minimum(np.clip(centre - half, 0, 1), p),
            np.maximum(np.clip(centre + half, 0, 1), p))


def binned_eff(df, xcol, flags, nbins):
    d = df[np.isfinite(pd.to_numeric(df[xcol], errors='coerce'))].copy()
    d[xcol] = pd.to_numeric(d[xcol])
    if len(d) == 0 or d[xcol].nunique() < 2:
        return None
    edges = np.linspace(d[xcol].min(), d[xcol].max(), nbins + 1)
    d['bin'] = pd.cut(d[xcol], edges, include_lowest=True, labels=False)
    rows = []
    for b in range(nbins):
        sub = d[d['bin'] == b]
        r = {'lo': edges[b], 'hi': edges[b + 1],
             'mid': 0.5 * (edges[b] + edges[b + 1]), 'n': len(sub)}
        for f in flags:
            v = sub[f].dropna()          # NaN = flag undefined, excluded
            nv = len(v)
            p, lo, hi = wilson(int(v.sum()), max(nv, 1))
            r[f], r[f + '_lo'], r[f + '_hi'] = (p, lo, hi) if nv else (np.nan,) * 3
        rows.append(r)
    return pd.DataFrame(rows)


def plot_efficiency(df, xcol, xlabel, outpath, nbins, min_per_bin, invert_x=False):
    classes = [c for c in NCLASSES if c in set(df['class'])]
    classes = [c for c in classes
               if np.isfinite(pd.to_numeric(df.loc[df['class'] == c, xcol], errors='coerce')).sum() >= 2]
    if not classes:
        print(f'No classes with usable {xcol}; skipping plot.')
        return
    ncol = min(4, len(classes))
    nrow = int(np.ceil(len(classes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), squeeze=False)
    styles = {'db_ok': ('C3', 'in database'),
              'color_ok': ('C0', 'GP colour'), 'peak_ok': ('C1', 'GP peak'),
              'gate_ok': ('C2', 'colour + point gate')}

    for ax, cname in zip(axes.ravel(), classes):
        sub = df[df['class'] == cname]
        eff = binned_eff(sub, xcol, list(styles), nbins)
        ax2 = ax.twinx()
        if eff is not None:
            ax2.bar(eff['mid'], eff['n'], width=(eff['hi'] - eff['lo']) * 0.9,
                    color='lightgrey', zorder=0)
            ax2.set_ylabel('N', color='grey', fontsize=8)
            ax.set_zorder(ax2.get_zorder() + 1)
            ax.patch.set_visible(False)
            ok = eff['n'] >= min_per_bin
            for f, (col, lab) in styles.items():
                e = eff[ok]
                ax.errorbar(e['mid'], e[f],
                            yerr=[e[f] - e[f + '_lo'], e[f + '_hi'] - e[f]],
                            color=col, marker='o', ms=3, lw=1, capsize=2, label=lab)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f'{cname} (N={len(sub)})', fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel('Fraction', fontsize=8)
        if invert_x:
            ax.invert_xaxis()
    for ax in axes.ravel()[len(classes):]:
        ax.axis('off')
    axes[0, 0].legend(fontsize=7, loc='lower left')
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print('Wrote', outpath)


def _numeric_flags(df):
    """Flags -> float 1/0/NaN (NaN = undefined). Derives db_ok for older tables."""
    for f in ('db_ok', 'color_ok', 'peak_ok', 'gate_ok'):
        if f in df.columns:
            df[f] = df[f].map({True: 1.0, False: 0.0, 'True': 1.0, 'False': 0.0})
    if 'db_ok' not in df.columns:
        print('NOTE: no db_ok column (older table); deriving from status.')
        df['db_ok'] = 1.0
        df.loc[df['status'] == 'no_photometry', 'db_ok'] = 0.0
        df.loc[df['status'].isin(['vetoed', 'error']), 'db_ok'] = np.nan
    return df


def summarize(df):
    g = df.groupby('class')
    summ = g.agg(n=('ZTFID', 'size'))
    for f in ('db_ok', 'color_ok', 'peak_ok', 'gate_ok'):
        summ[f] = g[f].sum()
        summ[f + '_frac'] = g[f].sum() / g[f].count()   # NaN rows excluded
    d = df[df['db_ok'] == 1]
    summ['gate_given_db_frac'] = d.groupby('class')['gate_ok'].mean()
    status = pd.crosstab(df['class'], df['status'])
    summ = summ.join(status)
    return summ.reindex([c for c in NCLASSES if c in summ.index])


# ----------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    suffix = '_dbonly' if args.db_only else ''
    objfile = os.path.join(args.outdir, f'efficiency_objects_{args.version}{suffix}.csv')

    if args.plot_only:
        df = pd.read_csv(objfile)
        if args.classes:
            df = df[df['class'].isin(args.classes)]
    else:
        import pymongo
        s1 = load_stage1(args.stage1, args.ampel_path)
        classes = args.classes or NCLASSES
        bad = [c for c in classes if c not in NCLASSES]
        if bad:
            raise SystemExit(f'Unknown classes: {bad}')

        tabulators = [s1.ZTFFPTabulator(inclusion_sigma=3)]
        client = pymongo.MongoClient()
        databases = s1.get_databases(args.mongodbs, client)
        sfd = s1.sfdmap.SFDMap()  # build once; it is slow to construct
        to_reject = [x for l in s1.SN_REJECT.values() for x in l]
        s1.register_all()  # harmless here; keeps sncosmo bandpasses/templates consistent

        frames = []
        for cname in classes:
            cdf = run_class(s1, cname, args, databases, tabulators, sfd, to_reject, args.limit_timeframe)
            frames.append(cdf)
            # checkpoint after each class
            pd.concat(frames, ignore_index=True).to_csv(objfile, index=False)
        df = pd.concat(frames, ignore_index=True)
        print('Wrote', objfile)

    df = _numeric_flags(df)

    summ = summarize(df)
    summpath = os.path.join(args.outdir, f'efficiency_summary_{args.version}{suffix}.csv')
    summ.to_csv(summpath)
    print(summ.to_string())
    print('Wrote', summpath)

    # Binned tables + plots
    for xcol, xlabel, inv in (('redshift', 'Redshift', False),
                              ('peakmag', 'Catalogue peak mag', True)):
        plot_efficiency(df, xcol, xlabel,
                        os.path.join(args.outdir, f'efficiency_vs_{xcol}_{args.version}{suffix}.png'),
                        args.nbins, args.min_per_bin, invert_x=inv)
        tabs = []
        for cname in NCLASSES:
            sub = df[df['class'] == cname]
            if len(sub) == 0:
                continue
            e = binned_eff(sub, xcol, ['db_ok', 'color_ok', 'peak_ok', 'gate_ok'], args.nbins)
            if e is not None:
                e.insert(0, 'class', cname)
                tabs.append(e)
        if tabs:
            pd.concat(tabs).to_csv(
                os.path.join(args.outdir, f'efficiency_binned_{xcol}_{args.version}{suffix}.csv'),
                index=False)


if __name__ == '__main__':
    main()
