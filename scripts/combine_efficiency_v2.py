#!/usr/bin/env python
# coding: utf-8

"""
Stage III: combined GP peak/colour efficiency for groups of classes.

Reads the per-object table written by peak_color_efficiency.py
(efficiency_objects_<ver>.csv) and pools any chosen set(s) of classes into
combined efficiency curves vs redshift and vs catalogue peak magnitude, plus a
2D (redshift x peakmag) efficiency map per group.

Examples
  # everything pooled into one curve
  python combine_efficiency.py --objects .../efficiency_objects_v7.csv

  # one pooled group from a class list
  python combine_efficiency.py --objects ... --classes "SN Ib" "SN Ic" "SN Ib/c" --label StrippedEnv

  # several groups compared on the same axes
  python combine_efficiency.py --objects ... \
      --group "Stripped=SN Ib,SN Ic,SN Ib/c,SN Ic-BL" \
      --group "Interacting=SN IIn,SN Ibn,SN Ia-CSM" \
      --group "Superluminous=SLSN-I,SLSN-II"

Weighting (--weight)
  pooled : every object counts equally, so classes with many objects dominate
           (efficiency of the sample as it exists in the catalogue).
  class  : each class gets equal total weight inside a group, so the curve is
           the average class efficiency regardless of class size.
  Errors are Wilson 1-sigma intervals using the Kish effective sample size.

Metrics: db_ok (object found in the local database -- the pure database
efficiency), color_ok (GP g-r colour), peak_ok (GP r peak), gate_ok (colour AND
enough points around peak -- what reaches the sncosmo loop), all measured over
ALL input objects (total efficiency); and gate_cond = gate_ok restricted to
objects that are in the database (conditional efficiency), so that
total gate ~ db_ok x gate_cond.

Objects without a catalogue peakmag (e.g. SLSN/TDE from the alt catalogue) are
excluded from the magnitude plots and 2D maps only; the number dropped is
printed per group.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRICS = {'db_ok': 'In local database',
           'color_ok': 'GP colour (all input)',
           'peak_ok': 'GP peak (all input)',
           'gate_ok': 'Colour + gate (all input)',
           'gate_cond': 'Colour + gate | in database'}


def load_objects(path):
    """Read the per-object table; flags become float 1/0/NaN (NaN = undefined
    and excluded from that metric). Adds gate_cond (gate_ok restricted to
    objects found in the database). Derives db_ok for older tables."""
    df = pd.read_csv(path)
    for c in ('redshift', 'peakmag'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in ('db_ok', 'color_ok', 'peak_ok', 'gate_ok'):
        if c in df.columns:
            df[c] = df[c].map({True: 1.0, False: 0.0, 'True': 1.0, 'False': 0.0})
    if 'db_ok' not in df.columns:
        print('NOTE: no db_ok column (older table); deriving from status '
              '(vetoed/error rows excluded from the database metric).')
        df['db_ok'] = 1.0
        df.loc[df['status'] == 'no_photometry', 'db_ok'] = 0.0
        df.loc[df['status'].isin(['vetoed', 'error']), 'db_ok'] = np.nan
    df['gate_cond'] = df['gate_ok'].where(df['db_ok'] == 1)
    return df


def parse_args():
    env = os.environ.get
    p = argparse.ArgumentParser(description='Combined efficiency over class groups.')
    p.add_argument('--objects', default=env('EFF_OBJECTS',
                   '/Users/jnordin/data/models/sncosmo/efficiency/efficiency_objects_v8.csv'))
    p.add_argument('--outdir', default=None, help='Default: directory of --objects')
    p.add_argument('--version', '-v', default=env('VERSION', 'v8'))
    p.add_argument('--classes', nargs='*', default=None,
                   help='Classes pooled into one group (see --label)')
    p.add_argument('--label', default='Combined', help='Name for the --classes group')
    p.add_argument('--group', action='append', default=[],
                   help='"Name=class1,class2,..." ; repeat for several groups')
    p.add_argument('--weight', choices=['pooled', 'class'], default='pooled')
    p.add_argument('--nbins', type=int, default=10)
    p.add_argument('--zrange', nargs=2, type=float, default=None)
    p.add_argument('--magrange', nargs=2, type=float, default=None)
    p.add_argument('--quantile-bins', action='store_true',
                   help='Equal-count bins instead of equal-width (per group)')
    p.add_argument('--min-per-bin', type=int, default=5,
                   help='Bins with fewer objects are not drawn')
    p.add_argument('--metric-2d', choices=list(METRICS), default='gate_ok')
    p.add_argument('--nbins-2d', type=int, default=6)
    p.add_argument('--min-per-cell', type=int, default=3)
    return p.parse_args()


# ----------------------------------------------------------------------------
def build_groups(df, args):
    avail = set(df['class'])
    groups = {}

    def add(name, classes):
        missing = [c for c in classes if c not in avail]
        if missing:
            print(f'WARNING: group "{name}": no objects for {missing}')
        sub = df[df['class'].isin(classes)]
        if len(sub):
            groups[name] = sub.copy()
        else:
            print(f'WARNING: group "{name}" is empty, skipped.')

    for g in args.group:
        if '=' not in g:
            raise SystemExit(f'--group needs "Name=cls1,cls2": got {g!r}')
        name, cl = g.split('=', 1)
        add(name.strip(), [c.strip() for c in cl.split(',') if c.strip()])
    if args.classes:
        add(args.label, args.classes)
    if not groups:
        add('All classes', sorted(avail))
    return groups


def add_weights(sub, mode):
    if mode == 'class':
        n = sub.groupby('class')['ZTFID'].transform('size')
        sub['w'] = 1.0 / n
    else:
        sub['w'] = 1.0
    return sub


def wilson(p, n, z=1.0):
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    # min/max with p guards against float rounding at p=0 or 1 (negative yerr)
    return min(max(centre - half, 0.0), p), max(min(centre + half, 1.0), p)


def make_edges(x, nbins, rng, quantile):
    lo, hi = (rng if rng else (np.nanmin(x), np.nanmax(x)))
    if quantile:
        e = np.unique(np.quantile(x[(x >= lo) & (x <= hi)], np.linspace(0, 1, nbins + 1)))
        return e
    return np.linspace(lo, hi, nbins + 1)


def weighted_eff(sub, flag):
    k = sub[flag].to_numpy(float)
    ok = np.isfinite(k)          # NaN = flag undefined for this object
    if not ok.any():
        return np.nan, np.nan, np.nan
    w = sub['w'].to_numpy()[ok]
    k = k[ok]
    sw = w.sum()
    p = (w * k).sum() / sw
    neff = sw**2 / (w**2).sum()
    lo, hi = wilson(p, neff)
    return p, lo, hi


def binned(sub, xcol, edges):
    rows = []
    b = pd.cut(sub[xcol], edges, include_lowest=True, labels=False)
    for i in range(len(edges) - 1):
        s = sub[b == i]
        r = {'lo': edges[i], 'hi': edges[i + 1],
             'mid': 0.5 * (edges[i] + edges[i + 1]), 'n': len(s)}
        for m in METRICS:
            if len(s):
                r[m], r[m + '_lo'], r[m + '_hi'] = weighted_eff(s, m)
            else:
                r[m] = r[m + '_lo'] = r[m + '_hi'] = np.nan
        rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
def plot_1d(groups, xcol, xlabel, args, outdir, rng, invert=False):
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.6 * len(METRICS), 4.2), sharey=True)
    tables = []
    for gi, (gname, sub) in enumerate(groups.items()):
        s = sub[np.isfinite(sub[xcol])]
        ndrop = len(sub) - len(s)
        if ndrop:
            print(f'[{gname}] {ndrop}/{len(sub)} objects lack {xcol}; excluded from {xcol} plot.')
        if len(s) < 2 or s[xcol].nunique() < 2:
            continue
        edges = make_edges(s[xcol].to_numpy(), args.nbins, rng, args.quantile_bins)
        t = binned(s, xcol, edges)
        t.insert(0, 'group', gname)
        tables.append(t)
        ok = t['n'] >= args.min_per_bin
        off = (gi - (len(groups) - 1) / 2) * 0.01 * (edges[-1] - edges[0])  # small x-jitter
        for ax, (m, mlabel) in zip(axes, METRICS.items()):
            e = t[ok]
            ax.errorbar(e['mid'] + off, e[m],
                        yerr=[e[m] - e[m + '_lo'], e[m + '_hi'] - e[m]],
                        color=f'C{gi}', marker='o', ms=4, lw=1.2, capsize=2,
                        label=f'{gname} (N={len(s)})')
            ax.set_title(mlabel)
    for ax in axes:
        ax.set_xlabel(xlabel)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        if invert:
            ax.invert_xaxis()
    axes[0].set_ylabel(f'Fraction ({args.weight} weighting)')
    axes[0].legend(fontsize=8, loc='best')
    fig.tight_layout()
    path = os.path.join(outdir, f'combined_vs_{xcol}_{args.version}.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print('Wrote', path)
    if tables:
        tp = os.path.join(outdir, f'combined_binned_{xcol}_{args.version}.csv')
        pd.concat(tables).to_csv(tp, index=False)
        print('Wrote', tp)


def plot_2d(groups, args, outdir):
    m = args.metric_2d
    items = []
    for gname, sub in groups.items():
        s = sub[np.isfinite(sub['redshift']) & np.isfinite(sub['peakmag'])]
        if len(s) >= 4:
            items.append((gname, s))
        else:
            print(f'[{gname}] too few objects with both z and peakmag for 2D map.')
    if not items:
        return
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.4), squeeze=False)
    for ax, (gname, s) in zip(axes.ravel(), items):
        ze = make_edges(s['redshift'].to_numpy(), args.nbins_2d, args.zrange, args.quantile_bins)
        me = make_edges(s['peakmag'].to_numpy(), args.nbins_2d, args.magrange, args.quantile_bins)
        zi = pd.cut(s['redshift'], ze, include_lowest=True, labels=False)
        mi = pd.cut(s['peakmag'], me, include_lowest=True, labels=False)
        eff = np.full((len(me) - 1, len(ze) - 1), np.nan)
        cnt = np.zeros_like(eff)
        for i in range(len(me) - 1):
            for j in range(len(ze) - 1):
                c = s[(mi == i) & (zi == j)]
                cnt[i, j] = len(c)
                if len(c) >= args.min_per_cell:
                    eff[i, j] = weighted_eff(c, m)[0]
        im = ax.pcolormesh(ze, me, np.ma.masked_invalid(eff), vmin=0, vmax=1, cmap='viridis')
        for i in range(len(me) - 1):
            for j in range(len(ze) - 1):
                if cnt[i, j] > 0:
                    ax.text(0.5 * (ze[j] + ze[j + 1]), 0.5 * (me[i] + me[i + 1]),
                            f'{int(cnt[i, j])}', ha='center', va='center',
                            fontsize=6, color='w' if (np.isnan(eff[i, j]) or eff[i, j] < 0.6) else 'k')
        ax.invert_yaxis()
        ax.set_xlabel('Redshift')
        ax.set_ylabel('Catalogue peak mag')
        ax.set_title(f'{gname}: {METRICS[m]}', fontsize=9)
        fig.colorbar(im, ax=ax, label='Fraction')
    fig.suptitle(f'Cell numbers = N objects; cells with N<{args.min_per_cell} masked', fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, f'combined_2d_{m}_{args.version}.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print('Wrote', path)


# ----------------------------------------------------------------------------
def main():
    args = parse_args()
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.objects))
    os.makedirs(outdir, exist_ok=True)

    df = load_objects(args.objects)

    groups = build_groups(df, args)
    groups = {k: add_weights(v, args.weight) for k, v in groups.items()}

    # Overall numbers per group
    rows = []
    for g, s in groups.items():
        r = {'group': g, 'n': len(s), 'classes': ','.join(sorted(s['class'].unique()))}
        for m in METRICS:
            p, lo, hi = weighted_eff(s, m)
            r[m], r[m + '_lo'], r[m + '_hi'] = p, lo, hi
        rows.append(r)
    summ = pd.DataFrame(rows)
    sp = os.path.join(outdir, f'combined_summary_{args.version}.csv')
    summ.to_csv(sp, index=False)
    print(summ.drop(columns='classes').to_string(index=False))
    print('Wrote', sp)

    plot_1d(groups, 'redshift', 'Redshift', args, outdir, args.zrange)
    plot_1d(groups, 'peakmag', 'Catalogue peak mag', args, outdir, args.magrange, invert=True)
    plot_2d(groups, args, outdir)


if __name__ == '__main__':
    main()
