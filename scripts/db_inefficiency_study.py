#!/usr/bin/env python
# coding: utf-8

"""
Why are objects from the input catalogue missing from the local database?

Takes the per-object table (peak_color_efficiency.py; needs db_ok, so --db-only
output is fine), joins it to the catalogue metadata (--bts-csv and/or --alt-csv)
and looks for what separates objects found in the database (db_ok=1) from
missing ones (db_ok=0):

  * time        catalogue peak/discovery time if a column is found (or given
                with --time-col), the ZTF-name year, and the ZTF-name sequence
                rank (ZTF names are assigned in order, so it is a monotonic
                time proxy even without dates)
  * position    RA, Dec, galactic latitude |b|, sky maps
  * brightness  peak magnitude, redshift
  * class and ANY other column in the catalogue files (automatic screening)

Outputs (in --outdir)
  dbstudy_screen_numeric_<ver>.csv   per numeric variable: AUC(found vs missing)
                                     [0.5 = no information], Mann-Whitney p, medians
  dbstudy_screen_categ_<ver>.csv     per category level: N, efficiency (Wilson),
                                     with chi2 p-value per variable
  dbstudy_regression_<ver>.csv       multivariate logistic regression of db_ok
                                     (separates correlated explanations)
  dbstudy_missing_<ver>.csv          the missing objects with their covariates
  dbstudy_vs_covariates_<ver>.png    efficiency vs key covariates (equal-count bins)
  dbstudy_sky_<ver>.png              sky maps (equatorial, galactic)
  dbstudy_time_<ver>.png             peak magnitude vs time, found vs missing

Use --maglim to restrict to the bright subsample where the catalogue should be
complete in the database (e.g. --maglim 19).

Caveat: BTS-explorer column names are auto-detected (the list of columns found
is printed); override with --time-col / --time-format if the guess is wrong.
"""

import argparse
import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.stats import mannwhitneyu, chi2_contingency

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import combine_efficiency_v2 as ce

TIME_CANDIDATES = ['peakt', 'peak_mjd', 'peakmjd', 'peak_jd', 'jd_peak', 'discovery_mjd',
                   'disc_mjd', 'mjd', 'jd', 'peak_time', 'discovery_date']
OBJ_ONLY = {'ZTFID', 'class', 'status', 'error', 'db_ok', 'vetoed', 'color_ok', 'peak_ok',
            'gate_ok', 'gate_cond', 'ndet', 'nbr_bands', 'presum', 'postsum'}


def parse_args():
    env = os.environ.get
    p = argparse.ArgumentParser(description='Study covariates of database (in)efficiency.')
    p.add_argument('--objects', default=env('EFF_OBJECTS',
                   '/Users/jnordin/data/models/sncosmo/efficiency/efficiency_objects_v8.csv'))
    p.add_argument('--bts-csv', default=env('BTS_CSV', '/Users/jnordin/data/ztf/bts/bts_explorer_241122.csv'))
    # We do not have the later files in the database, so do not use for efficiency
    #p.add_argument('--bts-csv', default=env('BTS_CSV', '/Users/jnordin/data/ztf/bts/bts_explorer_260601.csv'))
    p.add_argument('--alt-csv', default=env('ALT_CSV', '/Users/jnordin/data/ztf/dr4/dr4_slsntde_coordlist.csv'))
    p.add_argument('--outdir', default=None, help='Default: directory of --objects')
    p.add_argument('--version', '-v', default=env('VERSION', 'v8'))
    p.add_argument('--classes', nargs='*', default=None, help='Restrict to these classes')
    p.add_argument('--maglim', type=float, default=None,
                   help='Only objects with catalogue peakmag <= this')
    p.add_argument('--time-col', default=None, help='Catalogue column holding a time')
    p.add_argument('--time-format', choices=['auto', 'mjd', 'jd'], default='auto')
    p.add_argument('--nbins', type=int, default=8)
    p.add_argument('--regress-class', action='store_true',
                   help='Add class dummies (classes with N>=20) to the regression')
    return p.parse_args()


# ----------------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------------
def load_meta(bts_csv, alt_csv):
    from astropy.coordinates import SkyCoord
    frames = []
    if bts_csv and os.path.exists(bts_csv):
        b = pd.read_csv(bts_csv)
        print(f'BTS columns ({len(b.columns)}): {list(b.columns)}')
        try:
            c = SkyCoord(b['RA'], b['Dec'], unit=('hour', 'deg'))
            b['ra'], b['dec'] = c.ra.deg, c.dec.deg
        except Exception as e:
            print('WARNING: could not parse BTS RA/Dec:', e)
        b['meta_src'] = 'BTS'
        frames.append(b)
    if alt_csv and os.path.exists(alt_csv):
        a = pd.read_csv(alt_csv, index_col=0)
        a['ra'], a['dec'] = a['RAdeg'], a['Decdeg']
        a['meta_src'] = a['source'] if 'source' in a.columns else 'ALT'
        frames.append(a)
    if not frames:
        raise SystemExit('No metadata files found.')
    return pd.concat(frames, ignore_index=True, sort=False).drop_duplicates('ZTFID')


def to_mjd(v, fmt):
    v = pd.to_numeric(v, errors='coerce')
    med = np.nanmedian(v)
    if fmt == 'auto':
        fmt = 'jd' if med > 2.4e6 else 'mjd'
        print(f'  time format guessed as {fmt.upper()} (median {med:.1f}); '
              f'override with --time-format')
    return v - 2400000.5 if fmt == 'jd' else v


def add_derived(df, args):
    # ZTF-name based time proxies
    m = df['ZTFID'].str.extract(r'ZTF(\d\d)([a-z]+)')
    df['ztf_year'] = 2000 + pd.to_numeric(m[0], errors='coerce')
    df['ztf_seq'] = m[1].map(lambda s: sum((ord(ch) - 97) * 26 ** i
                                           for i, ch in enumerate(reversed(s))) if isinstance(s, str) else np.nan)
    order = df.sort_values(['ztf_year', 'ztf_seq']).reset_index().index
    df['ztf_order'] = np.nan
    df.loc[df.sort_values(['ztf_year', 'ztf_seq']).index, 'ztf_order'] = np.arange(len(df)) / max(len(df) - 1, 1)

    # catalogue time
    tcol = args.time_col
    if tcol is None:
        low = {c.lower(): c for c in df.columns}
        tcol = next((low[c] for c in TIME_CANDIDATES if c in low), None)
    if tcol is not None and tcol in df.columns:
        print(f'Using catalogue time column "{tcol}"')
        df['mjd'] = to_mjd(df[tcol], args.time_format)
    else:
        print('No catalogue time column found; using ZTF-name year / rank as time proxies.')
        df['mjd'] = np.nan

    # galactic latitude
    if 'ra' in df.columns:
        from astropy.coordinates import SkyCoord
        ok = df['ra'].notna() & df['dec'].notna()
        df['gal_b'] = np.nan
        df['gal_l'] = np.nan
        if ok.any():
            g = SkyCoord(df.loc[ok, 'ra'].values, df.loc[ok, 'dec'].values, unit='deg').galactic
            df.loc[ok, 'gal_b'] = g.b.deg
            df.loc[ok, 'gal_l'] = g.l.deg
        df['abs_b'] = df['gal_b'].abs()
    return df


# ----------------------------------------------------------------------------
# Screening
# ----------------------------------------------------------------------------
def screen_numeric(df, cols):
    f, mi = df[df['db_ok'] == 1], df[df['db_ok'] == 0]
    rows = []
    for c in cols:
        x1, x0 = f[c].dropna().to_numpy(float), mi[c].dropna().to_numpy(float)
        if len(x1) < 5 or len(x0) < 5 or np.nanstd(np.r_[x1, x0]) == 0:
            continue
        U, p = mannwhitneyu(x1, x0, alternative='two-sided')
        rows.append({'variable': c, 'n_found': len(x1), 'n_missing': len(x0),
                     'auc_found_gt_missing': U / (len(x1) * len(x0)),
                     'mannwhitney_p': p, 'median_found': np.median(x1),
                     'median_missing': np.median(x0)})
    t = pd.DataFrame(rows)
    if len(t):
        t['abs_auc_dev'] = (t['auc_found_gt_missing'] - 0.5).abs()
        t = t.sort_values('abs_auc_dev', ascending=False)
    return t


def screen_categorical(df, cols):
    rows = []
    for c in cols:
        ct = pd.crosstab(df[c].astype(str), df['db_ok'])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        p = chi2_contingency(ct)[1]
        for lvl, r in ct.iterrows():
            n, k = int(r.sum()), int(r.get(1.0, 0))
            e, lo, hi = ce.wilson(k / n, n) if False else (k / n,) + ce.wilson(k / n, n)
            rows.append({'variable': c, 'level': lvl, 'n': n, 'n_found': k,
                         'eff': e, 'eff_lo': lo, 'eff_hi': hi, 'chi2_p': p})
    return pd.DataFrame(rows)


def logit_fit(X, y, ridge=1e-6, iters=100):
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = expit(X @ beta)
        W = p * (1 - p) + 1e-9
        H = X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p) - ridge * beta)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta, np.sqrt(np.diag(np.linalg.inv(H)))


def regression(df, covs, use_class):
    d = df.dropna(subset=covs + ['db_ok']).copy()
    X = [np.ones(len(d))]
    names = ['intercept']
    for c in covs:
        sd = d[c].std()
        if sd == 0:
            continue
        X.append(((d[c] - d[c].mean()) / sd).to_numpy())
        names.append(f'{c} (per 1 sd)')
    if use_class:
        vc = d['class'].value_counts()
        keep = [c for c in vc.index if vc[c] >= 20]
        for c in keep[1:]:                     # most common class = reference
            X.append((d['class'] == c).astype(float).to_numpy())
            names.append(f'class={c} (vs {keep[0]})')
    X = np.column_stack(X)
    y = d['db_ok'].to_numpy(float)
    if y.sum() in (0, len(y)):
        return pd.DataFrame()
    beta, se = logit_fit(X, y)
    out = pd.DataFrame({'term': names, 'coef': beta, 'se': se, 'z': beta / se,
                        'odds_ratio': np.exp(beta)})
    out.attrs['n'] = len(d)
    return out


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def eff_bins(x, y, nbins):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2 * nbins or np.unique(x).size < 2:
        return None
    edges = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for i in range(len(edges) - 1):
        s = y[idx == i]
        if len(s) == 0:
            continue
        p = s.mean()
        lo, hi = ce.wilson(p, len(s))
        rows.append((x[idx == i].mean(), p, lo, hi, edges[i], edges[i + 1]))
    return np.array(rows)


def plot_covariates(df, covs, nbins, path):
    covs = [c for c in covs if c in df.columns and df[c].notna().sum() > 20]
    ncol = min(4, len(covs))
    nrow = int(np.ceil(len(covs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.3 * nrow), squeeze=False)
    y = df['db_ok'].to_numpy(float)
    for ax, c in zip(axes.ravel(), covs):
        r = eff_bins(df[c].to_numpy(float), y, nbins)
        if r is None:
            ax.set_title(f'{c}: insufficient data', fontsize=8)
            continue
        ax.errorbar(r[:, 0], r[:, 1], xerr=[r[:, 0] - r[:, 4], r[:, 5] - r[:, 0]],
                    yerr=[r[:, 1] - r[:, 2], r[:, 3] - r[:, 1]], fmt='o', ms=4, capsize=2, lw=0.8)
        ax.axhline(np.nanmean(y), color='grey', ls=':', lw=1)
        # Full range
#        ax.set_ylim(-0.05, 1.05)
        # For efficient 
        ax.set_ylim(0.9, 1.02)
        ax.set_xlabel(c)
        ax.set_ylabel('Fraction in database')
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(covs):]:
        ax.axis('off')
    fig.suptitle(f'Database efficiency vs covariates (equal-count bins; dotted = overall '
                 f'{np.nanmean(y):.2f}; N={len(df)})', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print('Wrote', path)


def plot_sky(df, path):
    d = df.dropna(subset=['ra', 'dec'])
    if len(d) == 0:
        return
    fig = plt.figure(figsize=(14, 4.8))
    for k, (xc, yc, title, wrap) in enumerate([('ra', 'dec', 'Equatorial', True),
                                               ('gal_l', 'gal_b', 'Galactic', True)]):
        ax = fig.add_subplot(1, 2, k + 1, projection='mollweide')
        for flag, col, sz, lab in ((1, 'lightgrey', 6, 'in database'), (0, 'crimson', 14, 'missing')):
            s = d[d['db_ok'] == flag].dropna(subset=[xc, yc])
            lon = np.radians(((s[xc].to_numpy() + 180) % 360) - 180)
            ax.scatter(lon, np.radians(s[yc].to_numpy()), s=sz, c=col, label=f'{lab} ({len(s)})',
                       alpha=0.8, lw=0)
        ax.grid(alpha=0.3)
        ax.set_title(title, fontsize=10)
        if k == 0:
            ax.legend(fontsize=8, loc='lower left')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print('Wrote', path)


def plot_time(df, path):
    use_mjd = df['mjd'].notna().sum() > 20
    xc = 'mjd' if use_mjd else 'ztf_order'
    d = df.dropna(subset=[xc, 'peakmag'])
    if len(d) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for flag, col, sz, lab in ((1, 'lightgrey', 8, 'in database'), (0, 'crimson', 16, 'missing')):
        s = d[d['db_ok'] == flag]
        ax.scatter(s[xc], s['peakmag'], s=sz, c=col, alpha=0.8, lw=0, label=f'{lab} ({len(s)})')
    ax.invert_yaxis()
    ax.set_xlabel('Catalogue time (MJD)' if use_mjd else 'ZTF-ID rank (time proxy, 0..1)')
    ax.set_ylabel('Catalogue peak mag')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print('Wrote', path)


# ----------------------------------------------------------------------------
def main():
    args = parse_args()
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.objects))
    os.makedirs(outdir, exist_ok=True)
    v = args.version

    obj = ce.load_objects(args.objects)
    meta = load_meta(args.bts_csv, args.alt_csv)
    extra = [c for c in meta.columns if c not in obj.columns or c == 'ZTFID']
    df = obj.merge(meta[extra], on='ZTFID', how='left')
    print(f'{len(df)} objects; {df["ra"].notna().sum() if "ra" in df else 0} matched to catalogue coordinates')

    df = df[df['db_ok'].notna()].copy()
    if args.classes:
        df = df[df['class'].isin(args.classes)]
    if args.maglim is not None:
        df = df[df['peakmag'] <= args.maglim]
        print(f'Restricted to peakmag <= {args.maglim}: {len(df)} objects')
    df = add_derived(df, args)

    n, k = len(df), int(df['db_ok'].sum())
    print(n,k)
    print(f'\nOverall: {k}/{n} in database ({k / n:.3f}); {n - k} missing')
    print(df.groupby('class')['db_ok'].agg(['size', 'sum', 'mean']).rename(
        columns={'size': 'n', 'sum': 'n_found', 'mean': 'eff'}).round(3).to_string())

    # --- screening -----------------------------------------------------
    derived = ['redshift', 'peakmag', 'ra', 'dec', 'gal_b', 'abs_b', 'gal_l',
               'ztf_year', 'ztf_seq', 'ztf_order', 'mjd']
    meta_num = [c for c in extra if c not in OBJ_ONLY and c not in derived
                and pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in derived + meta_num if c in df.columns]
    t = screen_numeric(df, num_cols)
    t.to_csv(os.path.join(outdir, f'dbstudy_screen_numeric_{v}.csv'), index=False)
    print('\nNumeric screening (AUC 0.5 = no information; sorted by |AUC-0.5|):')
    print(t.head(15).round(4).to_string(index=False))

    categ = ['class', 'meta_src', 'ztf_year'] + [
        c for c in extra if c not in OBJ_ONLY and c not in derived and c not in ('type', 'IAUID')
        and not pd.api.types.is_numeric_dtype(df[c]) and 1 < df[c].nunique() <= 30]
    categ = [c for c in dict.fromkeys(categ) if c in df.columns]
    tc = screen_categorical(df, categ)
    tc.to_csv(os.path.join(outdir, f'dbstudy_screen_categ_{v}.csv'), index=False)
    if len(tc):
        print('\nCategorical screening (chi2 p-value per variable):')
        print(tc.groupby('variable')['chi2_p'].first().sort_values().round(5).to_string())
        yr = tc[tc['variable'] == 'ztf_year']
        if len(yr):
            print('\nEfficiency by ZTF-name year:')
            print(yr[['level', 'n', 'eff']].round(3).to_string(index=False))

    # --- regression ----------------------------------------------------
    timecov = 'mjd' if df['mjd'].notna().sum() > 20 else 'ztf_order'
    covs = [c for c in ['peakmag', 'redshift', 'abs_b', 'dec', timecov] if c in df.columns]
    reg = regression(df, covs, args.regress_class)
    if len(reg):
        reg.to_csv(os.path.join(outdir, f'dbstudy_regression_{v}.csv'), index=False)
        print(f'\nLogistic regression of db_ok (N={reg.attrs["n"]}; coefficients per 1 sd):')
        print(reg.round(3).to_string(index=False))

    # --- missing list and plots ----------------------------------------
    cols = [c for c in ['ZTFID', 'class', 'peakmag', 'redshift', 'ra', 'dec', 'gal_b', 'mjd',
                        'ztf_year', 'status', 'vetoed'] if c in df.columns]
    df[df['db_ok'] == 0][cols].sort_values('peakmag').to_csv(
        os.path.join(outdir, f'dbstudy_missing_{v}.csv'), index=False)

    plot_covariates(df, ['peakmag', 'redshift', 'mjd', 'ztf_order', 'ra', 'dec', 'abs_b', 'ztf_year'],
                    args.nbins, os.path.join(outdir, f'dbstudy_vs_covariates_{v}.png'))
    plot_sky(df, os.path.join(outdir, f'dbstudy_sky_{v}.png'))
    plot_time(df, os.path.join(outdir, f'dbstudy_time_{v}.png'))


if __name__ == '__main__':
    main()
