#!/usr/bin/env python
# coding: utf-8

"""
Stage IV: parametric fits of efficiency vs magnitude, with model selection.

Reads the per-object table from peak_color_efficiency.py and fits, for each
class group (same --classes / --group syntax as combine_efficiency.py), one or
more of these efficiency curves (--models, default: sigmoid floor logistic):

  sigmoid   eps * expit(-(m - m50)/s)                         3 params
            sharp survival cut-off that goes to ZERO at the faint end
  floor     eps * [f + (1 - f) * expit(-(m - m50)/s)]         4 params
            same, but settles at a faint floor eps*f instead of zero
            (f = floorfrac in [0,1]; the floor efficiency is eps*f)
  logistic  expit(a + b*(m - mref))                           2 params
            smooth monotonic decline, no plateau
  linear    clip(p0 + slope*(m - mref), 0, 1)                 2 params
            gentle straight-line decline

mref is the median magnitude of the fitted sample (stored in the output).
eps (bright-end plateau) is fixed to 1 with --fix-eps (sigmoid, floor).

All models are fitted by UNBINNED weighted Bernoulli maximum likelihood on the
per-object flag; plotted points use equal-count bins by default (--equal-width
for equal-width). The model with the lowest AIC is the "best"; it is drawn
thick with a bootstrap 16-84% band, the others thin dashed (ΔAIC in legend).
AIC uses the weighted likelihood, so with --weight class it is approximate.
A constant-efficiency model is also compared (daic_vs_const; negative favours
the best model over a flat line).

Efficiencies (--flags, one figure each):
    db_ok      pure database efficiency (found in the local database)
    gate_cond  colour+point gate efficiency GIVEN in the database
    gate_ok    total colour+point gate efficiency over all input objects
For gate_ok the plot overlays the product eff_db(m) x eff_gate|db(m) (dashed
black-edged) of the best separate fits as a factorisation check.

Output table: per group/flag the best model's parameters (p_*, bootstrap errors
pe_*), aic_<model> for every model, the efficiency at the bright and faint end
of the data, and the magnitude where the efficiency first drops below 0.95,
0.90 and 0.80 (NaN if it never does within the grid).

Publication plots: for every flag and group the best model is also written as
a clean single-panel figure (pub_<flag>_<group>_vs_<xcol>_<ver>.pdf/.png) with
data points, best-fit curve and bootstrap band, and NO text other than the axis
labels (no title, legend or annotations; one group per figure so none is
needed). Axes are auto-zoomed on the active region (data extent in magnitude;
efficiency range spanned by the points, curve and band). Override with
--pub-xlim / --pub-ylim, relabel with --pub-xlabel / --pub-ylabel, disable
with --no-pub.

Class independence: the intended use is ONE pooled fit (no --group/--classes ->
group "All classes") that is applied to every simulated class. Whenever a group
holds several classes, the assumption is tested and printed / written to
class_independence_<flag>_<group>_<ver>.csv:
  * per class: observed vs expected successes under the pooled curve, z-score
  * summed z^2 test over classes (chi2, dof = number of classes)
  * likelihood-ratio / dAIC test: pooled curve vs a separate fit per class
    (same model, unit weights; classes with fewer than --class-test-min-n
    objects, or all-same outcomes, only enter the z-test)
A small p-value / negative dAIC means the classes are NOT consistent with one
curve. Disable with --no-class-test.

Magnitude: default 'peakmag' (catalogue), deliberately NOT the GP peak, since GP
failures have no measured peak (would bias the efficiency).

Requires combine_efficiency_v2.py (or combine_efficiency.py) in this directory.
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
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import combine_efficiency_v2 as ce
except ImportError:
    import combine_efficiency as ce

LEVELS_ABS = (0.95, 0.90, 0.80)   # absolute efficiency levels to report magnitudes for


# ----------------------------------------------------------------------------
# Models: f(m, theta, ctx) -> efficiency; ctx = {'mref': ...}
# ----------------------------------------------------------------------------
def _epsb(fix):
    return (1 - 1e-6, 1.0) if fix else (0.05, 1.0)


MODELS = {
    'sigmoid': {
        'names': ['m50', 's', 'eps'],
        'f': lambda m, t, c: t[2] * expit(-(np.asarray(m) - t[0]) / t[1]),
        'bounds': lambda m, c, fix: [(m.min() - 3, m.max() + 3), (0.05, 5.0), _epsb(fix)],
        'starts': lambda m, c, fix: [(q, s0, 1.0 if fix else 0.95)
                                     for q in np.quantile(m, [0.25, 0.5, 0.75]) for s0 in (0.3, 1.0)],
    },
    'floor': {
        'names': ['m50', 's', 'eps', 'floorfrac'],
        'f': lambda m, t, c: t[2] * (t[3] + (1 - t[3]) * expit(-(np.asarray(m) - t[0]) / t[1])),
        'bounds': lambda m, c, fix: [(m.min() - 3, m.max() + 3), (0.05, 5.0), _epsb(fix), (0.0, 1.0)],
        'starts': lambda m, c, fix: [(q, s0, 1.0 if fix else 0.97, f0)
                                     for q in np.quantile(m, [0.25, 0.5, 0.75])
                                     for s0 in (0.3, 1.0) for f0 in (0.5, 0.85)],
    },
    'logistic': {
        'names': ['a', 'b'],
        'f': lambda m, t, c: expit(t[0] + t[1] * (np.asarray(m) - c['mref'])),
        'bounds': lambda m, c, fix: [(-10, 10), (-5, 5)],
        'starts': lambda m, c, fix: [(a0, b0) for a0 in (1.0, 2.5) for b0 in (-0.1, -0.5)],
    },
    'linear': {
        'names': ['p0', 'slope'],
        'f': lambda m, t, c: np.clip(t[0] + t[1] * (np.asarray(m) - c['mref']), 0, 1),
        'bounds': lambda m, c, fix: [(0.0, 1.0), (-1.0, 1.0)],
        'starts': lambda m, c, fix: [(0.9, -0.02), (0.8, -0.1)],
    },
}


def parse_args():
    env = os.environ.get
    p = argparse.ArgumentParser(description='Efficiency vs magnitude: fits + model selection.')
    p.add_argument('--objects', default=env('EFF_OBJECTS',
                   '/Users/jnordin/data/models/sncosmo/efficiency/efficiency_objects_v8.csv'))
    p.add_argument('--outdir', default=None, help='Default: directory of --objects')
    p.add_argument('--version', '-v', default=env('VERSION', 'v8'))
    p.add_argument('--xcol', default='peakmag', help='Magnitude column to fit against')
    p.add_argument('--flags', nargs='+', choices=list(ce.METRICS),
                   default=['db_ok', 'gate_cond', 'gate_ok'])
    p.add_argument('--models', nargs='+', choices=list(MODELS),
                   default=['sigmoid', 'floor', 'logistic'],
                   help='Candidate models; lowest AIC is the best (default: sigmoid floor logistic)')
    p.add_argument('--classes', nargs='*', default=None)
    p.add_argument('--label', default='Combined')
    p.add_argument('--group', action='append', default=[])
    p.add_argument('--joint-all', action='store_true')
    p.add_argument('--weight', choices=['pooled', 'class'], default='pooled')
    p.add_argument('--fix-eps', action='store_true', help='Fix bright plateau eps=1 (sigmoid, floor)')
    p.add_argument('--nboot', type=int, default=300)
    p.add_argument('--nbins', type=int, default=10, help='Display bins only')
    p.add_argument('--equal-width', action='store_true')
    p.add_argument('--min-per-bin', type=int, default=5)
    p.add_argument('--magrange', nargs=2, type=float, default=None)
    p.add_argument('--ylim', nargs=2, type=float, default=None,
                   help='y-axis range of the efficiency panel (default -0.05 1.08)')
    p.add_argument('--no-class-test', action='store_true',
                   help='Skip the class-independence test')
    p.add_argument('--class-test-min-n', type=int, default=30,
                   help='Min objects for a class to get its own fit in the test')
    p.add_argument('--no-pub', action='store_true', help='Skip publication plots')
    p.add_argument('--pub-formats', nargs='+', default=['pdf', 'png'])
    p.add_argument('--pub-size', nargs=2, type=float, default=[3.5, 2.7], help='inches')
    p.add_argument('--pub-xlim', nargs=2, type=float, default=None)
    p.add_argument('--pub-ylim', nargs=2, type=float, default=None)
    p.add_argument('--pub-xlabel', default=None)
    p.add_argument('--pub-ylabel', default=None)
    p.add_argument('--seed', type=int, default=1)
    return p.parse_args()


# ----------------------------------------------------------------------------
def _bern_nll(p, k, w):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.sum(w * (k * np.log(p) + (1 - k) * np.log(1 - p)))


def n_free(name, fix_eps):
    n = len(MODELS[name]['names'])
    return n - 1 if (fix_eps and 'eps' in MODELS[name]['names']) else n


def fit_model(name, m, k, w, ctx, fix_eps, starts=None):
    M = MODELS[name]
    bounds = M['bounds'](m, ctx, fix_eps)
    starts = starts or M['starts'](m, ctx, fix_eps)
    obj = lambda t: _bern_nll(M['f'](m, t, ctx), k, w)
    best = None
    for st in starts:
        r = minimize(obj, st, method='L-BFGS-B', bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    return best.x, best.fun


def mag_at_level(grid, curve, level):
    """First magnitude where the curve falls below `level` (linear interp)."""
    below = np.where(curve < level)[0]
    if len(below) == 0 or below[0] == 0:
        return np.nan
    i = below[0]
    x0, x1, y0, y1 = grid[i - 1], grid[i], curve[i - 1], curve[i]
    return x0 + (level - y0) * (x1 - x0) / (y1 - y0)


def fit_group(s, flag, args, rng, grid):
    m = s[args.xcol].to_numpy(float)
    k = s[flag].to_numpy(float)
    w = s['w'].to_numpy(float)
    if k.sum() == 0 or k.sum() == len(k):
        print('  all objects have the same outcome; fits not constrained, skipping.')
        return None
    ctx = {'mref': float(np.median(m))}

    fits = {}
    for name in args.models:
        th, f = fit_model(name, m, k, w, ctx, args.fix_eps)
        npar = n_free(name, args.fix_eps)
        fits[name] = {'theta': th, 'nll': f, 'npar': npar, 'aic': 2 * npar + 2 * f,
                      'curve': MODELS[name]['f'](grid, th, ctx)}
    best = min(fits, key=lambda n: fits[n]['aic'])
    for n in fits:
        fits[n]['daic'] = fits[n]['aic'] - fits[best]['aic']

    pc = np.sum(w * k) / np.sum(w)
    aic_const = 2 + 2 * _bern_nll(np.full_like(k, pc), k, w)

    # bootstrap the best model only
    curves, thetas = [], []
    n = len(m)
    for _ in range(args.nboot):
        idx = rng.integers(0, n, n)
        if k[idx].sum() in (0, n):
            continue
        try:
            th, _ = fit_model(best, m[idx], k[idx], w[idx], ctx, args.fix_eps,
                              starts=[tuple(fits[best]['theta'])])
        except Exception:
            continue
        thetas.append(th)
        curves.append(MODELS[best]['f'](grid, th, ctx))
    thetas, curves = np.array(thetas), np.array(curves)
    return {'fits': fits, 'best': best, 'ctx': ctx, 'const': pc,
            'daic_const': fits[best]['aic'] - aic_const,
            'band': np.percentile(curves, [16, 84], axis=0) if len(curves) else None,
            'theta_boot': thetas, 'curves_boot': curves}


# ----------------------------------------------------------------------------
PUB_YLABELS = {'db_ok': 'Database efficiency', 'gate_cond': 'Gate efficiency (in database)',
               'gate_ok': 'Total gate efficiency', 'color_ok': 'Colour efficiency',
               'peak_ok': 'Peak efficiency'}
PUB_XLABELS = {'peakmag': 'Peak magnitude', 'redshift': 'Redshift'}


def pub_plot(pathbase, e, flag, grid, curve, band, args):
    """Clean single-panel figure of the best model: no text but axis labels."""
    xlo, xhi = args.pub_xlim if args.pub_xlim else (e['lo'].min(), e['hi'].max())
    sel = (grid >= xlo) & (grid <= xhi)
    if args.pub_ylim:
        ylo, yhi = args.pub_ylim
    else:
        lo = [e[flag + '_lo'].min(), curve[sel].min()]
        hi = [e[flag + '_hi'].max(), curve[sel].max()]
        if band is not None:
            lo.append(band[0][sel].min())
            hi.append(band[1][sel].max())
        span = max(max(hi) - min(lo), 0.05)
        ylo, yhi = max(min(lo) - 0.08 * span, 0.0), min(max(hi) + 0.08 * span, 1.0 + 0.02)
    rc = {'font.size': 9, 'axes.labelsize': 10, 'xtick.direction': 'in', 'ytick.direction': 'in',
          'xtick.top': True, 'ytick.right': True, 'axes.linewidth': 0.8,
          'pdf.fonttype': 42, 'ps.fonttype': 42}
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=tuple(args.pub_size))
        if band is not None:
            ax.fill_between(grid, *band, color='C0', alpha=0.25, lw=0)
        ax.plot(grid, curve, color='C0', lw=1.6)
        ax.errorbar(e['mid'], e[flag],
                    xerr=[e['mid'] - e['lo'], e['hi'] - e['mid']],
                    yerr=[e[flag] - e[flag + '_lo'], e[flag + '_hi'] - e[flag]],
                    fmt='o', ms=3.5, color='k', ecolor='0.3', capsize=0, elinewidth=0.8)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.set_xlabel(args.pub_xlabel or PUB_XLABELS.get(args.xcol, args.xcol))
        ax.set_ylabel(args.pub_ylabel or PUB_YLABELS.get(flag, ce.METRICS.get(flag, flag)))
        fig.tight_layout(pad=0.4)
        for fmt in args.pub_formats:
            path = f'{pathbase}.{fmt}'
            fig.savefig(path, dpi=300)
            print('Wrote', path)
        plt.close(fig)


def class_independence_test(s, flag, best, ctx, args):
    """Is one pooled curve consistent with all classes? -> (table, summary dict)."""
    M = MODELS[best]
    m = s[args.xcol].to_numpy(float)
    k = s[flag].to_numpy(float)
    cls = s['class'].to_numpy()
    one = np.ones_like(k)
    theta, _ = fit_model(best, m, k, one, ctx, args.fix_eps)      # pooled, unit weights
    p = np.clip(M['f'](m, theta, ctx), 1e-9, 1 - 1e-9)
    npar = n_free(best, args.fix_eps)

    rows, nll_pool, nll_sep, nfit = [], 0.0, 0.0, 0
    for c in np.unique(cls):
        sel = cls == c
        n, O = int(sel.sum()), float(k[sel].sum())
        E, V = p[sel].sum(), (p[sel] * (1 - p[sel])).sum()
        row = {'class': c, 'n': n, 'observed': O, 'expected': E,
               'eff_observed': O / n, 'eff_pooled_mean': E / n,
               'z': (O - E) / np.sqrt(V) if V > 0 else np.nan, 'separate_fit': False}
        if n >= args.class_test_min_n and 0 < O < n:
            th_c, f_c = fit_model(best, m[sel], k[sel], one[sel], ctx, args.fix_eps,
                                  starts=[tuple(theta)] + M['starts'](m[sel], ctx, args.fix_eps))
            nll_sep += f_c
            nll_pool += _bern_nll(p[sel], k[sel], one[sel])
            nfit += 1
            row['separate_fit'] = True
        rows.append(row)
    tab = pd.DataFrame(rows)
    zz = tab['z'].dropna()
    summ = {'ci_chi2_z': float((zz ** 2).sum()), 'ci_chi2_z_dof': int(len(zz))}
    summ['ci_chi2_z_p'] = float(chi2.sf(summ['ci_chi2_z'], max(len(zz), 1)))
    if nfit >= 2:
        lr = 2 * (nll_pool - nll_sep)
        dof = (nfit - 1) * npar
        summ.update(ci_lr=float(lr), ci_lr_dof=int(dof), ci_lr_p=float(chi2.sf(max(lr, 0), dof)),
                    ci_daic=float((2 * npar * nfit + 2 * nll_sep) - (2 * npar + 2 * nll_pool)))
    return tab, summ


def run_flag(flag, groups, args, rng, outdir, store):
    fig, (ax, axn) = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True,
                                  gridspec_kw={'height_ratios': [3, 1]})
    rows = []
    for gi, (gname, sub) in enumerate(groups.items()):
        s_all = sub[np.isfinite(sub[args.xcol])]
        s = s_all[np.isfinite(s_all[flag])].copy()    # drop undefined flags
        print(f'[{flag}][{gname}] N={len(s)} ({len(sub) - len(s_all)} without {args.xcol}, '
              f'{len(s_all) - len(s)} with undefined {flag}), positives={int(s[flag].sum())}')
        if len(s) < 10:
            print('  too few objects, skipping.')
            continue
        s = ce.add_weights(s, args.weight)
        col = f'C{gi}'
        grid = np.linspace(s_all[args.xcol].min() - 0.5, s_all[args.xcol].max() + 0.5, 200)

        edges = ce.make_edges(s[args.xcol].to_numpy(), args.nbins, args.magrange,
                              quantile=not args.equal_width)
        t = ce.binned(s, args.xcol, edges)
        e = t[t['n'] >= args.min_per_bin]
        ax.errorbar(e['mid'], e[flag],
                    xerr=[e['mid'] - e['lo'], e['hi'] - e['mid']],
                    yerr=[e[flag] - e[flag + '_lo'], e[flag + '_hi'] - e[flag]],
                    fmt='o', ms=4, color=col, capsize=2, alpha=0.8, elinewidth=0.8)
        axn.vlines(edges, gi - 0.35, gi + 0.35, color=col, lw=1)
        axn.text(1.01, gi, f'n/bin≈{int(t["n"].median())}', transform=axn.get_yaxis_transform(),
                 va='center', fontsize=7, color=col)

        res = fit_group(s, flag, args, rng, grid)
        store[(gname, flag)] = None
        if res is None:
            continue
        best, fits = res['best'], res['fits']
        names = MODELS[best]['names']
        th, tb = fits[best]['theta'], res['theta_boot']
        err = tb.std(axis=0) if len(tb) > 1 else np.full(len(th), np.nan)
        store[(gname, flag)] = (grid, fits[best]['curve'])

        pstr = ', '.join(f'{n}={v:.2f}' for n, v in zip(names, th))
        ax.plot(grid, fits[best]['curve'], color=col, lw=2,
                label=f'{gname} [{best}]: {pstr}  (dAIC vs const {res["daic_const"]:+.1f})')
        if res['band'] is not None:
            ax.fill_between(grid, *res['band'], color=col, alpha=0.2)
        for n, ft in fits.items():
            if n != best:
                ax.plot(grid, ft['curve'], color=col, lw=1, ls='--', alpha=0.8,
                        label=f'   {n}: dAIC=+{ft["daic"]:.1f}')
        print(f'  best={best} ({pstr}); dAIC: ' +
              ', '.join(f'{n}={ft["daic"]:+.1f}' for n, ft in fits.items()) +
              f'; vs const {res["daic_const"]:+.1f}  (nboot ok={len(tb)})')

        if flag == 'gate_ok':   # factorisation check
            a, b = store.get((gname, 'db_ok')), store.get((gname, 'gate_cond'))
            if a is not None and b is not None:
                ax.plot(grid, a[1] * b[1], color=col, ls=':', lw=2.2,
                        label=f'{gname}: product db × gate|db')

        slug = re.sub(r'[^A-Za-z0-9]+', '_', gname).strip('_')
        ci = None
        if not args.no_class_test and s['class'].nunique() > 1:
            ci_tab, ci = class_independence_test(s, flag, best, res['ctx'], args)
            ci_tab.to_csv(os.path.join(outdir, f'class_independence_{flag}_{slug}_{args.version}.csv'),
                          index=False)
            msg = f'  class independence: sum z^2={ci["ci_chi2_z"]:.1f} (dof {ci["ci_chi2_z_dof"]}, p={ci["ci_chi2_z_p"]:.3f})'
            if 'ci_lr' in ci:
                msg += (f'; LR={ci["ci_lr"]:.1f} (dof {ci["ci_lr_dof"]}, p={ci["ci_lr_p"]:.3f}), '
                        f'dAIC(separate - pooled)={ci["ci_daic"]:+.1f}')
            print(msg)
            print(ci_tab[['class', 'n', 'eff_observed', 'eff_pooled_mean', 'z', 'separate_fit']]
                  .round(3).to_string(index=False))
        if not args.no_pub:
            pub_plot(os.path.join(outdir, f'pub_{flag}_{slug}_vs_{args.xcol}_{args.version}'),
                     e, flag, grid, fits[best]['curve'], res['band'], args)

        mdat = s[args.xcol].to_numpy(float)
        r = {'flag': flag, 'group': gname, 'classes': ','.join(sorted(s['class'].unique())),
             'n': len(s), 'n_positive': int(s[flag].sum()), 'weight': args.weight,
             'model': best, 'mref': res['ctx']['mref'], 'nll': fits[best]['nll'],
             'const_eff': res['const'], 'daic_vs_const': res['daic_const'],
             'nboot_ok': len(tb),
             'mag_min': float(mdat.min()), 'mag_max': float(mdat.max()),
             'eff_bright_end': float(np.interp(mdat.min(), grid, fits[best]['curve'])),
             'eff_faint_end': float(np.interp(mdat.max(), grid, fits[best]['curve']))}
        for n, v, ee in zip(names, th, err):
            r[f'p_{n}'], r[f'pe_{n}'] = v, ee
        if best == 'floor':
            r['floor_eff'] = th[2] * th[3]
        for n, ft in fits.items():
            r[f'aic_{n}'] = ft['aic']
        for lv in LEVELS_ABS:
            r[f'mag_at_eff_{int(lv * 100)}'] = mag_at_level(grid, fits[best]['curve'], lv)
            if len(res['curves_boot']) > 1:
                mb = [mag_at_level(grid, c, lv) for c in res['curves_boot']]
                r[f'mag_at_eff_{int(lv * 100)}_err'] = np.nanstd(mb) if np.isfinite(mb).any() else np.nan
        if ci:
            r.update(ci)
        rows.append(r)

    ax.set_ylim(*(args.ylim if args.ylim else (-0.05, 1.08)))
    ax.set_ylabel(ce.METRICS[flag])
    ax.set_title(f'{ce.METRICS[flag]}: best of {", ".join(args.models)} by AIC; '
                 f'{args.weight} weighting', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, loc='best')
    axn.set_xlabel(args.xcol)
    axn.set_yticks([])
    axn.set_ylabel('bin edges')
    axn.set_ylim(-0.7, len(groups) - 0.3)
    fig.tight_layout()
    path = os.path.join(outdir, f'sigmoid_fit_{flag}_vs_{args.xcol}_{args.version}.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print('Wrote', path)
    return rows


def main():
    args = parse_args()
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.objects))
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = ce.load_objects(args.objects)
    df[args.xcol] = pd.to_numeric(df[args.xcol], errors='coerce')

    groups = ce.build_groups(df, args)
    if args.joint_all and len(groups) > 1:
        groups['Joint (all)'] = pd.concat(groups.values()).drop_duplicates('ZTFID')

    # gate_ok last so the db x gate|db product can be overlaid on it
    flags = [f for f in args.flags if f != 'gate_ok'] + (['gate_ok'] if 'gate_ok' in args.flags else [])
    store, rows = {}, []
    for flag in flags:
        rows += run_flag(flag, groups, args, rng, outdir, store)

    if rows:
        sp = os.path.join(outdir, f'sigmoid_fit_params_{args.version}.csv')
        pd.DataFrame(rows).to_csv(sp, index=False)
        print('Wrote', sp)


if __name__ == '__main__':
    main()

    