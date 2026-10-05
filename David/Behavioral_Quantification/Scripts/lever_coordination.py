"""
lever_coordination.py

Standalone module to test whether rat lever presses are coordinated beyond chance
using a within-session shuffle null model.

Usage:
    python lever_coordination.py --input path/to/trials.csv --outdir outputs

Assumptions about input CSV columns:
- Session identifier: `Session` or `session`
- Trial identifier: `Trial` or `trial`
- Lever onset absolute time: `LeverOnsetAbsTime` or `LeverOnset_Abs` or use `AbsTime` + `TrialTime` inference
- Press times: for rat1 and rat2 absolute times or relative to lever onset: prefer `rat1_press_abs`, `rat2_press_abs`, or `rat1_PressTime`, `rat2_PressTime`.
- Also accepts `AbsTime` and `TrialTime` columns as present in the repo; code will try to derive relative times from these.

The module produces PNG and PDF figures and CSV summary outputs.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

np.random.seed(0)


def safe_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_data(path):
    df = pd.read_csv(path)
    return df


def compute_relative_press_metrics(df, session_col='Session', trial_col='Trial',
                                   abs_time_col='AbsTime', trial_time_col='TrialTime',
                                   rat1_col_candidates=['rat1_press_abs','rat1_press_time','Rat1PressAbs','Rat1Press'],
                                   rat2_col_candidates=['rat2_press_abs','rat2_press_time','Rat2PressAbs','Rat2Press']):
    # Try to detect columns
    rat1_col = safe_column(df, rat1_col_candidates)
    rat2_col = safe_column(df, rat2_col_candidates)

    # If explicit rat press absolute times not present, try to derive from AbsTime + TrialTime
    # Many datasets have one press per trial per rat; we assume provided columns refer to press absolute time.

    df2 = df.copy()

    # Attempt to infer Lever onset absolute time. If not present, assume trial start is when lever appears (TrialTime==0)
    # If df has AbsTime and TrialTime, then lever onset per trial is the AbsTime where TrialTime==0 for that trial.
    lever_onset = None
    if 'AbsTime' in df.columns and 'TrialTime' in df.columns:
        # find rows where TrialTime == 0 per session/trial
        # We'll group by session and trial and take min AbsTime where TrialTime <= small threshold
        lever_onset = df.groupby([session_col, trial_col]).apply(
            lambda g: g.loc[g['TrialTime'] <= 1e-6, 'AbsTime'].min() if (g['TrialTime'] <= 1e-6).any() else g['AbsTime'].min()
        ).rename('lever_onset_abs').reset_index()
        df2 = df2.merge(lever_onset, on=[session_col, trial_col], how='left')
    else:
        # try common lever onset columns
        lever_onset_col = safe_column(df, ['LeverOnsetAbsTime','LeverOnset_Abs','lever_onset_abs'])
        if lever_onset_col:
            df2 = df2.rename(columns={lever_onset_col: 'lever_onset_abs'})
        else:
            df2['lever_onset_abs'] = np.nan

    # Create rat press absolute time columns if available
    if rat1_col:
        df2['rat1_press_abs'] = df2[rat1_col]
    if rat2_col:
        df2['rat2_press_abs'] = df2[rat2_col]

    # If press times are relative rather than absolute, try to detect columns with 'rel' or 'TrialTime'
    # If ratX_press_abs missing but have ratX_press_rel, convert
    if 'rat1_press_rel' in df2.columns and 'lever_onset_abs' in df2.columns:
        df2['rat1_press_abs'] = df2['lever_onset_abs'] + df2['rat1_press_rel']
    if 'rat2_press_rel' in df2.columns and 'lever_onset_abs' in df2.columns:
        df2['rat2_press_abs'] = df2['lever_onset_abs'] + df2['rat2_press_rel']

    # Compute press relative times
    df2['rat1_pressed'] = ~df2['rat1_press_abs'].isna()
    df2['rat2_pressed'] = ~df2['rat2_press_abs'].isna()
    df2['both_pressed'] = df2['rat1_pressed'] & df2['rat2_pressed']

    # Compute relative times to lever onset
    df2['rat1_press_rel'] = df2['rat1_press_abs'] - df2['lever_onset_abs']
    df2['rat2_press_rel'] = df2['rat2_press_abs'] - df2['lever_onset_abs']

    # Optionally drop negative relative times (press before lever onset)
    df2.loc[df2['rat1_press_rel'] < 0, 'rat1_pressed'] = False
    df2.loc[df2['rat2_press_rel'] < 0, 'rat2_pressed'] = False
    df2['both_pressed'] = df2['rat1_pressed'] & df2['rat2_pressed']

    # Compute lag metrics where both pressed
    def compute_lags(group):
        if not group['both_pressed'].any():
            return pd.Series()

    # Only compute per-row
    df2['first_press_rel'] = np.nan
    df2['second_press_rel'] = np.nan
    df2['lag_signed'] = np.nan
    df2['lag_abs'] = np.nan

    both_idx = df2['both_pressed']
    r1 = df2.loc[both_idx, 'rat1_press_rel']
    r2 = df2.loc[both_idx, 'rat2_press_rel']
    df2.loc[both_idx, 'first_press_rel'] = np.minimum(r1, r2)
    df2.loc[both_idx, 'second_press_rel'] = np.maximum(r1, r2)
    df2.loc[both_idx, 'lag_signed'] = r2 - r1
    df2.loc[both_idx, 'lag_abs'] = np.abs(df2.loc[both_idx, 'lag_signed'])

    return df2


def shuffle_press_times_within_session(trials_df, session_col='Session', n_shuffle=1000,
                                      random_state=None):
    rng = np.random.default_rng(random_state)
    # We'll collect null lag_abs and lag_signed for each shuffle in long format
    sessions = trials_df[session_col].unique()
    records = []
    per_shuffle_summary = []

    # Preselect trials where both pressed
    both = trials_df[trials_df['both_pressed']].copy()

    for s in sessions:
        sess_trials = both[both[session_col] == s]
        if sess_trials.empty:
            continue
        # arrays of rat1 and rat2 relative press times
        r1 = sess_trials['rat1_press_rel'].values
        r2 = sess_trials['rat2_press_rel'].values
        trials_idx = sess_trials.index.values

        # For each shuffle, preserve r1 and permute r2
        for sh in range(n_shuffle):
            perm = rng.permutation(len(r2))
            r2_perm = r2[perm]
            lag_signed = r2_perm - r1
            lag_abs = np.abs(lag_signed)
            for ti, ls, la in zip(trials_idx, lag_signed, lag_abs):
                records.append({
                    'Session': s,
                    'trial_index': int(ti),
                    'shuffle': sh,
                    'lag_signed_null': ls,
                    'lag_abs_null': la,
                })

    null_long = pd.DataFrame.from_records(records)

    return null_long


def summarize_null(null_long):
    # Compute per-shuffle summaries across all sessions/trials
    grouped = null_long.groupby('shuffle')
    summaries = grouped['lag_abs_null'].agg(['mean','median'])
    # fraction below thresholds
    thresholds = [0.1, 0.25, 0.5, 1.0]
    for t in thresholds:
        summaries[f'fract_below_{t}'] = grouped.apply(lambda g, thr=t: (g['lag_abs_null'] <= thr).mean())
    return summaries.reset_index()


def compute_global_real_metrics(trials_df):
    both = trials_df[trials_df['both_pressed']]
    out = {}
    out['n_trials_both'] = len(both)
    out['mean_lag_abs'] = both['lag_abs'].mean()
    out['median_lag_abs'] = both['lag_abs'].median()
    thresholds = [0.1,0.25,0.5,1.0]
    for t in thresholds:
        out[f'fract_below_{t}'] = (both['lag_abs'] <= t).mean()
    out['fract_rat1_first'] = (both['lag_signed'] < 0).mean()
    out['fract_rat2_first'] = (both['lag_signed'] > 0).mean()
    return pd.Series(out)


def empirical_pvalues(real_metric, null_vals, higher_better=False):
    # For metric where higher values indicate stronger effect (e.g. fraction below? depends), compute p-value
    null_vals = np.asarray(null_vals)
    if higher_better:
        p = (null_vals >= real_metric).sum() / len(null_vals)
    else:
        p = (null_vals <= real_metric).sum() / len(null_vals)
    # two-sided could be considered, but we'll report one-sided depending on hypothesis
    return p


# Plotting utilities

def compute_ecdf(data, xs=None):
    data = np.sort(np.asarray(data))
    n = len(data)
    if xs is None:
        xs = data
        ys = np.arange(1, n+1) / n
        return xs, ys
    ys = np.searchsorted(data, xs, side='right') / n
    return xs, ys


def plot_abs_lag_histogram(real_lags, null_long, outdir, bins=None, fname_prefix='fig1_abs_lag_hist'):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if bins is None:
        bins = np.linspace(0, np.percentile(real_lags, 99), 40)
    # compute null histogram per shuffle
    sh_groups = null_long.groupby('shuffle')
    hist_null = []
    for sh, g in sh_groups:
        h, _ = np.histogram(g['lag_abs_null'], bins=bins, density=True)
        hist_null.append(h)
    hist_null = np.array(hist_null)
    mean_null = hist_null.mean(axis=0)
    lower = np.percentile(hist_null, 2.5, axis=0)
    upper = np.percentile(hist_null, 97.5, axis=0)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(real_lags, bins=bins, density=True, alpha=0.7, color='C0', label='Real')
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax.plot(bin_centers, mean_null, color='C1', lw=2, label='Shuffled mean')
    ax.fill_between(bin_centers, lower, upper, color='C1', alpha=0.25, label='95% null')
    ax.set_xlabel('Absolute lag (s)')
    ax.set_ylabel('Density')
    ax.legend()
    for ext in ('.png', '.pdf'):
        fig.savefig(outdir / (fname_prefix + ext), bbox_inches='tight')
    plt.close(fig)


def plot_abs_lag_cdf(real_lags, null_long, outdir, xs=None, fname_prefix='fig2_abs_lag_cdf'):
    outdir = Path(outdir)
    if xs is None:
        xs = np.linspace(0, np.percentile(real_lags,99), 200)
    real_xs, real_ys = compute_ecdf(real_lags, xs=xs)
    # null cdfs per shuffle
    sh_groups = null_long.groupby('shuffle')
    null_ys = []
    for sh, g in sh_groups:
        _, ys = compute_ecdf(g['lag_abs_null'], xs=xs)
        null_ys.append(ys)
    null_ys = np.array(null_ys)
    mean_null = null_ys.mean(axis=0)
    lower = np.percentile(null_ys, 2.5, axis=0)
    upper = np.percentile(null_ys, 97.5, axis=0)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(real_xs, real_ys, color='C0', lw=2, label='Real')
    ax.plot(xs, mean_null, color='C1', lw=2, label='Shuffled mean')
    ax.fill_between(xs, lower, upper, color='C1', alpha=0.25, label='95% null')
    ax.set_xlabel('Absolute lag (s)')
    ax.set_ylabel('Cumulative fraction')
    ax.legend()
    for ext in ('.png', '.pdf'):
        fig.savefig(outdir / (fname_prefix + ext), bbox_inches='tight')
    plt.close(fig)


def plot_excess_synchrony(real_lags, null_long, outdir, xs=None, fname_prefix='fig3_excess_sync'):
    outdir = Path(outdir)
    if xs is None:
        xs = np.linspace(0, np.percentile(real_lags,99), 200)
    real_frac = [np.mean(np.array(real_lags) <= x) for x in xs]
    sh_groups = null_long.groupby('shuffle')
    null_fracs = []
    for sh, g in sh_groups:
        null_fracs.append([np.mean(g['lag_abs_null'].values <= x) for x in xs])
    null_fracs = np.array(null_fracs)
    mean_null = null_fracs.mean(axis=0)
    lower = np.percentile(null_fracs, 2.5, axis=0)
    upper = np.percentile(null_fracs, 97.5, axis=0)

    excess = np.array(real_frac) - mean_null

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(xs, excess, color='C0', lw=2)
    ax.fill_between(xs, (np.array(real_frac) - upper), (np.array(real_frac) - lower), color='C1', alpha=0.25)
    ax.axhline(0, color='k', ls='--', lw=1)
    ax.set_xlabel('Threshold (s)')
    ax.set_ylabel('Excess synchrony (Real - Mean null)')
    for ext in ('.png', '.pdf'):
        fig.savefig(outdir / (fname_prefix + ext), bbox_inches='tight')
    plt.close(fig)


def plot_signed_lag_distribution(real_signed, outdir, fname_prefix='fig4_signed_lag'):
    outdir = Path(outdir)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(real_signed, bins=40, color='C0', alpha=0.7)
    ax.set_xlabel('Signed lag (s) (rat2 - rat1)')
    ax.set_ylabel('Count')
    for ext in ('.png', '.pdf'):
        fig.savefig(outdir / (fname_prefix + ext), bbox_inches='tight')
    plt.close(fig)


def plot_first_press_binned_comparison(trials_df, null_long, bins=[0,0.5,1.0,2.0, 9999], outdir='outputs', fname_prefix='fig5_first_press_binned'):
    outdir = Path(outdir)
    both = trials_df[trials_df['both_pressed']].copy()
    both['first_bin'] = pd.cut(both['first_press_rel'], bins=bins, right=False)
    # real stats per bin
    bin_stats = both.groupby('first_bin')['lag_abs'].agg(['mean','median','count']).reset_index()

    # null: need to map null_long trial_index to first_bin. null_long currently stores trial_index as original df index
    null_with_bin = null_long.merge(both.reset_index()[['index','first_bin']], left_on='trial_index', right_on='index', how='left')
    sh_groups = null_with_bin.groupby(['shuffle','first_bin'])['lag_abs_null'].agg(['mean','median','count']).reset_index()

    # compute per-bin null mean and 95% interval
    bins_unique = bin_stats['first_bin']
    means = []
    lowers = []
    uppers = []
    for b in bins_unique:
        arr = sh_groups[sh_groups['first_bin'] == b]['mean'].values
        if len(arr)==0:
            means.append(np.nan); lowers.append(np.nan); uppers.append(np.nan)
        else:
            means.append(np.nanmean(arr)); lowers.append(np.nanpercentile(arr,2.5)); uppers.append(np.nanpercentile(arr,97.5))

    fig, ax = plt.subplots(figsize=(6,4))
    x = np.arange(len(bins_unique))
    ax.errorbar(x, bin_stats['mean'], yerr=0, fmt='o', color='C0', label='Real mean')
    ax.errorbar(x, means, yerr=[np.array(means)-np.array(lowers), np.array(uppers)-np.array(means)], fmt='s', color='C1', label='Null mean (95% CI)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in bins_unique], rotation=45)
    ax.set_ylabel('Mean abs lag (s)')
    ax.legend()
    for ext in ('.png', '.pdf'):
        fig.savefig(outdir / (fname_prefix + ext), bbox_inches='tight')
    plt.close(fig)


def plot_session_level_summary(trials_df, null_long, outdir, threshold=0.25, fname_prefix='fig6_session_summary'):
    outdir = Path(outdir)
    both = trials_df[trials_df['both_pressed']].copy()
    sess_real = both.groupby('Session')['lag_abs'].agg(['mean','median', lambda x, thr=threshold: (x<=thr).mean()])
    sess_real.columns = ['mean','median', f'fract_below_{threshold}']

    # null per session: group null_long by shuffle and session
    null_sessions = null_long.groupby(['Session','shuffle'])['lag_abs_null'].agg(['mean','median', lambda x, thr=threshold: (x<=thr).mean()]).reset_index()
    null_sessions = null_sessions.rename(columns={'<lambda_0>':f'fract_below_{threshold}'})

    # compute per-session null mean and 95% interval for the fraction
    sess_stats = []
    for s, g in null_sessions.groupby('Session'):
        arr = g[f'fract_below_{threshold}'].values
        sess_stats.append({'Session': s, 'null_mean': np.nanmean(arr), 'null_lo': np.nanpercentile(arr,2.5), 'null_hi': np.nanpercentile(arr,97.5)})
    sess_stats = pd.DataFrame(sess_stats).set_index('Session')

    merged = sess_real.merge(sess_stats, left_index=True, right_index=True, how='left')
    merged = merged.reset_index()

    fig, ax = plt.subplots(figsize=(8,4))
    x = np.arange(len(merged))
    ax.errorbar(x, merged[f'fract_below_{threshold}'], yerr=0, fmt='o', color='C0', label='Real')
    ax.errorbar(x, merged['null_mean'], yerr=[merged['null_mean']-merged['null_lo'], merged['null_hi']-merged['null_mean']], fmt='s', color='C1', label='Null (95% CI)')
    ax.set_xticks(x)
    ax.set_xticklabels(merged['Session'].astype(str), rotation=45)
    ax.set_ylabel(f'Fraction lag_abs <= {threshold} s')
    ax.legend()
    for ext in ('.png', '.pdf'):
        fig.savefig(outdir / (fname_prefix + ext), bbox_inches='tight')
    plt.close(fig)


def main(args):
    df = load_data(args.input)
    print(f'Loaded {len(df)} rows from {args.input}')
    trials = compute_relative_press_metrics(df)
    trials.to_csv(Path(args.outdir)/'trial_level_metrics.csv', index=False)
    print('Computed trial-level metrics; saved trial_level_metrics.csv')

    null_long = shuffle_press_times_within_session(trials, n_shuffle=args.n_shuffle, random_state=args.seed)
    null_long.to_csv(Path(args.outdir)/'null_long.csv', index=False)
    print('Generated null shuffles; saved null_long.csv')

    null_summary = summarize_null(null_long)
    null_summary.to_csv(Path(args.outdir)/'null_summary_by_shuffle.csv', index=False)

    real_metrics = compute_global_real_metrics(trials)
    real_metrics.to_csv(Path(args.outdir)/'real_global_metrics.csv')

    # compute p-values for key metrics
    pvals = {}
    null_means = null_summary['mean'].values
    pvals['p_mean_lag_abs_smaller'] = empirical_pvalues(real_metrics['mean_lag_abs'], null_means, higher_better=False)
    for t in [0.1,0.25,0.5,1.0]:
        col = f'fract_below_{t}'
        pvals[f'p_{col}_larger'] = empirical_pvalues(real_metrics[col], null_summary[f'fract_below_{t}'].values, higher_better=True)
    pvals = pd.Series(pvals)
    pvals.to_csv(Path(args.outdir)/'pvalues.csv')

    # Figures
    both = trials[trials['both_pressed']].copy()
    real_lags = both['lag_abs'].dropna().values
    plot_abs_lag_histogram(real_lags, null_long, args.outdir)
    plot_abs_lag_cdf(real_lags, null_long, args.outdir)
    plot_excess_synchrony(real_lags, null_long, args.outdir)
    plot_signed_lag_distribution(both['lag_signed'].dropna().values, args.outdir)
    plot_first_press_binned_comparison(trials, null_long, bins=[0,0.5,1.0,2.0,9999], outdir=args.outdir)
    plot_session_level_summary(trials, null_long, outdir=args.outdir)

    # concise summary
    summary_lines = []
    summary_lines.append(f"n_trials_both: {int(real_metrics['n_trials_both'])}")
    summary_lines.append(f"mean_lag_abs: {real_metrics['mean_lag_abs']:.3f} s (p={pvals['p_mean_lag_abs_smaller']:.3f} for smaller than null)")
    for t in [0.1,0.25,0.5,1.0]:
        summary_lines.append(f"fract_below_{t}: {real_metrics[f'fract_below_{t}']:.3f} (p={pvals[f'p_fract_below_{t}']:.3f} null larger)")

    summary_text = '\n'.join(summary_lines)
    with open(Path(args.outdir)/'concise_summary.txt','w') as f:
        f.write(summary_text)
    print('Saved concise summary to concise_summary.txt')
    print(summary_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='path to trial-level CSV')
    parser.add_argument('--outdir', default='outputs', help='output directory')
    parser.add_argument('--n_shuffle', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    main(args)
