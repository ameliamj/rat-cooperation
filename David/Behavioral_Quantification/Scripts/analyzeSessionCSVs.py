"""
analyzeSessionCSVs.py — Aggregate analysis across coop / ineq / comp session CSVs.

Input: 6 CSV files (3 session types × 2 CSV kinds)
    - Gazing CSVs      (coop, ineq, comp) — per-session gaze columns
    - Interacting CSVs (coop, ineq, comp) — per-session interaction + other columns

Output:
    - Bar charts of per-session averages for each numeric column, with the 3 session
      types (coop, ineq, comp) shown side by side.
    - Also computes a "non-coop pooled" group (ineq + comp combined) and includes it
      alongside the 3 base types.
    - Histograms for each numeric column: 3 histograms side by side (one per session type).

Everything is built from small modular plotting helpers so new graphs are easy to add.

Usage:
    Edit the SETTINGS block at the top of the file and run:
        python analyzeSessionCSVs.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════
# SETTINGS — Edit paths here
# ══════════════════════════════════════════════════════════════
CSV_DIR = "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/DataStorage"

#New CSVs
'''
GAZE_CSVS = {
    "coop": f"{CSV_DIR}/AllGazeDateNew/coop_sessions_allGazeData_uncorrectedData.csv",
    "ineq": f"{CSV_DIR}/AllGazeDataNew/ineq_sessions_allGazeData_uncorrectedData.csv",
    "comp": f"{CSV_DIR}/AllGazeDataNew/comp_sessions_allGazeData_uncorrectedData.csv",
}

INTER_CSVS = {
    "coop": f"{CSV_DIR}/InteractionAndOtherDataNew/coop_allInteractingAndOtherData_uncorrectedData.csv",
    "ineq": f"{CSV_DIR}/InteractionAndOtherDataNew/ineq_allInteractingAndOtherData_uncorrectedData.csv",
    "comp": f"{CSV_DIR}/InteractionAndOtherDataNew/comp_allInteractingAndOtherData_uncorrectedData.csv",
}
'''

#Old CSVs
GAZE_CSVS = {
    "coop": f"{CSV_DIR}/AllGazeData/coop_sessions_allGazeData_updated.csv",
    "ineq": f"{CSV_DIR}/AllGazeData/ineq_sessions_allGazeData_updated.csv",
    "comp": f"{CSV_DIR}/AllGazeData/comp_sessions_allGazeData_updated.csv",
}

INTER_CSVS = {
    "coop": f"{CSV_DIR}/InteractionAndOtherData/coop_allInteractingAndOtherData.csv",
    "ineq": f"{CSV_DIR}/InteractionAndOtherData/ineq_allInteractingAndOtherData.csv",
    "comp": f"{CSV_DIR}/InteractionAndOtherData/comp_allInteractingAndOtherData.csv",
}

OUTPUT_DIR = f"{CSV_DIR}/Analysis_Outputs/session_csv_analysis"

# Colors used consistently across all plots, one per session-type group
GROUP_COLORS = {
    "coop": "#2ca02c",       # green
    "ineq": "#ff7f0e",       # orange
    "comp": "#1f77b4",       # blue
    "non-coop pooled": "#9467bd",  # purple
}

HIST_BINS = 20

# Show plots interactively (e.g. in Spyder) in addition to saving to disk.
SHOW_PLOTS = True

# If True, bar charts show only 2 bars: coop vs the ineq+comp pooled "non-coop" group.
# If False, bar charts show coop / ineq / comp (and the pooled bar if include_pooled=True).
COOP_VS_NONCOOP_ONLY = True


# ══════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════

def load_csvs(csv_dict):
    """Load a dict of {group_name: csv_path} into {group_name: DataFrame}."""
    out = {}
    for name, path in csv_dict.items():
        if not os.path.exists(path):
            print(f"  [WARN] File not found: {path}  (skipping)")
            continue
        df = pd.read_csv(path)
        print(f"  Loaded {name}: {len(df)} rows, {len(df.columns)} cols from {os.path.basename(path)}")
        out[name] = df
    return out


def numeric_columns(df, exclude=None):
    """Return list of numeric column names, optionally excluding some."""
    exclude = set(exclude or [])
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def build_pooled_non_coop(dfs_by_group):
    """Return a pooled DataFrame from 'ineq' + 'comp' rows, if both exist."""
    frames = []
    for g in ("ineq", "comp"):
        if g in dfs_by_group:
            frames.append(dfs_by_group[g])
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


# ══════════════════════════════════════════════════════════════
# Modular plotting helpers
# ══════════════════════════════════════════════════════════════

def plot_column_averages_bar(dfs_by_group, column, title, savepath,
                             include_pooled=True, ylabel=None,
                             coop_vs_pooled_only=False):
    """
    Bar chart: one bar per group, height = mean of `column` across sessions in that group.
    Error bars = SEM.

    If `coop_vs_pooled_only` is True, the chart shows only two bars:
    'coop' vs the pooled 'non-coop pooled' group (ineq + comp combined).
    """
    if coop_vs_pooled_only:
        pooled = build_pooled_non_coop(dfs_by_group)
        new_dfs = {}
        if "coop" in dfs_by_group:
            new_dfs["coop"] = dfs_by_group["coop"]
        if pooled is not None:
            new_dfs["non-coop pooled"] = pooled
        dfs_by_group = new_dfs
        groups = list(dfs_by_group.keys())
    else:
        groups = list(dfs_by_group.keys())
        if include_pooled:
            pooled = build_pooled_non_coop(dfs_by_group)
            if pooled is not None:
                groups = groups + ["non-coop pooled"]
                dfs_by_group = dict(dfs_by_group)  # shallow copy — don't mutate caller
                dfs_by_group["non-coop pooled"] = pooled

    means, sems, ns = [], [], []
    for g in groups:
        if column not in dfs_by_group[g].columns:
            means.append(np.nan); sems.append(0); ns.append(0)
            continue
        vals = pd.to_numeric(dfs_by_group[g][column], errors="coerce").dropna().values
        if len(vals) == 0:
            means.append(np.nan); sems.append(0); ns.append(0)
        else:
            means.append(np.mean(vals))
            sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
            ns.append(len(vals))

    colors = [GROUP_COLORS.get(g, "#777777") for g in groups]
    x = np.arange(len(groups))

    fig, ax = plt.subplots(figsize=(max(5, len(groups) * 1.3), 4.5))
    ax.bar(x, means, yerr=sems, color=colors, edgecolor="black", capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n(n={ns[i]})" for i, g in enumerate(groups)])
    ax.set_ylabel(ylabel or column)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    # Annotate bars with the mean value
    for i, m in enumerate(means):
        if not np.isnan(m):
            ax.text(x[i], m, f"{m:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def plot_column_histograms_sidebyside(dfs_by_group, column, title, savepath,
                                      bins=HIST_BINS, share_x=True):
    """
    Create one figure with N subplots side by side — one histogram per group
    for the given column. Shared x-axis if share_x.
    """
    groups = list(dfs_by_group.keys())
    n = len(groups)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4), sharex=share_x, sharey=False)
    if n == 1:
        axes = [axes]

    # Shared x-range across subplots if requested
    all_vals = []
    for g in groups:
        if column not in dfs_by_group[g].columns:
            all_vals.append(np.array([]))
            continue
        vals = pd.to_numeric(dfs_by_group[g][column], errors="coerce").dropna().values
        all_vals.append(vals)
    if share_x:
        concat = np.concatenate([v for v in all_vals if len(v) > 0]) if any(len(v) for v in all_vals) else np.array([])
        if concat.size > 0:
            xmin, xmax = np.min(concat), np.max(concat)
            if xmin == xmax:
                xmax = xmin + 1
            bin_edges = np.linspace(xmin, xmax, bins + 1)
        else:
            bin_edges = bins
    else:
        bin_edges = bins

    for ax, g, vals in zip(axes, groups, all_vals):
        color = GROUP_COLORS.get(g, "#777777")
        if len(vals) > 0:
            ax.hist(vals, bins=bin_edges, color=color, edgecolor="black", alpha=0.85)
            ax.axvline(np.mean(vals), color="red", linestyle="--", linewidth=1,
                       label=f"mean={np.mean(vals):.2f}")
            ax.legend(fontsize=8, loc="upper right")
        ax.set_title(f"{g}  (n={len(vals)})")
        ax.set_xlabel(column)
        ax.set_ylabel("count")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(savepath, dpi=150)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def plot_zone_pie_chart(dfs_by_group, zone_columns, title, savepath):
    """
    Side-by-side pie charts of how a behavior is distributed across zones
    (lever / center / mag), one pie for 'coop' and one for the pooled non-coop group.

    zone_columns: dict mapping zone label -> list of candidate CSV column names
                  (the first candidate found in a given DataFrame is used).
    """
    pooled = build_pooled_non_coop(dfs_by_group)
    groups = {}
    if "coop" in dfs_by_group:
        groups["coop"] = dfs_by_group["coop"]
    if pooled is not None:
        groups["non-coop pooled"] = pooled

    n = len(groups)
    if n == 0:
        return

    zone_colors = {"lever": "#1f77b4", "center": "#2ca02c", "mag": "#ff7f0e"}

    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.5))
    if n == 1:
        axes = [axes]

    for ax, (gname, df) in zip(axes, groups.items()):
        labels, means, colors = [], [], []
        for zone_label, candidates in zone_columns.items():
            col = next((c for c in candidates if c in df.columns), None)
            if col is None:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").dropna().values
            labels.append(zone_label)
            means.append(float(np.mean(vals)) if len(vals) > 0 else 0.0)
            colors.append(zone_colors.get(zone_label, "#777777"))

        total = sum(means)
        if total <= 0 or not means:
            ax.text(0.5, 0.5, f"{gname}\n(no data)", ha="center", va="center")
            ax.axis("off")
            continue

        wedge_labels = [f"{lab}\n({m:.4f})" for lab, m in zip(labels, means)]
        ax.pie(
            means, labels=wedge_labels, autopct="%1.1f%%",
            colors=colors, startangle=90,
            wedgeprops={"edgecolor": "black", "linewidth": 1},
            textprops={"fontsize": 10},
        )
        ax.set_title(f"{gname}  (n={len(df)})")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(savepath, dpi=150)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


# ══════════════════════════════════════════════════════════════
# Pretty-printed summary table (side-by-side per column)
# ══════════════════════════════════════════════════════════════

def _column_stats(vals):
    """Return (mean, sem, std, min, max, n) for a numeric array (ignoring NaNs)."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    if n == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, 0)
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 1 else 0.0
    return (mean, sem, std, float(np.min(vals)), float(np.max(vals)), n)


def _fmt_cell(mean, sem, n):
    if n == 0 or np.isnan(mean):
        return "—"
    return f"{mean:>9.3f} ± {sem:<7.3f} (n={n})"


def print_summary_table(dfs_by_group, columns, kind_name, include_pooled=True):
    """
    Print a side-by-side summary table for every numeric column:
    each row = one column; each sub-column block = one group (coop / ineq / comp / pooled).
    """
    groups = list(dfs_by_group.keys())
    if include_pooled:
        pooled = build_pooled_non_coop(dfs_by_group)
        if pooled is not None:
            groups = groups + ["non-coop pooled"]
            dfs_by_group = dict(dfs_by_group)
            dfs_by_group["non-coop pooled"] = pooled

    # Column widths
    name_w = max(28, max((len(c) for c in columns), default=10) + 2)
    cell_w = 28
    header = "Column".ljust(name_w) + "".join(g.ljust(cell_w) for g in groups)
    sep = "─" * len(header)

    print("\n" + "═" * len(header))
    print(f"[{kind_name}] Summary statistics  —  mean ± SEM (n)")
    print("═" * len(header))
    print(header)
    print(sep)

    for col in columns:
        row = col.ljust(name_w)
        for g in groups:
            if col not in dfs_by_group[g].columns:
                row += "—".ljust(cell_w)
                continue
            vals = pd.to_numeric(dfs_by_group[g][col], errors="coerce").values
            mean, sem, _, _, _, n = _column_stats(vals)
            row += _fmt_cell(mean, sem, n).ljust(cell_w)
        print(row)
    print(sep)

    # Min / max table
    print(f"\n[{kind_name}] Range (min … max)")
    print(sep)
    print(header)
    print(sep)
    for col in columns:
        row = col.ljust(name_w)
        for g in groups:
            if col not in dfs_by_group[g].columns:
                row += "—".ljust(cell_w)
                continue
            vals = pd.to_numeric(dfs_by_group[g][col], errors="coerce").values
            _, _, _, vmin, vmax, n = _column_stats(vals)
            if n == 0:
                cell = "—"
            else:
                cell = f"{vmin:>9.3f} … {vmax:<9.3f}"
            row += cell.ljust(cell_w)
        print(row)
    print(sep + "\n")


# ══════════════════════════════════════════════════════════════
# Orchestration — generate a full figure set for one CSV kind
# ══════════════════════════════════════════════════════════════

# Columns that exist in these CSVs but aren't meaningful to average/histogram.
NON_DATA_COLS = {
    "label", "date", "session", "ratPair", "familiarity", "transparency",
    "sessionType", "pos_file", "isKL",
}


def generate_plots_for_csv_kind(dfs_by_group, kind_name, output_dir):
    """
    For each numeric column in the CSVs of this kind, produce:
      - one bar chart (averages side-by-side: coop / ineq / comp / non-coop pooled)
      - one histogram figure (3 histograms side-by-side: coop / ineq / comp)
    """
    out_bars = Path(output_dir) / kind_name / "bars"
    out_hist = Path(output_dir) / kind_name / "histograms"
    out_bars.mkdir(parents=True, exist_ok=True)
    out_hist.mkdir(parents=True, exist_ok=True)

    # Union of numeric columns across all groups (ignoring metadata columns)
    all_cols = []
    seen = set()
    for df in dfs_by_group.values():
        for c in numeric_columns(df, exclude=NON_DATA_COLS):
            if c not in seen:
                seen.add(c)
                all_cols.append(c)

    print(f"\n[{kind_name}] {len(all_cols)} numeric columns to plot:")
    for col in all_cols:
        print(f"  - {col}")

    # Pretty-print summary stats side-by-side BEFORE opening any plot windows
    print_summary_table(dfs_by_group, all_cols, kind_name, include_pooled=False)

    for col in all_cols:
        safe_col = col.replace("/", "_").replace(" ", "_").replace("%", "pct")

        bar_path = out_bars / f"{safe_col}_bar.png"
        plot_column_averages_bar(
            dfs_by_group, col,
            title=f"{kind_name}: mean {col} per session",
            savepath=bar_path,
            include_pooled=True,
            coop_vs_pooled_only=COOP_VS_NONCOOP_ONLY,
        )

        hist_path = out_hist / f"{safe_col}_hist.png"
        plot_column_histograms_sidebyside(
            dfs_by_group, col,
            title=f"{kind_name}: distribution of {col}",
            savepath=hist_path,
        )

    print(f"[{kind_name}] Saved plots to {out_bars.parent}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("Loading gaze CSVs...")
    gaze_dfs = load_csvs(GAZE_CSVS)

    print("\nLoading interacting CSVs...")
    inter_dfs = load_csvs(INTER_CSVS)

    if gaze_dfs:
        generate_plots_for_csv_kind(gaze_dfs, "gaze", OUTPUT_DIR)

        gaze_zone_cols = {
            "lever":  ["social_gaze_at_lev",    "socialGazingAtLev"],
            "center": ["social_gaze_at_center", "socialGazingAtCenter"],
            "mag":    ["social_gaze_at_mag",    "socialGazingAtMag"],
        }
        gaze_pie_path = Path(OUTPUT_DIR) / "gaze" / "gaze_zone_pies.png"
        gaze_pie_path.parent.mkdir(parents=True, exist_ok=True)
        plot_zone_pie_chart(
            gaze_dfs, gaze_zone_cols,
            title="Social gaze: distribution across zones (coop vs non-coop pooled)",
            savepath=gaze_pie_path,
        )
    else:
        print("\nNo gaze CSVs loaded — skipping gaze plots.")

    if inter_dfs:
        generate_plots_for_csv_kind(inter_dfs, "interacting", OUTPUT_DIR)

        inter_zone_cols = {
            "lever":  ["% interacting at lev",    "interacting_at_lev",    "interactingAtLev"],
            "center": ["% interacting at center", "interacting_at_center", "interactingAtCenter"],
            "mag":    ["% interacting at mag",    "interacting_at_mag",    "interactingAtMag"],
        }
        inter_pie_path = Path(OUTPUT_DIR) / "interacting" / "interacting_zone_pies.png"
        inter_pie_path.parent.mkdir(parents=True, exist_ok=True)
        plot_zone_pie_chart(
            inter_dfs, inter_zone_cols,
            title="Interacting: distribution across zones (coop vs non-coop pooled)",
            savepath=inter_pie_path,
        )
    else:
        print("\nNo interacting CSVs loaded — skipping interacting plots.")

    print(f"\nDone. All outputs under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
