#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mixin methods for threshold-stratified and learning-based gaze plots
used by MicePairGraphs.
"""

from collections import defaultdict

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem


class MicePairGazeThresholdGraphs:
    def __init__(self, experimentGroups=None, prefix="filtered_", save=True):
        """
        Optional standalone initializer.
        When used as a mixin by MicePairGraphs, that class can overwrite these fields.
        """
        self.experimentGroups = experimentGroups if experimentGroups is not None else []
        self.prefix = prefix
        self.save = save

    def gazeAroundLeverPressAcrossLearning(
        self,
        window_sec=5,
        sample_times_sec=(-3, 0, 3),  # Relative times (in seconds) to extract gaze metrics around each anchor event
        gaze_mode="lever",  # "social" or "lever"
        anchor_type="first_press_each_trial",  # "first_press_each_trial", "all_presses", or "success_hit"
        min_sessions_per_pair=4,  # Minimum number of sessions required for a rat pair to be included in the analysis
        save_csv=False,
        csv_path="gaze_around_press_across_learning.csv",
    ):
        """
        Quantify event-locked gaze around lever presses and track it across learning.

        For each rat-pair (group), this method computes session-level gaze frequency at
        specific time offsets from lever-press anchors (default: -3s, 0s, +3s). It then
        plots those metrics across session index with:
        1) Thin per-rat-pair trajectories
        2) Cohort mean +/- SEM trajectory

        Parameters
        ----------
        window_sec : int or float
            Half-window around each anchor event, in seconds.
        sample_times_sec : tuple
            Relative times (seconds) to extract from each event-locked window.
        gaze_mode : str
            "social"  -> uses pos.returnIsGazing()
            "lever"   -> uses pos.returnIsLookingAtObjects(useMinDist=False)
        anchor_type : str
            "first_press_each_trial", "all_presses", or "success_hit"
        min_sessions_per_pair : int
            Skip groups with fewer sessions than this count.
        save_csv : bool
            If True, writes long-format session-level values.
        csv_path : str
            Path for session-level output table.
        """
        print("\nStarting gazeAroundLeverPressAcrossLearning")

        if not self.experimentGroups:
            print("No experiment groups found.")
            return pd.DataFrame()

        if gaze_mode not in {"social", "lever"}:
            raise ValueError("gaze_mode must be 'social' or 'lever'")

        if anchor_type not in {"first_press_each_trial", "all_presses", "success_hit"}:
            raise ValueError(
                "anchor_type must be one of: "
                "'first_press_each_trial', 'all_presses', 'success_hit'"
            )

        # Keep only offsets that fit inside the requested window.
        sample_times_sec = tuple(t for t in sample_times_sec if abs(t) <= window_sec)
        if len(sample_times_sec) == 0:
            raise ValueError("No sample times fall within +/- window_sec.")

        max_sessions = max(len(group) for group in self.experimentGroups)
        # Aggregate per session-index across all rat-pair groups
        pooled_by_time = {t: [[] for _ in range(max_sessions)] for t in sample_times_sec}
        # Per-group trajectories (used for thin gray background lines)
        pair_lines_by_time = {t: [] for t in sample_times_sec}

        rows = []

        for group_idx, group in enumerate(self.experimentGroups):
            if len(group) < min_sessions_per_pair:
                print(
                    f"Skipping group {group_idx}: only {len(group)} session(s), "
                    f"minimum required is {min_sessions_per_pair}."
                )
                continue

            group_vals = {t: [np.nan] * len(group) for t in sample_times_sec}

            for session_idx, exp in enumerate(group):
                lev_df = exp.lev.data if exp.lev is not None else None
                if lev_df is None or lev_df.empty:
                    continue

                lev = lev_df.dropna(subset=["RatID", "AbsTime"]).copy()
                if lev.empty:
                    continue

                lev["RatID"] = pd.to_numeric(lev["RatID"], errors="coerce")
                lev = lev.dropna(subset=["RatID"])
                if lev.empty:
                    continue

                if anchor_type == "first_press_each_trial":
                    press_df = (
                        lev.sort_values("AbsTime")
                        .groupby("TrialNum", as_index=False)
                        .head(1)
                    )
                elif anchor_type == "all_presses":
                    press_df = lev.sort_values("AbsTime")
                else:  # "success_hit"
                    if "coopSucc" not in lev.columns or "Hit" not in lev.columns:
                        continue
                    press_df = lev[(lev["coopSucc"] == 1) & (lev["Hit"] == 1)].sort_values("AbsTime")

                if press_df.empty:
                    continue

                pos = exp.pos
                fps = exp.fps
                pre = int(round(window_sec * fps))
                post = int(round(window_sec * fps))
                total = pre + post
                center_idx = pre

                if gaze_mode == "social":
                    gaze_data = {
                        0: np.array(pos.returnIsGazing(0), dtype=float),
                        1: np.array(pos.returnIsGazing(1), dtype=float),
                    }
                else:
                    gaze_data = {
                        0: np.array(pos.returnIsLookingAtObjects(0, useMinDist=False), dtype=float),
                        1: np.array(pos.returnIsLookingAtObjects(1, useMinDist=False), dtype=float),
                    }

                per_time_values = {t: [] for t in sample_times_sec}
                valid_anchor_count = 0

                for _, row in press_df.iterrows():
                    presser_id = int(row["RatID"])
                    if presser_id not in (0, 1):
                        continue

                    center_frame = int(round(row["AbsTime"] * fps))
                    start = center_frame - pre
                    end = center_frame + post
                    if start < 0 or end > len(gaze_data[presser_id]):
                        continue

                    event_window = gaze_data[presser_id][start:end]
                    if len(event_window) != total:
                        continue

                    valid_anchor_count += 1
                    for t_sec in sample_times_sec:
                        rel_idx = center_idx + int(round(t_sec * fps))
                        if 0 <= rel_idx < total:
                            per_time_values[t_sec].append(event_window[rel_idx])

                if valid_anchor_count == 0:
                    continue

                for t_sec in sample_times_sec:
                    values = per_time_values[t_sec]
                    if not values:
                        continue

                    session_val = float(np.mean(values))
                    group_vals[t_sec][session_idx] = session_val
                    pooled_by_time[t_sec][session_idx].append(session_val)

                row = {
                    "group_idx": group_idx,
                    "session_idx": session_idx,
                    "ratPair": exp.ratPair,
                    "sessionID": exp.sessionID,
                    "date": exp.date,
                    "anchor_type": anchor_type,
                    "gaze_mode": gaze_mode,
                    "num_valid_anchors": valid_anchor_count,
                }
                for t_sec in sample_times_sec:
                    row[f"gaze_at_{t_sec:+g}s"] = group_vals[t_sec][session_idx]
                rows.append(row)

            for t_sec in sample_times_sec:
                pair_lines_by_time[t_sec].append(group_vals[t_sec])

        if not rows:
            print("No valid event-locked gaze data found.")
            return pd.DataFrame()

        # Plot learning trajectories for each sampled time point.
        fig, ax = plt.subplots(figsize=(10, 6))
        color_map = {-3: "teal", 0: "royalblue", 3: "indianred"}
        x_full = np.arange(max_sessions) + 1

        for t_sec in sample_times_sec:
            color = color_map.get(int(t_sec), None)
            if color is None:
                color = cm.viridis((t_sec - min(sample_times_sec)) / (max(sample_times_sec) - min(sample_times_sec) + 1e-8))

            # Per-rat-pair light lines
            for per_pair_vals in pair_lines_by_time[t_sec]:
                y = np.array(per_pair_vals, dtype=float)
                x_pair = np.arange(len(y)) + 1
                valid = ~np.isnan(y)
                if np.any(valid):
                    ax.plot(
                        x_pair[valid],
                        y[valid] * 100,
                        color=color,
                        alpha=0.2,
                        linewidth=1
                    )

            # Cohort mean +/- SEM
            means = []
            sems = []
            for sess_idx in range(max_sessions):
                vals = pooled_by_time[t_sec][sess_idx]
                if len(vals) == 0:
                    means.append(np.nan)
                    sems.append(np.nan)
                else:
                    means.append(float(np.mean(vals)))
                    sems.append(float(sem(vals)) if len(vals) > 1 else 0.0)

            means = np.array(means, dtype=float)
            sems = np.array(sems, dtype=float)
            valid = ~np.isnan(means)
            if not np.any(valid):
                continue

            x = x_full[valid]
            y = means[valid] * 100
            y_sem = sems[valid] * 100

            ax.plot(
                x,
                y,
                marker="o",
                linewidth=3,
                color=color,
                label=f"{t_sec:+g}s",
                zorder=5,
            )
            ax.fill_between(
                x,
                y - y_sem,
                y + y_sem,
                color=color,
                alpha=0.2,
                linewidth=0,
                zorder=4,
            )

        gaze_label = "Social Gaze" if gaze_mode == "social" else "Lever Gaze"
        ax.set_title(
            f"{gaze_label} Around Lever Press Across Learning\n"
            f"Anchor: {anchor_type.replace('_', ' ')}"
        )
        ax.set_xlabel("Session Index")
        ax.set_ylabel("Event-Locked Gaze Frequency (%)")
        ax.grid(True, alpha=0.2)
        ax.legend(title="Relative Time")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        filename = (
            f"{self.prefix}gazeAroundPressAcrossLearning_"
            f"{gaze_mode}_{anchor_type}.png"
        )
        if self.save:
            plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

        out_df = pd.DataFrame(rows)
        if save_csv:
            out_df.to_csv(csv_path, index=False)
        return out_df

    def _collect_event_locked_gaze_windows(self, experiments, gaze_mode="social", window_sec=5):
        """
        Collect event-locked gaze windows for the three conditions used in:
        1) first press in successful trials
        2) hit press in successful trials
        3) first press in unsuccessful trials
        """
        if gaze_mode not in {"social", "lever"}:
            raise ValueError("gaze_mode must be 'social' or 'lever'")

        p_to_o_data = [[], [], []]
        o_to_p_data = [[], [], []]

        PRE = int(30 * window_sec)
        POST = int(30 * window_sec)
        TOTAL = PRE + POST

        for exp in experiments:
            if exp.lev is None or exp.pos is None:
                continue

            lev = exp.lev.data.dropna(subset=["RatID", "AbsTime"]).copy()
            if lev.empty:
                continue

            lev["RatID"] = pd.to_numeric(lev["RatID"], errors="coerce")
            lev = lev.dropna(subset=["RatID"])
            if lev.empty:
                continue

            pos = exp.pos
            fps = exp.fps

            if "coopSucc" not in lev.columns or "Hit" not in lev.columns:
                continue

            if gaze_mode == "social":
                gaze_data = {
                    0: np.array(pos.returnIsGazing(0)),
                    1: np.array(pos.returnIsGazing(1)),
                }
            else:
                gaze_data = {
                    0: np.array(pos.returnIsLookingAtObjects(0, useMinDist=False)),
                    1: np.array(pos.returnIsLookingAtObjects(1, useMinDist=False)),
                }

            succ_trials = lev[lev["coopSucc"] == 1]
            first_succ_presses = (
                succ_trials.sort_values("AbsTime").groupby("TrialNum").head(1)
            )
            succ_presses = lev[(lev["coopSucc"] == 1) & (lev["Hit"] == 1)]
            unsucc_trials = lev[lev["coopSucc"] == 0]
            first_unsucc_presses = (
                unsucc_trials.sort_values("AbsTime").groupby("TrialNum").head(1)
            )

            for cond_idx, press_df in enumerate([first_succ_presses, succ_presses, first_unsucc_presses]):
                for _, row in press_df.iterrows():
                    presser_id = int(row["RatID"])
                    if presser_id not in (0, 1):
                        continue

                    other_id = 1 - presser_id
                    center_frame = int(row["AbsTime"] * fps)
                    start, end = center_frame - PRE, center_frame + POST

                    if start < 0 or end > len(gaze_data[0]):
                        continue

                    p_to_o_window = gaze_data[presser_id][start:end]
                    o_to_p_window = gaze_data[other_id][start:end]

                    if len(p_to_o_window) != TOTAL or len(o_to_p_window) != TOTAL:
                        continue

                    p_to_o_data[cond_idx].append(p_to_o_window)
                    o_to_p_data[cond_idx].append(o_to_p_window)

        time_axis = np.linspace(-PRE / 30, POST / 30, TOTAL)
        return p_to_o_data, o_to_p_data, time_axis

    def _plot_event_locked_windows(self, p_to_o_data, o_to_p_data, time_axis, gaze_mode, threshold_key, phase):
        colors = ["teal", "royalblue", "indianred"]

        if gaze_mode == "social":
            plot_configs = [
                (p_to_o_data, "Presser_to_Partner", "Gaze: Pressing Rat -> Partner"),
                (o_to_p_data, "Partner_to_Presser", "Gaze: Partner -> Pressing Rat"),
            ]
            labels = ["Successful (First)", "Successful (Second)", "Unsuccessful"]
            y_label = "Gaze Frequency"
            file_prefix = "gaze"
        else:
            plot_configs = [
                (p_to_o_data, "Presser_to_Lever", "Gaze: Pressing Rat -> Lever"),
                (o_to_p_data, "Partner_to_Lever", "Gaze: Partner -> Lever"),
            ]
            labels = ["Successful (First)", "Successful (Hit)", "Unsuccessful"]
            y_label = "Gaze at Lever Frequency"
            file_prefix = "gazeAtLever"

        for all_data, filename, title in plot_configs:
            plt.figure(figsize=(10, 6))

            for cond_idx in range(3):
                if not all_data[cond_idx]:
                    continue

                data_arr = np.array(all_data[cond_idx])
                grand_mean = np.mean(data_arr, axis=0)
                if data_arr.shape[0] > 1:
                    err = np.std(data_arr, axis=0, ddof=1) / np.sqrt(data_arr.shape[0])
                else:
                    err = np.zeros_like(grand_mean, dtype=float)

                plt.fill_between(
                    time_axis,
                    grand_mean - err,
                    grand_mean + err,
                    color=colors[cond_idx],
                    alpha=0.25,
                    linewidth=0,
                    zorder=2
                )

                plt.plot(
                    time_axis,
                    grand_mean,
                    color=colors[cond_idx],
                    linewidth=3,
                    label=f"{labels[cond_idx]} (Mean +/- SEM)",
                    zorder=5
                )

            plt.axvline(0, color="black", linestyle="--", alpha=0.5)
            plt.title(f"{title}\nThreshold={threshold_key} | {phase} session per rat pair")
            plt.xlabel("Time from Press (s)")
            plt.ylabel(y_label)
            plt.legend()
            plt.grid(True, alpha=0.2)
            plt.tight_layout()

            out_name = f"{self.prefix}{file_prefix}_{filename}_threshold_{threshold_key}_{phase}.png"
            if self.save:
                plt.savefig(out_name, dpi=300)
            plt.show()
            plt.close()

    def gazeAroundLeverPressByThresholdFirstLast(
        self,
        gaze_mode="social",
        window_sec=5,
        save_report=True,
        report_path="gaze_threshold_first_last_report.txt",
    ):
        """
        Recreate the event-locked gaze plots (social or lever gaze) but split by:
        1) threshold (levLoader.threshold)
        2) first vs last session per rat pair within each threshold
        """
        print("\nStarting gazeAroundLeverPressByThresholdFirstLast")

        if gaze_mode not in {"social", "lever"}:
            raise ValueError("gaze_mode must be 'social' or 'lever'")

        threshold_to_pairs = defaultdict(list)
        report_lines = []

        for group_idx, group in enumerate(self.experimentGroups):
            if not group:
                continue

            rat_pair = group[0].ratPair if getattr(group[0], "ratPair", None) else f"group_{group_idx}"
            idx_by_threshold = defaultdict(list)

            for idx, exp in enumerate(group):
                if exp.lev is None:
                    continue

                threshold_val = getattr(exp.lev, "threshold", exp.lev.returnSuccThreshold())
                if threshold_val is None:
                    continue
                if isinstance(threshold_val, float) and np.isnan(threshold_val):
                    continue

                if isinstance(threshold_val, (int, float, np.number)):
                    th_key = f"{float(threshold_val):g}"
                else:
                    th_key = str(threshold_val)

                idx_by_threshold[th_key].append(idx)

            for th_key, idxs in idx_by_threshold.items():
                first_idx = idxs[0]
                last_idx = idxs[-1]
                threshold_to_pairs[th_key].append({
                    "ratPair": rat_pair,
                    "count": len(idxs),
                    "first_exp": group[first_idx],
                    "last_exp": group[last_idx],
                    "first_sessionID": getattr(group[first_idx], "sessionID", None),
                    "last_sessionID": getattr(group[last_idx], "sessionID", None),
                })

        if not threshold_to_pairs:
            print("No threshold-stratified sessions found.")
            return {}

        def _th_key_sort(v):
            try:
                return (0, float(v))
            except Exception:
                return (1, str(v))

        results = {}
        for th_key in sorted(threshold_to_pairs.keys(), key=_th_key_sort):
            pair_records = threshold_to_pairs[th_key]
            first_exps = [r["first_exp"] for r in pair_records]
            last_exps = [r["last_exp"] for r in pair_records]

            report_lines.append("=" * 80)
            report_lines.append(f"THRESHOLD: {th_key}")
            report_lines.append(f"Rat pairs included: {len(pair_records)}")
            report_lines.append("RatPair details:")

            print("\n" + "=" * 80)
            print(f"THRESHOLD: {th_key}")
            print(f"Rat pairs included: {len(pair_records)}")

            for rec in sorted(pair_records, key=lambda x: str(x["ratPair"])):
                line = (
                    f"  {rec['ratPair']}: sessions_at_threshold={rec['count']}, "
                    f"first_session={rec['first_sessionID']}, "
                    f"last_session={rec['last_sessionID']}"
                )
                report_lines.append(line)
                print(line)

            p_to_o_first, o_to_p_first, time_axis_first = self._collect_event_locked_gaze_windows(
                first_exps, gaze_mode=gaze_mode, window_sec=window_sec
            )
            self._plot_event_locked_windows(
                p_to_o_first, o_to_p_first, time_axis_first, gaze_mode, th_key, "first"
            )

            p_to_o_last, o_to_p_last, time_axis_last = self._collect_event_locked_gaze_windows(
                last_exps, gaze_mode=gaze_mode, window_sec=window_sec
            )
            self._plot_event_locked_windows(
                p_to_o_last, o_to_p_last, time_axis_last, gaze_mode, th_key, "last"
            )

            results[th_key] = {
                "pair_records": pair_records,
                "num_first_sessions": len(first_exps),
                "num_last_sessions": len(last_exps),
            }

        if save_report:
            with open(report_path, "w") as f:
                f.write("\n".join(report_lines) + "\n")
            print(f"\nSaved report: {report_path}")

        return results

    def run_gaze_option(self, option, window_sec=5):
        """
        Convenience dispatcher for common analysis options.

        Options:
          - "threshold_first_last_social": social gaze by threshold (first vs last sessions)
          - "threshold_first_last_lever": lever gaze by threshold (first vs last sessions)
          - "learning_social": social gaze at -3/0/+3 seconds across session index
          - "learning_lever": lever gaze at -3/0/+3 seconds across session index
        """
        if option == "threshold_first_last_social":
            return self.gazeAroundLeverPressByThresholdFirstLast(
                gaze_mode="social",
                window_sec=window_sec,
            )
        if option == "threshold_first_last_lever":
            return self.gazeAroundLeverPressByThresholdFirstLast(
                gaze_mode="lever",
                window_sec=window_sec,
            )
        if option == "learning_social":
            return self.gazeAroundLeverPressAcrossLearning(
                window_sec=window_sec,
                sample_times_sec=(-3, 0, 3),
                gaze_mode="social",
                anchor_type="first_press_each_trial",
                min_sessions_per_pair=5,
                save_csv=False,
            )
        if option == "learning_lever":
            return self.gazeAroundLeverPressAcrossLearning(
                window_sec=window_sec,
                sample_times_sec=(-3, 0, 3),
                gaze_mode="lever",
                anchor_type="first_press_each_trial",
                min_sessions_per_pair=5,
                save_csv=False,
            )

        raise ValueError(
            "Unknown option. Use one of: "
            "threshold_first_last_social, threshold_first_last_lever, "
            "learning_social, learning_lever"
        )


if __name__ == "__main__":
    # Import here to avoid circular imports during normal module import.
    from graph_creator import MicePairGraphs, getGroupRatPairs

    data = getGroupRatPairs()
    pairGraphs = MicePairGraphs(
        data[0], data[1], data[2], data[3], data[4],
        data[5], data[6], data[7], data[8], data[9]
    )

    # Social gaze by threshold; compares first vs last session per rat pair.
    # pairGraphs.gazeAroundLeverPressByThresholdFirstLast(gaze_mode="social", window_sec=5)

    # Lever gaze by threshold; compares first vs last session per rat pair.
    # pairGraphs.gazeAroundLeverPressByThresholdFirstLast(gaze_mode="lever", window_sec=5)

    # Social gaze across learning; tracks -3s, 0s, +3s around press over sessions.
    # pairGraphs.gazeAroundLeverPressAcrossLearning(
    #     window_sec=5, sample_times_sec=(-3, 0, 3),
    #     gaze_mode="social", anchor_type="first_press_each_trial",
    #     min_sessions_per_pair=5, save_csv=False
    # )

    # Lever gaze across learning; tracks -3s, 0s, +3s around press over sessions.
    # pairGraphs.gazeAroundLeverPressAcrossLearning(
    #     window_sec=5, sample_times_sec=(-3, 0, 3),
    #     gaze_mode="lever", anchor_type="first_press_each_trial",
    #     min_sessions_per_pair=5, save_csv=False
    # )
