#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-file graph utilities plus lever coordination shuffle analysis.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_class import singleExperiment
from file_extractor_class import fileExtractor


# Paths used by helper dataset selectors
filtered = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/Filtered.csv"
only_TrainingCoop_filtered = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_TrainingCooperation_filtered.csv"
only_unfamiliar_filtered = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_unfamiliar_partners_filtered.csv"


class multiFileGraphs:
    def __init__(
        self,
        magFiles: List[str],
        levFiles: List[str],
        posFiles: List[str],
        fpsList: List[int],
        totFramesList: List[int],
        initialNanList: List[int],
        dates: List[int],
        sessions: List[int],
        ratPairs: List[int],
        fiberFiles=None,
        prefix="",
        save=True,
        saveAsPDF=False,
    ):
        self.experiments = []
        self.prefix = prefix
        self.save = save
        self.saveAsPDF = saveAsPDF
        self.NUM_BINS = 30
        self.labelSize = 17
        self.titleSize = 18

        self.real_color = "#1f77b4"
        self.null_color = "#d62728"
        self.null_band_alpha = 0.18

        deleted_count = 0

        print("There are ", len(magFiles), " experiments in this data session. ")
        print("")

        if len(magFiles) != len(levFiles) or len(magFiles) != len(posFiles):
            raise ValueError("Different number of mag, lev, and pos files")

        if (
            len(magFiles) != len(fpsList)
            or len(magFiles) != len(totFramesList)
            or len(magFiles) != len(initialNanList)
        ):
            print("lenDataFiles: ", len(magFiles))
            print("len(fpsList)", len(fpsList))
            print("len(totFramesList)", len(totFramesList))
            print("len(initialNanList)", len(initialNanList))
            raise ValueError("Different number of fpsList, totFramesList, or initialNanList values")

        if fiberFiles is not None and len(magFiles) != len(fiberFiles):
            print("len(fiber Files): ", len(fiberFiles))
            raise ValueError("Diff Length of fiberFiles")

        SKIP_POS_FILES = {
            "/gpfs/radev/pi/saxena/aj764/PairedTestingSessions/051024_EB023B-021R-019Y_TimeOut_CNO/Tracking/h5_corrected/051024_Cam1_TrNum7_Coop_EB023B-EB021R.predictions.h5",
            "/gpfs/radev/pi/saxena/aj764/PairedTestingSessions/053124_EB027Y-029R-023B_TimeOut_CNO/Tracking/h5_corrected/053124_Cam1_TrNum12_Coop_EB027Y-EB003B.predictions.h5",
        }

        for i in range(len(magFiles)):
            if posFiles[i] in SKIP_POS_FILES:
                deleted_count += 1
                print(f"Skipping corrupted h5 file: {posFiles[i]}")
                continue

            if fiberFiles is not None and fiberFiles[i] is not None:
                exp = singleExperiment(
                    magFiles[i],
                    levFiles[i],
                    posFiles[i],
                    fpsList[i],
                    totFramesList[i],
                    initialNanList[i],
                    fp_files=fiberFiles[i],
                )
            else:
                exp = singleExperiment(
                    magFiles[i],
                    levFiles[i],
                    posFiles[i],
                    fpsList[i],
                    totFramesList[i],
                    initialNanList[i],
                    date=dates[i],
                    sessionID=sessions[i],
                    ratPair=ratPairs[i],
                )

            mag_missing = [col for col in exp.mag.categories if col not in exp.mag.data.columns]
            lev_missing = [col for col in exp.lev.categories if col not in exp.lev.data.columns]

            if mag_missing or lev_missing:
                deleted_count += 1
                print("Skipping experiment due to missing categories:")
                if mag_missing:
                    print(f"  MagFile missing: {mag_missing}")
                    print(f"  Mag File: {magFiles[i]}")
                if lev_missing:
                    print(f"  LevFile missing: {lev_missing}")
                    print(f"  Lev File: {levFiles[i]}")
                continue

            self.experiments.append(exp)

        print(f"Deleted {deleted_count} experiment(s) due to missing categories.")

    def _saveCurrentFigure(self, baseFileName):
        if not self.save:
            return
        ext = "pdf" if self.saveAsPDF else "png"
        plt.savefig(f"{self.prefix}{baseFileName}.{ext}", bbox_inches="tight")

    def _save_figure_png_pdf(self, fig: plt.Figure, outdir: Path, base_name: str):
        if not self.save:
            plt.close(fig)
            return
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / f"{base_name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(outdir / f"{base_name}.pdf", bbox_inches="tight")
        plt.close(fig)

    def _quantile_band(self, arr_2d: np.ndarray, low: float = 2.5, high: float = 97.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if arr_2d.ndim != 2:
            raise ValueError("arr_2d must be 2D")
        mean = np.nanmean(arr_2d, axis=0)
        lo = np.nanpercentile(arr_2d, low, axis=0)
        hi = np.nanpercentile(arr_2d, high, axis=0)
        return mean, lo, hi

    def _empty_trial_metrics(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "experiment_index",
                "session_key",
                "sessionID",
                "date",
                "ratPair",
                "trial_num",
                "rat1_id",
                "rat2_id",
                "lever_onset_abs",
                "rat1_press_abs",
                "rat2_press_abs",
                "rat1_press_rel",
                "rat2_press_rel",
                "rat1_pressed",
                "rat2_pressed",
                "both_pressed",
                "successful_trial",
                "first_press_rel",
                "second_press_rel",
                "lag_signed",
                "lag_abs",
                "trial_uid",
            ]
        )

    def compute_relative_press_metrics(self, drop_negative_press_times: bool = True) -> pd.DataFrame:
        """
        Build one row per trial, with rat press times relative to lever onset.

        Lever onset is estimated per trial as min(AbsTime - TrialTime) from the trial's lever rows.
        Rat press times are first lever press absolute times for each rat in the trial.
        """
        records = []
        trial_uid = 0

        for exp_idx, exp in enumerate(self.experiments):
            lev = exp.lev.data.copy()
            required_cols = {"TrialNum", "RatID", "AbsTime", "TrialTime"}
            if not required_cols.issubset(lev.columns):
                print(
                    f"Skipping experiment {exp_idx}: missing required columns "
                    f"{required_cols - set(lev.columns)}"
                )
                continue

            lev = lev.dropna(subset=["TrialNum", "RatID", "AbsTime", "TrialTime"]).copy()
            if lev.empty:
                continue

            lev["TrialNum"] = pd.to_numeric(lev["TrialNum"], errors="coerce")
            lev["RatID"] = pd.to_numeric(lev["RatID"], errors="coerce")
            lev["AbsTime"] = pd.to_numeric(lev["AbsTime"], errors="coerce")
            lev["TrialTime"] = pd.to_numeric(lev["TrialTime"], errors="coerce")
            lev = lev.dropna(subset=["TrialNum", "RatID", "AbsTime", "TrialTime"])
            if lev.empty:
                continue

            lev = lev.sort_values(["TrialNum", "AbsTime"])
            rat_ids = sorted(lev["RatID"].unique().tolist())
            if len(rat_ids) < 2:
                print(f"Skipping experiment {exp_idx}: fewer than 2 rat IDs present.")
                continue

            rat1_id, rat2_id = rat_ids[0], rat_ids[1]
            session_key = f"{exp_idx}_{exp.sessionID}_{exp.date}_{exp.ratPair}"

            for trial_num, trial_df in lev.groupby("TrialNum", sort=True):
                onset_candidates = trial_df["AbsTime"] - trial_df["TrialTime"]
                lever_onset_abs = float(onset_candidates.min()) if len(onset_candidates) > 0 else np.nan

                rat1_df = trial_df[trial_df["RatID"] == rat1_id]
                rat2_df = trial_df[trial_df["RatID"] == rat2_id]

                rat1_press_abs = rat1_df["AbsTime"].min() if not rat1_df.empty else np.nan
                rat2_press_abs = rat2_df["AbsTime"].min() if not rat2_df.empty else np.nan

                rat1_press_rel = rat1_press_abs - lever_onset_abs if pd.notna(rat1_press_abs) else np.nan
                rat2_press_rel = rat2_press_abs - lever_onset_abs if pd.notna(rat2_press_abs) else np.nan

                if drop_negative_press_times:
                    if pd.notna(rat1_press_rel) and rat1_press_rel < 0:
                        rat1_press_abs = np.nan
                        rat1_press_rel = np.nan
                    if pd.notna(rat2_press_rel) and rat2_press_rel < 0:
                        rat2_press_abs = np.nan
                        rat2_press_rel = np.nan

                rat1_pressed = pd.notna(rat1_press_rel)
                rat2_pressed = pd.notna(rat2_press_rel)
                both_pressed = bool(rat1_pressed and rat2_pressed)

                if both_pressed:
                    first_press_rel = float(min(rat1_press_rel, rat2_press_rel))
                    second_press_rel = float(max(rat1_press_rel, rat2_press_rel))
                    lag_signed = float(rat2_press_rel - rat1_press_rel)
                    lag_abs = float(abs(lag_signed))
                else:
                    first_press_rel = np.nan
                    second_press_rel = np.nan
                    lag_signed = np.nan
                    lag_abs = np.nan

                successful_trial = np.nan
                if "coopSucc" in trial_df.columns:
                    successful_trial = bool(pd.to_numeric(trial_df["coopSucc"], errors="coerce").fillna(0).max() >= 1)

                records.append(
                    {
                        "experiment_index": exp_idx,
                        "session_key": session_key,
                        "sessionID": exp.sessionID,
                        "date": exp.date,
                        "ratPair": exp.ratPair,
                        "trial_num": int(trial_num),
                        "rat1_id": rat1_id,
                        "rat2_id": rat2_id,
                        "lever_onset_abs": lever_onset_abs,
                        "rat1_press_abs": rat1_press_abs,
                        "rat2_press_abs": rat2_press_abs,
                        "rat1_press_rel": rat1_press_rel,
                        "rat2_press_rel": rat2_press_rel,
                        "rat1_pressed": bool(rat1_pressed),
                        "rat2_pressed": bool(rat2_pressed),
                        "both_pressed": both_pressed,
                        "successful_trial": successful_trial,
                        "first_press_rel": first_press_rel,
                        "second_press_rel": second_press_rel,
                        "lag_signed": lag_signed,
                        "lag_abs": lag_abs,
                        "trial_uid": trial_uid,
                    }
                )
                trial_uid += 1

        if not records:
            return self._empty_trial_metrics()

        trial_metrics = pd.DataFrame.from_records(records)
        return trial_metrics.sort_values(["experiment_index", "trial_num"]).reset_index(drop=True)

    def _compute_metric_dict(
        self,
        lag_abs: np.ndarray,
        lag_signed: np.ndarray,
        thresholds: Sequence[float],
    ) -> Dict[str, float]:
        metrics = {
            "n_trials_both": float(len(lag_abs)),
            "mean_lag_abs": float(np.nanmean(lag_abs)) if len(lag_abs) else np.nan,
            "median_lag_abs": float(np.nanmedian(lag_abs)) if len(lag_abs) else np.nan,
            "fract_rat1_first": float(np.mean(lag_signed < 0)) if len(lag_signed) else np.nan,
            "fract_rat2_first": float(np.mean(lag_signed > 0)) if len(lag_signed) else np.nan,
            "fract_tie": float(np.mean(lag_signed == 0)) if len(lag_signed) else np.nan,
        }
        for thr in thresholds:
            metrics[f"fract_below_{thr}"] = float(np.mean(lag_abs <= thr)) if len(lag_abs) else np.nan
        return metrics

    def shuffle_press_times_within_session(
        self,
        trial_metrics: pd.DataFrame,
        n_shuffles: int = 1000,
        random_state: int = 0,
        thresholds: Sequence[float] = (0.1, 0.25, 0.5, 1.0),
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Within each session, preserve rat1 press times and permute rat2 press times across trials.
        Returns:
        1) long-format per-trial null table
        2) per-shuffle summary metrics
        """
        both = trial_metrics[trial_metrics["both_pressed"]].copy()
        if both.empty:
            return pd.DataFrame(), pd.DataFrame()

        rng = np.random.default_rng(random_state)

        long_parts = []
        summary_records = []
        grouped_sessions = list(both.groupby("session_key", sort=False))

        for shuffle_idx in range(n_shuffles):
            shuffle_parts = []
            lag_abs_all = []
            lag_signed_all = []

            for session_key, session_df in grouped_sessions:
                r1 = session_df["rat1_press_rel"].to_numpy(dtype=float)
                r2 = session_df["rat2_press_rel"].to_numpy(dtype=float)
                trial_uid = session_df["trial_uid"].to_numpy(dtype=int)

                perm_idx = rng.permutation(len(r2))
                r2_perm = r2[perm_idx]

                lag_signed = r2_perm - r1
                lag_abs = np.abs(lag_signed)
                first_press_rel = np.minimum(r1, r2_perm)

                lag_abs_all.append(lag_abs)
                lag_signed_all.append(lag_signed)

                shuffle_parts.append(
                    pd.DataFrame(
                        {
                            "shuffle": shuffle_idx,
                            "session_key": session_key,
                            "trial_uid": trial_uid,
                            "lag_signed_null": lag_signed,
                            "lag_abs_null": lag_abs,
                            "first_press_rel_null": first_press_rel,
                        }
                    )
                )

            if shuffle_parts:
                one_shuffle_long = pd.concat(shuffle_parts, ignore_index=True)
                long_parts.append(one_shuffle_long)

                lag_abs_concat = np.concatenate(lag_abs_all) if lag_abs_all else np.array([])
                lag_signed_concat = np.concatenate(lag_signed_all) if lag_signed_all else np.array([])
                metrics = self._compute_metric_dict(lag_abs_concat, lag_signed_concat, thresholds)
                metrics["shuffle"] = shuffle_idx
                summary_records.append(metrics)

        null_long = pd.concat(long_parts, ignore_index=True) if long_parts else pd.DataFrame()
        null_summary = pd.DataFrame(summary_records)
        return null_long, null_summary

    def compute_ecdf(self, data: np.ndarray, xs: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        data = np.asarray(data, dtype=float)
        data = data[np.isfinite(data)]
        if data.size == 0:
            if xs is None:
                return np.array([]), np.array([])
            return xs, np.zeros_like(xs, dtype=float)

        data = np.sort(data)
        n = data.size
        if xs is None:
            xs = data
            ys = np.arange(1, n + 1) / n
            return xs, ys

        ys = np.searchsorted(data, xs, side="right") / n
        return xs, ys

    def compute_excess_synchrony(
        self,
        real_lag_abs: np.ndarray,
        null_long: pd.DataFrame,
        xs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        real_cdf = np.array([np.mean(real_lag_abs <= x) for x in xs], dtype=float)

        null_cdfs = []
        for _, g in null_long.groupby("shuffle"):
            lag_abs_null = g["lag_abs_null"].to_numpy(dtype=float)
            null_cdfs.append(np.array([np.mean(lag_abs_null <= x) for x in xs], dtype=float))

        null_cdfs = np.asarray(null_cdfs)
        null_mean, null_lo, null_hi = self._quantile_band(null_cdfs)
        excess = real_cdf - null_mean
        excess_lo = real_cdf - null_hi
        excess_hi = real_cdf - null_lo
        return excess, excess_lo, excess_hi, real_cdf

    def _empirical_pvalue(self, real_value: float, null_values: np.ndarray, tail: str) -> float:
        null_values = np.asarray(null_values, dtype=float)
        null_values = null_values[np.isfinite(null_values)]
        if null_values.size == 0 or not np.isfinite(real_value):
            return np.nan

        # +1 correction avoids p=0 for finite shuffle counts.
        if tail == "smaller":
            count = np.sum(null_values <= real_value)
        elif tail == "larger":
            count = np.sum(null_values >= real_value)
        else:
            raise ValueError("tail must be 'smaller' or 'larger'")

        return float((count + 1) / (null_values.size + 1))

    def summarize_real_vs_null(
        self,
        trial_metrics: pd.DataFrame,
        null_summary: pd.DataFrame,
        thresholds: Sequence[float] = (0.1, 0.25, 0.5, 1.0),
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        both = trial_metrics[trial_metrics["both_pressed"]].copy()
        lag_abs = both["lag_abs"].to_numpy(dtype=float)
        lag_signed = both["lag_signed"].to_numpy(dtype=float)

        real_metrics = self._compute_metric_dict(lag_abs, lag_signed, thresholds)
        real_metrics_df = pd.DataFrame([real_metrics])

        rows = []
        for metric_name, real_value in real_metrics.items():
            if metric_name == "n_trials_both":
                continue
            if metric_name not in null_summary.columns:
                continue

            null_vals = null_summary[metric_name].to_numpy(dtype=float)
            null_mean = float(np.nanmean(null_vals))
            null_lo = float(np.nanpercentile(null_vals, 2.5))
            null_hi = float(np.nanpercentile(null_vals, 97.5))

            if metric_name in {"mean_lag_abs", "median_lag_abs"}:
                tail = "smaller"
            elif metric_name.startswith("fract_below_"):
                tail = "larger"
            else:
                tail = "larger"

            pval = self._empirical_pvalue(real_value, null_vals, tail)
            rows.append(
                {
                    "metric": metric_name,
                    "real_value": real_value,
                    "null_mean": null_mean,
                    "null_ci_low": null_lo,
                    "null_ci_high": null_hi,
                    "tail_test": tail,
                    "empirical_p": pval,
                }
            )

        global_summary = pd.DataFrame(rows)

        return real_metrics_df, global_summary, both

    def _get_default_bins(self, values: np.ndarray, n_bins: int = 40) -> np.ndarray:
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return np.linspace(0, 1, n_bins)
        max_v = np.nanpercentile(finite_values, 99.5)
        max_v = max(max_v, 0.5)
        return np.linspace(0, max_v, n_bins)

    def plot_abs_lag_histogram(
        self,
        real_lag_abs: np.ndarray,
        null_long: pd.DataFrame,
        outdir: Path,
        bins: Optional[np.ndarray] = None,
    ):
        if bins is None:
            bins = self._get_default_bins(real_lag_abs)

        null_hists = []
        for _, g in null_long.groupby("shuffle"):
            h, _ = np.histogram(g["lag_abs_null"].to_numpy(dtype=float), bins=bins, density=True)
            null_hists.append(h)

        null_hists = np.asarray(null_hists)
        null_mean, null_lo, null_hi = self._quantile_band(null_hists)

        fig, ax = plt.subplots(figsize=(7, 4.8))
        ax.hist(
            real_lag_abs,
            bins=bins,
            density=True,
            color=self.real_color,
            alpha=0.55,
            label="Real",
        )

        centers = (bins[:-1] + bins[1:]) / 2
        ax.plot(centers, null_mean, color=self.null_color, lw=2, label="Shuffled null mean")
        ax.fill_between(
            centers,
            null_lo,
            null_hi,
            color=self.null_color,
            alpha=self.null_band_alpha,
            label="Shuffled null 95%",
        )
        ax.set_xlabel("Absolute lag from lever onset (s)", fontsize=self.labelSize)
        ax.set_ylabel("Density", fontsize=self.labelSize)
        ax.set_title("Figure 1. Absolute Lag Histogram", fontsize=self.titleSize)
        ax.legend(frameon=False)
        self._save_figure_png_pdf(fig, outdir, "figure1_absolute_lag_histogram")

    def plot_abs_lag_cdf(
        self,
        real_lag_abs: np.ndarray,
        null_long: pd.DataFrame,
        outdir: Path,
        xs: Optional[np.ndarray] = None,
    ):
        if xs is None:
            x_max = np.nanpercentile(real_lag_abs, 99.5) if len(real_lag_abs) else 1.0
            xs = np.linspace(0, max(x_max, 0.5), 250)

        _, real_ecdf = self.compute_ecdf(real_lag_abs, xs)

        null_ecdfs = []
        for _, g in null_long.groupby("shuffle"):
            _, ecdf = self.compute_ecdf(g["lag_abs_null"].to_numpy(dtype=float), xs)
            null_ecdfs.append(ecdf)

        null_ecdfs = np.asarray(null_ecdfs)
        null_mean, null_lo, null_hi = self._quantile_band(null_ecdfs)

        fig, ax = plt.subplots(figsize=(7, 4.8))
        ax.plot(xs, real_ecdf, color=self.real_color, lw=2.2, label="Real")
        ax.plot(xs, null_mean, color=self.null_color, lw=2, label="Shuffled null mean")
        ax.fill_between(
            xs,
            null_lo,
            null_hi,
            color=self.null_color,
            alpha=self.null_band_alpha,
            label="Shuffled null 95%",
        )
        ax.set_xlabel("Absolute lag from lever onset (s)", fontsize=self.labelSize)
        ax.set_ylabel("Cumulative fraction", fontsize=self.labelSize)
        ax.set_title("Figure 2. Absolute Lag CDF", fontsize=self.titleSize)
        ax.legend(frameon=False, loc="lower right")
        self._save_figure_png_pdf(fig, outdir, "figure2_absolute_lag_cdf")

    def plot_excess_synchrony(
        self,
        real_lag_abs: np.ndarray,
        null_long: pd.DataFrame,
        outdir: Path,
        xs: Optional[np.ndarray] = None,
    ):
        if xs is None:
            x_max = np.nanpercentile(real_lag_abs, 99.5) if len(real_lag_abs) else 1.0
            xs = np.linspace(0, max(x_max, 0.5), 250)

        excess, excess_lo, excess_hi, _ = self.compute_excess_synchrony(real_lag_abs, null_long, xs)

        fig, ax = plt.subplots(figsize=(7, 4.8))
        ax.plot(xs, excess, color=self.real_color, lw=2.2, label="Excess synchrony")
        ax.fill_between(
            xs,
            excess_lo,
            excess_hi,
            color=self.null_color,
            alpha=self.null_band_alpha,
            label="95% null band",
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Absolute lag threshold x (s)", fontsize=self.labelSize)
        ax.set_ylabel("P(real lag <= x) - P(null lag <= x)", fontsize=self.labelSize)
        ax.set_title("Figure 3. Excess Synchrony Curve", fontsize=self.titleSize)
        ax.legend(frameon=False)
        self._save_figure_png_pdf(fig, outdir, "figure3_excess_synchrony_curve")

    def plot_signed_lag_distribution(
        self,
        real_lag_signed: np.ndarray,
        null_long: pd.DataFrame,
        outdir: Path,
    ):
        signed_max = np.nanpercentile(np.abs(real_lag_signed), 99.5) if len(real_lag_signed) else 1.0
        signed_max = max(signed_max, 0.5)
        bins = np.linspace(-signed_max, signed_max, 41)

        null_hists = []
        for _, g in null_long.groupby("shuffle"):
            h, _ = np.histogram(g["lag_signed_null"].to_numpy(dtype=float), bins=bins, density=True)
            null_hists.append(h)

        null_hists = np.asarray(null_hists)
        null_mean, null_lo, null_hi = self._quantile_band(null_hists)

        fig, ax = plt.subplots(figsize=(7, 4.8))
        ax.hist(
            real_lag_signed,
            bins=bins,
            density=True,
            color=self.real_color,
            alpha=0.55,
            label="Real",
        )
        centers = (bins[:-1] + bins[1:]) / 2
        ax.plot(centers, null_mean, color=self.null_color, lw=2, label="Shuffled null mean")
        ax.fill_between(
            centers,
            null_lo,
            null_hi,
            color=self.null_color,
            alpha=self.null_band_alpha,
            label="Shuffled null 95%",
        )
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("Signed lag (rat2 - rat1) from lever onset (s)", fontsize=self.labelSize)
        ax.set_ylabel("Density", fontsize=self.labelSize)
        ax.set_title("Figure 4. Signed Lag Distribution", fontsize=self.titleSize)
        ax.legend(frameon=False)
        self._save_figure_png_pdf(fig, outdir, "figure4_signed_lag_distribution")

    def plot_first_press_binned_comparison(
        self,
        trial_metrics: pd.DataFrame,
        null_long: pd.DataFrame,
        outdir: Path,
        bins: Sequence[float] = (0.0, 0.5, 1.0, 2.0, np.inf),
    ) -> pd.DataFrame:
        both = trial_metrics[trial_metrics["both_pressed"]].copy()
        both["first_press_bin"] = pd.cut(both["first_press_rel"], bins=bins, right=False)

        real_bin_stats = (
            both.groupby("first_press_bin", observed=False)["lag_abs"]
            .agg(real_mean="mean", real_median="median", real_count="count")
            .reset_index()
        )

        null_long = null_long.copy()
        null_long["first_press_bin"] = pd.cut(null_long["first_press_rel_null"], bins=bins, right=False)
        null_per_shuffle = (
            null_long.groupby(["shuffle", "first_press_bin"], observed=False)["lag_abs_null"]
            .agg(null_mean="mean", null_median="median", null_count="count")
            .reset_index()
        )

        rows = []
        for first_bin, g in null_per_shuffle.groupby("first_press_bin", observed=False):
            rows.append(
                {
                    "first_press_bin": first_bin,
                    "null_mean_of_mean": float(np.nanmean(g["null_mean"])),
                    "null_mean_ci_low": float(np.nanpercentile(g["null_mean"], 2.5)),
                    "null_mean_ci_high": float(np.nanpercentile(g["null_mean"], 97.5)),
                    "null_median_of_median": float(np.nanmean(g["null_median"])),
                    "null_median_ci_low": float(np.nanpercentile(g["null_median"], 2.5)),
                    "null_median_ci_high": float(np.nanpercentile(g["null_median"], 97.5)),
                }
            )

        null_bin_summary = pd.DataFrame(rows)
        merged = real_bin_stats.merge(null_bin_summary, on="first_press_bin", how="left")

        plot_df = merged.copy()
        plot_df["bin_label"] = plot_df["first_press_bin"].astype(str)
        x = np.arange(len(plot_df))

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)

        axes[0].plot(x, plot_df["real_mean"], "o-", color=self.real_color, lw=1.8, label="Real mean")
        axes[0].plot(x, plot_df["null_mean_of_mean"], "s-", color=self.null_color, lw=1.8, label="Null mean")
        axes[0].fill_between(
            x,
            plot_df["null_mean_ci_low"],
            plot_df["null_mean_ci_high"],
            color=self.null_color,
            alpha=self.null_band_alpha,
        )
        axes[0].set_ylabel("Mean absolute lag (s)", fontsize=self.labelSize)
        axes[0].set_title("Conditioned Mean", fontsize=self.titleSize - 1)
        axes[0].legend(frameon=False)

        axes[1].plot(x, plot_df["real_median"], "o-", color=self.real_color, lw=1.8, label="Real median")
        axes[1].plot(x, plot_df["null_median_of_median"], "s-", color=self.null_color, lw=1.8, label="Null median")
        axes[1].fill_between(
            x,
            plot_df["null_median_ci_low"],
            plot_df["null_median_ci_high"],
            color=self.null_color,
            alpha=self.null_band_alpha,
        )
        axes[1].set_ylabel("Median absolute lag (s)", fontsize=self.labelSize)
        axes[1].set_title("Conditioned Median", fontsize=self.titleSize - 1)
        axes[1].legend(frameon=False)

        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels(plot_df["bin_label"], rotation=25, ha="right")
            ax.set_xlabel("First press time from lever onset bin (s)", fontsize=self.labelSize)

        fig.suptitle("Figure 5. Conditioning on First Press Time", fontsize=self.titleSize)
        self._save_figure_png_pdf(fig, outdir, "figure5_first_press_binned_comparison")

        return merged

    def _compute_session_level_summaries(
        self,
        trial_metrics: pd.DataFrame,
        null_long: pd.DataFrame,
        thresholds: Sequence[float],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        both = trial_metrics[trial_metrics["both_pressed"]].copy()

        real_rows = []
        for session_key, g in both.groupby("session_key", sort=False):
            vals_abs = g["lag_abs"].to_numpy(dtype=float)
            vals_signed = g["lag_signed"].to_numpy(dtype=float)
            metrics = self._compute_metric_dict(vals_abs, vals_signed, thresholds)
            metrics["session_key"] = session_key
            metrics["experiment_index"] = g["experiment_index"].iloc[0]
            metrics["sessionID"] = g["sessionID"].iloc[0]
            metrics["date"] = g["date"].iloc[0]
            metrics["ratPair"] = g["ratPair"].iloc[0]
            real_rows.append(metrics)

        session_real = pd.DataFrame(real_rows)

        null_rows = []
        grouped = null_long.groupby(["session_key", "shuffle"], sort=False)
        for (session_key, shuffle_idx), g in grouped:
            vals_abs = g["lag_abs_null"].to_numpy(dtype=float)
            vals_signed = g["lag_signed_null"].to_numpy(dtype=float)
            metrics = self._compute_metric_dict(vals_abs, vals_signed, thresholds)
            metrics["session_key"] = session_key
            metrics["shuffle"] = shuffle_idx
            null_rows.append(metrics)

        session_null_per_shuffle = pd.DataFrame(null_rows)

        agg_rows = []
        metrics_to_agg = [
            "mean_lag_abs",
            "median_lag_abs",
            "fract_rat1_first",
            "fract_rat2_first",
            "fract_tie",
        ] + [f"fract_below_{thr}" for thr in thresholds]

        for session_key, g in session_null_per_shuffle.groupby("session_key", sort=False):
            row = {"session_key": session_key}
            for metric in metrics_to_agg:
                arr = g[metric].to_numpy(dtype=float)
                row[f"null_mean_{metric}"] = float(np.nanmean(arr))
                row[f"null_ci_low_{metric}"] = float(np.nanpercentile(arr, 2.5))
                row[f"null_ci_high_{metric}"] = float(np.nanpercentile(arr, 97.5))
            agg_rows.append(row)

        session_null_summary = pd.DataFrame(agg_rows)
        session_combined = session_real.merge(session_null_summary, on="session_key", how="left")

        return session_real, session_null_per_shuffle, session_combined

    def plot_session_level_summary(
        self,
        session_combined: pd.DataFrame,
        outdir: Path,
        main_threshold: float = 0.25,
    ):
        real_col = f"fract_below_{main_threshold}"
        null_mean_col = f"null_mean_{real_col}"
        null_lo_col = f"null_ci_low_{real_col}"
        null_hi_col = f"null_ci_high_{real_col}"

        if real_col not in session_combined.columns:
            raise ValueError(f"{real_col} not found in session summary")

        plot_df = session_combined.sort_values("experiment_index").reset_index(drop=True)
        x = np.arange(len(plot_df))
        labels = plot_df["sessionID"].astype(str).tolist()

        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.plot(x, plot_df[real_col], "o-", color=self.real_color, lw=2, label="Real")
        ax.plot(x, plot_df[null_mean_col], "s-", color=self.null_color, lw=2, label="Shuffled null mean")
        ax.fill_between(
            x,
            plot_df[null_lo_col],
            plot_df[null_hi_col],
            color=self.null_color,
            alpha=self.null_band_alpha,
            label="Shuffled null 95%",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Session", fontsize=self.labelSize)
        ax.set_ylabel(f"Fraction absolute lag <= {main_threshold}s", fontsize=self.labelSize)
        ax.set_title("Figure 6. Session-Level Coordination Summary", fontsize=self.titleSize)
        ax.legend(frameon=False)
        self._save_figure_png_pdf(fig, outdir, "figure6_session_level_summary")

    def run_lever_coordination_analysis(
        self,
        output_dir: str = "lever_coordination_outputs",
        n_shuffles: int = 1000,
        thresholds: Sequence[float] = (0.1, 0.25, 0.5, 1.0),
        first_press_bins: Sequence[float] = (0.0, 0.5, 1.0, 2.0, np.inf),
        main_threshold: float = 0.25,
        random_state: int = 0,
        drop_negative_press_times: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        trial_metrics = self.compute_relative_press_metrics(
            drop_negative_press_times=drop_negative_press_times
        )
        if trial_metrics.empty:
            raise ValueError("No valid trial metrics were found across experiments.")

        both = trial_metrics[trial_metrics["both_pressed"]].copy()
        if both.empty:
            raise ValueError("No trials with both rats pressing were found.")

        null_long, null_summary = self.shuffle_press_times_within_session(
            trial_metrics,
            n_shuffles=n_shuffles,
            random_state=random_state,
            thresholds=thresholds,
        )
        if null_long.empty:
            raise ValueError("Null shuffle generation failed (empty result).")

        real_metrics_df, global_summary, both_only = self.summarize_real_vs_null(
            trial_metrics,
            null_summary,
            thresholds=thresholds,
        )

        self.plot_abs_lag_histogram(both_only["lag_abs"].to_numpy(dtype=float), null_long, outdir)
        self.plot_abs_lag_cdf(both_only["lag_abs"].to_numpy(dtype=float), null_long, outdir)
        self.plot_excess_synchrony(both_only["lag_abs"].to_numpy(dtype=float), null_long, outdir)
        self.plot_signed_lag_distribution(both_only["lag_signed"].to_numpy(dtype=float), null_long, outdir)
        first_press_bin_summary = self.plot_first_press_binned_comparison(
            trial_metrics,
            null_long,
            outdir,
            bins=first_press_bins,
        )

        session_real, session_null_per_shuffle, session_combined = self._compute_session_level_summaries(
            trial_metrics,
            null_long,
            thresholds,
        )
        self.plot_session_level_summary(session_combined, outdir, main_threshold=main_threshold)

        trial_metrics.to_csv(outdir / "trial_level_metrics_cleaned.csv", index=False)
        null_long.to_csv(outdir / "null_long_results.csv", index=False)
        null_summary.to_csv(outdir / "null_summary_per_shuffle.csv", index=False)
        real_metrics_df.to_csv(outdir / "global_real_metrics.csv", index=False)
        global_summary.to_csv(outdir / "global_summary_stats.csv", index=False)
        first_press_bin_summary.to_csv(outdir / "first_press_binned_summary.csv", index=False)
        session_real.to_csv(outdir / "session_level_real_summary.csv", index=False)
        session_null_per_shuffle.to_csv(outdir / "session_level_null_per_shuffle.csv", index=False)
        session_combined.to_csv(outdir / "session_level_summary_combined.csv", index=False)

        mean_row = global_summary[global_summary["metric"] == "mean_lag_abs"]
        frac_row = global_summary[global_summary["metric"] == f"fract_below_{main_threshold}"]

        mean_text = "n/a"
        frac_text = "n/a"
        first_text = "rat1 first=n/a, rat2 first=n/a"
        supports_coord = False
        p_mean = np.nan
        p_frac = np.nan

        if not mean_row.empty:
            real_mean = float(mean_row["real_value"].iloc[0])
            p_mean = float(mean_row["empirical_p"].iloc[0])
            mean_text = f"mean absolute lag={real_mean:.3f}s vs null (p={p_mean:.4f}, smaller-tail)"

        if not frac_row.empty:
            real_frac = float(frac_row["real_value"].iloc[0])
            p_frac = float(frac_row["empirical_p"].iloc[0])
            frac_text = f"fraction lag_abs<={main_threshold}s={real_frac:.3f} vs null (p={p_frac:.4f}, larger-tail)"

        rat1_first = real_metrics_df["fract_rat1_first"].iloc[0] if "fract_rat1_first" in real_metrics_df.columns else np.nan
        rat2_first = real_metrics_df["fract_rat2_first"].iloc[0] if "fract_rat2_first" in real_metrics_df.columns else np.nan
        if np.isfinite(rat1_first) and np.isfinite(rat2_first):
            first_text = f"rat1 first={rat1_first:.3f}, rat2 first={rat2_first:.3f}"

        supports_coord = bool(np.isfinite(p_mean) and np.isfinite(p_frac) and (p_mean < 0.05) and (p_frac < 0.05))

        summary_line = (
            "Coordination summary: "
            + mean_text
            + "; "
            + frac_text
            + "; "
            + first_text
            + ". "
            + (
                "Result supports coordination beyond chance under the within-session shuffle null."
                if supports_coord
                else "Result does not reach the default p<0.05 support criterion for coordination beyond chance."
            )
        )

        print("\n" + summary_line)
        with open(outdir / "concise_summary.txt", "w", encoding="utf-8") as f:
            f.write(summary_line + "\n")

        return {
            "trial_metrics": trial_metrics,
            "null_long": null_long,
            "null_summary": null_summary,
            "real_metrics": real_metrics_df,
            "global_summary": global_summary,
            "first_press_binned_summary": first_press_bin_summary,
            "session_real_summary": session_real,
            "session_null_per_shuffle": session_null_per_shuffle,
            "session_summary_combined": session_combined,
        }


# Testing Multi File Graphs


def getFiltered():
    fe = fileExtractor(filtered)
    fe.data = fe.deleteBadNaN()
    fpsList, totFramesList = fe.returnFPSandTotFrames()
    initial_nan_list = fe.returnNaNPercentage()
    dates = fe.getDatesList()
    sessions = fe.getSessionIDList()
    ratPairs = fe.getRatPairList()
    familiarity = fe.getFamiliarityList()
    transparency = fe.getBarrierTransparencyList()
    return [
        fe.getLevsDatapath(),
        fe.getMagsDatapath(),
        fe.getPosDatapath(),
        fpsList,
        totFramesList,
        initial_nan_list,
        dates,
        sessions,
        ratPairs,
        familiarity,
        transparency,
    ]


def trainingCoopData():
    fe = fileExtractor(only_TrainingCoop_filtered)
    fe.data = fe.deleteBadNaN()
    fpsList, totFramesList = fe.returnFPSandTotFrames()
    initial_nan_list = fe.returnNaNPercentage()
    return [
        fe.getLevsDatapath(),
        fe.getMagsDatapath(),
        fe.getPosDatapath(),
        fpsList,
        totFramesList,
        initial_nan_list,
    ]


def trainingCoopDataThresh1():
    fe = fileExtractor(only_TrainingCoop_filtered)
    fe.keepOnlyThresh1()
    fe.data = fe.deleteBadNaN()
    fpsList, totFramesList = fe.returnFPSandTotFrames()
    initial_nan_list = fe.returnNaNPercentage()
    return [
        fe.getLevsDatapath(),
        fe.getMagsDatapath(),
        fe.getPosDatapath(),
        fpsList,
        totFramesList,
        initial_nan_list,
    ]


def getUnfamiliar():
    fe = fileExtractor(only_unfamiliar_filtered)
    fe.data = fe.deleteBadNaN()
    fpsList, totFramesList = fe.returnFPSandTotFrames()
    initial_nan_list = fe.returnNaNPercentage()
    dates = fe.getDatesList()
    sessions = fe.getSessionIDList()
    dates = dates.tolist()
    ratPairs = fe.getRatPairList()
    familiarity = fe.getFamiliarityList()
    transparency = fe.getBarrierTransparencyList()
    return [
        fe.getLevsDatapath(),
        fe.getMagsDatapath(),
        fe.getPosDatapath(),
        fpsList,
        totFramesList,
        initial_nan_list,
        dates,
        sessions,
        ratPairs,
        familiarity,
        transparency,
    ]


fiberPhoto = "/gpfs/radev/home/drb83/project/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/fiber_photo.csv"


def getFiberPhoto():
    fe = fileExtractor(fiberPhoto)
    fpsList, totFramesList = fe.returnFPSandTotFrames()
    initial_nan_list = fe.returnNaNPercentage()
    fiberFiles = fe.getFiberPhotoDataPath()
    print("fiberFiles: ", fiberFiles)
    return [
        fe.getLevsDatapath(),
        fe.getMagsDatapath(),
        fe.getPosDatapath(),
        fpsList,
        totFramesList,
        initial_nan_list,
        fiberFiles,
    ]


if __name__ == "__main__":
    arr = getFiltered()
    # arr = trainingCoopData()
    # arr = trainingCoopDataThresh1()
    # arr = getUnfamiliar()
    # arr = getFiberPhoto()

    lev_files = arr[0]
    mag_files = arr[1]
    pos_files = arr[2]
    fpsList = arr[3]
    totFramesList = arr[4]
    initialNanList = arr[5]
    dates = arr[6]
    sessions = arr[7]
    ratPairs = arr[8]

    experiment = multiFileGraphs(
        mag_files,
        lev_files,
        pos_files,
        fpsList,
        totFramesList,
        initialNanList,
        dates,
        sessions,
        ratPairs,
        prefix="GazeFigures_",
        save=True,
        saveAsPDF=True,
    )

    experiment.run_lever_coordination_analysis(
        output_dir="GazeFigures_lever_coordination_outputs",
        n_shuffles=1000,
        thresholds=(0.1, 0.25, 0.5, 1.0),
        first_press_bins=(0.0, 0.5, 1.0, 2.0, np.inf),
        main_threshold=0.25,
        random_state=0,
        drop_negative_press_times=True,
    )
