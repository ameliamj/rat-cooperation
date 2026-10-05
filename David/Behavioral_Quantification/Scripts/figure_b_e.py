#!/usr/bin/env python3
"""Create cooperative paired-testing pose examples (B) and partner decoding (E)."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import logging
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from partner_features import (
    DEFAULT_DATA_ROOT, DEFAULT_INDEX, E_CHANNELS, FEATURES, INDIVIDUAL_FEATURES,
    JOINT_FEATURES, PROJECT, event_indices, frame_features, load_session,
    local_file_index, session_index, trial_features,
)


LOG = logging.getLogger("figure_b_e")
POSE_COLORS = {"blue": (230, 110, 45), "green": (70, 205, 80),
               "red": (55, 75, 230), "yellow": (35, 220, 240)}  # OpenCV BGR
SKELETON_EDGES = ((0, 1), (0, 2), (1, 3), (2, 3), (3, 4))
WINDOWS = {"pre": range(5), "full": range(10)}


def args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX,
                   help="Session index; defaults to all valid cooperative paired-testing sessions")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                   help="Root containing PairedTestingSessions on the cluster")
    p.add_argument("--output", type=Path, default=None,
                   help="Output directory (default: Graphs/PartnerSpecific; local examples use its local_examples subfolder)")
    p.add_argument("--local-examples", action="store_true",
                   help="Explicitly use example files in DataStorage and Example_Data_Files")
    p.add_argument("--analysis", choices=("both", "b", "e"), default="both")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--min-trials", type=int, default=12,
                   help="Minimum usable trials per session for an E comparison")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-sessions", type=int, default=None,
                   help="Optional smoke-test limit; omit for all sessions")
    p.add_argument("--max-pairs", type=int, default=None,
                   help="Optional smoke-test limit; omit for all pairs")
    return p


def _safe_name(*parts: str) -> str:
    raw = "__".join(parts)
    short = "__".join(p.replace("/", "_")[:35] for p in parts)
    return f"{short}__{hashlib.sha1(raw.encode()).hexdigest()[:8]}"


def _track_label(session, track_idx: int) -> tuple[str, tuple[int, int, int]]:
    name = session.track_animal[track_idx]
    color_name = next((k for k, suffix in (("blue", "B"), ("green", "G"),
                                          ("red", "R"), ("yellow", "Y")) if name.endswith(suffix)), "blue")
    return name, POSE_COLORS[color_name]


def _point(xy: np.ndarray) -> tuple[int, int] | None:
    if not np.isfinite(xy).all():
        return None
    return int(round(float(xy[0]))), int(round(float(xy[1])))


def annotate_frame(frame: np.ndarray, session, frame_idx: int, feature: str) -> np.ndarray:
    frame = frame.copy()
    _, w = frame.shape[:2]
    points = session.tracks[:, :, :, frame_idx]
    for track_idx in (0, 1):
        name, color = _track_label(session, track_idx)
        for i, j in SKELETON_EDGES:
            p, q = _point(points[track_idx, :, i]), _point(points[track_idx, :, j])
            if p and q:
                cv2.line(frame, p, q, color, 3, cv2.LINE_AA)
        for part in range(5):
            p = _point(points[track_idx, :, part])
            if p:
                cv2.circle(frame, p, 5, color, -1, cv2.LINE_AA)
        head = _point(points[track_idx, :, 3])
        if head:
            cv2.putText(frame, name, (min(head[0] + 10, w - 110), max(head[1] - 8, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    active = session.animal_track(session.animal_a if feature.startswith("a_") else session.animal_b)
    other = 1 - active
    head = _point(points[active, :, 3])
    partner_head = _point(points[other, :, 3])
    accent = (255, 255, 255)
    if feature == "distance_px":
        if head and partner_head:
            cv2.line(frame, head, partner_head, accent, 2, cv2.LINE_AA)
            cv2.circle(frame, head, 7, accent, 2, cv2.LINE_AA)
            cv2.circle(frame, partner_head, 7, accent, 2, cv2.LINE_AA)
    elif feature == "interacting":
        candidates = []
        for rat in (0, 1):
            nose = _point(points[rat, :, 0])
            if nose:
                for part in range(5):
                    target = _point(points[1 - rat, :, part])
                    if target:
                        candidates.append((np.linalg.norm(np.subtract(nose, target)), nose, target))
        if candidates:
            _, p, q = min(candidates, key=lambda item: item[0])
            cv2.line(frame, p, q, accent, 2, cv2.LINE_AA)
            cv2.circle(frame, p, 10, accent, 2, cv2.LINE_AA)
            cv2.circle(frame, q, 10, accent, 2, cv2.LINE_AA)
    elif "orientation" in feature:
        nose = _point(points[active, :, 0])
        if head and partner_head and nose:
            direction = np.asarray(nose, float) - np.asarray(head, float)
            norm = np.linalg.norm(direction)
            if norm > 0:
                tip = tuple(np.round(np.asarray(head) + direction / norm * 90).astype(int))
                cv2.arrowedLine(frame, head, tip, accent, 3, cv2.LINE_AA, tipLength=0.18)
                cv2.line(frame, head, partner_head, (50, 210, 255), 2, cv2.LINE_AA)
    elif "speed" in feature:
        lag = max(1, round(session.fps))
        if head and frame_idx >= lag:
            old = _point(session.tracks[active, :, 3, frame_idx - lag])
            if old:
                cv2.arrowedLine(frame, old, head, accent, 3, cv2.LINE_AA, tipLength=0.25)
                cv2.circle(frame, old, 6, accent, 2, cv2.LINE_AA)
    elif "is_gazing" in feature:
        nose = _point(points[active, :, 0])
        if head and nose:
            direction = np.asarray(nose, float) - np.asarray(head, float)
            norm = np.linalg.norm(direction)
            if norm > 0:
                start = tuple(np.round(np.asarray(head) + direction / norm * 150).astype(int))
                end = tuple(np.round(np.asarray(head) + direction / norm * 425).astype(int))
                cv2.arrowedLine(frame, start, end, accent, 2, cv2.LINE_AA, tipLength=0.08)

    value = frame_features(session)[feature][frame_idx]
    label = {
        "a_orientation_deg": f"{session.animal_a} orientation toward {session.animal_b}: {value:.1f} deg",
        "b_orientation_deg": f"{session.animal_b} orientation toward {session.animal_a}: {value:.1f} deg",
        "a_speed_px_1s": f"{session.animal_a} head-base displacement over 1 s: {value:.1f} px",
        "b_speed_px_1s": f"{session.animal_b} head-base displacement over 1 s: {value:.1f} px",
        "a_is_gazing": f"{session.animal_a} gazing at {session.animal_b}: {'yes' if value else 'no'}",
        "b_is_gazing": f"{session.animal_b} gazing at {session.animal_a}: {'yes' if value else 'no'}",
        "distance_px": f"Inter-rat head distance: {value:.1f} px",
        "interacting": f"Interacting: {'yes' if value else 'no'}",
    }[feature]
    cv2.rectangle(frame, (0, 0), (w, 72), (15, 15, 15), -1)
    cv2.putText(frame, label, (15, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.7, accent, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{session.vid} | frame {frame_idx} | near cooperative press",
                (15, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.53, accent, 1, cv2.LINE_AA)
    return frame


def save_b(candidates: list[dict], rows: dict[str, pd.Series], data_root: Path,
           local: dict[str, Path], output: Path, fps: float, seed: int) -> None:
    out = output / "B_frames"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    records = []
    cache = {}
    for feature in FEATURES:
        options = [c for c in candidates if c["feature"] == feature and np.isfinite(c["value"])]
        rng.shuffle(options)
        if feature.endswith("is_gazing") or feature == "interacting":
            options.sort(key=lambda item: item["value"] < 0.5)
        chosen = options[:5]
        if len(chosen) < 5:
            LOG.warning("B: %s has only %d usable video instances; requested 5", feature, len(chosen))
        for ordinal, item in enumerate(chosen, 1):
            vid = item["vid"]
            if vid not in cache:
                cache[vid] = load_session(rows[vid], data_root, local, fps=fps)
            session = cache[vid]
            capture = cv2.VideoCapture(str(session.video_path))
            capture.set(cv2.CAP_PROP_POS_FRAMES, item["frame"])
            ok, frame = capture.read()
            capture.release()
            if not ok:
                LOG.warning("B: video frame unreadable: %s frame %s", vid, item["frame"])
                continue
            decorated = annotate_frame(frame, session, item["frame"], feature)
            filename = f"{feature}__{ordinal:02d}__{_safe_name(vid, str(item['trial']))}.png"
            cv2.imwrite(str(out / filename), decorated)
            records.append({"feature": feature, "instance": ordinal, "vid": vid,
                            "trial": item["trial"], "frame": item["frame"],
                            "coop_press_s": item["coop_press_s"], "value": item["value"],
                            "file": filename})
    pd.DataFrame(records).to_csv(output / "B_frame_index.csv", index=False)
    LOG.info("Saved %d annotated B frames to %s", len(records), out)


class BTimeSeries:
    """Keep trial and session sums without retaining every frame of every trial."""

    def __init__(self, fps: float):
        self.fps = fps
        self.offsets = np.arange(-round(5 * fps), round(5 * fps) + 1) / fps
        self.sessions: dict[str, dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}

    def _add(self, vid: str, feature: str, role: str, values: np.ndarray) -> None:
        bucket = self.sessions.setdefault(vid, {})
        key = (feature, role)
        if key not in bucket:
            bucket[key] = (np.zeros(len(self.offsets)), np.zeros(len(self.offsets)),
                           np.zeros(len(self.offsets)))
        sums, counts, sum_squares = bucket[key]
        good = np.isfinite(values)
        sums[good] += values[good]
        counts[good] += 1
        sum_squares[good] += values[good] ** 2

    def add_session(self, session) -> None:
        for trial in session.trials.itertuples(index=False):
            if not trial.success or not np.isfinite(trial.coop_press_s):
                continue
            frames = event_indices(session, trial.coop_press_s)
            if frames is None:
                continue
            for feature in INDIVIDUAL_FEATURES:
                data = session.features[feature]
                for track in (0, 1):
                    self._add(session.vid, feature, "merged", data[track, frames])
                if np.isfinite(trial.first_track):
                    first = int(trial.first_track)
                    self._add(session.vid, feature, "first", data[first, frames])
                    self._add(session.vid, feature, "second", data[1 - first, frames])
            for feature in JOINT_FEATURES:
                self._add(session.vid, feature, "merged", session.features[feature][frames])

    def save(self, output: Path) -> None:
        out = output / "B_timeseries"
        for layout in ("merged", "first_vs_second"):
            (out / layout).mkdir(parents=True, exist_ok=True)
        labels = {
            "orientation_deg": ("Orientation to partner", "Angle (degrees)"),
            "speed_px_1s": ("Head-base displacement", "Pixels in preceding 1 s"),
            "is_gazing": ("Gazing at partner", "Fraction of rats gazing"),
            "distance_px": ("Inter-rat head-base distance", "Distance (pixels)"),
            "interacting": ("Interacting", "Fraction of pairs interacting"),
        }
        rows = []
        for feature in (*INDIVIDUAL_FEATURES, *JOINT_FEATURES):
            layouts = ("merged", "first_vs_second") if feature in INDIVIDUAL_FEATURES else ("merged",)
            for layout in layouts:
                roles = ("merged",) if layout == "merged" else ("first", "second")
                for weighting in ("trial", "session"):
                    fig, ax = plt.subplots(figsize=(8.3, 4.4), constrained_layout=True)
                    plotted = False
                    binary_peak = 0.0
                    for role in roles:
                        sums = np.zeros(len(self.offsets))
                        counts = np.zeros(len(self.offsets))
                        sum_squares = np.zeros(len(self.offsets))
                        session_curves = []
                        for bucket in self.sessions.values():
                            if (feature, role) not in bucket:
                                continue
                            s, c, ss = bucket[(feature, role)]
                            sums += s
                            counts += c
                            sum_squares += ss
                            session_curves.append(np.divide(s, c, out=np.full_like(s, np.nan), where=c > 0))
                        if weighting == "trial":
                            mean = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
                            n = counts
                            variance = np.divide(sum_squares - sums * mean, n - 1,
                                                 out=np.full_like(sums, np.nan), where=n > 1)
                            variance = np.maximum(variance, 0)
                            sem = np.sqrt(np.divide(variance, n, out=np.full_like(sums, np.nan), where=n > 1))
                        elif session_curves:
                            curves = np.asarray(session_curves)
                            n = np.isfinite(curves).sum(axis=0)
                            mean = np.divide(np.nansum(curves, axis=0), n,
                                             out=np.full(len(n), np.nan), where=n > 0)
                            centered = curves - mean
                            centered[~np.isfinite(curves)] = np.nan
                            squared_deviations = np.nansum(centered ** 2, axis=0)
                            variance = np.divide(squared_deviations, n - 1,
                                                 out=np.full(len(n), np.nan), where=n > 1)
                            sem = np.sqrt(np.divide(variance, n,
                                                    out=np.full(len(n), np.nan), where=n > 1))
                        else:
                            mean = np.full(len(self.offsets), np.nan)
                            n = np.zeros(len(self.offsets))
                            sem = np.full(len(self.offsets), np.nan)
                        if np.isfinite(mean).any():
                            color = {"merged": "#2f6e9c", "first": "#d05b39", "second": "#527b49"}[role]
                            lower, upper = mean - sem, mean + sem
                            if feature in ("is_gazing", "interacting"):
                                lower, upper = np.clip(lower, 0, 1), np.clip(upper, 0, 1)
                                visible_upper = np.where(np.isfinite(upper), upper, mean)
                                if np.isfinite(visible_upper).any():
                                    binary_peak = max(binary_peak, float(np.nanmax(visible_upper)))
                            ax.fill_between(self.offsets, lower, upper, color=color, alpha=0.22,
                                            linewidth=0)
                            ax.plot(self.offsets, mean, label={"merged": "Both rats", "first": "First presser",
                                                              "second": "Second presser"}[role], color=color, linewidth=2)
                            plotted = True
                        rows.extend({"feature": feature, "layout": layout, "weighting": weighting,
                                     "role": role, "time_s": time, "mean": value,
                                     "sem": error, "n": int(count)}
                                    for time, value, error, count in zip(self.offsets, mean, sem, n))
                    ax.axvline(0, color="black", linestyle="--", linewidth=1)
                    ax.set(xlim=(-5, 5), xlabel="Time from cooperative press (s)", ylabel=labels[feature][1],
                           title=f"{labels[feature][0]} | {weighting}-weighted | {layout.replace('_', ' ')}\nMean ± SEM")
                    if feature in ("is_gazing", "interacting"):
                        ax.set_ylim(0, min(1.02, max(0.001, binary_peak * 1.1)))
                    if plotted and layout == "first_vs_second":
                        ax.legend(frameon=False)
                    fig.savefig(out / layout / f"{feature}__{layout}__{weighting}.png", dpi=200)
                    plt.close(fig)
        pd.DataFrame(rows).to_csv(output / "B_timeseries.csv", index=False)
        LOG.info("Saved 16 B time-series plots to %s", out)


def model_columns(window: str) -> list[str]:
    return [f"{channel}_t{bin_num - 5:+d}"
            for channel in E_CHANNELS for bin_num in WINDOWS[window]]


def grouped_importance(model, X: pd.DataFrame, y: np.ndarray, window: str,
                       seed: int, repeats: int = 20) -> pd.DataFrame:
    """Shuffle all time bins of one behavior together on held-out trials."""
    rng = np.random.default_rng(seed)
    baseline = balanced_accuracy_score(y, model.predict(X))
    records = []
    for channel in E_CHANNELS:
        columns = [col for col in model_columns(window) if col.startswith(channel + "_t")]
        drops = []
        for _ in range(repeats):
            permuted = X.copy()
            order = rng.permutation(len(X))
            permuted.loc[:, columns] = X.iloc[order][columns].to_numpy()
            drops.append(baseline - balanced_accuracy_score(y, model.predict(permuted)))
        records.append({"feature": channel, "importance_mean": float(np.mean(drops)),
                        "importance_sd": float(np.std(drops))})
    return pd.DataFrame(records)


def _pair_figure(pair_id: str, window: str, focal: str, partner0: str, partner1: str,
                 y_true: np.ndarray, y_pred: np.ndarray, importance: pd.DataFrame,
                 score: float, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    axes[0].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    axes[0].set_xticks([0, 1], [partner0, partner1], rotation=30)
    axes[0].set_yticks([0, 1], [partner0, partner1])
    axes[0].set_xlabel("Predicted partner")
    axes[0].set_ylabel("Actual partner")
    axes[0].set_title(f"Focal rat {focal}; balanced accuracy {score:.2f}")
    imp = importance.sort_values("importance_mean")
    axes[1].barh(imp["feature"], imp["importance_mean"], xerr=imp["importance_sd"], color="#3976a8")
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Drop in test balanced accuracy when shuffled")
    axes[1].set_title("Feature importance")
    fig.suptitle(f"Exploratory partner-plus-session classification | {window} window", fontsize=12)
    fig.savefig(outdir / f"{pair_id}.png", dpi=180)
    plt.close(fig)


def save_e(table: pd.DataFrame, output: Path, min_trials: int, seed: int,
           max_pairs: int | None) -> None:
    if table.empty:
        LOG.warning("E: no usable trial feature rows")
        return
    out = output / "E_pairs"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "E_trial_features.csv", index=False)
    summaries, predictions, importances, skipped = [], [], [], []
    attempted = 0
    for focal, focal_rows in table.groupby("focal", sort=True):
        sessions = sorted(focal_rows["vid"].unique())
        for vid0, vid1 in itertools.combinations(sessions, 2):
            left = focal_rows[focal_rows["vid"] == vid0]
            right = focal_rows[focal_rows["vid"] == vid1]
            partner0, partner1 = left["partner"].iloc[0], right["partner"].iloc[0]
            if partner0 == partner1:
                continue
            pair_id = _safe_name(focal, vid0, vid1)
            if len(left) < min_trials or len(right) < min_trials:
                skipped.append({"pair_id": pair_id, "reason": "too_few_trials",
                                "n0": len(left), "n1": len(right)})
                continue
            attempted += 1
            if max_pairs is not None and attempted > max_pairs:
                break
            combined = pd.concat([left, right], ignore_index=True)
            y = (combined["partner"] == partner1).astype(int).to_numpy()
            idx_train, idx_test = train_test_split(np.arange(len(combined)), test_size=0.25,
                                                    random_state=seed, stratify=y)
            for window in WINDOWS:
                X = combined[model_columns(window)]
                model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                      LogisticRegression(C=0.1, max_iter=1000,
                                                         class_weight="balanced", random_state=seed))
                model.fit(X.iloc[idx_train], y[idx_train])
                test_x = X.iloc[idx_test]
                pred = model.predict(test_x)
                probability = model.predict_proba(test_x)[:, 1]
                balanced = balanced_accuracy_score(y[idx_test], pred)
                importance = grouped_importance(model, test_x, y[idx_test], window, seed)
                window_out = out / window
                window_out.mkdir(parents=True, exist_ok=True)
                _pair_figure(pair_id, window, focal, partner0, partner1,
                             y[idx_test], pred, importance, balanced, window_out)
                summaries.append({"pair_id": pair_id, "window": window, "focal": focal,
                                  "partner0": partner0, "partner1": partner1,
                                  "session0": vid0, "session1": vid1,
                                  "familiarity0": left["familiarity"].iloc[0],
                                  "familiarity1": right["familiarity"].iloc[0],
                                  "divider0": left["divider"].iloc[0],
                                  "divider1": right["divider"].iloc[0],
                                  "date0": left["date"].iloc[0], "date1": right["date"].iloc[0],
                                  "n0": len(left), "n1": len(right), "n_train": len(idx_train),
                                  "n_test": len(idx_test), "accuracy": accuracy_score(y[idx_test], pred),
                                  "balanced_accuracy": balanced,
                                  "roc_auc": roc_auc_score(y[idx_test], probability),
                                  "figure": f"E_pairs/{window}/{pair_id}.png"})
                for i, prediction, prob in zip(idx_test, pred, probability):
                    row = combined.iloc[i]
                    predictions.append({"pair_id": pair_id, "window": window, "focal": focal,
                                        "vid": row["vid"], "trial": row["trial"],
                                        "actual_partner": row["partner"],
                                        "predicted_partner": partner1 if prediction else partner0,
                                        "probability_partner1": prob})
                for item in importance.itertuples(index=False):
                    importances.append({"pair_id": pair_id, "window": window,
                                        "feature": item.feature, "importance_mean": item.importance_mean,
                                        "importance_sd": item.importance_sd})
        if max_pairs is not None and attempted > max_pairs:
            break
    pd.DataFrame(skipped, columns=["pair_id", "reason", "n0", "n1"]).to_csv(
        output / "E_skipped_pairs.csv", index=False)
    pd.DataFrame(summaries, columns=[
        "pair_id", "window", "focal", "partner0", "partner1", "session0", "session1",
        "familiarity0", "familiarity1", "divider0", "divider1", "date0", "date1",
        "n0", "n1", "n_train", "n_test", "accuracy", "balanced_accuracy", "roc_auc", "figure",
    ]).to_csv(output / "E_pair_results.csv", index=False)
    pd.DataFrame(predictions, columns=[
        "pair_id", "window", "focal", "vid", "trial", "actual_partner", "predicted_partner",
        "probability_partner1",
    ]).to_csv(output / "E_test_predictions.csv", index=False)
    pd.DataFrame(importances, columns=[
        "pair_id", "window", "feature", "importance_mean", "importance_sd",
    ]).to_csv(output / "E_feature_importance.csv", index=False)
    if not summaries:
        LOG.warning("E: no session pairs had enough usable trials with different partners")
        return
    results = pd.DataFrame(summaries)
    importance_df = pd.DataFrame(importances)
    for window in WINDOWS:
        these_results = results[results["window"] == window]
        these_importances = importance_df[importance_df["window"] == window]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        axes[0].hist(these_results["balanced_accuracy"], bins=np.linspace(0, 1, 21), color="#3976a8")
        axes[0].axvline(0.5, color="black", linestyle="--", label="Balanced chance")
        axes[0].set(xlabel="Test balanced accuracy", ylabel="Session-pair comparisons",
                    title=f"{len(these_results)} comparisons")
        axes[0].legend(frameon=False)
        aggregate = these_importances.groupby("feature")["importance_mean"].agg(["mean", "std"]).fillna(0)
        aggregate = aggregate.sort_values("mean")
        axes[1].barh(aggregate.index, aggregate["mean"], xerr=aggregate["std"], color="#3976a8")
        axes[1].axvline(0, color="black", linewidth=0.8)
        axes[1].set(xlabel="Mean drop in test balanced accuracy", title="Features across comparisons")
        fig.suptitle(f"Exploratory partner-plus-session classification | {window} window")
        fig.savefig(output / f"E_summary_{window}.png", dpi=200)
        plt.close(fig)
    LOG.info("E: saved %d session-pair comparisons in each window", len(results) // 2)


def main() -> None:
    args = args_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not (args.data_root / "PairedTestingSessions").is_dir() and not args.local_examples:
        raise SystemExit(
            f"Raw data root {args.data_root} is unavailable. Run on Misha, pass --data-root "
            "for another full-data location, or add --local-examples for the small local examples."
        )
    if args.output is None:
        args.output = PROJECT / "Graphs" / "PartnerSpecific" / "revised"
        if args.local_examples:
            args.output /= "local_examples"
    args.output.mkdir(parents=True, exist_ok=True)
    index = session_index(args.index)
    if args.max_sessions is not None:
        index = index.iloc[:args.max_sessions]
    index.to_csv(args.output / "selected_sessions.csv", index=False)
    LOG.info("Selected %d cooperative paired-testing recordings in %d session folders",
             len(index), index["session"].nunique())
    local = local_file_index(PROJECT) if args.local_examples else {}
    rows = {str(row["vid"]): row for _, row in index.iterrows()}
    all_trials, b_candidates, status = [], [], []
    b_series = BTimeSeries(args.fps)
    for count, (_, row) in enumerate(index.iterrows(), 1):
        vid = str(row["vid"])
        try:
            session = load_session(row, args.data_root, local, fps=args.fps)
            status.append({"vid": vid, "status": "loaded", "reason": "",
                           "trials": len(session.trials), "video": bool(session.video_path)})
            if args.analysis in ("both", "e"):
                for focal in (session.animal_a, session.animal_b):
                    table = trial_features(session, focal)
                    if not table.empty:
                        table["familiarity"] = row.get("familiarity", "")
                        table["divider"] = row.get("dividers", "")
                        table["date"] = vid[:6]
                        all_trials.append(table)
            if args.analysis in ("both", "b"):
                b_series.add_session(session)
            if args.analysis in ("both", "b") and session.video_path is not None:
                features = frame_features(session)
                for trial in session.trials.itertuples(index=False):
                    if not trial.success or not np.isfinite(trial.coop_press_s):
                        continue
                    press_frame = round(trial.coop_press_s * session.fps)
                    standard = press_frame - max(1, round(0.1 * session.fps))
                    for feature in FEATURES:
                        frame = standard
                        if feature.endswith("is_gazing") or feature == "interacting":
                            search = range(max(0, press_frame - round(5 * session.fps)),
                                           max(0, press_frame - 1))
                            active = [t for t in search if t < len(features[feature])
                                      and np.isfinite(features[feature][t])
                                      and features[feature][t] > 0.5]
                            if active:
                                frame = active[-1]
                        if not (0 <= frame < session.tracks.shape[-1]):
                            continue
                        if all(np.isfinite(arr[frame]) for arr in features.values()):
                            b_candidates.append({"vid": vid, "trial": trial.trial,
                                                 "feature": feature, "frame": frame,
                                                 "coop_press_s": trial.coop_press_s,
                                                 "value": float(features[feature][frame])})
        except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
            status.append({"vid": vid, "status": "skipped", "reason": str(exc),
                           "trials": 0, "video": False})
        if count % 25 == 0:
            LOG.info("Processed %d/%d indexed sessions", count, len(index))
    pd.DataFrame(status).to_csv(args.output / "session_status.csv", index=False)
    if args.analysis in ("both", "b"):
        save_b(b_candidates, rows, args.data_root, local, args.output, args.fps, args.seed)
        b_series.save(args.output)
    if args.analysis in ("both", "e"):
        table = pd.concat(all_trials, ignore_index=True) if all_trials else pd.DataFrame()
        save_e(table, args.output, args.min_trials, args.seed, args.max_pairs)


if __name__ == "__main__":
    main()
