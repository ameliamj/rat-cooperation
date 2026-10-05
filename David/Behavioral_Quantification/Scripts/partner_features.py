"""Shared pose and trial features for cooperative paired-testing figures B and E."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import shapely


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = PROJECT / "Sorted_Data_Files" / "only_PairedTesting.csv"
DEFAULT_DATA_ROOT = Path("/gpfs/radev/pi/saxena/aj764")
FEATURES = (
    "a_orientation_deg",
    "b_orientation_deg",
    "a_speed_px_1s",
    "b_speed_px_1s",
    "a_is_gazing",
    "b_is_gazing",
    "distance_px",
    "interacting",
)
INDIVIDUAL_FEATURES = ("orientation_deg", "speed_px_1s", "is_gazing")
JOINT_FEATURES = ("distance_px", "interacting")
COLOR_SUFFIX = {"blue": "B", "green": "G", "red": "R", "yellow": "Y"}


@dataclass
class Session:
    vid: str
    session_id: str
    animal_a: str
    animal_b: str
    track_animal: tuple[str, str]
    h5_path: Path
    lever_path: Path
    video_path: Path | None
    fps: float
    tracks: np.ndarray
    features: dict[str, np.ndarray]
    trials: pd.DataFrame

    def animal_track(self, animal: str) -> int:
        return self.track_animal.index(animal)


def session_index(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"vid": str, "session": str})
    needed = {"vid", "session", "color pair", "trial type", "test/train", "pred", "correct", "levers"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Session index missing columns: {sorted(missing)}")
    yes = lambda col: df[col].astype(str).str.lower().eq("true")
    keep = df["trial type"].astype(str).str.lower().eq("coop")
    keep &= df["test/train"].astype(str).str.lower().eq("test")
    for col in ("pred", "correct", "levers"):
        keep &= yes(col)
    return df.loc[keep].drop_duplicates("vid").reset_index(drop=True)


def local_file_index(project: Path) -> dict[str, Path]:
    """Find local example files once; cluster paths take priority when present."""
    found: dict[str, Path] = {}
    for folder in (project / "DataStorage", project / "Example_Data_Files"):
        if folder.exists():
            for path in folder.rglob("*"):
                if path.suffix.lower() in {".h5", ".csv", ".mp4"} and path.is_file():
                    found.setdefault(path.name, path)
    return found


def resolve_files(row: pd.Series, data_root: Path, local: dict[str, Path]) -> tuple[Path, Path, Path | None]:
    vid, session = str(row["vid"]), str(row["session"])
    root = data_root / "PairedTestingSessions" / session
    h5_candidates = [
        root / "Tracking" / sub / f"{vid}.predictions.h5"
        for sub in ("h5_uncorrected", "h5_corrected")
    ]
    h5 = next((p for p in h5_candidates if p.is_file()), None) or local.get(f"{vid}.predictions.h5")
    lever = root / "Behavioral" / "processed" / "lever" / f"{vid}_lever.csv"
    if not lever.is_file():
        lever = local.get(lever.name)
    video = root / "Videos" / f"{vid}.mp4"
    if not video.is_file():
        video = local.get(video.name)
    if h5 is None or lever is None:
        raise FileNotFoundError(f"Missing pose or lever data for {vid}")
    return h5, lever, video


def _animal_tracks(h5: h5py.File, animal_a: str, animal_b: str) -> tuple[str, str]:
    if "track_names" not in h5 or len(h5["track_names"]) != 2:
        raise ValueError("Missing two named pose tracks")
    names = [x.decode().lower() if isinstance(x, bytes) else str(x).lower() for x in h5["track_names"][:]]
    animals = []
    for name in names:
        suffix = COLOR_SUFFIX.get(name)
        hits = [a for a in (animal_a, animal_b) if a.upper().endswith(suffix or "#")]
        if len(hits) != 1:
            raise ValueError(f"Cannot uniquely map pose track {name!r} to {animal_a}, {animal_b}")
        animals.append(hits[0])
    if set(animals) != {animal_a, animal_b}:
        raise ValueError(f"Both pose tracks map to the same animal: {names}")
    return tuple(animals)


def _long_runs(hits: np.ndarray, minimum: int, include_start: bool) -> np.ndarray:
    """Mark runs of at least minimum frames, matching the legacy gaze/interaction rules."""
    output = np.zeros(len(hits), dtype=bool)
    start = 0
    for end in range(len(hits) + 1):
        if end < len(hits) and hits[end]:
            continue
        if end - start >= minimum:
            output[start if include_start else start + minimum - 1:end] = True
        start = end + 1
    return output


def _zones(head: np.ndarray) -> np.ndarray:
    """Legacy arena zones: 0 other, 1/2 lever, 3/4 magazine, 5 middle."""
    x, y = head
    zone = np.zeros(len(x), dtype=np.int8)
    zone[(351 <= x) & (x < 1041)] = 5
    zone[(10 <= x) & (x <= 350) & (10 <= y) & (y <= 310)] = 1
    zone[(10 <= x) & (x <= 350) & (330 <= y) & (y <= 630)] = 2
    zone[(1042 <= x) & (x <= 1382) & (10 <= y) & (y <= 310)] = 3
    zone[(1042 <= x) & (x <= 1382) & (330 <= y) & (y <= 630)] = 4
    return zone


def _gaze_intersects(tracks: np.ndarray, source: int, valid: np.ndarray) -> np.ndarray:
    """Vectorized version of the legacy 150–425 px gaze ray/body-polygon test."""
    target = 1 - source
    origin = tracks[source, :, 3, :].T
    heading = (tracks[source, :, 0, :] - tracks[source, :, 3, :]).T
    norm = np.linalg.norm(heading, axis=1)
    direction = heading / np.where(norm[:, None] > 1e-6, norm[:, None], np.nan)
    ray = np.stack((origin + 150 * direction, origin + 425 * direction), axis=1)
    body = np.stack([tracks[target, :, part, :].T for part in (1, 0, 2, 4, 1)], axis=1)
    good = valid[source, 0] & valid[source, 3]
    good &= valid[target, [0, 1, 2, 4]].all(axis=0)
    good &= np.isfinite(ray).all(axis=(1, 2)) & np.isfinite(body).all(axis=(1, 2))
    result = np.zeros(len(good), dtype=bool)
    if good.any():
        polygons = shapely.polygons(body[good])
        lines = shapely.linestrings(ray[good])
        result[good] = shapely.intersects(lines, polygons)
    return result


def _fill_for_events(tracks: np.ndarray) -> np.ndarray:
    """Linearly fill pose gaps as the legacy event detector does."""
    filled = tracks.astype(float).copy()
    frames = np.arange(tracks.shape[-1])
    for rat in range(2):
        for coord in range(2):
            for part in range(5):
                values = filled[rat, coord, part]
                good = np.isfinite(values)
                if good.sum() >= 2:
                    filled[rat, coord, part] = np.interp(frames, frames[good], values[good])
    return filled


def _pose_features(tracks: np.ndarray, fps: float, point_scores: np.ndarray | None, min_score: float) -> dict[str, np.ndarray]:
    """Frame-level orientation, one-second speed, gaze, distance, and interaction."""
    n = tracks.shape[-1]
    head = tracks[:, :, 3, :].astype(float).copy()
    nose = tracks[:, :, 0, :].astype(float).copy()
    valid = np.isfinite(tracks).all(axis=1)
    if point_scores is not None and point_scores.shape == (2, 5, n):
        valid &= np.isfinite(point_scores) & (point_scores >= min_score)
    head = np.where(valid[:, 3, :][:, None, :], head, np.nan)
    nose = np.where(valid[:, 0, :][:, None, :], nose, np.nan)
    delta = head[1] - head[0]
    distance = np.linalg.norm(delta, axis=0)
    unit = delta / np.where(distance > 1e-6, distance, np.nan)
    heading = nose - head
    heading_norm = np.linalg.norm(heading, axis=1)
    heading = heading / np.where(heading_norm > 1e-6, heading_norm, np.nan)[:, None, :]
    toward = np.stack((unit, -unit))
    cosine = np.einsum("icf,icf->if", heading, toward)
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    lag = max(1, round(fps))
    speed = np.full((2, n), np.nan)
    speed[:, lag:] = np.linalg.norm(head[:, :, lag:] - head[:, :, :-lag], axis=1)

    filled = _fill_for_events(tracks)
    event_valid = np.isfinite(filled).all(axis=1)
    event_head = filled[:, :, 3, :]
    # Either nose within 90 px of any body point, subject to the legacy zone
    # exclusions. An interaction starts on the tenth consecutive close frame.
    d0 = np.linalg.norm(filled[1] - filled[0, :, 0, :][:, None, :], axis=0)
    d1 = np.linalg.norm(filled[0] - filled[1, :, 0, :][:, None, :], axis=0)
    closest0 = np.min(np.where(np.isfinite(d0), d0, np.inf), axis=0)
    closest1 = np.min(np.where(np.isfinite(d1), d1, np.inf), axis=0)
    near = np.minimum(closest0, closest1) < 90
    near &= event_valid.all(axis=(0, 1))
    z0, z1 = _zones(event_head[0]), _zones(event_head[1])
    opposite = (((z0 == 1) & (z1 == 2)) | ((z0 == 2) & (z1 == 1)) |
                ((z0 == 3) & (z1 == 4)) | ((z0 == 4) & (z1 == 3)))
    near &= (z0 != 0) & (z1 != 0) & ~opposite
    interacting = _long_runs(near, 10, include_start=False)

    gaze = np.stack([
        _long_runs(_gaze_intersects(filled, rat, event_valid), 10, include_start=True)
        for rat in (0, 1)
    ]) & ~interacting[None, :]
    gaze_values = gaze.astype(float)
    for rat in (0, 1):
        target = 1 - rat
        gaze_valid = valid[rat, [0, 3]].all(axis=0)
        gaze_valid &= valid[target, [0, 1, 2, 4]].all(axis=0)
        gaze_values[rat, ~gaze_valid] = np.nan
    interaction_values = interacting.astype(float)
    interaction_values[~valid.all(axis=(0, 1))] = np.nan
    return {
        "orientation_deg": angle,
        "speed_px_1s": speed,
        "is_gazing": gaze_values,
        "distance_px": distance,
        "interacting": interaction_values,
    }


def _trial_table(lever_path: Path, fps: float, nframes: int) -> pd.DataFrame:
    raw = pd.read_csv(lever_path)
    needed = {"TrialNum", "AbsTime", "TrialTime", "coopSucc", "Hit", "RatID"}
    if not needed.issubset(raw.columns):
        raise ValueError(f"Lever file missing {sorted(needed - set(raw.columns))}")
    for col in needed:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["TrialNum", "AbsTime", "TrialTime"])
    records = []
    for number, group in raw.groupby("TrialNum", sort=True):
        group = group.sort_values("AbsTime")
        onset = float((group["AbsTime"] - group["TrialTime"]).median())
        success = bool(group["coopSucc"].fillna(0).eq(1).any())
        hits = group.loc[group["Hit"].eq(1), "AbsTime"]
        coop_press = float(hits.iloc[1]) if success and len(hits) >= 2 else np.nan
        first_hit_rows = group.loc[group["Hit"].eq(1)]
        first_track = first_hit_rows["RatID"].iloc[0] if not first_hit_rows.empty else np.nan
        first_track = int(first_track) if pd.notna(first_track) and first_track in (0, 1) else np.nan
        first_press = float(group["AbsTime"].min())
        if not (np.isfinite(onset) and 0 <= onset * fps < nframes):
            continue
        records.append({"trial": int(number), "onset_s": onset, "first_press_s": first_press,
                        "coop_press_s": coop_press, "success": success,
                        "first_track": first_track})
    return pd.DataFrame.from_records(records)


def load_session(row: pd.Series, data_root: Path, local: dict[str, Path], fps: float = 30.0,
                 min_score: float = 0.2) -> Session:
    pair = str(row["color pair"]).split("-")
    if len(pair) != 2:
        raise ValueError(f"Invalid animal pair for {row['vid']}")
    h5_path, lever_path, video_path = resolve_files(row, data_root, local)
    with h5py.File(h5_path) as h5:
        tracks = h5["tracks"][:]
        if tracks.ndim != 4 or tracks.shape[:3] != (2, 2, 5):
            raise ValueError(f"Unexpected tracks shape {tracks.shape}")
        track_animal = _animal_tracks(h5, pair[0], pair[1])
        scores = h5["point_scores"][:] if "point_scores" in h5 else None
        features = _pose_features(tracks, fps, scores, min_score)
    trials = _trial_table(lever_path, fps, tracks.shape[-1])
    return Session(str(row["vid"]), str(row["session"]), pair[0], pair[1], track_animal,
                   h5_path, lever_path, video_path, fps, tracks, features, trials)


def frame_features(session: Session) -> dict[str, np.ndarray]:
    a, b = session.animal_track(session.animal_a), session.animal_track(session.animal_b)
    f = session.features
    return {
        "a_orientation_deg": f["orientation_deg"][a],
        "b_orientation_deg": f["orientation_deg"][b],
        "a_speed_px_1s": f["speed_px_1s"][a],
        "b_speed_px_1s": f["speed_px_1s"][b],
        "a_is_gazing": f["is_gazing"][a],
        "b_is_gazing": f["is_gazing"][b],
        "distance_px": f["distance_px"],
        "interacting": f["interacting"],
    }


E_CHANNELS = (
    "focal_orientation_deg", "partner_orientation_deg",
    "focal_speed_px_1s", "partner_speed_px_1s",
    "focal_is_gazing", "partner_is_gazing", "distance_px", "interacting",
)


def event_indices(session: Session, press_s: float, before_s: int = 5,
                  after_s: int = 5) -> np.ndarray | None:
    """Inclusive [-before,+after] frame indices around a cooperative press."""
    anchor = round(press_s * session.fps)
    frames = anchor + np.arange(-round(before_s * session.fps), round(after_s * session.fps) + 1)
    if frames[0] < 0 or frames[-1] >= session.tracks.shape[-1]:
        return None
    return frames


def trial_features(session: Session, focal: str,
                   min_valid_fraction: float = 0.8) -> pd.DataFrame:
    """One successful trial per row; one-second bins of its event-aligned traces."""
    focal_track = session.animal_track(focal)
    partner_track = 1 - focal_track
    partner = session.track_animal[partner_track]
    f = session.features
    source = {
        "focal_orientation_deg": f["orientation_deg"][focal_track],
        "partner_orientation_deg": f["orientation_deg"][partner_track],
        "focal_speed_px_1s": f["speed_px_1s"][focal_track],
        "partner_speed_px_1s": f["speed_px_1s"][partner_track],
        "focal_is_gazing": f["is_gazing"][focal_track],
        "partner_is_gazing": f["is_gazing"][partner_track],
        "distance_px": f["distance_px"],
        "interacting": f["interacting"],
    }
    records = []
    bin_width = round(session.fps)
    for trial in session.trials.itertuples(index=False):
        if not trial.success or not np.isfinite(trial.coop_press_s):
            continue
        frames = event_indices(session, trial.coop_press_s)
        if frames is None:
            continue
        record = {"vid": session.vid, "session_id": session.session_id,
                  "trial": trial.trial, "focal": focal, "partner": partner}
        complete = True
        for name, array in source.items():
            segment = array[frames[:-1]]  # 300 samples, bins [-5,-4), ... [4,5)
            for bin_num in range(10):
                values = segment[bin_num * bin_width:(bin_num + 1) * bin_width]
                if len(values) != bin_width or np.isfinite(values).mean() < min_valid_fraction:
                    complete = False
                    break
                record[f"{name}_t{bin_num - 5:+d}"] = float(np.nanmean(values))
            if not complete:
                break
        if complete:
            records.append(record)
    return pd.DataFrame.from_records(records)
