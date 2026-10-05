"""
simulateSingleSession.py — Interactive frame-by-frame session viewer for rat cooperation experiments.

Usage:
    python simulateSingleSession.py --video VIDEO_PATH --h5 H5_PATH --lev LEV_CSV --mag MAG_CSV [options]

Controls:
    Right Arrow / D  — Next frame
    Left Arrow / A   — Previous frame
    Space            — Play/pause at video framerate
    S                — Save video from current start_frame to end_frame
    Q / Escape       — Quit

Overlays:
    - Gaze vectors for both rats (color-coded by state)
    - Interaction region circles around each rat's body parts (red/blue, purple overlap)
    - Arena zone boundaries (white lines) with labels
    - Body polygon of each rat
    - Per-frame status text: gazing, interacting, region, distance, velocity, state

Session statistics are printed to the terminal on startup and optionally saved to CSV.
"""

import argparse
import sys
import csv
import time
import subprocess
import shutil
from pathlib import Path

import cv2
import numpy as np

from pos_class import posLoader
from lev_class import levLoader
from mag_class import magLoader


# ──────────────────────────────────────────────────────────────
# Helpers: compute gaze/interaction event periods
# ──────────────────────────────────────────────────────────────

def extract_event_periods(bool_array):
    """
    Given a boolean array, return a list of (start_frame, end_frame, length)
    tuples for each contiguous True run.
    """
    periods = []
    in_event = False
    start = 0
    for i, val in enumerate(bool_array):
        if val and not in_event:
            start = i
            in_event = True
        elif not val and in_event:
            periods.append((start, i - 1, i - start))
            in_event = False
    if in_event:
        periods.append((start, len(bool_array) - 1, len(bool_array) - start))
    return periods


def compute_session_stats(loader, lev, mag):
    """
    Compute and return a dict of session-level statistics.
    """
    num_frames = loader.returnNumFrames()
    raw_data_shape = loader.data.shape
    total_values = np.prod(raw_data_shape)
    # Reload raw to count NaNs (the loader already interpolated)
    # Use the percentage from how many frames have any NaN body part
    # Approximate: count NaN percentage from the stored (interpolated) data won't work,
    # so we report what the loader provides or recompute from file
    import h5py
    with h5py.File(loader.filename, "r") as f:
        raw = f['tracks'][:]
        nan_count = np.isnan(raw).sum()
        nan_pct = 100.0 * nan_count / np.prod(raw.shape)

    stats = {}
    stats['num_frames'] = num_frames
    stats['nan_percentage'] = round(nan_pct, 2)

    # Gazing stats (both rats)
    gaze0 = loader.returnIsGazing(0)
    gaze1 = loader.returnIsGazing(1)
    total_gazing_frames = np.sum(gaze0) + np.sum(gaze1)
    stats['social_gazing_pct'] = round(100.0 * total_gazing_frames / (2 * num_frames), 2)

    # Per-rat gaze events for average length
    events0 = extract_event_periods(gaze0)
    events1 = extract_event_periods(gaze1)
    all_gaze_lengths = [e[2] for e in events0] + [e[2] for e in events1]
    stats['avg_social_gaze_length'] = round(np.mean(all_gaze_lengths), 2) if all_gaze_lengths else 0
    stats['num_social_gaze_events'] = len(all_gaze_lengths)

    # Lever gazing — returnIsLookingAtObjects already returns a bool array of frames,
    # so total / (2 * num_frames) is already the fraction; just multiply by 100 once.
    lev_gaze0 = loader.returnIsLookingAtObjects(0, target="lever")
    lev_gaze1 = loader.returnIsLookingAtObjects(1, target="lever")
    stats['lever_gazing_pct'] = round((np.sum(lev_gaze0) + np.sum(lev_gaze1)) / (2 * num_frames) * 100, 2)
    print("(np.sum(lev_gaze0): ", (np.sum(lev_gaze0)))
    print("(np.sum(lev_gaze1): ", (np.sum(lev_gaze1)))
    # Magazine gazing
    mag_gaze0 = loader.returnIsLookingAtObjects(0, target="mag")
    mag_gaze1 = loader.returnIsLookingAtObjects(1, target="mag")
    stats['mag_gazing_pct'] = round((np.sum(mag_gaze0) + np.sum(mag_gaze1)) / (2 * num_frames) * 100, 2)

    # Interaction
    is_interacting = np.array(loader.returnIsInteracting(), dtype=bool)
    stats['interacting_pct'] = round(100.0 * np.sum(is_interacting) / num_frames, 2)

    # Interaction events
    interaction_events = extract_event_periods(is_interacting)
    interaction_lengths = [e[2] for e in interaction_events]
    stats['num_interaction_events'] = len(interaction_lengths)
    stats['avg_interaction_event_length'] = round(np.mean(interaction_lengths), 2) if interaction_lengths else 0

    # Proximity (average inter-mouse distance)
    distances = loader.returnInterMouseDistance()
    stats['avg_proximity'] = round(np.mean(distances), 2)

    # Average distance moved per frame (headbase velocity)
    vel0 = loader.computeVelocity(0)
    vel1 = loader.computeVelocity(1)
    stats['avg_distance_per_frame_rat0'] = round(np.mean(vel0), 2)
    stats['avg_distance_per_frame_rat1'] = round(np.mean(vel1), 2)

    # Trial stats
    stats['total_trials'] = int(lev.returnNumTotalTrials())
    stats['successful_trials'] = int(lev.returnNumSuccessfulTrials())
    stats['success_pct'] = round(100.0 * lev.returnSuccessPercentage(), 2)

    return stats, gaze0, gaze1, is_interacting


def save_events_csv(filepath, gaze0, gaze1, is_interacting):
    """
    Save gazing and interaction periods to a CSV file.
    Columns: event_type, rat_id, start_frame, end_frame, length
    """
    rows = []
    for start, end, length in extract_event_periods(gaze0):
        rows.append(('social_gaze', 0, start, end, length))
    for start, end, length in extract_event_periods(gaze1):
        rows.append(('social_gaze', 1, start, end, length))
    for start, end, length in extract_event_periods(is_interacting):
        rows.append(('interaction', -1, start, end, length))

    # Sort by start frame
    rows.sort(key=lambda r: r[2])

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['event_type', 'rat_id', 'start_frame', 'end_frame', 'length'])
        writer.writerows(rows)
    print(f"Events CSV saved to {filepath}")


# ──────────────────────────────────────────────────────────────
# Frame annotation
# ──────────────────────────────────────────────────────────────

def annotate_frame(frame, frame_idx, loader, lev, mag, precomputed, width, height):
    """
    Draw all overlays on a single frame. Returns the annotated frame.

    precomputed is a dict with pre-calculated arrays to avoid recomputing per frame.
    """
    pc = precomputed
    INTERACTION_RADIUS = 90

    # --- Interaction region circles ---
    rat0_parts = loader.data[0, :, :, frame_idx].T  # shape (5, 2)
    rat1_parts = loader.data[1, :, :, frame_idx].T

    mask0 = np.zeros_like(frame, dtype=np.uint8)
    mask1 = np.zeros_like(frame, dtype=np.uint8)

    for pt in rat0_parts:
        x, y = float(pt[0]), float(pt[1])
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(mask0, (int(round(x)), int(round(y))), INTERACTION_RADIUS, (0, 0, 255), -1)
    for pt in rat1_parts:
        x, y = float(pt[0]), float(pt[1])
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(mask1, (int(round(x)), int(round(y))), INTERACTION_RADIUS, (255, 0, 0), -1)

    overlap = cv2.bitwise_and(mask0, mask1)
    only0 = cv2.subtract(mask0, overlap)
    only1 = cv2.subtract(mask1, overlap)
    # Make overlap purple
    purple_overlap = np.zeros_like(frame, dtype=np.uint8)
    overlap_gray = cv2.cvtColor(overlap, cv2.COLOR_BGR2GRAY)
    purple_overlap[overlap_gray > 0] = (180, 0, 180)

    overlay = cv2.add(cv2.add(only0, only1), purple_overlap)
    frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

    # --- Arena zone boundaries (white) ---
    white = (255, 255, 255)
    cv2.line(frame, (loader.levBoundary, 0), (loader.levBoundary, height), white, 1)
    cv2.line(frame, (loader.magBoundary, 0), (loader.magBoundary, height), white, 1)
    cv2.line(frame, (0, loader.topWall), (width, loader.topWall), white, 1)
    cv2.line(frame, (0, loader.bottomWall), (width, loader.bottomWall), white, 1)

    # Zone labels
    cv2.putText(frame, "Lev", (loader.levBoundary // 2 - 20, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)
    cv2.putText(frame, "Mid", ((loader.levBoundary + loader.magBoundary) // 2 - 20, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)
    cv2.putText(frame, "Mag", (loader.magBoundary + (width - loader.magBoundary) // 2 - 20, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)

    # Sub-zone rectangles
    zone_color = (200, 200, 200)
    zones = [
        ("levTop", loader.levTopTR, loader.levTopBL),
        ("levBot", loader.levBotTR, loader.levBotBL),
        ("magTop", loader.magTopTR, loader.magTopBL),
        ("magBot", loader.magBotTR, loader.magBotBL),
    ]
    for label, tr, bl in zones:
        top_left = (bl[0], tr[1])
        bottom_right = (tr[0], bl[1])
        cv2.rectangle(frame, top_left, bottom_right, zone_color, 1)

    # Lever and magazine object rectangles
    lev_rects = [((100, 180), (160, 240)), ((100, 415), (160, 475))]
    mag_rects = [((1240, 185), (1300, 245)), ((1240, 415), (1300, 475))]
    for (p1, p2) in lev_rects:
        cv2.rectangle(frame, p1, p2, (100, 100, 255), 2)
    for (p1, p2) in mag_rects:
        cv2.rectangle(frame, p1, p2, (255, 100, 100), 2)

    # --- Gaze vectors for both rats ---
    for rat_id in (0, 1):
        gaze_vec = pc['gaze_vectors'][rat_id][:, frame_idx]
        gaze_origin = loader.data[rat_id, :, loader.HB_INDEX, frame_idx]
        target_body = loader.data[1 - rat_id, :, :, frame_idx]

        gaze_dir = gaze_vec / (np.linalg.norm(gaze_vec) + 1e-8)
        p1 = gaze_origin + 150 * gaze_dir
        p2 = gaze_origin + loader.vectorLength * gaze_dir
        p1_int = tuple(np.round(p1).astype(int))
        p2_int = tuple(np.round(p2).astype(int))

        is_gazing_now = bool(pc['gaze'][rat_id][frame_idx])
        is_intersecting = loader._gaze_intersects_body(gaze_origin, gaze_vec, target_body)

        if is_gazing_now:
            color = (0, 0, 255) if rat_id == 0 else (0, 140, 255)  # red / orange
        elif is_intersecting:
            color = (0, 255, 255)  # yellow — intersecting but not gazing (< 10 frames)
        else:
            color = (0, 255, 0) if rat_id == 0 else (255, 165, 0)  # green / light blue

        cv2.line(frame, p1_int, p2_int, color, 2)

    # --- Body polygons for both rats ---
    polygon_indices = [loader.earL_INDEX, loader.NOSE_INDEX, loader.earR_INDEX, loader.TB_INDEX]
    for rat_id in (0, 1):
        body = loader.data[rat_id, :, :, frame_idx]
        pts = [tuple(np.round(body[:, idx]).astype(int)) for idx in polygon_indices]
        pts.append(pts[0])
        poly_color = (128, 128, 255) if rat_id == 0 else (255, 128, 128)
        for j in range(len(pts) - 1):
            cv2.line(frame, pts[j], pts[j + 1], poly_color, 1)

    # --- Text overlays ---
    y_offset = 22
    line = 0

    def put(text, color=(255, 255, 255)):
        nonlocal line
        cv2.putText(frame, text, (10, y_offset + line * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        line += 1

    # Frame info
    put(f"Frame: {frame_idx}")

    # Social gazing status + running count
    g0 = bool(pc['gaze'][0][frame_idx])
    g1 = bool(pc['gaze'][1][frame_idx])
    g0_count = int(np.sum(pc['gaze'][0][:frame_idx + 1]))
    g1_count = int(np.sum(pc['gaze'][1][:frame_idx + 1]))
    put(f"Rat0 Social Gaze: {g0}  (total: {g0_count})", (0, 0, 255) if g0 else (150, 150, 150))
    put(f"Rat1 Social Gaze: {g1}  (total: {g1_count})", (0, 140, 255) if g1 else (150, 150, 150))

    # Lever gazing status + running count
    lg0 = bool(pc['lev_gaze'][0][frame_idx])
    lg1 = bool(pc['lev_gaze'][1][frame_idx])
    lg0_count = int(np.sum(pc['lev_gaze'][0][:frame_idx + 1]))
    lg1_count = int(np.sum(pc['lev_gaze'][1][:frame_idx + 1]))
    put(f"Rat0 Lever Gaze: {lg0}  (total: {lg0_count})", (100, 100, 255) if lg0 else (150, 150, 150))
    put(f"Rat1 Lever Gaze: {lg1}  (total: {lg1_count})", (100, 100, 255) if lg1 else (150, 150, 150))

    # Magazine gazing status + running count
    mg0 = bool(pc['mag_gaze'][0][frame_idx])
    mg1 = bool(pc['mag_gaze'][1][frame_idx])
    mg0_count = int(np.sum(pc['mag_gaze'][0][:frame_idx + 1]))
    mg1_count = int(np.sum(pc['mag_gaze'][1][:frame_idx + 1]))
    put(f"Rat0 Mag Gaze: {mg0}  (total: {mg0_count})", (255, 100, 100) if mg0 else (150, 150, 150))
    put(f"Rat1 Mag Gaze: {mg1}  (total: {mg1_count})", (255, 100, 100) if mg1 else (150, 150, 150))

    # Interacting + running count
    interacting = bool(pc['interacting'][frame_idx])
    inter_count = int(np.sum(pc['interacting'][:frame_idx + 1]))
    put(f"Interacting: {interacting}  (total: {inter_count})", (255, 0, 255) if interacting else (150, 150, 150))

    # Regions
    put(f"Rat0: {pc['regions'][0][frame_idx]}")
    put(f"Rat1: {pc['regions'][1][frame_idx]}")

    # Distance
    dist = pc['inter_mouse_dist'][frame_idx]
    put(f"Distance: {dist:.0f}px")

    # Velocities
    v0 = pc['velocities'][0][frame_idx]
    v1 = pc['velocities'][1][frame_idx]
    put(f"Vel Rat0: {v0:.1f} px/f")
    put(f"Vel Rat1: {v1:.1f} px/f")

    # Trial info — find which trial this frame is in
    trial_text = ""
    for i, (ts, te) in enumerate(zip(pc['trial_starts_frames'], pc['trial_ends_frames'])):
        if ts <= frame_idx <= te:
            succ = pc['trial_success'][i] if i < len(pc['trial_success']) else '?'
            trial_text = f"Trial {i+1} ({'SUCCESS' if succ == 1 else 'FAIL' if succ == 0 else '?'})"
            break
    if trial_text:
        put(trial_text, (0, 255, 0) if 'SUCCESS' in trial_text else (100, 100, 255))

    return frame


# ──────────────────────────────────────────────────────────────
# Precompute expensive arrays once
# ──────────────────────────────────────────────────────────────

def precompute(loader, lev, mag):
    """Precompute all per-frame arrays needed for annotation."""
    print("Precomputing session data...")
    num_frames = loader.returnNumFrames()

    pc = {}
    pc['gaze_vectors'] = {
        0: loader.returnGazeVector(0),
        1: loader.returnGazeVector(1),
    }

    print("  Computing gazing...")
    pc['gaze'] = {
        0: loader.returnIsGazing(0),
        1: loader.returnIsGazing(1),
    }

    print("  Computing lever gazing...")
    pc['lev_gaze'] = {
        0: loader.returnIsLookingAtObjects(0, target="lever"),
        1: loader.returnIsLookingAtObjects(1, target="lever"),
    }

    print("  Computing mag gazing...")
    pc['mag_gaze'] = {
        0: loader.returnIsLookingAtObjects(0, target="mag"),
        1: loader.returnIsLookingAtObjects(1, target="mag"),
    }

    print("  Computing interactions...")
    pc['interacting'] = np.array(loader.returnIsInteracting(), dtype=bool)

    pc['regions'] = {
        0: loader.returnMouseLocation(0),
        1: loader.returnMouseLocation(1),
    }

    pc['inter_mouse_dist'] = loader.returnInterMouseDistance()

    pc['velocities'] = {
        0: loader.computeVelocity(0),
        1: loader.computeVelocity(1),
    }

    # Trial timing
    fps = lev.fps
    trial_starts = lev.returnTimeStartTrials()
    trial_ends = lev.returnTimeEndTrials()
    pc['trial_starts_frames'] = [int(t * fps) for t in trial_starts]
    pc['trial_ends_frames'] = [int(t * fps) for t in trial_ends]
    pc['trial_success'] = lev.returnSuccessTrials()

    print("  Precomputation done.")
    return pc


# ──────────────────────────────────────────────────────────────
# Interactive viewer
# ──────────────────────────────────────────────────────────────

def run_viewer(video_path, loader, lev, mag, start_frame=0):
    """
    Interactive OpenCV window.
    Controls:
        Right / D      — next frame
        Left / A       — previous frame
        Space          — play/pause
        Up / W         — speed up (2x, 4x, 8x)
        Down / S       — slow down (0.5x, 0.25x)
        R              — reset to 1x speed
        Trackbar       — drag to scrub to any frame
        Q / Esc        — quit
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frame = min(total_video_frames, loader.returnNumFrames()) - 1

    print(f"Video: {width}x{height}, {fps:.1f} fps, {total_video_frames} frames")
    print(f"Position data: {loader.returnNumFrames()} frames")
    print(f"Usable frames: 0 to {max_frame}")
    print()
    print("Controls:")
    print("  Right/D = next frame    Left/A = prev frame")
    print("  Space   = play/pause    Q/Esc  = quit")
    print("  Up/W    = speed up      Down/S = slow down    R = reset 1x")
    print("  Trackbar = scrub to any frame")

    pc = precompute(loader, lev, mag)

    # Compute and print session stats
    stats, gaze0, gaze1, is_interacting = compute_session_stats(loader, lev, mag)
    print("\n" + "=" * 50)
    print("SESSION STATISTICS")
    print("=" * 50)
    for k, v in stats.items():
        label = k.replace('_', ' ').title()
        print(f"  {label}: {v}")
    print("=" * 50 + "\n")

    frame_idx = min(start_frame, max_frame)
    playing = False
    speed_mult = 1.0
    trackbar_updating = False  # flag to avoid feedback loop

    window_name = "Session Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def read_frame(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        if not ret:
            return None
        return annotate_frame(f, idx, loader, lev, mag, pc, width, height)

    # --- Trackbar callback ---
    def on_trackbar(pos):
        nonlocal frame_idx, trackbar_updating
        if trackbar_updating:
            return
        frame_idx = min(pos, max_frame)
        annotated = read_frame(frame_idx)
        if annotated is not None:
            cv2.imshow(window_name, annotated)

    cv2.createTrackbar("Frame", window_name, frame_idx, max_frame, on_trackbar)

    annotated = read_frame(frame_idx)
    if annotated is not None:
        cv2.imshow(window_name, annotated)

    while True:
        if playing:
            effective_fps = fps * speed_mult
            wait_ms = max(1, int(1000 / effective_fps))
        else:
            wait_ms = 50  # poll at 20Hz so trackbar stays responsive

        key = cv2.waitKey(wait_ms) & 0xFF

        # Check if window was closed
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        moved = False

        if key == ord('q') or key == 27:  # Q or Escape
            break
        elif key == ord(' '):  # Space — toggle play
            playing = not playing
            if playing:
                print(f"  Playing at {speed_mult}x speed")
            continue
        elif key == ord('w') or key == 82 or key == 0:  # Up arrow or W — speed up
            speed_mult = min(speed_mult * 2, 128.0)
            print(f"  Speed: {speed_mult}x")
        elif key == ord('s') or key == 84 or key == 1:  # Down arrow or S — slow down
            speed_mult = max(speed_mult / 2, 0.125)
            print(f"  Speed: {speed_mult}x")
        elif key == ord('r'):  # R — reset speed
            speed_mult = 1.0
            print(f"  Speed: {speed_mult}x")
        elif key == ord('d') or key == 83 or key == 3:  # Right arrow or D
            if frame_idx < max_frame:
                frame_idx += 1
                moved = True
            playing = False
        elif key == ord('a') or key == 81 or key == 2:  # Left arrow or A
            if frame_idx > 0:
                frame_idx -= 1
                moved = True
            playing = False
        elif playing:
            if frame_idx < max_frame:
                frame_idx += 1
                moved = True
            else:
                playing = False

        if moved:
            annotated = read_frame(frame_idx)
            if annotated is not None:
                cv2.imshow(window_name, annotated)
            # Sync trackbar position
            trackbar_updating = True
            cv2.setTrackbarPos("Frame", window_name, frame_idx)
            trackbar_updating = False

    cap.release()
    cv2.destroyAllWindows()

    return stats, gaze0, gaze1, is_interacting, pc


# ──────────────────────────────────────────────────────────────
# Video export
# ──────────────────────────────────────────────────────────────

def save_video(video_path, loader, lev, mag, pc, start_frame, end_frame, save_path):
    """
    Render annotated frames from start_frame to end_frame and save as mp4.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_dir = Path("temp_session_frames")
    temp_dir.mkdir(exist_ok=True)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    count = 0
    for idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret or idx >= loader.returnNumFrames():
            break
        annotated = annotate_frame(frame, idx, loader, lev, mag, pc, width, height)
        cv2.imwrite(str(temp_dir / f"frame_{count:06d}.png"), annotated)
        if count % 100 == 0:
            print(f"  Rendered frame {count} / {end_frame - start_frame + 1}")
        count += 1

    cap.release()

    print(f"Encoding video with ffmpeg...")
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(int(fps)),
        "-i", str(temp_dir / "frame_%06d.png"),
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        str(save_path)
    ]
    result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("FFmpeg error:", result.stderr.decode())
        raise RuntimeError("FFmpeg failed.")

    shutil.rmtree(temp_dir)
    print(f"Video saved to {save_path} ({count} frames)")


# ══════════════════════════════════════════════════════════════
# SETTINGS — Edit these directly, then just run: python simulateSingleSession.py
# ══════════════════════════════════════════════════════════════

DATA_DIR = "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/DataStorage/SingleRatExampleDatafiles"
SESSION_NAME = "041624_Cam4_TrNum10_Coop_KL001B-KL001Y"

VIDEO_PATH = f"{DATA_DIR}/{SESSION_NAME}.mp4"
H5_PATH    = f"{DATA_DIR}/{SESSION_NAME}.predictions.h5"
LEV_PATH   = f"{DATA_DIR}/{SESSION_NAME}_lever.csv"
MAG_PATH   = f"{DATA_DIR}/{SESSION_NAME}_mag.csv"

FPS = 30
START_FRAME = 0

# Video export (set SAVE_VIDEO_PATH to None for interactive mode)
SAVE_VIDEO_PATH = None          # e.g. "output.mp4"
END_FRAME = None                # e.g. 2000 (required if SAVE_VIDEO_PATH is set)

# Events CSV export (set to None to skip)
SAVE_EVENTS_CSV = None          # e.g. "events.csv"


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    # Check if CLI args were provided — if so, use them; otherwise use settings above
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            description="Interactive session viewer for rat cooperation experiments."
        )
        parser.add_argument("--video", required=True, help="Path to the video file (.mp4)")
        parser.add_argument("--h5", required=True, help="Path to the H5 position file")
        parser.add_argument("--lev", required=True, help="Path to the lever CSV file")
        parser.add_argument("--mag", required=True, help="Path to the magazine CSV file")
        parser.add_argument("--start", type=int, default=0, help="Starting frame index (default: 0)")
        parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
        parser.add_argument("--save-video", type=str, default=None,
                            help="Save annotated video to this path instead of interactive mode")
        parser.add_argument("--end", type=int, default=None,
                            help="End frame for video export (required with --save-video)")
        parser.add_argument("--save-events-csv", type=str, default=None,
                            help="Save gazing/interaction periods to this CSV path")
        args = parser.parse_args()
        video_path = args.video
        h5_path = args.h5
        lev_path = args.lev
        mag_path = args.mag
        fps = args.fps
        start_frame = args.start
        save_video_path = args.save_video
        end_frame_arg = args.end
        save_events_csv = args.save_events_csv
    else:
        # Use the hardcoded settings above
        video_path = VIDEO_PATH
        h5_path = H5_PATH
        lev_path = LEV_PATH
        mag_path = MAG_PATH
        fps = FPS
        start_frame = START_FRAME
        save_video_path = SAVE_VIDEO_PATH
        end_frame_arg = END_FRAME
        save_events_csv = SAVE_EVENTS_CSV

    # Load data
    print("Loading position data...")
    loader = posLoader(h5_path)
    print("Loading lever data...")
    lev_data = levLoader(lev_path, endFrame=loader.returnNumFrames(), fps=fps)
    print("Loading magazine data...")
    mag_data = magLoader(mag_path, fps=fps)

    if save_video_path:
        # Non-interactive: just render and save
        end_frame = end_frame_arg if end_frame_arg is not None else loader.returnNumFrames() - 1
        pc = precompute(loader, lev_data, mag_data)

        # Print stats
        stats, gaze0, gaze1, is_interacting = compute_session_stats(loader, lev_data, mag_data)
        print("\n" + "=" * 50)
        print("SESSION STATISTICS")
        print("=" * 50)
        for k, v in stats.items():
            label = k.replace('_', ' ').title()
            print(f"  {label}: {v}")
        print("=" * 50 + "\n")

        save_video(video_path, loader, lev_data, mag_data, pc,
                   start_frame, end_frame, save_video_path)

        if save_events_csv:
            save_events_csv(save_events_csv, gaze0, gaze1, is_interacting)
    else:
        # Interactive viewer
        stats, gaze0, gaze1, is_interacting, pc = run_viewer(
            video_path, loader, lev_data, mag_data, start_frame=start_frame
        )

        if save_events_csv:
            save_events_csv(save_events_csv, gaze0, gaze1, is_interacting)


if __name__ == "__main__":
    main()
