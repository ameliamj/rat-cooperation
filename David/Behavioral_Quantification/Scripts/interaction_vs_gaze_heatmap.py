#!/usr/bin/env python3
"""
Interaction vs Gazing heatmaps for coop vs non-coop sessions.

Single-session plots: 2 traces (gazing, interaction).
Combined plot: 4 traces — coop pair (warm) vs non-coop pair (cool).

Each figure has a 2D flat version and a 3D perspective version.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from scipy.ndimage import gaussian_filter
from pos_class import posLoader

# ---------- Config ----------
COOP_H5 = "/Users/david/Downloads/041024_COOPTRAIN_LARGEARENA_EB009B-EB019Y_Camera2.predictions.h5"
NONCOOP_H5 = "/Users/david/Downloads/041624_Cam4_TrNum6_Comp_KL001B-KL001Y.predictions.h5"

BIN_SIZE = 8
SIGMA = 3.0
WIDTH, HEIGHT = 1392, 640
LEV_BOUNDARY = 351
MAG_BOUNDARY = 1041

# Single-session colors
COLOR_GAZING = "#8E44AD"       # purple
COLOR_INTERACTION = "#E74C3C"  # red

# Combined-plot colors (warm = coop, cool = non-coop)
COLOR_COOP_GAZING = "#FF6F00"          # bold orange
COLOR_COOP_INTERACTION = "#D50000"     # bold red
COLOR_NONCOOP_GAZING = "#00BFA5"       # teal
COLOR_NONCOOP_INTERACTION = "#304FFE"  # bold indigo


# ============================================================
#  Shared helpers
# ============================================================

def _make_cmap(hex_color):
    """Transparent-to-color colormap."""
    rgba = to_rgba(hex_color)
    return LinearSegmentedColormap.from_list(
        "custom",
        [(rgba[0], rgba[1], rgba[2], 0.0),
         (rgba[0], rgba[1], rgba[2], 1.0)],
        N=256,
    )


def build_session_heatmaps(pos):
    """Return (hm_gazing, hm_interaction) for one session, combining both rats."""
    hm_h, hm_w = HEIGHT // BIN_SIZE, WIDTH // BIN_SIZE
    num_frames = pos.returnNumFrames()

    isGazing0 = pos.returnIsGazing(0)
    isGazing1 = pos.returnIsGazing(1)
    isInteracting = np.array(pos.returnIsInteracting(), dtype=bool)

    hm_gaze = np.zeros((hm_h, hm_w))
    hm_interact = np.zeros((hm_h, hm_w))

    for t in range(num_frames):
        for rat_id in [0, 1]:
            x, y = pos.returnRatHBPosition(rat_id, t)
            if np.isnan(x) or np.isnan(y):
                continue
            xb = int(min(max(x // BIN_SIZE, 0), hm_w - 1))
            yb = int(min(max(y // BIN_SIZE, 0), hm_h - 1))

            gazing = isGazing0[t] if rat_id == 0 else isGazing1[t]
            if gazing:
                hm_gaze[yb, xb] += 1
            if isInteracting[t]:
                hm_interact[yb, xb] += 1

    hm_gaze = gaussian_filter(hm_gaze, sigma=SIGMA)
    hm_interact = gaussian_filter(hm_interact, sigma=SIGMA)
    if hm_gaze.max() > 0:
        hm_gaze /= hm_gaze.max()
    if hm_interact.max() > 0:
        hm_interact /= hm_interact.max()

    return hm_gaze, hm_interact


# ============================================================
#  2D flat arena
# ============================================================

def _draw_arena_2d(ax):
    """Draw a clean 2D arena outline with zone dividers and labels."""
    arena = mpatches.FancyBboxPatch(
        (0, 0), WIDTH, HEIGHT,
        boxstyle=mpatches.BoxStyle.Round(pad=0, rounding_size=20),
        linewidth=2.5, edgecolor='#2C3E50', facecolor='#F7F9FA', zorder=0,
    )
    ax.add_patch(arena)

    for xb in [LEV_BOUNDARY, MAG_BOUNDARY]:
        ax.axvline(xb, color='#7F8C8D', linewidth=1.2, linestyle='--', zorder=1)

    label_y = HEIGHT + 28
    ax.text(LEV_BOUNDARY / 2, label_y, 'lever', ha='center', va='center',
            fontsize=13, fontstyle='italic', color='#2C3E50')
    ax.text((LEV_BOUNDARY + MAG_BOUNDARY) / 2, label_y, 'center', ha='center',
            va='center', fontsize=13, fontstyle='italic', color='#2C3E50')
    ax.text((MAG_BOUNDARY + WIDTH) / 2, label_y, 'mag', ha='center', va='center',
            fontsize=13, fontstyle='italic', color='#2C3E50')

    ax.set_xlim(-30, WIDTH + 30)
    ax.set_ylim(HEIGHT + 50, -30)
    ax.set_aspect('equal')
    ax.axis('off')


def _overlay_heatmaps_2d(ax, traces, alpha=0.7):
    """Overlay list of (heatmap, color, label, vmax) onto a 2D arena axis."""
    extent = [0, WIDTH, HEIGHT, 0]
    _draw_arena_2d(ax)
    for hm, color, _label, vmax in traces:
        cmap = _make_cmap(color)
        hm_plot = (hm / vmax if vmax > 0 else hm).copy()
        hm_plot[hm_plot < 0.08] = 0
        ax.imshow(hm_plot, cmap=cmap, origin='upper', extent=extent,
                  aspect='auto', zorder=2, alpha=alpha, vmin=0, vmax=1,
                  interpolation='bilinear')


# ============================================================
#  3D perspective arena  (cv2.warpPerspective approach)
# ============================================================

def _perspective_transform(x, y, canvas_w, canvas_h):
    """Map arena coords to perspective canvas coords."""
    u = x / WIDTH
    v = y / HEIGHT

    shrink_far = 0.55
    y_near = canvas_h * 0.85
    y_far = canvas_h * 0.30

    w_near = canvas_w * 0.92
    w_far = canvas_w * shrink_far

    w = w_near + (w_far - w_near) * v
    cx = canvas_w / 2

    px = cx + (u - 0.5) * w
    py = y_near + (y_far - y_near) * v

    return px, py


def _draw_arena_3d(ax, canvas_w, canvas_h):
    """Draw a 3D box arena — walls, floor outline, zone labels, notches."""
    fl_bl = _perspective_transform(0, HEIGHT, canvas_w, canvas_h)
    fl_br = _perspective_transform(WIDTH, HEIGHT, canvas_w, canvas_h)
    fl_tl = _perspective_transform(0, 0, canvas_w, canvas_h)
    fl_tr = _perspective_transform(WIDTH, 0, canvas_w, canvas_h)

    wall_h = canvas_h * 0.28

    bw_tl = (fl_tl[0], fl_tl[1] - wall_h)
    bw_tr = (fl_tr[0], fl_tr[1] - wall_h)

    sw_front_h = canvas_h * 0.15

    lw_tl_pt = (fl_bl[0], fl_bl[1] - sw_front_h)
    rw_tl_pt = (fl_br[0], fl_br[1] - sw_front_h)

    wall_color = '#E8ECF0'
    edge_color = '#2C3E50'
    lw = 2.0

    # Back wall
    ax.add_patch(plt.Polygon([fl_tl, fl_tr, bw_tr, bw_tl], closed=True,
                              facecolor=wall_color, edgecolor=edge_color,
                              linewidth=lw, zorder=0))
    # Left wall
    ax.add_patch(plt.Polygon([fl_bl, fl_tl, bw_tl, lw_tl_pt], closed=True,
                              facecolor=wall_color, edgecolor=edge_color,
                              linewidth=lw, zorder=0))
    # Right wall
    ax.add_patch(plt.Polygon([fl_br, fl_tr, bw_tr, rw_tl_pt], closed=True,
                              facecolor=wall_color, edgecolor=edge_color,
                              linewidth=lw, zorder=0))

    # Floor — light fill behind heatmaps
    ax.add_patch(plt.Polygon([fl_bl, fl_br, fl_tr, fl_tl], closed=True,
                              facecolor='#F0F2F4', edgecolor='none', zorder=1))

    # Floor zone dividers
    for boundary in [LEV_BOUNDARY, MAG_BOUNDARY]:
        top = _perspective_transform(boundary, 0, canvas_w, canvas_h)
        bot = _perspective_transform(boundary, HEIGHT, canvas_w, canvas_h)
        ax.plot([bot[0], top[0]], [bot[1], top[1]],
                color='#B0B8C0', linewidth=1, linestyle='--', zorder=2)

    # Depth stripes
    for frac in np.linspace(0.1, 0.9, 8):
        left = _perspective_transform(0, frac * HEIGHT, canvas_w, canvas_h)
        right = _perspective_transform(WIDTH, frac * HEIGHT, canvas_w, canvas_h)
        ax.plot([left[0], right[0]], [left[1], right[1]],
                color='#DDE1E5', linewidth=0.4, zorder=2)

    # Floor outline on top of everything
    ax.add_patch(plt.Polygon([fl_bl, fl_br, fl_tr, fl_tl], closed=True,
                              facecolor='none', edgecolor=edge_color,
                              linewidth=lw, zorder=8))

    # Zone labels
    label_y = fl_bl[1] + 22
    lev_x = (fl_bl[0] + _perspective_transform(LEV_BOUNDARY, HEIGHT, canvas_w, canvas_h)[0]) / 2
    ctr_x = (_perspective_transform(LEV_BOUNDARY, HEIGHT, canvas_w, canvas_h)[0] +
             _perspective_transform(MAG_BOUNDARY, HEIGHT, canvas_w, canvas_h)[0]) / 2
    mag_x = (_perspective_transform(MAG_BOUNDARY, HEIGHT, canvas_w, canvas_h)[0] + fl_br[0]) / 2

    for x, txt in [(lev_x, 'lever'), (ctr_x, 'center'), (mag_x, 'mag')]:
        ax.text(x, label_y, txt, ha='center', va='top',
                fontsize=12, fontstyle='italic', color='#2C3E50')

    # Lever/mag notches on side walls
    notch_w, notch_h = 12, 8
    for base_pt, top_pt, x_off in [(fl_bl, lw_tl_pt, -2), (fl_br, rw_tl_pt, -notch_w + 2)]:
        mid_y = (base_pt[1] + top_pt[1]) / 2
        for dy in [-20, 20]:
            ax.add_patch(mpatches.FancyBboxPatch(
                (base_pt[0] + x_off, mid_y + dy - notch_h / 2), notch_w, notch_h,
                boxstyle=mpatches.BoxStyle.Round(pad=0, rounding_size=2),
                facecolor='#5D6D7E', edgecolor='#2C3E50', linewidth=1, zorder=9,
            ))

    ax.set_xlim(0, canvas_w)
    ax.set_ylim(canvas_h + 10, -canvas_h * 0.15)
    ax.set_aspect('equal')
    ax.axis('off')


def _heatmap_to_rgba(hm, color, vmax):
    """Convert a 2D heatmap to a float32 RGBA image with proper transparency."""
    rgba = to_rgba(color)
    hm_norm = np.clip((hm / vmax) if vmax > 0 else hm, 0, 1)
    h, w = hm.shape
    img = np.zeros((h, w, 4), dtype=np.float32)
    img[:, :, 0] = rgba[0]
    img[:, :, 1] = rgba[1]
    img[:, :, 2] = rgba[2]
    img[:, :, 3] = hm_norm * 0.85
    img[hm_norm < 0.08, 3] = 0  # kill noise
    return img


def _overlay_heatmaps_3d(ax, traces, canvas_w, canvas_h):
    """Warp heatmaps onto the 3D floor using cv2.warpPerspective."""
    hm_h, hm_w = HEIGHT // BIN_SIZE, WIDTH // BIN_SIZE

    # Source: corners of the heatmap image (TL, TR, BR, BL)
    src = np.array([
        [0, 0], [hm_w, 0], [hm_w, hm_h], [0, hm_h]
    ], dtype=np.float32)

    # Destination: perspective-transformed floor corners on canvas
    dst = np.array([
        list(_perspective_transform(0, 0, canvas_w, canvas_h)),
        list(_perspective_transform(WIDTH, 0, canvas_w, canvas_h)),
        list(_perspective_transform(WIDTH, HEIGHT, canvas_w, canvas_h)),
        list(_perspective_transform(0, HEIGHT, canvas_w, canvas_h)),
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    out_w, out_h = int(canvas_w), int(canvas_h)

    # Composite all traces into one RGBA canvas
    composite = np.zeros((out_h, out_w, 4), dtype=np.float32)

    for hm, color, _label, vmax in traces:
        rgba_img = _heatmap_to_rgba(hm, color, vmax)
        warped = cv2.warpPerspective(
            rgba_img, M, (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        # Alpha-over compositing
        a = warped[:, :, 3:4]
        composite[:, :, :3] = composite[:, :, :3] * (1 - a) + warped[:, :, :3] * a
        composite[:, :, 3:4] = np.clip(composite[:, :, 3:4] + a, 0, 1)

    # Draw arena structure first
    _draw_arena_3d(ax, canvas_w, canvas_h)

    # Overlay the composited warped heatmap
    ax.imshow(composite, extent=[0, canvas_w, canvas_h, 0],
              aspect='auto', zorder=5, interpolation='bilinear')


# ============================================================
#  Plot functions
# ============================================================

def plot_single_session(hm_gaze, hm_interact, label, filename):
    """One session: gazing + interaction, flat and 3D."""
    traces = [
        (hm_gaze, COLOR_GAZING, 'Gazing', 1.0),
        (hm_interact, COLOR_INTERACTION, 'Interaction', 1.0),
    ]
    legend_info = [
        (COLOR_GAZING, 'Gazing'),
        (COLOR_INTERACTION, 'Interaction'),
    ]

    # --- 2D ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    _overlay_heatmaps_2d(ax, traces)
    patches = [mpatches.Patch(color=c, alpha=0.7, label=l) for c, l in legend_info]
    ax.legend(handles=patches, loc='upper right', fontsize=11, framealpha=0.9)
    ax.set_title(label, fontsize=18, fontweight='bold', pad=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()

    # --- 3D ---
    canvas_w, canvas_h = 600, 400
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    _overlay_heatmaps_3d(ax, traces, canvas_w, canvas_h)
    patches = [mpatches.Patch(color=c, alpha=0.7, label=l) for c, l in legend_info]
    ax.legend(handles=patches, loc='upper right', fontsize=11, framealpha=0.9)
    ax.set_title(label, fontsize=18, fontweight='bold', pad=14)
    plt.tight_layout()
    fname_3d = filename.replace('.png', '_3d.png')
    plt.savefig(fname_3d, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()


def plot_combined(hm_gaze_coop, hm_interact_coop,
                  hm_gaze_noncoop, hm_interact_noncoop,
                  filename="interactVsGaze_combined.png"):
    """4 traces on one arena: coop (warm) vs non-coop (cool)."""
    combined_traces = [
        (hm_gaze_coop, COLOR_COOP_GAZING, 'Coop Gazing', 1.0),
        (hm_interact_coop, COLOR_COOP_INTERACTION, 'Coop Interaction', 1.0),
        (hm_gaze_noncoop, COLOR_NONCOOP_GAZING, 'Non-Coop Gazing', 1.0),
        (hm_interact_noncoop, COLOR_NONCOOP_INTERACTION, 'Non-Coop Interaction', 1.0),
    ]
    legend_info = [(c, l) for _, c, l, _ in combined_traces]

    # --- 2D ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    _overlay_heatmaps_2d(ax, combined_traces)
    patches = [mpatches.Patch(color=c, alpha=0.7, label=l) for c, l in legend_info]
    ax.legend(handles=patches, loc='upper right', fontsize=10, framealpha=0.9)
    ax.set_title('Coop vs Non-Coop', fontsize=18, fontweight='bold', pad=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()

    # --- 3D ---
    canvas_w, canvas_h = 600, 400
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    _overlay_heatmaps_3d(ax, combined_traces, canvas_w, canvas_h)
    patches = [mpatches.Patch(color=c, alpha=0.7, label=l) for c, l in legend_info]
    ax.legend(handles=patches, loc='upper right', fontsize=10, framealpha=0.9)
    ax.set_title('Coop vs Non-Coop', fontsize=18, fontweight='bold', pad=14)
    plt.tight_layout()
    fname_3d = filename.replace('.png', '_3d.png')
    plt.savefig(fname_3d, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    print("Loading coop session...")
    pos_coop = posLoader(COOP_H5)
    print("Loading non-coop session...")
    pos_noncoop = posLoader(NONCOOP_H5)

    print("Building coop heatmaps...")
    gaze_coop, interact_coop = build_session_heatmaps(pos_coop)
    print("Building non-coop heatmaps...")
    gaze_noncoop, interact_noncoop = build_session_heatmaps(pos_noncoop)

    print("Plotting coop session...")
    plot_single_session(gaze_coop, interact_coop,
                        "Coop Session", "interactVsGaze_coop.png")

    print("Plotting non-coop session...")
    plot_single_session(gaze_noncoop, interact_noncoop,
                        "Non-Coop Session", "interactVsGaze_noncoop.png")

    print("Plotting combined...")
    plot_combined(gaze_coop, interact_coop, gaze_noncoop, interact_noncoop)

    print("Done.")
