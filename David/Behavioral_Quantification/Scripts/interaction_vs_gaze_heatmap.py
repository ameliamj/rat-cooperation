#!/usr/bin/env python3
"""
Interaction vs Gazing heatmaps for two sessions (e.g. coop vs non-coop).
Produces:
  1. A single-session 1x3 figure per session (Gazing | Interaction | Difference)
  2. A 2x3 comparison figure across the two sessions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from pos_class import posLoader

# ---------- Config ----------
COOP_H5 = "/Users/david/Downloads/041024_COOPTRAIN_LARGEARENA_EB009B-EB019Y_Camera2.predictions.h5"
NONCOOP_H5 = "/Users/david/Downloads/041624_Cam4_TrNum6_Comp_KL001B-KL001Y.predictions.h5"

BIN_SIZE = 5
SIGMA = 1.5
WIDTH, HEIGHT = 1392, 640


def build_heatmaps(pos):
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


def plot_single_session(hm_gaze, hm_interact, label, filename):
    extent = [0, WIDTH, HEIGHT, 0]
    diff = hm_interact - hm_gaze

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    im0 = axes[0].imshow(hm_gaze, cmap='magma', origin='upper',
                          extent=extent, aspect='auto')
    axes[0].set_title('Gazing', fontsize=17)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(hm_interact, cmap='magma', origin='upper',
                          extent=extent, aspect='auto')
    axes[1].set_title('Interaction', fontsize=17)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    vmax = max(np.abs(diff).max(), 1e-8)
    im2 = axes[2].imshow(diff, cmap='seismic', origin='upper',
                          extent=extent, aspect='auto', vmin=-vmax, vmax=vmax)
    axes[2].set_title('Interaction - Gazing', fontsize=17)
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_xlabel('X (px)')
        ax.set_ylabel('Y (px)')

    fig.suptitle(f'Interaction vs Gazing — {label}', fontsize=19)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_comparison(gaze_a, interact_a, gaze_b, interact_b,
                    label_a="Coop", label_b="Non-Coop", filename="comparison.png"):
    extent = [0, WIDTH, HEIGHT, 0]
    diff_a = interact_a - gaze_a
    diff_b = interact_b - gaze_b

    gaze_vmax = max(gaze_a.max(), gaze_b.max(), 1e-8)
    interact_vmax = max(interact_a.max(), interact_b.max(), 1e-8)
    diff_vmax = max(np.abs(diff_a).max(), np.abs(diff_b).max(), 1e-8)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    for row, (gaze_hm, interact_hm, diff_hm, label) in enumerate([
        (gaze_a, interact_a, diff_a, label_a),
        (gaze_b, interact_b, diff_b, label_b),
    ]):
        im0 = axes[row, 0].imshow(gaze_hm, cmap='magma', origin='upper',
                                   extent=extent, aspect='auto', vmin=0, vmax=gaze_vmax)
        axes[row, 0].set_title(f'{label} — Gazing', fontsize=17)
        fig.colorbar(im0, ax=axes[row, 0], fraction=0.046)

        im1 = axes[row, 1].imshow(interact_hm, cmap='magma', origin='upper',
                                   extent=extent, aspect='auto', vmin=0, vmax=interact_vmax)
        axes[row, 1].set_title(f'{label} — Interaction', fontsize=17)
        fig.colorbar(im1, ax=axes[row, 1], fraction=0.046)

        im2 = axes[row, 2].imshow(diff_hm, cmap='seismic', origin='upper',
                                   extent=extent, aspect='auto', vmin=-diff_vmax, vmax=diff_vmax)
        axes[row, 2].set_title(f'{label} — Interact - Gaze', fontsize=17)
        fig.colorbar(im2, ax=axes[row, 2], fraction=0.046)

        for ax in axes[row]:
            ax.set_xlabel('X (px)')
            ax.set_ylabel('Y (px)')

    fig.suptitle(f'{label_a} vs {label_b}: Interaction & Gazing Heatmaps', fontsize=19)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == "__main__":
    print("Loading coop session...")
    pos_coop = posLoader(COOP_H5)
    print("Loading non-coop session...")
    pos_noncoop = posLoader(NONCOOP_H5)

    print("Building coop heatmaps...")
    gaze_coop, interact_coop = build_heatmaps(pos_coop)
    print("Building non-coop heatmaps...")
    gaze_noncoop, interact_noncoop = build_heatmaps(pos_noncoop)

    print("Plotting single-session figures...")
    plot_single_session(gaze_coop, interact_coop, "Coop", "interactVsGaze_coop.png")
    plot_single_session(gaze_noncoop, interact_noncoop, "Non-Coop", "interactVsGaze_noncoop.png")

    print("Plotting comparison figure...")
    plot_comparison(gaze_coop, interact_coop, gaze_noncoop, interact_noncoop,
                    filename="interactVsGaze_comparison.png")

    print("Done.")
