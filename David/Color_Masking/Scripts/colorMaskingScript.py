#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 13:52:18 2025

@author: david
"""

#Color Masking for an Entire Video with Fail Safe's
import cv2
import numpy as np
import imageio
import os

def color_distance_rgb(rgb1, rgb2):
    # Convert to float32 and scale to [0, 1]
    rgb1 = np.array([[rgb1]], dtype=np.float32) / 255.0
    rgb2 = np.array([[rgb2]], dtype=np.float32) / 255.0

    lab1 = cv2.cvtColor(rgb1, cv2.COLOR_RGB2Lab)[0][0]
    lab2 = cv2.cvtColor(rgb2, cv2.COLOR_RGB2Lab)[0][0]

    return np.linalg.norm(lab1 - lab2)

def extract_frame_colors(video_path, frame_idx=0, output_path=None):
    """
    Extract a single frame for manual color picking.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret and output_path:
        cv2.imwrite(output_path, frame)
        print(f"Saved frame {frame_idx} to {output_path}. Use an image editor to pick RGB colors of dyed areas.")
    return frame if ret else None

def normalize_video(
    input_path,
    output_path,
    color1_rgb,
    color1_rgb2,
    color2_rgb,
    color2_rgb2,
    threshold=16,
    var_threshold=2.0,
    sample_rate=1,
    diagnostics_dir=None,
    verbose=False,
    distance_threshold=5  # New parameter for proximity distance in pixels
):
    # Create diagnostics directory and save test frame
    if diagnostics_dir:
        os.makedirs(diagnostics_dir, exist_ok=True)
        extract_frame_colors(input_path, frame_idx=0, output_path=os.path.join(diagnostics_dir, "test_frame.jpg"))

    # Compute background mask

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open input video at: {input_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if fps == 0 or fps is None:
        fps = 30  # fallback
    
    print(f"Video properties - FPS: {fps}, Width: {width}, Height: {height}")
    
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', format='ffmpeg')

    # Precompute LAB reference colors
    c1 = np.array([[color1_rgb]], dtype=np.float32) / 255.0
    c3 = np.array([[color1_rgb2]], dtype=np.float32) / 255.0
    
    c2 = np.array([[color2_rgb]], dtype=np.float32) / 255.0
    c4 = np.array([[color2_rgb2]], dtype=np.float32) / 255.0
    
    color1_lab = cv2.cvtColor(c1, cv2.COLOR_RGB2Lab)[0][0]
    color1_lab2 = cv2.cvtColor(c3, cv2.COLOR_RGB2Lab)[0][0]
    
    color2_lab = cv2.cvtColor(c2, cv2.COLOR_RGB2Lab)[0][0]
    color2_lab2 = cv2.cvtColor(c4, cv2.COLOR_RGB2Lab)[0][0]
    
    
    prev_mask1 = None
    prev_mask3 = None
    
    prev_mask2 = None
    prev_mask4 = None
    
    frame_idx = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    # Kernel for proximity dilation
    proximity_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * distance_threshold + 1, 2 * distance_threshold + 1))

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_f = frame_rgb.astype(np.float32) / 255.0
        lab = cv2.cvtColor(rgb_f, cv2.COLOR_RGB2Lab)

        # Compute distances
        dist1 = np.linalg.norm(lab - color1_lab, axis=2)
        dist3 = np.linalg.norm(lab - color1_lab2, axis=2)
        
        dist2 = np.linalg.norm(lab - color2_lab, axis=2)
        dist4 = np.linalg.norm(lab - color2_lab2, axis=2)
        
        # Create mask for bright pixels (all RGB > 251)
        bright_mask = np.all(frame_rgb > 248, axis=2)

        # Create proximity masks based on previous frame's colored pixels
        if prev_mask1 is not None and prev_mask2 is not None and prev_mask3 is not None and prev_mask4 is not None:
            # Proximity mask for color1_rgb and color1_rgb2 (mask1 and mask3)
            prev_colored1 = (prev_mask1 | prev_mask3).astype(np.uint8)
            proximity_mask1a = cv2.dilate(prev_colored1, proximity_kernel, iterations=1).astype(bool)
            # Proximity mask for color2_rgb (mask2)
            prev_colored2 = (prev_mask2 | prev_mask4).astype(np.uint8)
            proximity_mask2a = cv2.dilate(prev_colored2, proximity_kernel, iterations=1).astype(bool)
            # Restrict proximity masks to leftmost 1/8th or rightmost 1/8th of the screen
            left_mask = np.zeros_like(proximity_mask1a, dtype=bool)
            left_mask[:, :width // 8] = True  # Leftmost 1/8th of the frame
            right_mask = np.zeros_like(proximity_mask1a, dtype=bool)
            right_mask[:, -width // 12:] = True  # Rightmost 1/8th of the frame
            left_or_right_mask = ~(left_mask | right_mask)
            #cv2.imwrite(os.path.join(diagnostics_dir, f"frame_{frame_idx}_proximity_mask1_before.png"), proximity_mask1.astype(np.uint8) * 255)
            #cv2.imwrite(os.path.join(diagnostics_dir, f"frame_{frame_idx}_proximity_mask2_before.png"), proximity_mask2.astype(np.uint8) * 255)
            proximity_mask1 = (proximity_mask1a & (left_or_right_mask))
            proximity_mask2 = (proximity_mask2a & (left_or_right_mask))
            
        else:
            proximity_mask1 = np.zeros_like(dist1, dtype=bool)
            proximity_mask2 = np.zeros_like(dist2, dtype=bool)
            # Define dummy masks for diagnostics output
            proximity_mask1a = np.zeros_like(proximity_mask1, dtype=bool)
            proximity_mask2a = np.zeros_like(proximity_mask1, dtype=bool)
            left_or_right_mask = np.zeros_like(proximity_mask1, dtype=bool)
            left_mask = np.zeros_like(proximity_mask1, dtype=bool)
            right_mask = np.zeros_like(proximity_mask1, dtype=bool)
            left_or_right_mask = np.zeros_like(proximity_mask1, dtype=bool)

        # Apply different thresholds based on proximity
        threshold_near = threshold + 0    # Threshold for pixels near previously colored pixels
        mask1 = np.where(proximity_mask1, dist1 < threshold_near, dist1 < threshold) & (dist1 < dist2) & (~bright_mask) #& (~static_mask)
        mask2 = np.where(proximity_mask2, dist2 < threshold_near, dist2 < threshold) & (dist2 < dist1) & (~bright_mask) #& (~static_mask)
        mask3 = np.where(proximity_mask1, dist3 < threshold_near, dist3 < threshold) & (dist3 < dist2) & (~bright_mask) #& (~static_mask)
        mask4 = np.where(proximity_mask2, dist4 < threshold_near, dist4 < threshold) & (dist4 < dist1) & (~bright_mask) #& (~static_mask)
        
        '''right_mask2 = np.zeros_like(proximity_mask1a, dtype=bool)
        right_mask2[:, -width // 10:] = True  # Rightmost 1/8th of the frame
        right_mask2 = ~right_mask2
        mask1 = mask1 & right_mask2
        mask2 = mask2 & right_mask2
        mask3 = mask3 & right_mask2
        mask4 = mask4'''
        
        # Apply morphological operations
        mask1 = cv2.morphologyEx(mask1.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
        mask2 = cv2.morphologyEx(mask2.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
        mask3 = cv2.morphologyEx(mask3.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
        mask4 = cv2.morphologyEx(mask4.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)

        prev_mask1 = mask1
        prev_mask2 = mask2
        prev_mask3 = mask3
        prev_mask4 = mask4

        # Apply coloring
        output = frame_rgb.copy() 
        output[mask1] = [255, 0, 0]  # Red for mouse 1
        output[mask3] = [255, 0, 0]  # Red for mouse 1
        
        
        output[mask2] = [0, 0, 255]  # Blue for mouse 2
        output[mask4] = [0, 0, 255]  # Blue for mouse 2

        # Diagnostic outputs
        if verbose:
            print(f"Frame {frame_idx}:")
            #print(f"  Mouse 1 (red) pixels: {np.sum(refined1)}")
            #print(f"  Mouse 2 (blue) pixels: {np.sum(refined2)}")
            #print(f"  Avg LAB distances - Color1: {np.mean(dist1):.2f}, Color2: {np.mean(dist2):.2f}")
            #print(f"  Min LAB distances - Color1: {np.min(dist1):.2f}, Color2: {np.min(dist2):.2f}")

        if diagnostics_dir and frame_idx % 100 == 0:
            cv2.imwrite(os.path.join(diagnostics_dir, f"frame_{frame_idx}_mask1.png"), mask1.astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(diagnostics_dir, f"frame_{frame_idx}_mask2.png"), mask2.astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(diagnostics_dir, f"frame_{frame_idx}_mask3.png"), mask3.astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(diagnostics_dir, f"frame_{frame_idx}_mask4.png"), mask4.astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(diagnostics_dir, f"frame_{frame_idx}_output.jpg"), cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(diagnostics_dir, f"frame_{frame_idx}_proximity_mask1_before.png"), proximity_mask1a.astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(diagnostics_dir, f"frame_{frame_idx}_proximity_mask2_before.png"), proximity_mask2a.astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(diagnostics_dir, f"frame_{frame_idx}_proximity_mask1.png"), proximity_mask1.astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(diagnostics_dir, f"frame_{frame_idx}_proximity_mask2.png"), proximity_mask2.astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(diagnostics_dir, f"frame_{frame_idx}_bright_mask.png"), bright_mask.astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(diagnostics_dir, f"frame_{frame_idx}_left_mask.png"), left_mask.astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(diagnostics_dir, f"frame_{frame_idx}_right_mask.png"), right_mask.astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(diagnostics_dir, f"frame_{frame_idx}_leftorright_mask.png"), left_or_right_mask.astype(np.uint8) * 255)


        writer.append_data(output)
        frame_idx += 1

    cap.release()
    writer.close()

if __name__ == "__main__":
    print("Start Running: ")
    
    yellowcollar = [250, 244, 195]     #yellow collar (15, 22)
    bluecollar = [37, 140, 253]     #blue collar
    movingmouse_yellowcollar = [232, 200, 150]
    movingmouse_bluecollar = [40, 150, 245]
    redcollar = [220, 153, 180]
    redcollar1 = [255, 200, 255]
    redcollar2 = [224, 160, 190]
    movingmouse_redcollar = [210, 146, 185]
    #movingmouse_redcollar = [177, 130, 160]
    movingmouse_redcollar1 = [255, 155, 250]
    movingmouse_redcollar2 = [177, 130, 160]
    greencollar = [44, 202, 200]
    movingmouse_greencollar = [42, 145, 145]
    
    
    shade1 = [255, 230, 188]  # yellow (unused)
    shade2 = [188, 186, 248]  # blue
    shade3 = [178, 175, 169]  # green
    
    
    
    shade6 = [253, 215, 175] #yellow, test bad dyed
    shade62 = [254, 245, 210]
    shade7 = [200, 195, 250] #blue, test bad dyed
    
    shade8 = [255, 237, 202] #Yellow2, bad dyed (13, 18)
    shade8b = [223, 195, 170] 
    shade9 = [241, 150, 155] #red,  bad dyed
    shade9b = [255, 190, 245]
    
    
    normalize_video(
        input_path="/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Color_Masking/Input_Videos/red_green_collar_highquality.mp4",
        output_path="/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Color_Masking/Sample_Videos/red_green_collars_hq_masked.mp4",
        color1_rgb=redcollar,
        color1_rgb2 = movingmouse_redcollar,
        color2_rgb=greencollar,
        color2_rgb2 = movingmouse_greencollar,
        threshold=16,
        var_threshold=8.0,
        sample_rate=3,
        diagnostics_dir="/Users/david/Documents/Research/Saxena Lab/Color Masking/diagnostics",
        verbose=True,
        distance_threshold=1  # Set proximity distance to 5 pixels
    )
    
    #dist = color_distance_rgb(movingmouse_yellowcollar, shade4)
    #dist2 = color_distance_rgb(background1, shade5)
    #print("LAB distance:", dist)
    #print("LAB distance:", dist2)
    #print("Done writing masked_output.mp4")
