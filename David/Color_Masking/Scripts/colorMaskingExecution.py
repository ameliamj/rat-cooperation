#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 23:37:33 2025

@author: david
"""

import os
from colorMaskingScript import normalize_video

directory1 = "/gpfs/radev/pi/saxena/aj764/Nina_Model_Testing/Collars/Masking/videos"
directory2 = "/gpfs/radev/pi/saxena/aj764/David_Model_Testing/Collars/Masking/videos"

list_names = ["111824_COOPTRAIN_LARGEARENA_NM014B-NM014Y_Camera2.mp4", 
              "111924_COOPTRAIN_LARGEARENA_NF010B-NF010Y_Camera3.mp4", 
              "111924_COOPTRAIN_LARGEARENA_NM002B-NM002Y_Camera2.mp4", 
              "112024_COOPTRAIN_LARGEARENA_NF010B-NF010Y_Camera4.mp4", 
              "112124_COOPTRAIN_LARGEARENA_NF008B-NF008Y_Camera3.mp4", 
              "112524_COOPTRAIN_LARGEARENA_NM001R-NM001G_Camera2.mp4", 
              "112524_COOPTRAIN_LARGEARENA_NM016B-NM016Y_Camera1.mp4", 
              "112624_COOPTRAIN_LARGEARENA_NM002B-NM002Y_Camera2.mp4", 
              "112724_COOPTRAIN_LARGEARENA_NF031R-NF031G_Camera3.mp4", 
              "112724_COOPTRAIN_LARGEARENA_NM001R-NM001G_Camera1.mp4", 
              "112724_COOPTRAIN_LARGEARENA_NM003R-NM003G_Camera2.mp4", 
              "120224_COOPTRAIN_LARGEARENA_NF019R-NF019G_Camera3.mp4", 
              "120324_COOPTRAIN_LARGEARENA_NM002B-NM002Y_Camera2.mp4", 
              "120524_COOPTRAIN_LARGEARENA_NM001R-NM001G_Camera1.mp4", 
              "120524_COOPTRAIN_LARGEARENA_NM003R-NM003G_Camera1.mp4", 
              "120624_COOPTRAIN_LARGEARENA_NM001R-NM001G_Camera2.mp4"]

'''
Iterate through list_names, extract the two colors by looking the the character 
9 letters from the back and 16 letters from the back of the name in list_names. 
R – Red, B – Blue, Y – Yellow, G – Green. Then, run normalizeVideo for each name in listname, where the file is stored with the 
appropriate RGB values depending on the colors. 

'''

#Yellow
yellowcollar = [250, 244, 195]    
movingmouse_yellowcollar = [232, 200, 150]
 
#Blue
bluecollar = [37, 140, 253]
movingmouse_bluecollar = [40, 150, 245]

#Red
redcollar = [220, 153, 180]
movingmouse_redcollar = [214, 146, 185]

#Green
greencollar = [44, 202, 200]
movingmouse_greencollar = [42, 145, 145]

# Map from color code to collar RGB values
color_map = {
    "Y": (yellowcollar, movingmouse_yellowcollar),
    "B": (bluecollar, movingmouse_bluecollar),
    "R": (redcollar, movingmouse_redcollar),
    "G": (greencollar, movingmouse_greencollar)
}

#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

#Actual Code: 
    
for name in list_names:
    color1_code = name[-13]  # First collar
    color2_code = name[-20]   # Second collar

    # Safety check
    if color1_code not in color_map or color2_code not in color_map:
        print(f"Skipping {name} due to unknown color code(s): {color1_code}, {color2_code}")
        continue

    color1_rgb, color1_rgb2 = color_map[color1_code]
    color2_rgb, color2_rgb2 = color_map[color2_code]

    input_path = os.path.join(directory1, name)
    output_path = os.path.join(directory2, "masked_" + name)
    
    print(input_path)
    print(output_path)
    
    diagnostics_dir = os.path.join(directory2, "diagnostics", name.replace(".mp4", ""))

    print(f"Processing {name} with colors {color1_code} and {color2_code}...")

    normalize_video(
        input_path=input_path,
        output_path=output_path,
        color1_rgb=color1_rgb,
        color1_rgb2=color1_rgb2,
        color2_rgb=color2_rgb,
        color2_rgb2=color2_rgb2,
        threshold=15,
        var_threshold=8.0,
        sample_rate=1,
        diagnostics_dir=diagnostics_dir,
        verbose=True,
        distance_threshold=0
    )

print("Batch processing complete.")