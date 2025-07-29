#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 16:56:00 2025

@author: david
"""

import cv2
import numpy as np

def convert_blue_to_red(image_path, output_path):
    # Load the image
    img = cv2.imread(image_path)

    # Define blue color range in BGR
    lower_blue = np.array([100, 0, 0])
    upper_blue = np.array([255, 100, 100])

    # Create a mask for blue regions
    blue_mask = cv2.inRange(img, lower_blue, upper_blue)

    # Create red color (in BGR)
    red_color = [0, 0, 255]

    # Change blue pixels to red
    img[blue_mask > 0] = red_color

    # Save the result
    cv2.imwrite(output_path, img)
    print(f"Saved new image with blue turned to red at: {output_path}")

# Example usage

input_img = "/Users/david/Documents/Research/Saxena_Lab/WuTsai_Poster/filtered_InteractionByExperimentIndex_onlyAll.png"
convert_blue_to_red(input_img, "/Users/david/Documents/Research/Saxena_Lab/WuTsai_Poster/convertedFull_interactionByExperimentIndex_All.png")