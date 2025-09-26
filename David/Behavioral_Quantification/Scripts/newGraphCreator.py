#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 14:58:32 2025

@author: david
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import networkx as nx
from datetime import datetime

#import os

from experiment_class import singleExperiment
from collections import defaultdict
from collections import Counter
from typing import List
from file_extractor_class import fileExtractor
#from mag_class import magLoader
#from lev_class import levLoader
from scipy.stats import linregress, sem
from scipy.interpolate import make_interp_spline
from scipy.stats import mannwhitneyu, kruskal
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from scipy.stats import ttest_ind
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import itertools
import statistics

import statsmodels.api as sm
from itertools import combinations

import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import matplotlib.cm as cm
from matplotlib import animation

import seaborn as sns

import sys

class dataAnalysis:
    def __init__(self, magFiles: List[str], levFiles: List[str], posFiles: List[str], fpsList: List[int], totFramesList: List[int], initialNanList: List[int], dates: List[int], sessions: List[int], ratPairs: List[int], fiberFiles = None, prefix = "", save = True):
        self.experiments = []
        self.prefix = prefix
        self.save = save
        self.NUM_BINS = 30 # Number of time bins for trial chunking
        self.labelSize = 17
        self.titleSize = 18
        deleted_count = 0
        
        print("There are ", len(magFiles), " experiments in this data session. ")
        print("")
        
        if (len(magFiles) != len(levFiles) or len(magFiles) != len(posFiles)):
            raise ValueError("Different number of mag, lev, and pos files")
            
        if ((len(magFiles) != len(fpsList)) or (len(magFiles) != len(totFramesList)) or len(magFiles) != len(initialNanList)):
            print("lenDataFiles: ", len(magFiles))
            print("len(fpsList)", len(fpsList))
            print("len(totFramesList)", len(totFramesList))
            print("len(initialNanList)", len(initialNanList))
            raise ValueError("Different number of fpsList, totFramesList, or initialNanList values")
        
        if (fiberFiles is not None and len(magFiles) != len(fiberFiles)):
            print("len(fiber Files): ", len(fiberFiles))
            raise ValueError("Diff Length of fiberFiles")
        
        
        for i in range(len(magFiles)):
            if (fiberFiles is not None and fiberFiles[i] is not None):
                exp = singleExperiment(magFiles[i], levFiles[i], posFiles[i], fpsList[i], totFramesList[i], initialNanList[i], fp_files=fiberFiles[i])
            else:
                exp = singleExperiment(magFiles[i], levFiles[i], posFiles[i], fpsList[i], totFramesList[i], initialNanList[i], date = dates[i], sessionID = sessions[i], ratPair=ratPairs[i])
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
        
    def generateLeftRightHeatmapData(self, bodypart=3):
        """
        Generate concatenated left vs right position data across experiments.
        Returns: left_x, left_y, right_x, right_y
        """
        left_x, left_y = [], []
        right_x, right_y = [], []

        for i, exp in enumerate(self.experiments):
            pos = exp.pos
            
            #Get both rats x and y data
            x0 = pos.data[0, 0, bodypart, :]
            x1 = pos.data[1, 0, bodypart, :]
            y0 = pos.data[0, 1, bodypart, :]
            y1 = pos.data[1, 1, bodypart, :]
            
            '''x0 = [100, 200, 100, 400, 300, 400, 500, 600, 100, 400, 350, 550]
            y0 = [350, 400, 300, 100, 500, 550, 400, 500, 300, 400, 300, 300]
            
            x1 = [1300, 1200, 1250, 1100, 1000, 800, 400, 700, 900, 900, 1000, 1100]
            y1 = [100, 100, 30, 20, 50, 200, 175, 150, 125, 80, 100, 50]'''
            
            if (len(x0) != len(x1) or len(x0) != len(y0) or len(x0) != len(y1)):
                print("MISMATCH IN LENGTHS")
            
            #Remove NaNs
            '''mask = ~np.isnan(x0) & ~np.isnan(y0)
            x0, y0 = x0[mask], y0[mask]
            mask = ~np.isnan(x1) & ~np.isnan(y1)
            x1, y1 = x1[mask], y1[mask]'''
            
            averageX0 = np.mean(x0)
            averageX1 = np.mean(x1)
            
            print("averageX0: ", averageX0)
            print("averageX1: ", averageX1)
            
            if (averageX0 > averageX1):
                left_x.append(x1)
                left_y.append(y1)
                right_x.append(x0)
                right_y.append(y0)
            else:
                left_x.append(x0)
                left_y.append(y0)
                right_x.append(x1)
                right_y.append(y1)


        left_x = np.concatenate(left_x) if left_x else np.array([])
        left_y = np.concatenate(left_y) if left_y else np.array([])
        right_x = np.concatenate(right_x) if right_x else np.array([])
        right_y = np.concatenate(right_y) if right_y else np.array([])

        return left_x, left_y, right_x, right_y
        
    def generateUpDownHeatmapData(self, bodypart = 3):
        """
        Generate concatenated left vs right position data across experiments.
        Returns: up_x, up_y, down_x, down_y
        """
        up_x, up_y = [], []
        down_x, down_y = [], []

        for i, exp in enumerate(self.experiments):
            pos = exp.pos
            
            #Get both rats x and y data
            x0 = pos.data[0, 0, bodypart, :]
            x1 = pos.data[1, 0, bodypart, :]
            y0 = pos.data[0, 1, bodypart, :]
            y1 = pos.data[1, 1, bodypart, :]
            
            if (len(x0) != len(x1) or len(x0) != len(y0) or len(x0) != len(y1)):
                print("MISMATCH IN LENGTHS")
            
            averageY0 = np.mean(y0)
            averageY1 = np.mean(y1)
            
            print("averageY0: ", averageY0)
            print("averageY1: ", averageY1)
            
            if (averageY0 > averageY1):
                up_x.append(x1)
                up_y.append(y1)
                down_x.append(x0)
                down_y.append(y0)
            else:
                up_x.append(x0)
                up_y.append(y0)
                down_x.append(x1)
                down_y.append(y1)

        up_x = np.concatenate(up_x) if up_x else np.array([])
        up_y = np.concatenate(up_y) if up_y else np.array([])
        down_x = np.concatenate(down_x) if down_x else np.array([])
        down_y = np.concatenate(down_y) if down_y else np.array([])

        return up_x, up_y, down_x, down_y
    
class createGraphs:
    def __init__(self, arena_width=1392, arena_height=640):
        self.arena_width = arena_width
        self.arena_height = arena_height
        self.aspect_ratio = arena_width / arena_height

    def savePositionsCSV(self, left_x, left_y, right_x, right_y, savepath="positions.csv"):
        max_len = max(len(left_x), len(left_y), len(right_x), len(right_y))
        def pad(arr, length):
            return np.pad(arr, (0, length - len(arr)), constant_values=np.nan)

        df = pd.DataFrame({
            "left_x": pad(left_x, max_len),
            "left_y": pad(left_y, max_len),
            "right_x": pad(right_x, max_len),
            "right_y": pad(right_y, max_len)
        })
        df.to_csv(savepath, index=False)
        print(f"Saved positions to {savepath}")

    def makeHeatmap(self, x, y, bins=150, sigma=2, clip_percent=97, gamma=0.3):
        if len(x) == 0:
            return None
        xedges = np.linspace(0, self.arena_width, bins)
        yedges = np.linspace(0, self.arena_height, bins)
        H, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
        H = H.T #transpose so x→cols, y→rows
        H = gaussian_filter(H, sigma=sigma)
        vmax = np.percentile(H, clip_percent)
        H = np.clip(H, 0, vmax)
        H = H / H.max() if H.max() > 0 else H
        H = H ** gamma
        return H

    def saveHeatmap(self, H, title, filename):
        if H is None:
            print(f"Skipping {title}, no data")
            return
        fig_width = 6
        fig_height = fig_width / self.aspect_ratio
        plt.figure(figsize=(fig_width, fig_height))
        im = plt.imshow(
            np.flipud(H),
            extent=[0, self.arena_width, 0, self.arena_height],
            origin="upper",
            aspect="auto",
            cmap="hot"
        )
        plt.colorbar(im, label="Density")
        plt.xlabel("X position (px)")
        plt.ylabel("Y position (px)")
        plt.title(title)
        # Invert Y axis so 0 is top, max at bottom
        plt.gca().invert_yaxis()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        
    
        
        
        
#DATA ANALYSIS GENERATION
#
#

#Real

filtered = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/Filtered.csv"
def getFiltered():
    fe = fileExtractor(filtered)
    fe.data = fe.deleteBadNaN()
    fpsList, totFramesList = fe.returnFPSandTotFrames()
    initial_nan_list = fe.returnNaNPercentage()
    dates = fe.getDatesList()
    print("dates: ", dates)
    sessions = fe.getSessionIDList()
    print("sessions: ", sessions)
    #dates = dates.tolist()
    #sessions = sessions.tolist()
    ratPairs = fe.getRatPairList()
    familiarity = fe.getFamiliarityList()
    transparency = fe.getBarrierTransparencyList()
    #print("initial_nan_list: ", initial_nan_list)
    return [fe.getLevsDatapath(), fe.getMagsDatapath(), fe.getPosDatapath(), fpsList, totFramesList, initial_nan_list, dates, sessions, ratPairs, familiarity, transparency]
arr = getFiltered()
lev_files = arr[0]
mag_files = arr[1]
pos_files = arr[2]
fpsList = arr[3]
totFramesList = arr[4]
initialNanList = arr[5]
dates = arr[6]
sessions = arr[7]
ratPairs = arr[8]        


#Test
'''
lev_files = ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv"]

mag_files = ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv"] 

pos_files = ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5"]

h5_file1 = "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041624_Cam3_TrNum7_Coop_KL002B-KL002Y.predictions.h5"
h5_file2 = "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041624_Cam3_TrNum9_Coop_KL002B-KL002Y.predictions.h5"
h5_file = "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/4_nanerror_test.h5"
#pos_files = [h5_file]
pos_files = [h5_file, h5_file1]

fpsList = [30, 30]
totFramesList = [15000, 15000]
initialNanList = [0.15, 0.12]
dates = [datetime(2024, 4, 16, 0, 0), datetime(2024, 4, 18, 0, 0)] # 
sessions = ['000000_TrNum1', '000000_TrNum2'] #
ratPairs = ['KL001Y-KL001G', 'KL007Y-KL007G'] #
'''


#Generate Graphs
#
#

data = dataAnalysis(mag_files, lev_files, pos_files, fpsList, totFramesList, initialNanList, dates, sessions, ratPairs, prefix = "", save=True)

# Generate left/right position data
left_x, left_y, right_x, right_y = data.generateLeftRightHeatmapData()
up_x, up_y, down_x, down_y = data.generateUpDownHeatmapData()

# Create the graph object
graphs = createGraphs()

# Save CSV
#graphs.savePositionsCSV(left_x, left_y, right_x, right_y, savepath="left_vs_right_positions.csv")

# Make heatmaps
H_left = graphs.makeHeatmap(left_x, left_y)
H_right = graphs.makeHeatmap(right_x, right_y)
H_up = graphs.makeHeatmap(up_x, up_y)
H_down = graphs.makeHeatmap(down_x, down_y)

# Save heatmaps
graphs.saveHeatmap(H_left, "Left-preferring Group Heatmap", "left_heatmap.png")
graphs.saveHeatmap(H_right, "Right-preferring Group Heatmap", "right_heatmap.png")
graphs.saveHeatmap(H_up, "Up-preferring Group Heatmap", "up_heatmap.png")
graphs.saveHeatmap(H_down, "Down-preferring Group Heatmap", "down_heatmap.png")

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
   