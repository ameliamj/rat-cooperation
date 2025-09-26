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
        
    def generateLeftRightData(self, midline=696, bodypart=3):
        """
        Generate concatenated left vs right position data across experiments.
        Returns: left_x, left_y, right_x, right_y
        """
        left_x, left_y = [], []
        right_x, right_y = [], []

        for i, exp in enumerate(self.experiments):
            pos = exp.pos
            for r in range(2):  # iterate over two rats
                x = pos.data[r, 0, bodypart, :]
                y = pos.data[r, 1, bodypart, :]
                
                # Remove NaNs
                mask = ~np.isnan(x) & ~np.isnan(y)
                x, y = x[mask], y[mask]
                if len(x) == 0:
                    continue

                # Classify left vs right preference
                if np.mean(x < midline) > 0.5:
                    left_x.append(x)
                    left_y.append(y)
                else:
                    right_x.append(x)
                    right_y.append(y)

        left_x = np.concatenate(left_x) if left_x else np.array([])
        left_y = np.concatenate(left_y) if left_y else np.array([])
        right_x = np.concatenate(right_x) if right_x else np.array([])
        right_y = np.concatenate(right_y) if right_y else np.array([])

        return left_x, left_y, right_x, right_y
        
    
    
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

    def makeHeatmap(self, x, y, bins=100, sigma=2, clip_percent=97, gamma=0.2):
        if len(x) == 0:
            return None
        xedges = np.linspace(0, self.arena_width, bins)
        yedges = np.linspace(0, self.arena_height, bins)
        H, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
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
            origin="lower",
            aspect="auto",
            cmap="hot"
        )
        plt.colorbar(im, label="Density")
        plt.xlabel("X position (px)")
        plt.ylabel("Y position (px)")
        plt.title(title)
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

fpsList = [30, 30]
totFramesList = [15000, 15000]
initialNanList = [0.15, 0.12]
dates = ['04172006', '8172006'] # 
sessions = ['id1', 'id2'] #
ratPairs = [] #
'''     


data = dataAnalysis(mag_files, lev_files, pos_files, fpsList, totFramesList, initialNanList, dates, sessions, ratPairs, prefix = "", save=True)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
   