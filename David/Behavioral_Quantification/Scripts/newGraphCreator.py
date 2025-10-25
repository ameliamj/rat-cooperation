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
from typing import List
from file_extractor_class import fileExtractor
#from mag_class import magLoader
#from lev_class import levLoader
from scipy import stats
from scipy.stats import linregress, sem
from scipy.stats import ttest_ind, f_oneway
from scipy.ndimage import gaussian_filter


class dataAnalysisRegular:
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
        
    def _filterToLeverPressTrials(self, original_list, lev):
        """
        Filters a list of length lev.returnNumTotalTrials() down to only those trials
        that have lever press data (i.e., appear in lev.data['TrialNum']).
    
        Assumes original_list is 0-indexed, while TrialNum starts at 1.
    
        Args:
            original_list (list): Full list, one entry per trial (indexed from 0).
            lev (levLoader): The lever data loader object.
    
        Returns:
            list: Filtered list with entries only from trials that had lever presses.
        """
        if len(original_list) != lev.returnNumTotalTrials():
            raise ValueError("Length of input list does not match total number of trials.")
    
        # Convert trial numbers to integers and subtract 1 to use as 0-based indices
        lever_trials = sorted(lev.data['TrialNum'].dropna().unique().astype(int))
        filtered_list = [original_list[trial_num - 1] for trial_num in lever_trials]
    
        return filtered_list
    
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
    
    def percentOfTimeARatWasAtLeverFirstandPresseditFirst(self, onlyTrialsWithOneRatatCue = False):
        '''
        For each session, this function generates the percentage of the trials where the rat that was at the lever first before 
        the trial and pressed the lever first. 
        '''
        
        countFirstPressandWaitFirstMatching = 0
        countTrials = 0
        
        for exp_idx, exp in enumerate(self.experiments):
            #print("\nexp.lev_file: ", exp.lev_file)
            
            lev = exp.lev
            pos = exp.pos
            fps = exp.fps
            
            firstPressIDs = lev.returnRatIDFirstPressTrial()
            #print("firstPressIDs: ", firstPressIDs)
            start_times = lev.returnTimeStartTrials()  # Array of trial start times (in seconds) for all trials
            end_times = lev.returnTimeEndTrials()
            total_trials = lev.returnNumTotalTrialswithLeverPress()  # Total number of trials with at least one lever press
            success_trials = lev.returnSuccessTrials()  # Array indicating whether each trial was successful (True/False)
            success_trials = self._filterToLeverPressTrials(success_trials, lev)
            rat0_locations = pos.returnMouseLocation(0)
            rat1_locations = pos.returnMouseLocation(1)
            
            for trial_idx in range(total_trials):
                start_time = start_times[trial_idx]
                end_time = end_times[trial_idx]
                succ = success_trials[trial_idx]
                firstPressID = firstPressIDs[trial_idx]
                
                if (np.isnan(start_time) or np.isnan(end_time)):
                    continue
                
                start_frame = int(start_time * fps)
                end_frame = int(end_time * fps)
                numFrames = end_frame - start_frame
                
                #Wait Before Queue Analysis
                t = start_frame - 1
                rat0_waiting = 0
                rat1_waiting = 0
                rat0_active = True
                rat1_active = True

                while t >= 0 and t < len(rat0_locations) and t < len(rat1_locations) and rat0_locations[t] is not None:
                    if rat0_locations[t] in ['lev_top', 'lev_bottom'] and rat0_active:
                        rat0_waiting += 1
                    else:
                        rat0_active = False

                    if rat1_locations[t] in ['lev_top', 'lev_bottom'] and rat1_active:
                        rat1_waiting += 1
                    else:
                        rat1_active = False

                    if not (rat0_active or rat1_active):
                        break
                    t -= 1
                
                if (exp_idx == 0):
                    print("\nstart_time: ", start_time)
                    print("rat0_waiting: ", rat0_waiting)
                    print("rat1_waiting: ", rat1_waiting)
                    print("firstPressID: ", firstPressID)
                
                if (onlyTrialsWithOneRatatCue and min(rat0_waiting, rat1_waiting) == 0):
                    continue
                if (max(rat0_waiting, rat1_waiting) == 0):
                    continue
                
                countTrials += 1
                if ((rat0_waiting > rat1_waiting and firstPressID == 0) or (rat0_waiting < rat1_waiting and firstPressID == 1)):
                    countFirstPressandWaitFirstMatching += 1
        
        return countFirstPressandWaitFirstMatching, countTrials

    def pressFirstDisparity(self):
        '''
        '''
        
        pressFirstDisparityperSession = []
        
        for exp in self.experiments:
            lev = exp.lev
            pos = exp.pos
            fps = exp.fps
            
            firstPressIDs = lev.returnRatIDFirstPressTrial()
            count1 = sum(1 for rat_id in firstPressIDs if rat_id == 0)
            count2 = sum(1 for rat_id in firstPressIDs if rat_id == 1)
            disparity = (max(count1, count2) - min(count1, count2)) / (count1 + count2)
            pressFirstDisparityperSession.append(disparity)
        
        return pressFirstDisparityperSession
              
    def waitFirstBeforeCueDisparity(self):
        '''
        '''

        waitFirstDisparityperSession = []
        
        for exp in self.experiments:
            lev = exp.lev
            pos = exp.pos
            fps = exp.fps
            
            firstPressIDs = lev.returnRatIDFirstPressTrial()
            #print("firstPressIDs: ", firstPressIDs)
            start_times = lev.returnTimeStartTrials()  # Array of trial start times (in seconds) for all trials
            end_times = lev.returnTimeEndTrials()
            total_trials = lev.returnNumTotalTrialswithLeverPress()  # Total number of trials with at least one lever press
            success_trials = lev.returnSuccessTrials()  # Array indicating whether each trial was successful (True/False)
            success_trials = self._filterToLeverPressTrials(success_trials, lev)
            rat0_locations = pos.returnMouseLocation(0)
            rat1_locations = pos.returnMouseLocation(1)
            
            rat0waitingMore = 0
            rat1waitingMore = 0
            
            for trial_idx in range(total_trials):
                start_time = start_times[trial_idx]
                end_time = end_times[trial_idx]
                succ = success_trials[trial_idx]
                firstPressID = firstPressIDs[trial_idx]
                
                if (np.isnan(start_time) or np.isnan(end_time)):
                    continue
                
                start_frame = int(start_time * fps)
                end_frame = int(end_time * fps)
                numFrames = end_frame - start_frame
                
                #Wait Before Queue Analysis
                t = start_frame - 1
                rat0_waiting = 0
                rat1_waiting = 0
                rat0_active = True
                rat1_active = True

                while t >= 0 and t < len(rat0_locations) and t < len(rat1_locations) and rat0_locations[t] is not None:
                    if rat0_locations[t] in ['lev_top', 'lev_bottom'] and rat0_active:
                        rat0_waiting += 1
                    else:
                        rat0_active = False

                    if rat1_locations[t] in ['lev_top', 'lev_bottom'] and rat1_active:
                        rat1_waiting += 1
                    else:
                        rat1_active = False

                    if not (rat0_active or rat1_active):
                        break
                    t -= 1
                    
                if (rat0_waiting > rat1_waiting):
                    rat0waitingMore += 1
                elif(rat1_waiting > rat0_waiting):
                    rat1waitingMore += 1
            
            disparity = (max(rat0waitingMore, rat1waitingMore) - min(rat0waitingMore, rat1waitingMore)) / (rat0waitingMore + rat1waitingMore)
            waitFirstDisparityperSession.append(disparity)
        
        return waitFirstDisparityperSession

    def successRatePerSession(self):
        successRatePerSession = []
        
        for exp in self.experiments:
            lev = exp.lev
            succPercentage = lev.returnSuccessPercentage()
            successRatePerSession.append(succPercentage)
        
        return successRatePerSession
        
    def gazingAtOtherVsLevVsMag(self):
        otherGazingPerSession = []
        levGazingPerSession = []
        magGazingPerSession = []
        numFramesPerSession = []
        
        for exp in self.experiments:
            pos = exp.pos
            lev = exp.lev
            
            numFrames = exp.endFrame
            
            levGazingFrames0 = np.sum(pos.returnIsLookingAtObjects(0))
            magGazingFrames0 = np.sum(pos.returnIsLookingAtObjects(0, target="mag"))
            otherGazingFrames0 = np.sum(pos.returnIsGazing(0))
            
            levGazingFrames1 = np.sum(pos.returnIsLookingAtObjects(1))
            magGazingFrames1 = np.sum(pos.returnIsLookingAtObjects(1, target="mag"))
            otherGazingFrames1 = np.sum(pos.returnIsGazing(1))
            
            
            otherGazingPerSession.append(levGazingFrames0)
            otherGazingPerSession.append(levGazingFrames1)
            levGazingPerSession.append(levGazingFrames0)
            levGazingPerSession.append(levGazingFrames1)
            magGazingPerSession.append(magGazingFrames0)
            magGazingPerSession.append(magGazingFrames1)
            numFramesPerSession.append(numFrames)
            numFramesPerSession.append(numFrames)
        
        return otherGazingPerSession, levGazingPerSession, magGazingPerSession, numFramesPerSession
    
        

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
        
    def plot_bar(self, data_list, labels, ylabel, title, filename, colors=None, figsize=(6, 5)):
        """
        Create and save a bar plot comparing multiple datasets, including error bars, p-value, and mean values.
        
        Args:
            data_list (List[List[float]]): List of datasets to plot (each dataset is a list of values).
            labels (List[str]): Labels for each dataset (bar group).
            ylabel (str): Label for the y-axis.
            title (str): Title of the plot.
            filename (str): Path to save the plot.
            colors (List[str], optional): Colors for each bar group. Defaults to None (uses default colors).
            figsize (tuple): Figure size as (width, height) in inches (default: (6, 5)).
        """
        if not data_list or any(not data for data in data_list):
            print(f"No data for bar chart: {title}")
            return
        means = [np.mean(data) for data in data_list]
        sems = [sem(data) for data in data_list]
        if colors is None:
            colors = plt.cm.tab10(np.arange(len(data_list)))
        plt.figure(figsize=figsize)
        bars = plt.bar(labels, means, yerr=sems, capsize=5, color=colors)
        
        # Add mean values above bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + sems[0]/2, f'{mean:.2f}',
                     ha='center', va='bottom', fontsize=10)
        
        # Calculate p-value
        if len(data_list) == 2:
            _, p_value = ttest_ind(data_list[0], data_list[1], equal_var=False)
            p_text = f'p = {p_value:.3f}'
        elif len(data_list) > 2:
            f_stat, p_value = f_oneway(*data_list)
            p_text = f'ANOVA p = {p_value:.3f}'
        else:
            p_text = ''
        
        # Add p-value to the plot
        if p_text:
            plt.text(0.05, 0.95, p_text, transform=plt.gca().transAxes, fontsize=10,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    def plot_pie(self, data, labels, title, filename, colors=None, figsize=(6, 6)):
        """
        Create and save a pie chart for categorical data.
        
        Args:
            data (List[float]): List of values representing the size of each pie slice.
            labels (List[str]): Labels for each pie slice.
            title (str): Title of the plot.
            filename (str): Path to save the plot.
            colors (List[str], optional): Colors for each pie slice. Defaults to None (uses default colors).
            figsize (tuple): Figure size as (width, height) in inches (default: (6, 6)).
        """
        if not data or any(d <= 0 for d in data):
            print(f"No valid data for pie chart: {title}")
            return
        plt.figure(figsize=figsize)
        if colors is None:
            colors = plt.cm.tab20(np.arange(len(data)))
        plt.pie(data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title(title)
        plt.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
        
    def plot_scatter(self, x, y, *, xlabel="", ylabel="", filename="", title=""):
        ticksFontSize = 13
        legendsFontSize = 13
        labelFontSize = 15
        titleFontSize = 17
        
        # Convert to numpy arrays
        x = np.array(x)
        y = np.array(y)
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Line of best fit
        x_fit = np.linspace(min(x), max(x), 100)
        y_fit = slope * x_fit + intercept
        
        # Scatterplot
        plt.figure(figsize=(7, 5))
        plt.scatter(x, y, color="blue", alpha=0.7, label="Data points")
        plt.plot(x_fit, y_fit, color="red", label="Best fit line")
        
        # Labels & title
        plt.xlabel(xlabel, fontsize=labelFontSize)
        plt.ylabel(ylabel, fontsize=labelFontSize)
        plt.title(title, fontsize=titleFontSize, weight="bold")
        
        # Annotation with formula, R², p-value, N
        eqn = f"y = {slope:.3f}x + {intercept:.3f}"
        stats_text = f"{eqn}\nR² = {r_value**2:.3f}\np = {p_value:.3g}\nN = {len(x)}"
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment="top", bbox=dict(boxstyle="round", fc="w"))
        
        plt.legend(fontsize=legendsFontSize)
        plt.xticks(fontsize=ticksFontSize)
        plt.yticks(fontsize=ticksFontSize)
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
        plt.close()
        
    
#DATA ANALYSIS GENERATION
#
#

#Real

filtered = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/Filtered.csv"
minReq = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/dyed_preds_min_requirements_valid.csv"


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

def minRequirements():
    fe = fileExtractor(minReq)
    fe.data = fe.deleteBadNaN()
    fe.deleteOnlyFullyInvalid()
    fe.filterOutBadNums()
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

#arr = getFiltered()
arr = minRequirements()

lev_files = arr[0]
mag_files = arr[1]
pos_files = arr[2]
fpsList = arr[3]
totFramesList = arr[4]
initialNanList = arr[5]
dates = arr[6]
sessions = arr[7]
ratPairs = arr[8]        

#hi

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

# Create the data generation object
data = dataAnalysisRegular(mag_files, lev_files, pos_files, fpsList, totFramesList, initialNanList, dates, sessions, ratPairs, prefix = "", save=True)

# Create the graph object
graphs = createGraphs()


#
# gazing At Other vs. Lever vs. Magazine Across Sessions
#

# Extract data from your function
otherGazingPerSession, levGazingPerSession, magGazingPerSession, numFramesPerSession = data.gazingAtOtherVsLevVsMag()

# Convert to percentages for each session
percentGazingOther = [o / n * 100 for o, n in zip(otherGazingPerSession, numFramesPerSession)]
percentGazingLev   = [l / n * 100 for l, n in zip(levGazingPerSession, numFramesPerSession)]
percentGazingMag   = [m / n * 100 for m, n in zip(magGazingPerSession, numFramesPerSession)]

print("percentGazingOther: ", percentGazingOther)
print("percentGazingLev: ", percentGazingLev)
print("percentGazingMag: ", percentGazingMag)

# Labels and title
labels = ["Other", "Lever", "Magazine"]
data_list = [percentGazingOther, percentGazingLev, percentGazingMag]

ylabel = "Percent of Time Gazing (%)"
title = "Comparison of Gazing Behavior Across Sessions"
filename = "percent_gazing_objects_vs_rat_comparison.png"

# Call your plotting function
graphs.plot_bar(data_list, labels, ylabel, title, filename)



'''# Pie chart: Proportion of trials where the rat at lever first pressed it first
matching, total = data.percentOfTimeARatWasAtLeverFirstandPresseditFirst(onlyTrialsWithOneRatatCue=False)
if total > 0:
    data_pie = [matching, total - matching]
    labels_pie = ['First at Lever Pressed First', 'First at Lever Did Not Press First']
    graphs.plot_pie(
        data=data_pie,
        labels=labels_pie,
        title='Proportion of Trials: First at Lever vs. First Press',
        filename="first_lever_press_based_on_waiting_pie.png",
        colors=['green', 'red']
    )

pressFirstDisparity = data.pressFirstDisparity()
waitFirstBeforeCueDisparity = data.waitFirstBeforeCueDisparity()
successRate = data.successRatePerSession()

# Plot 1
graphs.plot_scatter(pressFirstDisparity, successRate,
                    xlabel="Press First Disparity", ylabel="Success Rate",
                    filename="scatter_pressFirstDisp_vs_SuccessRate",
                    title="Press First Disparity vs Success Rate")
# Plot 2
graphs.plot_scatter(
    waitFirstBeforeCueDisparity, successRate,
    xlabel="Wait First Before Cue Disparity",
    ylabel="Success Rate",
    filename="scatter_waitFirstDisp_vs_SuccessRate",
    title="Wait First Disparity vs. Success Rate"
)'''
  



'''
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
'''
        
        

        
        
   