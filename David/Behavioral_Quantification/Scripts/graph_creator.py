#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 17:57:51 2025

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
from typing import List
from file_extractor_class import fileExtractor
#from mag_class import magLoader
#from lev_class import levLoader
from scipy.stats import linregress
from scipy.interpolate import make_interp_spline
from scipy.stats import mannwhitneyu, kruskal
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit
import itertools

from matplotlib.patches import Patch
import seaborn as sns

import sys
sys.stdout.flush()
sys.stderr.flush()


#all_valid = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral Quantification/Sorted Data Files/dyed_preds_all_valid.csv"
all_valid = "/gpfs/radev/home/drb83/project/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/dyed_preds_all_valid.csv"

only_opaque = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_opaque_sessions.csv"
only_translucent = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_translucent_sessions.csv"
only_transparent = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_transparent_sessions.csv"

only_unfamiliar = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_unfamiliar_partners.csv"
only_trainingpartners = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_training_partners.csv"

only_PairedTesting = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_PairedTesting.csv"
only_TrainingCoop = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_TrainingCooperation.csv"


only_opaque_filtered = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_opaque_sessions_filtered.csv"
only_translucent_filtered = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_translucent_sessions_filtered.csv"
only_transparent_filtered = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_transparent_sessions_filtered.csv"

only_unfamiliar_filtered = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_unfamiliar_partners_filtered.csv"
only_trainingpartners_filtered = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_training_partners_filtered.csv"

only_PairedTesting_filtered = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_PairedTesting_filtered.csv"
only_TrainingCoop_filtered = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_TrainingCooperation_filtered.csv"


only_opaque_filtered_onlyFirst = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_opaque_sessions_filtered_onlyFirst.csv"
only_translucent_filtered_onlyFirst = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_translucent_sessions_filtered_onlyFirst.csv"
only_transparent_filtered_onlyFirst = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_transparent_sessions_filtered_onlyFirst.csv"

only_unfamiliar_filtered_onlyFirst = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_unfamiliar_partners_filtered_onlyFirst.csv"
only_trainingpartners_filtered_onlyFirst = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_training_partners_filtered_onlyFirst.csv"

only_PairedTesting_filtered_onlyFirst = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_PairedTesting_filtered_onlyFirst.csv"
only_TrainingCoop_filtered_onlyFirst = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_TrainingCooperation_filtered_onlyFirst.csv"


only_opaque_filtered_partiallyValid = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_opaque_sessions_filtered_partiallyValid.csv"
only_translucent_filtered_partiallyValid = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_translucent_sessions_filtered_partiallyValid.csv"
only_transparent_filtered_partiallyValid = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_transparent_sessions_filtered_partiallyValid.csv"

only_unfamiliar_filtered_partiallyValid = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_unfamiliar_partners_filtered_partiallyValid.csv"
only_trainingpartners_filtered_partiallyValid = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_training_partners_filtered_partiallyValid.csv"

only_PairedTesting_filtered_partiallyValid = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_PairedTesting_filtered_partiallyValid.csv"
only_TrainingCoop_filtered_partiallyValid = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_TrainingCooperation_filtered_partiallyValid.csv"



filtered = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/Filtered.csv"


def getAllValid():
    fe = fileExtractor(all_valid)
    fpsList, totFramesList = fe.returnFPSandTotFrames()
    return [fe.getLevsDatapath(), fe.getMagsDatapath(), fe.getPosDatapath(), fpsList, totFramesList]
    
def getOnlyOpaque(filtered = True, onlyFirst = False):
    if (onlyFirst):
        fe = fileExtractor(only_opaque_filtered_onlyFirst)
    elif (filtered):
        fe = fileExtractor(only_opaque_filtered)
    else:
        fe = fileExtractor(only_opaque)
    fpsList, totFramesList = fe.returnFPSandTotFrames()
    initial_nan_list = fe.returnNaNPercentage()
    return [fe.getLevsDatapath(), fe.getMagsDatapath(), fe.getPosDatapath(), fpsList, totFramesList, initial_nan_list]

def getOnlyTranslucent(filtered = True, onlyFirst = False):
    if (onlyFirst):
        fe = fileExtractor(only_translucent_filtered_onlyFirst)
    elif (filtered):
        fe = fileExtractor(only_translucent_filtered)
    else:
        fe = fileExtractor(only_translucent)
    fpsList, totFramesList = fe.returnFPSandTotFrames()
    initial_nan_list = fe.returnNaNPercentage()
    return [fe.getLevsDatapath(), fe.getMagsDatapath(), fe.getPosDatapath(), fpsList, totFramesList, initial_nan_list]

def getOnlyTransparent(filtered = True, onlyFirst = False):
    if (onlyFirst):
        fe = fileExtractor(only_transparent_filtered_onlyFirst)
    elif (filtered):
        fe = fileExtractor(only_transparent_filtered)
    else:
        fe = fileExtractor(only_transparent)
    fpsList, totFramesList = fe.returnFPSandTotFrames()
    initial_nan_list = fe.returnNaNPercentage()
    return [fe.getLevsDatapath(), fe.getMagsDatapath(), fe.getPosDatapath(), fpsList, totFramesList, initial_nan_list]

def getOnlyUnfamiliar(filtered = True, onlyFirst = False):
    if (onlyFirst):
        fe = fileExtractor(only_unfamiliar_filtered_onlyFirst)
    elif (filtered):
        fe = fileExtractor(only_unfamiliar_filtered)
    else:
        fe = fileExtractor(only_unfamiliar)
    fpsList, totFramesList = fe.returnFPSandTotFrames()
    initial_nan_list = fe.returnNaNPercentage()
    
    return [fe.getLevsDatapath(), fe.getMagsDatapath(), fe.getPosDatapath(), fpsList, totFramesList, initial_nan_list]

def getOnlyTrainingPartners(filtered = True, onlyFirst = False):
    if (onlyFirst):
        fe = fileExtractor(only_trainingpartners_filtered_onlyFirst)
    elif (filtered):
        fe = fileExtractor(only_trainingpartners_filtered)
    else:
        fe = fileExtractor(only_trainingpartners)
    fpsList, totFramesList = fe.returnFPSandTotFrames()
    initial_nan_list = fe.returnNaNPercentage()
    
    return [fe.getLevsDatapath(), fe.getMagsDatapath(), fe.getPosDatapath(), fpsList, totFramesList, initial_nan_list]

def getOnlyPairedTesting(filtered = True, onlyFirst = False):
    if (onlyFirst):
        fe = fileExtractor(only_PairedTesting_filtered_onlyFirst)
    elif (filtered):
        fe = fileExtractor(only_PairedTesting_filtered)
    else:
        fe = fileExtractor(only_PairedTesting)
    fpsList, totFramesList = fe.returnFPSandTotFrames()
    initial_nan_list = fe.returnNaNPercentage()
    
    return [fe.getLevsDatapath(), fe.getMagsDatapath(), fe.getPosDatapath(), fpsList, totFramesList, initial_nan_list]


def getOnlyTrainingCoop(filtered = True, onlyFirst = False):
    if (onlyFirst):
        fe = fileExtractor(only_TrainingCoop_filtered_onlyFirst)
    elif (filtered):
        fe = fileExtractor(only_TrainingCoop_filtered)
    else:
        fe = fileExtractor(only_TrainingCoop)
    fpsList, totFramesList = fe.returnFPSandTotFrames()
    initial_nan_list = fe.returnNaNPercentage()
    
    return [fe.getLevsDatapath(), fe.getMagsDatapath(), fe.getPosDatapath(), fpsList, totFramesList, initial_nan_list]     

def getAllTrainingCoop():
    fe = fileExtractor(only_TrainingCoop_filtered_partiallyValid)
    #fe.data = fe.deleteBadNaN()
    fpsList, totFramesList = fe.returnFPSandTotFrames()
    initial_nan_list = fe.returnNaNPercentage()
    #print("initial_nan_list: ", initial_nan_list)
    return [fe.getLevsDatapath(), fe.getMagsDatapath(), fe.getPosDatapath(), fpsList, totFramesList, initial_nan_list]

def getAllPairedTesting():
    fe = fileExtractor(only_PairedTesting_filtered_partiallyValid)
    #fe.data = fe.deleteBadNaN()
    fpsList, totFramesList = fe.returnFPSandTotFrames()
    initial_nan_list = fe.returnNaNPercentage()
    #print("initial_nan_list: ", initial_nan_list)
    return [fe.getLevsDatapath(), fe.getMagsDatapath(), fe.getPosDatapath(), fpsList, totFramesList, initial_nan_list]

def getAllTransparent():
    fe = fileExtractor(only_transparent_filtered_partiallyValid)
    #fe.data = fe.deleteBadNaN()
    fpsList, totFramesList = fe.returnFPSandTotFrames()
    initial_nan_list = fe.returnNaNPercentage()
    #print("initial_nan_list: ", initial_nan_list)
    return [fe.getLevsDatapath(), fe.getMagsDatapath(), fe.getPosDatapath(), fpsList, totFramesList, initial_nan_list]

def getAllTranslucent():
    fe = fileExtractor(only_translucent_filtered_partiallyValid)
    #fe.data = fe.deleteBadNaN()
    fpsList, totFramesList = fe.returnFPSandTotFrames()
    initial_nan_list = fe.returnNaNPercentage()
    #print("initial_nan_list: ", initial_nan_list)
    return [fe.getLevsDatapath(), fe.getMagsDatapath(), fe.getPosDatapath(), fpsList, totFramesList, initial_nan_list]

def getAllOpaque():
    fe = fileExtractor(only_opaque_filtered_partiallyValid)
    #fe.data = fe.deleteBadNaN()
    fpsList, totFramesList = fe.returnFPSandTotFrames()
    initial_nan_list = fe.returnNaNPercentage()
    #print("initial_nan_list: ", initial_nan_list)
    return [fe.getLevsDatapath(), fe.getMagsDatapath(), fe.getPosDatapath(), fpsList, totFramesList, initial_nan_list]

# ---------------------------------------------------------------------------------------------------------


##Class to create Graphs with data from multiple files with different categories + Functions to get data from all the different categories
#
#
        
class multiFileGraphsCategories:
    def __init__(self, magFiles: List[List[str]], levFiles: List[List[str]], posFiles: List[List[str]], categoryNames: List[str], save = True):
        self.allFileGroupExperiments = []
        self.categoryNames = categoryNames
        self.numCategories = len(magFiles)
        
        print("There are: ", len(magFiles), " categories!")
        
        if not (len(magFiles) == len(levFiles) == len(posFiles) == len(categoryNames)):
            raise ValueError("Mismatch between number of categories and provided file lists or category names.")

        for c in range(self.numCategories):
            file_group = []
            for mag, lev, pos in zip(magFiles[c], levFiles[c], posFiles[c]):
                exp = singleExperiment(mag, lev, pos)
                
                mag_missing = [col for col in exp.mag.categories if col not in exp.mag.data.columns]
                lev_missing = [col for col in exp.lev.categories if col not in exp.lev.data.columns]
                
                if mag_missing or lev_missing:
                    print("Skipping experiment due to missing categories:")
                    if mag_missing:
                        print(f"  MagFile missing: {mag_missing}")
                        print(f"  Mag File: {mag}")
                    if lev_missing:
                        print(f"  LevFile missing: {lev_missing}")
                        print(f"  Lev File: {lev}")
                    continue
                
                file_group.append(exp)
            self.allFileGroupExperiments.append(file_group)
        
        self.prefix = "filtered_"
        self.endSaveName = ""
        for cat in categoryNames:    
            self.endSaveName += f"_{cat}"
        
        self.save = save
        self.path = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Graphs/filtered_"
        #self.path = ""
        
        if not self.path:
            print("Warning: No save path specified. Saving plots to current directory.")
        
        print("Done with init")
        
    def compareGazeEventsCategories(self):
        '''
        Function Purpose:
        -----------------
        This function calculates and visualizes the average number of gaze events per minute 
        (assuming 1800 frames equals 1 minute) for each experimental category.
        
        For each category:
        - It sums all gaze events detected for both mice (ID 0 and ID 1).
        - It normalizes the total events by the total number of frames.
        - It scales the result to a 1-minute equivalent (1800 frames).
        - The result is plotted as a bar chart with each bar representing a category.
        '''
        
        avg_events = []  # Stores the average gaze events per minute for each category
        group_events = []  # List of lists: per-experiment values for each category
        FRAME_WINDOW = 1800  # Defines the number of frames corresponding to one minute
    
        # Iterate over each group of experiments (i.e., each category)
        for group in self.allFileGroupExperiments:
            sumEvents = 0     # Total number of gaze events for this category
            sumFrames = 0     # Total number of frames across all experiments in this category
            events_per_exp = []
            
            # Process each experiment in the current category
            for exp in group:
                loader = exp.pos  # Get positional data loader
    
                # Retrieve gaze event counts for both mice
                countEvents0 = loader.returnNumGazeEvents(0)
                countEvents1 = loader.returnNumGazeEvents(1)
                numFrames = loader.returnNumFrames()
    
                # Only include experiments with complete data
                if countEvents0 is not None and countEvents1 is not None and numFrames is not None:
                    sumEvents += countEvents0 + countEvents1
                    sumFrames += numFrames
                    events_per_exp.append((countEvents0 + countEvents1) / numFrames * FRAME_WINDOW)
    
            # Compute average gaze events per 1800 frames if data is available
            if sumFrames > 0 and events_per_exp:
                group_events.append(events_per_exp)
                avg_events.append(sumEvents / sumFrames * FRAME_WINDOW)
    
        # --- Plot: Bar chart of average gaze events per minute (1800 frames) ---
        plt.figure(figsize=(8, 6))
        plt.bar(range(len(avg_events)), avg_events, color='skyblue')
        
        if group_events is not None:
            for i, cat_data in enumerate(group_events):
                # Generate slight jitter for x-coordinates to avoid overlap
                x_jitter = np.random.normal(i, 0.1, size=len(cat_data))
                plt.scatter(x_jitter, cat_data, color='darkblue', alpha=0.5, s=50, label='Individual Data' if i == 0 else None)
        
        #plt.xlabel('Category')
        plt.ylabel('Avg. Gaze Events per 1800 Frames', fontsize=13)
        plt.title('Average Gaze Events (per Minute) per Category', fontsize=15)
        plt.xticks(range(len(avg_events)), self.categoryNames, fontsize = 13)
        
        # --- Statistical Significance Tests ---
        if (group_events is not None):
            y_max = 0
            for cat_data in group_events:
                y_max = max(y_max, max(cat_data))
        else: 
            y_max = max(avg_events)
        
        plt.ylim(0, y_max * 1.13)  # Adjust ylim so text doesn't overlap title

        if self.numCategories == 2:
            # Mann-Whitney U test
            stat, p = mannwhitneyu(group_events[0], group_events[1], alternative='two-sided')
            plt.text(0.5, y_max * 1.05, f"Mann-Whitney U: p = {p:.3g}", ha='center', fontsize=13)
            # Optional: line between the bars
            #plt.plot([0, 1], [y_max * 1.1, y_max * 1.1], color='black', lw=1.2)
        
        elif self.numCategories > 2:
            # Kruskal-Wallis test
            stat, p = kruskal(*group_events)
            plt.text(len(avg_events)/2 - 0.5, y_max * 1.05, f"Kruskal-Wallis: p = {p:.3g}", ha='center', fontsize=12)

        plt.tight_layout()
        
        # Save and display the plot
        if (self.save):
            plt.savefig(f'{self.path}GazeEventsPerMinute{self.endSaveName}')
        plt.show()
        plt.close()
    
        # --- Plot 2: Cumulative gaze events over time ---
        '''plt.figure(figsize=(10, 6))
        for idx, cum_series in enumerate(cum_event_data):
            plt.plot(cum_series, label=self.categoryNames[idx])
        plt.xlabel('Frame')
        plt.ylabel('Cumulative Gaze Events')
        plt.title('Cumulative Gaze Events Over Time per Category')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.path}GazeEventsoverTime{self.endSaveName}')
        plt.show()
        plt.close()'''

    def compareSuccesfulTrials(self):
        '''
        What? – Displays the comparison between the percent of trials that result in a cooperative 
        success for each of the categories inputted
              - Displays each individual data point for each file on top of the percentage bar
        '''
        # Initialize lists to store average success probabilities and individual datapoints per category
        probs = []
        individual_datapoints = []
        datapoint_colors = []
    
        # Iterate through each experimental group (category)
        for i, group in enumerate(self.allFileGroupExperiments):
            print("\n\n\nCategory: ", self.categoryNames[i])
            print("Quantity: ", len(self.allFileGroupExperiments[i]))
            
            individual_datapoints.append([])  # Holds datapoints for this category
            datapoint_colors.append([])  # Holds datapoints for this category
            totalSucc = 0
            totalTrials = 0
    
            # Iterate through each experiment in the group
            for j, exp in enumerate(group):
                loader = exp.lev
                
                # Add to totals for computing the average success rate across the category
                num_succ = loader.returnNumSuccessfulTrials()
                num_total = loader.returnNumTotalTrials()
    
                totalSucc += num_succ
                totalTrials += num_total
    
                # Store individual success probability for this experiment
                if num_total > 0:
                    #print("\n\nProb: ",  num_succ / num_total)
                    #print("Num Trials: ", num_total)
                    #print("Lev File: ", self.allFileGroupExperiments[i][j].lev_file)
                    individual_datapoints[i].append(num_succ / num_total)
                    
                    # Assign color based on threshold
                    thresh = loader.returnSuccThreshold()
                    if thresh > 3:
                        color = 'red'
                    elif thresh > 2:
                        color = 'orange'
                    elif thresh > 1:
                        color = 'blue'
                    elif thresh > 0:
                        color = 'black'
                    else:
                        color = 'gray'
                    datapoint_colors[i].append(color)
                    print(f"\nSuccess Rate is: {num_succ/num_total}")
                    print(f"Lev is: {exp.lev_file}")
                    
                else:
                    individual_datapoints[i].append(np.nan)
                    datapoint_colors[i].append('gray')
                    print("\nTotal Trials was 0")
                    print("Lev File: ", self.allFileGroupExperiments[i][j][1])
    
            # Compute overall success probability for the category
            prob = totalSucc / totalTrials if totalTrials > 0 else 0
            prob2 = np.mean(individual_datapoints[i])
            probs.append(prob2)
    
        # --- Plotting ---
        plt.figure(figsize=(8, 6))
    
        # Bar plot for average success probability per category
        bar_positions = range(len(probs))
        plt.bar(bar_positions, probs, color='green', label='Category Average')
    
        # Overlay individual datapoints as scatter plot
        for i, datapoints in enumerate(individual_datapoints):
            jittered_x = [i + (np.random.rand() - 0.5) * 0.2 for _ in datapoints]
            for x, y, c in zip(jittered_x, datapoints, datapoint_colors[i]):
                plt.scatter(x, y, color=c, alpha=0.7, s=40)
        '''for i, datapoints in enumerate(individual_datapoints):
            jittered_x = [i + (np.random.rand() - 0.5) * 0.2 for _ in datapoints]  # Add slight x jitter
            if i == 0:
                plt.scatter(jittered_x, datapoints, color='black', alpha=0.7, s=40, label='Individual Data')
            else:
                plt.scatter(jittered_x, datapoints, color='black', alpha=0.7, s=40)'''
    
        # Formatting
        #plt.xlabel('Category')
        plt.ylabel('Probability of Successful Trials')
        plt.title('Success Probability per Category')
        plt.xticks(bar_positions, self.categoryNames)
        plt.ylim(0, 1)
        '''legend_patches = [
            Patch(color='red', label='Threshold > 3'),
            Patch(color='orange', label='Threshold > 2'),
            Patch(color='blue', label='Threshold > 1'),
            Patch(color='black', label='Threshold > 0'),
            Patch(color='gray', label='Threshold ≤ 0')
        ]
        plt.legend(handles=legend_patches)'''
        plt.legend()
        plt.tight_layout()
    
        # Save and display the plot
        if (self.save):
            plt.savefig(f'{self.path}ProbofSuccesfulTrial_{self.endSaveName}')
        plt.show()
        plt.close()
        
    def compareIPI(self):
        """
        Function Purpose:
        -----------------
        This function computes and visualizes average inter-press intervals (IPIs) and success-related timing metrics 
        across multiple experimental categories. Specifically, it generates bar plots for:
        1. Average IPI across all lever presses.
        2. Average time between the first press and a successful trial.
        3. Average time between the last press and a successful trial.
        
        It aggregates these metrics across all files in each category, accounting for potential division-by-zero cases.
        """
        
        # Initialize lists to store per-category averages
        avg_ipi = []         # Average IPI per category
        avg_first_to = []    # Average time from first press to success per category
        avg_last_to = []     # Average time from last press to success per category
        
        # Iterate over each experimental group (i.e., each category)
        for group in self.allFileGroupExperiments:
            # Initialize cumulative totals for this group
            totalPresses = 0
            totalSucc = 0
            totalFirsttoSuccTime = 0
            totalLasttoSuccTime = 0
            totalIPITime = 0
        
            # Process each experiment in the group
            for exp in group:
                loader = exp.lev  # Get lever loader for this experiment
        
                # Get IPI stats and total lever presses
                ipiSum = loader.returnAvgIPI()
                numPresses = loader.returnTotalLeverPresses()
                
                # Accumulate total IPI time only if valid data exists
                if ipiSum is not None and numPresses > 0:
                    totalIPITime += ipiSum * numPresses  # Weighted sum
                    totalPresses += numPresses
        
                # Get success-related stats
                succ = loader.returnNumSuccessfulTrials()
                avgIPIFirst_to_Sucess = loader.returnAvgIPI_FirsttoSuccess()
                avgIPILast_to_Sucess = loader.returnAvgIPI_LasttoSuccess()
        
                # Accumulate total time from first/last press to success
                totalFirsttoSuccTime += succ * avgIPIFirst_to_Sucess
                totalLasttoSuccTime += succ * avgIPILast_to_Sucess
                totalSucc += succ
        
            # Compute per-category averages, handling division by zero
            avg_ipi.append(totalIPITime / totalPresses if totalPresses > 0 else 0)
            avg_first_to.append(totalFirsttoSuccTime / totalSucc if totalSucc > 0 else 0)
            avg_last_to.append(totalLasttoSuccTime / totalSucc if totalSucc > 0 else 0)
        
        # Define plot titles, corresponding data, and colors
        for title, data, color in zip(
            ['Avg IPI per Category', 'Avg First->Success per Category', 'Avg Last->Success per Category'],
            [avg_ipi, avg_first_to, avg_last_to],
            ['blue', 'skyblue', 'salmon']):
        
            # Create bar plot for the current metric
            plt.figure(figsize=(8, 6))
            plt.bar(range(len(data)), data, color=color)
            plt.xticks(range(len(data)), self.categoryNames)
            plt.xlabel('Category')
            plt.ylabel('Time (s)')
            plt.title(title)
            
            # Save and display the plot
            if (self.save):
                plt.savefig(f'{self.path}{title}{self.endSaveName}')
            plt.show()
            plt.close()
            
    def make_bar_plot(self, data, ylabel, title, saveFileName, individual_data = None):
        plt.figure(figsize=(8, 5))
        x = range(len(data))
        
        # Create bar plot
        plt.bar(x, data, color='skyblue', alpha=0.6, label='Mean')
        
        # Overlay scatter points for individual data if provided
        if individual_data is not None:
            for i, cat_data in enumerate(individual_data):
                # Generate slight jitter for x-coordinates to avoid overlap
                x_jitter = np.random.normal(i, 0.1, size=len(cat_data))
                plt.scatter(x_jitter, cat_data, color='darkblue', alpha=0.5, s=50, label='Individual Data' if i == 0 else None)
        
        plt.xticks(x, self.categoryNames, fontsize = 13)
        plt.ylabel(ylabel, fontsize = 13)
        plt.title(title, fontsize = 15)
        plt.legend()
        
        # --- Statistical Significance Tests ---
        if (individual_data is not None):
            y_max = 0
            for cat_data in individual_data:
                y_max = max(y_max, max(cat_data))
        else: 
            y_max = max(data)
        
        plt.ylim(0, y_max * 1.13)  # Adjust ylim so text doesn't overlap title
        
        if self.numCategories == 2:
            # Mann-Whitney U test
            stat, p = mannwhitneyu(individual_data[0], individual_data[1], alternative='two-sided')
            plt.text(0.5, y_max * 1.05, f"Mann-Whitney U: p = {p:.3g}", ha='center', fontsize=13)
            # Optional: line between the bars
            #plt.plot([0, 1], [y_max * 1.1, y_max * 1.1], color='black', lw=1.2)
        
        elif self.numCategories > 2:
            # Kruskal-Wallis test
            stat, p = kruskal(*individual_data)
            plt.text(len(data)/2 - 0.5, y_max * 1.05, f"Kruskal-Wallis: p = {p:.3g}", ha='center', fontsize=12)
        
        plt.tight_layout()
        if self.save:
            plt.savefig(f'{self.path}{saveFileName}{self.endSaveName}')
        plt.show()
        plt.close()
    
    def printSummaryStats(self):
        avg_gaze_lengths = []       # Stores average gaze duration (in frames) per category
        avg_gaze_lengths_alternate = [] # Stores average gaze duration (in frames) per category for alternate definition
        avg_lever_per_trial = []    # Stores average number of lever presses per trial
        avg_mag_per_trial = []      # Stores average number of magazine entries per trial
    
        percent_gazing = []         # Stores percent frames gazing per category
        percent_gazing_alternate = [] # Stores percent frames gazing (alternate) per category
        individual_gaze_lengths = [] # Stores individual gaze lengths per category
        individual_gaze_lengths_alternate = [] # Stores individual gaze lengths (alternate) per category
        individual_lever_per_trial = [] # Stores individual lever presses per trial
        individual_mag_per_trial = []   # Stores individual magazine entries per trial
        individual_percentGazing_per_trial = []
    
        # Loop through each experimental group (i.e., category)
        for idx, group in enumerate(self.allFileGroupExperiments):
            total_gaze_events = 0     # Total gaze events in the category
            total_gaze_events_alternate = 0     # Total gaze events in the category for alternate definition
            total_frames = 0          # Total number of frames across all sessions
            total_trials = 0          # Total number of trials across sessions
            successful_trials = 0     # Total number of cooperative successful trials
            total_lever_presses = 0   # Total number of lever presses
            total_mag_events = 0      # Total number of magazine entries
            total_gaze_frames = 0     # Total frames where gaze was detected
            total_gaze_frames_alternate = 0
            
            cat_gaze_lengths = []     # Individual gaze lengths for this category
            cat_gaze_lengths_alternate = [] # Individual gaze lengths (alternate) for this category
            cat_lever_per_trial = []  # Individual lever presses per trial
            cat_mag_per_trial = []    # Individual magazine entries per trial
            cat_percentGazing_per_trial = []
                        
            # Process each experiment within the category
            for exp in group:
                loader = exp.pos
                
                g0 = loader.returnIsGazing(0, alternateDef=False)
                g1 = loader.returnIsGazing(1, alternateDef=False)
                
                #print("g0: ", ', '.join(map(str, g0[2000:4000])))
                #print("g1: ", ', '.join(map(str, g1)))
                
                g2 = loader.returnIsGazing(0)
                g3 = loader.returnIsGazing(1)
                
                #print("\n" * 5)
                
                #print("g2 (alternate): ", ', '.join(map(str, g2[2000:4000])))
                #print("g3 (alternate): ", ', '.join(map(str, g3)))
                
                # Count gaze events and sum up the frames with gazing behavior
                curGazeEvents = loader.returnNumGazeEvents(0, alternateDef=False) + loader.returnNumGazeEvents(1, alternateDef=False)
                curGazeFrames = np.sum(g0) + np.sum(g1)
                
                curGazeEventsAlternate = loader.returnNumGazeEvents(0) + loader.returnNumGazeEvents(1)
                curGazeFramesAlternate = np.sum(g2) + np.sum(g3)
                
                curFrames = g0.shape[0]
                
                total_gaze_events += curGazeEvents
                total_gaze_frames += curGazeFrames
                total_frames += curFrames
                
                total_gaze_events_alternate += curGazeEventsAlternate
                total_gaze_frames_alternate += curGazeFramesAlternate
                
                # Calculate individual experiment metrics
                if total_gaze_events > 0:
                    cat_gaze_lengths.append(curGazeFrames / curGazeEvents)
                if total_gaze_events_alternate > 0:
                    cat_gaze_lengths_alternate.append(curGazeFramesAlternate / curGazeEventsAlternate)
                if total_frames > 0:
                    cat_percentGazing_per_trial.append(curGazeFramesAlternate / curFrames * 100) 
                
                # Access lever press data and compute trial/success counts
                lev = exp.lev.data
                trials = lev['TrialNum'].nunique()
                succ = lev.groupby('TrialNum').first().query('coopSucc == 1').shape[0]
                total_trials += trials
                successful_trials += succ
                total_lever_presses += lev.shape[0]
                if trials > 0:
                    cat_lever_per_trial.append(lev.shape[0] / trials)
    
                # Count magazine events
                mag = exp.mag.data
                total_mag_events += mag.shape[0]
                if trials > 0:
                    cat_mag_per_trial.append(mag.shape[0] / trials)
    
            # Calculate averages for the category
            avg_gaze_len = (total_gaze_frames / total_gaze_events) if total_gaze_events > 0 else 0
            avg_gaze_len_alternate = (total_gaze_frames_alternate / total_gaze_events_alternate) if total_gaze_events_alternate > 0 else 0
            avg_lever = (total_lever_presses / total_trials) if total_trials > 0 else 0
            avg_mag = (total_mag_events / total_trials) if total_trials > 0 else 0
            percent_gaze = (100 * total_gaze_frames / total_frames) if total_frames > 0 else 0
            percent_gaze_alt = (100 * total_gaze_frames_alternate / total_frames) if total_frames > 0 else 0
    
            # Store for plotting
            avg_gaze_lengths.append(avg_gaze_len)
            avg_gaze_lengths_alternate.append(avg_gaze_len_alternate)
            avg_lever_per_trial.append(avg_lever)
            avg_mag_per_trial.append(avg_mag)
            percent_gazing.append(percent_gaze)
            percent_gazing_alternate.append(percent_gaze_alt)
            individual_gaze_lengths.append(cat_gaze_lengths)
            individual_gaze_lengths_alternate.append(cat_gaze_lengths_alternate)
            individual_lever_per_trial.append(cat_lever_per_trial)
            individual_mag_per_trial.append(cat_mag_per_trial)
            individual_percentGazing_per_trial.append(cat_percentGazing_per_trial)
    
            # Print summary statistics for the current category
            print(f"\nCategory: {self.categoryNames[idx]}")
            print(f"  Number of Files: {len(group)}")
            print(f"  Total Frames: {total_frames}")
            print(f"  Total Trials: {total_trials}")
            print(f"  Successful Trials: {successful_trials}")
            print(f"  Percent Successful: {successful_trials / total_trials:.2f}")
            print(f"  Frames Gazing: {total_gaze_frames}")
            print(f"  Total Gaze Events: {total_gaze_events}")
            print(f"  Average Gaze Length: {total_gaze_frames / total_gaze_events:.2f}")
            print(f"  Percent Gazing: {100 * total_gaze_frames / total_frames:.2f}%")
            print(f"  Total Gaze Events (Alternate): {total_gaze_events_alternate}")
            print(f"  Average Gaze Length (Alternate): {total_gaze_frames_alternate / total_gaze_events_alternate:.2f}")
            print(f"  Percent Gazing (Alternate): {100 * total_gaze_frames_alternate / total_frames:.2f}%")
            print(f"  Avg Lever Presses per Trial: {total_lever_presses / total_trials:.2f}")
            print(f"  Total Lever Presses: {total_lever_presses}")
            print(f"  Avg Mag Events per Trial: {total_mag_events / total_trials:.2f}")
            print(f"  Total Mag Events: {total_mag_events}")
    
        self.make_bar_plot(
            avg_gaze_lengths,
            'Avg Gaze Length (frames)',
            'Average Gaze Length per Category',
            "Avg_Gaze_Length",
            individual_data=individual_gaze_lengths
        )
        
        self.make_bar_plot(
            avg_gaze_lengths_alternate,
            'Avg Gaze Length (frames)',
            'Average Gaze Length per Category (Alternate Def)',
            "Avg_Gaze_Length_Alternate",
            individual_data=individual_gaze_lengths_alternate
        )
    
        self.make_bar_plot(
            avg_lever_per_trial,
            'Avg Lever Presses per Trial',
            'Lever Presses per Trial per Category',
            "Avg_Lev_Presses_perTrial",
            individual_data=individual_lever_per_trial
        )
    
        self.make_bar_plot(
            avg_mag_per_trial,
            'Avg Mag Events per Trial',
            'Mag Events per Trial per Category',
            "Avg_Mag_Events_perTrial",
            individual_data=individual_mag_per_trial
        )
        
        self.make_bar_plot(
            percent_gazing_alternate,
            'Percent Gazing per Trial',
            'Percent Gazing (Alternate) per Category',
            "Avg_PercentGazing_Alternate_perTrial",
            individual_data=individual_percentGazing_per_trial
        )
        
    def rePressingBehavior(self):
        """
        Plots grouped bar charts for re-pressing behavior across multiple categories.
        1. Average re-presses by the first mouse (across all trials).
        2. Average re-presses by the second mouse in successful trials.
        3. Comparison of re-presses by the first mouse in successful vs. non-successful trials.
        """
        
        print("Starting quantifyRePressingBehavior")
        
        avg_repress_first = []
        avg_repress_second_success = []
        avg_repress_first_success = []
        avg_repress_first_non = []
        
        # Iterate through each experimental group (category)
        for i, group in enumerate(self.allFileGroupExperiments):
            avg_repress_first_temp = []
            avg_repress_second_success_temp = []
            avg_repress_first_success_temp = []
            avg_repress_first_non_temp = []
    
            # Iterate through each experiment in the group
            for j, exp in enumerate(group):
                lev = exp.lev
                avg_repress_first_temp.append(lev.returnAvgRepresses_FirstMouse())
                avg_repress_second_success_temp.append(lev.returnAvgRepresses_SecondMouse_Success())
                success, non_success = lev.returnAvgRepresses_FirstMouse_SuccessVsNon()
                avg_repress_first_success_temp.append(success)
                avg_repress_first_non_temp.append(non_success)
            
            avg_repress_first.append(avg_repress_first_temp)
            avg_repress_second_success.append(avg_repress_second_success_temp)
            avg_repress_first_success.append(avg_repress_first_success_temp)
            avg_repress_first_non.append(avg_repress_first_non_temp)
    
        categories = self.categoryNames

        # --- Plot 1: Avg re-presses by First Mouse across all trials (by category) ---
        avg_first_per_cat = [np.mean(group) if group else 0 for group in avg_repress_first]
        
        plt.figure(figsize=(8, 6))
        plt.bar(categories, avg_first_per_cat, color='steelblue')
        plt.title('Avg Re-Presses by First Mouse (All Trials) by Category', fontsize=14)
        plt.ylabel('Average Re-Presses')
        plt.tight_layout()
        if (self.save):
            plt.savefig(f"{self.path}avg_repress_first_mouse_by_category{self.endSaveName}.png")
        plt.show()
        
        # --- Plot 2: Avg re-presses by Second Mouse in successful trials (by category) ---
        avg_second_success_per_cat = [np.mean(group) if group else 0 for group in avg_repress_second_success]
        
        plt.figure(figsize=(8, 6))
        plt.bar(categories, avg_second_success_per_cat, color='seagreen')
        plt.title('Avg Re-Presses by Second Mouse (Success Only) by Category', fontsize=14)
        plt.ylabel('Average Re-Presses')
        plt.tight_layout()
        if (self.save):
            plt.savefig(f"{self.path}avg_repress_second_mouse_success_by_category{self.endSaveName}.png")
        plt.show()
        
        # --- Plot 3: First Mouse Re-Presses in Success vs. Non-Success Trials (by category) ---
        avg_success_per_cat = [np.mean(group) if group else 0 for group in avg_repress_first_success]
        avg_non_per_cat = [np.mean(group) if group else 0 for group in avg_repress_first_non]
        
        x = np.arange(len(categories))
        width = 0.35
        
        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, avg_success_per_cat, width=width, label='Success', color='green')
        plt.bar(x + width/2, avg_non_per_cat, width=width, label='Non-Success', color='red')
        plt.xticks(x, categories)
        plt.ylabel('Average Re-Presses')
        plt.title('First Mouse Re-Pressing by Category\n(Success vs Non-Success)', fontsize=14)
        plt.legend()
        plt.tight_layout()
        if (self.save):
            plt.savefig(f"{self.path}avg_repress_first_mouse_success_vs_non_by_category{self.endSaveName}.png")
        plt.show()
    
    def gazeAlignmentAngle(self, both_mice = True):
        """
        Computes and plots histograms of gaze-to-body angle alignment for each category.
    
        Parameters:
            both_mice (bool): If True, includes both mouse 0 and 1 from each experiment.
        """
        
        print("Starting compareGazeAlignmentAngleHistogram")

        num_bins = 36  # 0-180 degrees in 5° bins
        histograms = []  # To store histogram arrays for each category
        total_trials_per_category = []  # To store total trials per category
        
        for group in self.allFileGroupExperiments:
            total_hist = np.zeros(num_bins)
            total_trials = 0
            for exp in group:
                pos = exp.pos
                if both_mice:
                    for mouseID in [0, 1]:
                        total_hist += pos.returnGazeAlignmentHistogram(mouseID)
                else:
                    total_hist += pos.returnGazeAlignmentHistogram(mouseID=0)
                total_trials += exp.lev.returnNumTotalTrials()
                
            histograms.append(total_hist)
            total_trials_per_category.append(total_trials)
        
        # Normalize histograms by total trials in each category
        normalized_histograms = []
        for hist, total_trials in zip(histograms, total_trials_per_category):
            if total_trials > 0:
                normalized_hist = hist / total_trials
            else:
                print(f"Warning: Category has zero trials; using unnormalized histogram.")
                normalized_hist = hist
            normalized_histograms.append(normalized_hist)
        
        # Plotting
        bin_centers = np.arange(2.5, 180, 5)  # Centers of 5° bins
        plt.figure(figsize=(12, 7))
    
        # Plot each category's normalized histogram with some transparency
        colors = plt.cm.get_cmap('tab10', len(normalized_histograms))
        for idx, hist in enumerate(normalized_histograms):
            plt.bar(
                bin_centers + idx * 1.2,  # Shift bars slightly for side-by-side grouping
                hist,
                width=1.2,
                alpha=0.7,
                label=self.categoryNames[idx],
                color=colors(idx),
                edgecolor='black'
            )
    
        plt.xlabel("Angle between gaze and TB→HB vector (degrees)")
        plt.ylabel("Average Frames per Trial")
        plt.title("Normalized Gaze-Body Angle Distribution by Category")
        plt.xticks(np.arange(0, 181, 15))
        plt.legend()
        plt.tight_layout()
        if self.save:
            plt.savefig(f"{self.path}Gaze_Alignment_Angle_Histogram{self.endSaveName}.png")
        plt.show()
        plt.close()



#magFiles = [["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_lever.csv"], ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv"]]
#levFiles = [["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_mag.csv"], ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv"]]
#posFiles = [["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G.predictions.h5"], ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5"]]                   

levFiles = [["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv"],
            ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv"]]
magFiles = [["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv"], 
            ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv"]]
posFiles = [["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5"], 
            ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5"]]

'''
levFiles = [["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_lever.csv"], 
            ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv"]]
magFiles = [["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_mag.csv"], 
            ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv"]]
posFiles = [["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G.predictions.h5"], 
            ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5"]]
'''

#categoryExperiments = multiFileGraphsCategories(magFiles, levFiles, posFiles, ["Paired_Testing", "Training_Cooperation"], save=False)
#categoryExperiments.compareGazeEventsCategories()
#categoryExperiments.printSummaryStats()

#categoryExperiments.compareSuccesfulTrials()
#categoryExperiments.rePressingBehavior()
#categoryExperiments.gazeAlignmentAngle()

#Paired Testing vs. Training Cooperation
'''
print("Running Paired Testing vs Training Cooperation")
dataPT = getOnlyPairedTesting()
dataTC = getOnlyTrainingCoop()

levFiles = [dataPT[0], dataTC[0]]
magFiles = [dataPT[1], dataTC[1]]
posFiles = [dataPT[2], dataTC[2]]
categoryExperiments = multiFileGraphsCategories(magFiles, levFiles, posFiles, ["Paired_Testing", "Training_Cooperation"])
#categoryExperiments.compareSuccesfulTrials()
'''

'''
#Unfamiliar vs. Training Partners
print("Running UF vs TP")
dataUF = getOnlyUnfamiliar() #Unfamiliar
dataTP = getOnlyTrainingPartners() #Training Partners

levFiles = [dataUF[0], dataTP[0]]
magFiles = [dataUF[1], dataTP[1]]
posFiles = [dataUF[2], dataTP[2]]
categoryExperiments = multiFileGraphsCategories(magFiles, levFiles, posFiles, ["Unfamiliar", "Training Partners"])
#categoryExperiments.compareSuccesfulTrials()
'''


#Transparent vs. Translucent vs. Opaque
'''
print("Running Transparency")
dataTransparent = getOnlyTransparent() #Transparent
dataTranslucent = getOnlyTranslucent() #Translucent
dataOpaque = getOnlyOpaque() #Opaque

levFiles = [dataTransparent[0], dataTranslucent[0], dataOpaque[0]]
magFiles = [dataTransparent[1], dataTranslucent[1], dataOpaque[1]]
posFiles = [dataTransparent[2], dataTranslucent[2], dataOpaque[2]]
categoryExperiments = multiFileGraphsCategories(magFiles, levFiles, posFiles, ["Transparent", "Translucent", "Opaque"])
#categoryExperiments.compareSuccesfulTrials()
'''

'''
print("0")
categoryExperiments.compareGazeEventsCategories()
print("1")
#categoryExperiments.compareSuccesfulTrials()
print("2")
#categoryExperiments.compareIPI()
print("3")
#categoryExperiments.rePressingBehavior()
print("4")
#categoryExperiments.gazeAlignmentAngle()
print("5")
categoryExperiments.printSummaryStats()
print("Done")
'''

# ---------------------------------------------------------------------------------------------------------


##Class to create Graphs with data sorted by mice pairs
#
#


class MicePairGraphs:
    def __init__(self, magGroups, levGroups, posGroups):
        print("Initializing MicePairGraphs")
        assert len(magGroups) == len(levGroups) == len(posGroups), "Mismatched group lengths."
        self.experimentGroups = []
        self.prefix = "filtered_"
        deleted_count = 0

        for group_idx, (mag_list, lev_list, pos_list) in enumerate(zip(magGroups, levGroups, posGroups)):
            print(f"Creating group {group_idx + 1} for {len(mag_list)} files")
            group_exps = []
        
            for mag_path, lev_path, pos_path in zip(mag_list, lev_list, pos_list):
                exp = singleExperiment(lev_path, mag_path, pos_path)        
                mag_missing = [col for col in exp.mag.categories if col not in exp.mag.data.columns]
                lev_missing = [col for col in exp.lev.categories if col not in exp.lev.data.columns]
        
                if mag_missing or lev_missing:
                    deleted_count += 1
                    print("Skipping experiment due to missing categories:")
                    if mag_missing:
                        print(f"  MagFile missing: {mag_missing}")
                        print(f"  Mag File: {mag_path}")
                    if lev_missing:
                        print(f"  LevFile missing: {lev_missing}")
                        print(f"  Lev File: {lev_path}")
                    continue
        
                group_exps.append(exp)
        
            self.experimentGroups.append(group_exps)

        print(f"Deleted {deleted_count} experiment(s) due to missing categories.")

    def _make_boxplot(self, data, ylabel, title, filename):
        print(f"Creating boxplot: {title}")
        plt.figure(figsize=(5, 5))
        plt.boxplot(data, showfliers=False)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks([])
        plt.tight_layout()
        plt.savefig(f"{self.prefix}{filename}.png")
        plt.show()
        plt.close()

    def _make_histogram(self, data, xlabel, title, filename):
        print(f"Creating histogram: {title}")
        plt.figure(figsize=(10, 5))
        plt.hist(data, bins=20)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f"{self.prefix}{filename}_hist.png")
        plt.show()
        plt.close()

    def boxplot_avg_gaze_length(self):
        print("\n\nGenerating average gaze length boxplot")
        all_vals = []
        for group in self.experimentGroups:
            pair_vals = []
            for exp in group:
                l0 = exp.pos.returnAverageGazeLength(0)
                l1 = exp.pos.returnAverageGazeLength(1)
                print(f"Gaze lengths: {l0}, {l1}")
                if l0 is not None and l1 is not None:
                    pair_vals.append((l0 + l1) / 2)
            if pair_vals:
                all_vals.append(np.mean(pair_vals))
        print(f"All values: {all_vals}")
        self._make_boxplot(all_vals, "Frames per Gaze Event", "Avg Gaze Length per Pair", "Box_Gaze_Length")
        self._make_histogram(all_vals, "Frames per Gaze Event", "Gaze Length Distribution", "Hist_Gaze_Length")

    def boxplot_lever_presses_per_trial(self):
        print("\n\nGenerating lever presses per trial boxplot")
        vals = []
        for group in self.experimentGroups:
            pair_rates = []
            for exp in group:
                trials = exp.lev.returnNumTotalTrials()
                presses = exp.lev.returnTotalLeverPresses()
                print(f"Trials: {trials}, Presses: {presses}")
                if trials > 0:
                    pair_rates.append(presses / trials)
            if pair_rates:
                vals.append(np.mean(pair_rates))
        print(f"All values: {vals}")
        self._make_boxplot(vals, "Presses / Trial", "Lever Presses per Trial", "Box_LeverPerTrial")
        self._make_histogram(vals, "Presses / Trial", "Lever Press Distribution", "Hist_LeverPerTrial")

    def boxplot_mag_events_per_trial(self):
        print("\n\nGenerating mag events per trial boxplot")
        vals = []
        for group in self.experimentGroups:
            pair_rates = []
            for exp in group:
                trials = exp.lev.returnNumTotalTrials()
                mags = exp.mag.getTotalMagEvents()
                print(f"Trials: {trials}, Mag Events: {mags}")
                if trials > 0:
                    pair_rates.append(mags / trials)
            if pair_rates:
                vals.append(np.mean(pair_rates))
        print(f"All values: {vals}")
        self._make_boxplot(vals, "Mag Events / Trial", "Mag Events per Trial", "Box_MagPerTrial")
        self._make_histogram(vals, "Mag Events / Trial", "Mag Event Distribution", "Hist_MagPerTrial")

    def boxplot_avg_IPI(self):
        print("\n\nRunning boxplot_avg_IPI...")
        vals = []
        for group in self.experimentGroups:
            sum_weighted_ipi = 0.0
            sum_presses = 0
            for exp in group:
                mean_ipi = exp.lev.returnAvgIPI()
                n_presses = exp.lev.returnTotalLeverPresses()
                print(f"Avg IPI: {mean_ipi}, Presses: {n_presses}")
                if mean_ipi and n_presses > 0:
                    sum_weighted_ipi += mean_ipi * n_presses
                    sum_presses += n_presses
            print(f"Sum Weighted IPI: {sum_weighted_ipi}, Total Presses: {sum_presses}")
            if sum_presses > 0:
                vals.append(sum_weighted_ipi / sum_presses)
        print(f"Avg IPI per group: {vals}\n")
        self._make_boxplot(vals, "IPI (s)", "Avg Inter-Press Interval", "Box_IPI")
        self._make_histogram(vals, "IPI (s)", "IPI Distribution", "Hist_IPI")

    def boxplot_IPI_first_to_success(self):
        print("\n\nRunning boxplot_IPI_first_to_success...")
        vals = []
        for group in self.experimentGroups:
            sum_weighted = 0.0
            sum_success = 0
            for exp in group:
                v = exp.lev.returnAvgIPI_FirsttoSuccess()
                n_succ = exp.lev.returnNumSuccessfulTrials()
                print(f"First→Success IPI: {v}, Successes: {n_succ}")
                if v is not None and n_succ > 0:
                    sum_weighted += v * n_succ
                    sum_success += n_succ
            if sum_success > 0:
                vals.append(sum_weighted / sum_success)
        print(f"First→Success IPI per group: {vals}\n")
        self._make_boxplot(vals, "Time (s)", "IPI: First→Success", "Box_IPI_First")
        self._make_histogram(vals, "Time (s)", "First→Success Distribution", "Hist_IPI_First")

    def boxplot_IPI_last_to_success(self):
        print("\n\nRunning boxplot_IPI_last_to_success...")
        vals = []
        for group in self.experimentGroups:
            sum_weighted = 0.0
            sum_success = 0
            for exp in group:
                v = exp.lev.returnAvgIPI_LasttoSuccess()
                n_succ = exp.lev.returnNumSuccessfulTrials()
                print(f"Last→Success IPI: {v}, Successes: {n_succ}")
                if v is not None and n_succ > 0:
                    sum_weighted += v * n_succ
                    sum_success += n_succ
            if sum_success > 0:
                vals.append(sum_weighted / sum_success)
        print(f"Last→Success IPI per group: {vals}\n")
        self._make_boxplot(vals, "Time (s)", "IPI: Last→Success", "Box_IPI_Last")
        self._make_histogram(vals, "Time (s)", "Last→Success Distribution", "Hist_IPI_Last")
        
    def boxplot_gaze_events_per_minute(self): 
        print("\n\nRunning boxplot_gaze_events_per_minute...")
        FRAME_WINDOW = 1800
        vals = []
        for group in self.experimentGroups:
            sumEvents = 0
            sumFrames = 0
            for exp in group:
                countEvents0 = exp.pos.returnNumGazeEvents(0)
                countEvents1 = exp.pos.returnNumGazeEvents(1)
                numFrames = exp.pos.returnNumFrames()
                print(f"Gaze0: {countEvents0}, Gaze1: {countEvents1}, Frames: {numFrames}")
                if countEvents0 is not None and countEvents1 is not None and numFrames is not None:
                    sumEvents += countEvents0 + countEvents1
                    sumFrames += numFrames
            if sumFrames > 0:
                rate = sumEvents / sumFrames * FRAME_WINDOW
                print(f"Gaze Rate: {rate}")
                vals.append(rate)
        print(f"Gaze per minute values: {vals}\n")
        self._make_boxplot(vals, "Gaze Events / Min", "Gaze Rate per Pair", "Box_GazePerMin")
        self._make_histogram(vals, "Gaze Events / Min", "Gaze Rate Distribution", "Hist_GazePerMin")
     
    def boxplot_percent_successful_trials(self):
        print("\n\nRunning boxplot_percent_successful_trials...")
        vals = []
        for group in self.experimentGroups:
            sum_tot = 0
            sum_success = 0
            for exp in group:
                tot = exp.lev.returnNumTotalTrials()
                n_succ = exp.lev.returnNumSuccessfulTrials()
                print(f"Total Trials: {tot}, Successful Trials: {n_succ}")
                if n_succ is not None and tot > 0:
                    sum_tot += tot
                    sum_success += n_succ
            if sum_tot > 0:
                ratio = sum_success / sum_tot
                print(f"Success Rate: {ratio}")
                vals.append(ratio)
        print(f"Success rates: {vals}\n")
        self._make_boxplot(vals, "% Success", "Success Rate per Pair", "Box_Success")
        self._make_histogram(vals, "% Success", "Success Rate Distribution", "Hist_Success")
        
    def difference_last_vs_first(self):
        print("\n\n Running difference_last_vs_first...")
        
        def gaze_length(exp):
            df = exp.pos
            countEvents0 = df.returnNumGazeEvents(0)
            countEvents1 = df.returnNumGazeEvents(1)
            avgGaze0 = df.returnAverageGazeLength(0)
            avgGaze1 = df.returnAverageGazeLength(1)
            return (avgGaze0 * countEvents0 + avgGaze1 * countEvents1) / (countEvents0 + countEvents1)
        
        def lev_rate(exp):
            df = exp.lev
            return df.returnLevPressesPerTrial()
        
        def mag_rate(exp):
            df = exp.mag
            return df.getTotalMagEvents() / exp.lev.returnNumTotalTrials()
    
        def percent_success(exp):
            df = exp.lev
            return df.returnNumSuccessfulTrials() / df.returnNumTotalTrials()
    
        def gaze_events_per_min(exp):
            FRAME_WINDOW = 1800  # Normalization constant for converting to events/minute

            df = exp.pos
            countEvents0 = df.returnNumGazeEvents(0)
            countEvents1 = df.returnNumGazeEvents(1)
            numFrames = df.returnNumFrames()
            
            return (countEvents0 + countEvents1) / numFrames * FRAME_WINDOW
    
        def avg_ipi(exp):
            lev = exp.lev
            return lev.returnAvgIPI()
    
        def first_to_success(exp):
            lev = exp.lev
            return lev.returnAvgIPI_FirsttoSuccess()
    
        def last_to_success(exp):
            lev = exp.lev
            return lev.returnAvgIPI_LasttoSuccess()
    
        metrics = {
            "Gaze Length": gaze_length,
            "Lever Rate": lev_rate,
            "Mag Rate": mag_rate,
            "% Success": percent_success,
            "Gaze Rate": gaze_events_per_min,
            "Avg IPI": avg_ipi,
            "First → Success": first_to_success,
            "Last → Success": last_to_success
        }
    
        diffs = {name: [] for name in metrics}
    
        for idx, group in enumerate(self.experimentGroups):
            if len(group) < 5:
                print(f"Skipping group {idx}: only {len(group)} session(s)")
                continue
            first, second, second_last, last = group[0], group[1], group[-2], group[-1]
            for name, func in metrics.items():
                try:
                    v1a, v1b, v2a, v2b = func(first), func(second), func(second_last), func(last)
                    #print("v1a: ", v1a)
                    #print("v2a: ", v2a)
                    
                    v1 = np.mean([v1a, v1b])
                    v2 = np.mean([v2a, v2b])
                    #print("v1: ", v1)
                    #print("v2: ", v2)
                    
                    if v1 is not None and v2 is not None:
                        diffs[name].append(v2 - v1)
                except:
                    continue
                
        #Plot individual histograms
        print("Generating histograms for session differences...")

        print("Diffs.item(): ")
        print(diffs.items())
        
        for name, values in diffs.items():
            print(f"  [Histogram] Metric: {name} — Number of values: {len(values)}")
            plt.figure(figsize=(10, 4))
            plt.hist(values, bins=15)
            plt.title(f"Change in {name} (Last 2 - First 2)")
            plt.xlabel(f"Δ {name}")
            plt.tight_layout()
            filename = f"{self.prefix}Diff_2_{name.replace(' ', '_')}.png"
            print(f" Saving histogram to {filename}")
            plt.savefig(filename)
            plt.show()
            plt.close()    
        
        # Plot individual bar graphs
        print("\nGenerating bar plots for average session differences...")
        for name, values in diffs.items():
            print(f"  [Bar] Metric: {name}")
            if not values:
                print(f"    Skipping {name} — No data.")
                continue
            
            avg_diff = np.mean(values)
            error = np.std(values) / np.sqrt(len(values))
        
            plt.figure(figsize=(5, 6))
            
            # Plot the bar
            bar = plt.bar([0], [avg_diff], width=0.3, yerr=[error], capsize=10, 
                          color='lightgreen', edgecolor='black')
            
            # Plot individual scatter points
            x_jittered = np.random.normal(0, 0.04, size=len(values))  # Small jitter around x=0
            plt.scatter(x_jittered, values, color='black', zorder=3, label='Individual values')
            
            # Draw horizontal line at 0
            plt.axhline(0, color='gray', linestyle='--', linewidth=1)
            
            # Set x-ticks to the metric name
            plt.xticks([0], [name], fontsize=10)
            
            # Add vertical padding to y-limits based on both bar and individual points
            all_y = values + [avg_diff - error, avg_diff + error]
            ymin = min(all_y)
            ymax = max(all_y)
            yrange = ymax - ymin if ymax > ymin else 1.0  # Prevent zero-range
            plt.ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
            
            plt.xlim(-0.4, 0.4)
            
            # Clean labels and layout
            plt.ylabel(f"Δ {name} (Last 2 - First 2)", fontsize=12)
            plt.title(f"Avg Change in {name}", fontsize=14)
            plt.tight_layout()
        
            filename = f"{self.prefix}Bar_Change_2_{name.replace(' ', '_').replace('→', 'to')}.png"
            print(f"    Saving bar plot to {filename}")
            plt.savefig(filename)
            plt.show()
            plt.close()
        
        #Plot Individual Line Graphs
        print("\nPreparing line graphs showing metric progression across sessions...")

        
        # Track values across sessions per metric
        max_sessions = max(len(group) for group in self.experimentGroups)
        print(f"  Max sessions in any group: {max_sessions}")
        
        metric_over_sessions = {name: [[] for _ in range(max_sessions)] for name in metrics}
        
        # Fill values by session
        for idx, group in enumerate(self.experimentGroups):
            print(f"  [Group {idx}] Processing {len(group)} sessions")
            for i, exp in enumerate(group):
                for name, func in metrics.items():
                    try:
                        val = func(exp)
                        if val is not None:
                            metric_over_sessions[name][i].append(val)
                            print(f"    [Session {i}] {name}: {val}")
                        else:
                            print(f"    [Session {i}] {name}: None")
                    except Exception as e:
                        print(f"Error computing {name} for session {i} in group: {e}")
                        continue  # Optional: track or log failures more thoroughly
        
        # Average and plot averages per session
        print("\nPlotting average progression for each metric...")
        for name, session_lists in metric_over_sessions.items():
            averages = [np.mean(vals) if vals else None for vals in session_lists]
            counts = [len(vals) for vals in session_lists]
        
            # Only plot sessions with valid averages
            session_indices = [i+1 for i, v in enumerate(averages) if v is not None]
            y_values = [v for v in averages if v is not None]
            y_counts = [counts[i] for i, v in enumerate(averages) if v is not None]
        
            if not y_values:
                print(f"  Skipping {name} — No valid session data.")
                continue
            
            print(f"  Plotting {name} — {len(y_values)} sessions with valid data")
            
            plt.figure(figsize=(6, 4))
            plt.plot(session_indices, y_values, marker='o', linestyle='-', color='steelblue')
        
            # Optional: Annotate with n (sample size)
            for x, y, n in zip(session_indices, y_values, y_counts):
                plt.text(x, y, f"n={n}", fontsize=8, ha='center', va='bottom')
        
            plt.title(f"Avg {name} Over Sessions")
            plt.xlabel("Session Index")
            plt.ylabel(name)
            plt.grid(True)
            plt.tight_layout()
        
            filename = f"{self.prefix}Line_Progression_{name.replace(' ', '_').replace('→', 'to')}.png"
            print(f"    Saving line plot to {filename}")
            plt.savefig(filename)
            plt.show()
            plt.close()
    

groupMicePairs = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/group_mice_pairs.csv"

def getGroupMicePairs():
    fe = fileExtractor(groupMicePairs)
    return [fe.getLevsDatapath(grouped = True), fe.getMagsDatapath(grouped = True), fe.getPosDatapath(grouped = True)]


#data = getGroupMicePairs()
#pairGraphs = MicePairGraphs(data[0], data[1], data[2])


'''magFiles = [["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv"],
            ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv"]]
levFiles = [["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv"], 
            ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv"]]
posFiles = [["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5"], 
            ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5"]]

pairGraphs = MicePairGraphs(magFiles, levFiles, posFiles)'''


'''pairGraphs.boxplot_avg_gaze_length()
pairGraphs.boxplot_lever_presses_per_trial()
pairGraphs.boxplot_mag_events_per_trial()
pairGraphs.boxplot_percent_successful_trials()
pairGraphs.boxplot_gaze_events_per_minute()
pairGraphs.boxplot_avg_IPI()
pairGraphs.boxplot_IPI_first_to_success()
pairGraphs.boxplot_IPI_last_to_success()'''
#pairGraphs.difference_last_vs_first()



# ---------------------------------------------------------------------------------------------------------




#Class to create Graphs with data from multiple files + Functions to get data from all the different categories
#
#


class multiFileGraphs:
    def __init__(self, magFiles: List[str], levFiles: List[str], posFiles: List[str], fpsList: List[int], totFramesList: List[int], initialNanList: List[int], fiberFiles = None, prefix = "", save = True):
        self.experiments = []
        self.prefix = prefix
        self.save = save
        self.NUM_BINS = 30 # Number of time bins for trial chunking
        self.labelSize = 13
        self.titleSize = 15
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
                exp = singleExperiment(magFiles[i], levFiles[i], posFiles[i], fpsList[i], totFramesList[i], initialNanList[i])
            mag_missing = [col for col in exp.mag.categories if col not in exp.mag.data.columns]
            lev_missing = [col for col in exp.lev.categories if col not in exp.lev.data.columns]
            
            #print("mag.categories: ", exp.mag.categories)
            #print("lev.categories: ", exp.lev.categories)
            
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
    
    def magFileDataAvailabilityGraph(self):
        # Expected column structure
        expected_cats = self.experiments[0].mag.categories
    
        # Filter only experiments with correct columns
        filtered_experiments = []
        for exp in self.experiments:
            actual_cats = list(exp.mag.data.columns)
            if actual_cats == expected_cats:
                filtered_experiments.append(exp)
            #else:
                #print(f"Excluding {exp.mag.filename} due to mismatched columns.\nExpected: {expected_cats}\nGot: {actual_cats}")
    
        if not filtered_experiments:
            raise ValueError("No experiments with matching mag columns were found.")
    
        # Optional: update self.experiments to only valid ones
        self.experiments = filtered_experiments
    
        # Compute total rows and initialize null counter
        total_rows = sum(exp.mag.getNumRows() for exp in filtered_experiments)
        nulls_per_cat = {cat: 0 for cat in expected_cats}
    
        # Count nulls
        for exp in filtered_experiments:
            for cat in expected_cats:
                nulls_per_cat[cat] += exp.mag.countNullsinColumn(cat)
    
        # Compute non-null percentages
        pct = [(total_rows - nulls_per_cat[cat]) / total_rows * 100 for cat in expected_cats]
    
        # Plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(expected_cats, pct, color='steelblue')
        plt.xlabel('Categories')
        plt.ylabel('Percentage of Non-Null Data (%)')
        plt.title('Aggregated Data Availability in Mag Files')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 100)
        for bar in bars:
            y = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, y - 5, f'{y:.1f}%', ha='center', va='bottom')
        plt.tight_layout()
        plt.show()
        plt.savefig('mag_data_availability.png')
        plt.close()
        
    def levFileDataAvailabilityGraph(self):
        # Expected column structure
        expected_cats = self.experiments[0].lev.categories
    
        # Filter only experiments with correct columns
        filtered_experiments = []
        for exp in self.experiments:
            actual_cats = list(exp.lev.data.columns)
            if actual_cats == expected_cats:
                filtered_experiments.append(exp)
            #else:
                #print(f"Excluding {exp.lev.filename} due to mismatched columns.\nExpected: {expected_cats}\nGot: {actual_cats}")
    
        if not filtered_experiments:
            raise ValueError("No experiments with matching mag columns were found.")
    
        # Optional: update self.experiments to only valid ones
        self.experiments = filtered_experiments
    
        # Compute total rows and initialize null counter
        total_rows = sum(exp.lev.getNumRows() for exp in filtered_experiments)
        nulls_per_cat = {cat: 0 for cat in expected_cats}
    
        # Count nulls
        for exp in filtered_experiments:
            for cat in expected_cats:
                nulls_per_cat[cat] += exp.lev.countNullsinColumn(cat)
    
        # Compute non-null percentages
        pct = [(total_rows - nulls_per_cat[cat]) / total_rows * 100 for cat in expected_cats]
    
        # Plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(expected_cats, pct, color='steelblue')
        plt.xlabel('Categories')
        plt.ylabel('Percentage of Non-Null Data (%)')
        plt.title('Aggregated Data Availability in Lev Files')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0,100)
        for bar in bars:
            y = max(bar.get_height(), 6)
            plt.text(bar.get_x()+bar.get_width()/2, y-5, f'{y:.1f}%', ha='center')
        plt.tight_layout()
        plt.show()
        plt.savefig('lev_data_availability.png')
    
    def interpressIntervalPlot(self):
        '''
        
        '''
        
        # Initialize lists to store per-category averages
        avg_ipi = 0         # Average IPI
        avg_first_to = 0    # Average time from first press to success
        avg_last_to = 0     # Average time from last press to success
        
        # Initialize cumulative totals for this group
        totalPresses = 0
        totalSucc = 0
        totalFirsttoSuccTime = 0
        totalLasttoSuccTime = 0
        totalIPITime = 0
        
        for exp in self.experiments:
            loader = exp.lev  # Get lever loader for this experiment
    
            # Get IPI stats and total lever presses
            ipiSum = loader.returnAvgIPI()
            numPresses = loader.returnAvgIPI(returnLen = True)
            
            # Accumulate total IPI time only if valid data exists
            if ipiSum is not None and numPresses > 0:
                totalIPITime += ipiSum * numPresses  # Weighted sum
                totalPresses += numPresses
    
            # Get success-related stats
            succ = loader.returnNumSuccessfulTrials()
            avgIPIFirst_to_Sucess = loader.returnAvgIPI_FirsttoSuccess()
            avgIPILast_to_Sucess = loader.returnAvgIPI_LasttoSuccess()
    
            # Accumulate total time from first/last press to success
            totalFirsttoSuccTime += succ * avgIPIFirst_to_Sucess
            totalLasttoSuccTime += succ * avgIPILast_to_Sucess
            totalSucc += succ
        
        if (numPresses > 0):
            avg_ipi = totalIPITime / numPresses
            
        if (totalSucc > 0):
            avg_first_to = totalFirsttoSuccTime / totalSucc
            avg_last_to = totalLasttoSuccTime / totalSucc
        
        # Create figure and twin axes
        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax2 = ax1.twinx()
    
        # X-axis bar positions
        x = [0, 1, 2]
    
        # Plot Avg IPI on left axis
        ax1.bar(x[0], avg_ipi, width=0.4, color='blue', label='Avg IPI')
        ax1.set_ylabel('Avg IPI (s)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Avg IPI', 'First→Success', 'Last→Success'])
    
        # Plot First→Success and Last→Success on right axis
        ax2.bar(x[1], avg_first_to, width=0.4, color='skyblue', label='First→Success')
        ax2.bar(x[2], avg_last_to, width=0.4, color='salmon', label='Last→Success')
        ax2.set_ylabel('Success Timing (s)', color='darkred')
        ax2.tick_params(axis='y', labelcolor='darkred')
    
        # Title and legend
        plt.title('Inter-Press Interval and Success Timing Metrics')
    
        # Combine legends from both axes
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right')
    
        # Save and show
        plt.tight_layout()
        plt.savefig(f'{self.prefix}interpress_metrics_dualaxis')
        plt.show()
        plt.close()
      
    def percentSuccesfulTrials(self):
        #Incorrect I think, REDO
        
        all_lev = pd.concat([exp.lev.data for exp in self.experiments], ignore_index=True)
        
        grouped = all_lev.groupby("TrialNum")
        
        totalTrials, countSuccess = 0, 0
        for trial_num, trial_data in grouped:
            if (trial_data.iloc[0]['coopSucc'] == 1):
                countSuccess += 1
            
            totalTrials += 1
        
        countFail = totalTrials - countSuccess
        labels = ['Successful', 'Unsuccessful']
        sizes = [countSuccess, countFail]
        colors = ['green', 'red']
    
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 14})
        plt.title('Successful Cooperative Trials (%)', fontsize=16)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.tight_layout()
        plt.show()
        plt.savefig(f'{self.prefix}PercentSuccessful.png')
        plt.close()

    def mouseIDFirstPress(self):
        countRat0 = 0
        countRat1 = 0
        
        for exp in self.experiments:
            #print(countRat0)
            levLoader = exp.lev
            expRes = levLoader.returnRatFirstPress()
            countRat0 += expRes[0]
            countRat1 += expRes[1]
        
        #print(countRat0)
        #print(countRat1)
        
        labels = ['Rat 0', 'Rat 1']
        counts = [countRat0, countRat1]
        
        # Pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title('First Press Distribution Between Rats')
        plt.axis('equal')  # Equal aspect ratio ensures the pie is a circle
        plt.savefig('PercentFirstPressbyRat.png')
        plt.show()
        
    def compareGazeEventsbyRat(self):
       countRat0 = 0
       countRat1 = 0
       
       for exp in self.experiments:
           posLoader = exp.pos
           countRat0 += posLoader.returnNumGazeEvents(0)
           countRat1 += posLoader.returnNumGazeEvents(1)
       
       #print(countRat0)
       #print(countRat1) 
       
       labels = ['Rat 0', 'Rat 1']
       counts = [countRat0, countRat1]
       
       # Pie chart
       plt.figure(figsize=(6, 6))
       plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
       plt.title('Gaze Distribution Between Rats')
       plt.axis('equal')  # Equal aspect ratio ensures the pie is a circle
       plt.savefig("PercentGazingbyRat.png")
       plt.show()
                 
    def quantifyRePressingBehavior(self):
        """
        Generates graphs to quantify re-pressing behavior:
        1. Average re-presses by the first mouse (across all trials).
        2. Average re-presses by the second mouse in successful trials.
        3. Comparison of re-presses by the first mouse in successful vs. non-successful trials.
        """
        
        print("Starting quantifyRePressingBehavior")
        
        avg_repress_first = []
        avg_repress_second_success = []
        avg_repress_first_success = []
        avg_repress_first_non = []
    
        # Collect re-pressing data from each experiment
        for exp in self.experiments:
            lev = exp.lev
            avg_repress_first.append(lev.returnAvgRepresses_FirstMouse())
            avg_repress_second_success.append(lev.returnAvgRepresses_SecondMouse_Success())
            success, non_success = lev.returnAvgRepresses_FirstMouse_SuccessVsNon()
            avg_repress_first_success.append(success)
            avg_repress_first_non.append(non_success)
            
    
        # --- Plot 1: Avg re-presses by First Mouse across all trials ---
        overall_first_avg = sum(avg_repress_first) / len(avg_repress_first) if avg_repress_first else 0
    
        plt.figure(figsize=(6, 6))
        plt.bar(x=[0], height=[overall_first_avg], width=0.4, color='steelblue')
        plt.xticks([0], ['First Mouse'])
        # Adjust the x-axis limits to create space around the bar
        plt.xlim(-0.5, 0.5)
        plt.title('Average Re-Presses by First Mouse (All Trials)', fontsize=14)
        plt.ylabel('Average Re-Presses')
        plt.tight_layout()
        plt.savefig(f"{self.prefix}avg_repress_first_mouse.png")
        plt.show()
    
        # --- Plot 2: Avg re-presses by Second Mouse in successful trials ---
        overall_second_success_avg = sum(avg_repress_second_success) / len(avg_repress_second_success) if avg_repress_second_success else 0
    
        plt.figure(figsize=(6, 6))
        plt.bar(x=[0], height=[overall_second_success_avg], width=0.4, color='seagreen')
        plt.xticks([0], ['Second Mouse (Success Only)'])
        # Adjust the x-axis limits to create space around the bar
        plt.xlim(-0.5, 0.5)
        plt.title('Average Re-Presses by Second Mouse (Successful Trials)', fontsize=14)
        plt.ylabel('Average Re-Presses')
        plt.tight_layout()
        plt.savefig(f"{self.prefix}avg_repress_second_mouse_success.png")
        plt.show()
    
        # --- Plot 3: First Mouse Re-Presses in Success vs. Non-Success Trials ---
        overall_success_avg = sum(avg_repress_first_success) / len(avg_repress_first_success) if avg_repress_first_success else 0
        overall_non_avg = sum(avg_repress_first_non) / len(avg_repress_first_non) if avg_repress_first_non else 0
        overall_combined_avg = (overall_success_avg + overall_non_avg) / 2  # Or use avg_repress_first instead if preferred
    
        labels = ['Success', 'Non-Success', 'Overall']
        values = [overall_success_avg, overall_non_avg, overall_combined_avg]
        colors = ['green', 'red', 'gray']
    
        plt.figure(figsize=(7, 6))
        plt.bar(labels, values, color=colors)
        plt.title('First Mouse Re-Pressing:\nSuccess vs Non-Success vs Overall', fontsize=14)
        plt.ylabel('Average Re-Presses')
        plt.tight_layout()
        plt.savefig(f"{self.prefix}avg_repress_first_mouse_success_vs_non.png")
        plt.show()
        plt.close()

    def gazeAlignmentAngleHistogram(self, both_mice=True):
        """
        Computes and plots a combined histogram of gaze-to-body angle alignment
        across all experiments.
    
        Parameters:
            both_mice (bool): If True, includes both mouse 0 and 1 from each experiment.
        """
        print("Starting gazeAlignment Angle Histogram")
        
        total_hist = np.zeros(36)  # 36 bins for 0–180 degrees in 5° intervals
    
        for exp in self.experiments:
            pos = exp.pos  # posLoader object
            if both_mice:
                for mouseID in [0, 1]:
                    total_hist += pos.returnGazeAlignmentHistogram(mouseID)
            else:
                total_hist += pos.returnGazeAlignmentHistogram(mouseID=0)
        
        # Plot
        bin_centers = np.arange(2.5, 180, 5)  # Centers of 5° bins
        plt.figure(figsize=(10, 6))
        plt.bar(bin_centers, total_hist, width=5, color='mediumseagreen', edgecolor='black')
        plt.xlabel("Angle between gaze and TB→HB vector (degrees)")
        plt.ylabel("Total Frame Count")
        plt.title("Gaze-Body Angle Distribution Across All Experiments")
        plt.xticks(np.arange(0, 181, 15))
        plt.tight_layout()
        plt.savefig("Gaze_Alignment_Angle_Histogram.png")
        plt.show()
        plt.close()
        
    def rePressingbyDistance(self):
        #Graph 1: RePressing by Region
        #use posLoader.returnMouseLocation to get the location of the mouse at eachFrame. 
        #Then using abs time in levLoader and posLoader.returnFPS() find which frame the first press happened at for each trial. 
        #Then quantify the average re-presses by using levLoader.returnAvgRepresses_FirstMouse(returnArr = True) and make a bar graph displaying the average represses by location at first press
        
        
        #Graph 2: RePressing by Dsitance 
        #Find a way to quantify the rePressing by distance by using posLoader.returnInterMouseDistance and the functions above
        '''
        Quantify the rePressing Behavior of the rats given the location of the other mouse. 
        '''
        
        # Initialize containers to hold repressing values for each location and distance
        location_dict = {'lev_other': [], 'lev_same': [], 'mid': [], 'mag_other': [], 'mag_same': [], 'other': []}
        distance_list = []  # Inter-mouse distance at first press
        repress_list = []   # Avg number of re-presses by first mouse in each trial
        
        location_counts = {key: 0 for key in location_dict}
        
        for exp in self.experiments:
            pos = exp.pos
            lev = exp.lev
            
            #Data we Keep Track of
            fps = exp.fps
            num_trials = lev.returnNumTotalTrialswithLeverPress()
            first_press_times = lev.returnFirstPressAbsTimes()
            first_press_ids = lev.returnRatIDFirstPressTrial()
            represses = lev.returnAvgRepresses_FirstMouse(returnArr=True)
            inter_mouse_dist = pos.returnInterMouseDistance()
            locationPresser = lev.returnRatLocationFirstPressTrial()
    
            # Skip if mismatch in data length
            if len(first_press_times) != len(represses):
                print("Mismatch between number of trials and repress array")
                continue
            
            if (len(first_press_times) != num_trials):
                print("Mismatch between number of trials and repress array")
                print(num_trials)
                print(len(first_press_times))
                continue
            
            #Iterate through Trials to classify location
            for i in range(num_trials):
                press_time = first_press_times[i]
                rat_id_val = first_press_ids.iloc[i]
                
                if math.isnan(rat_id_val):
                    # Skip or handle this trial because no first press rat ID
                    continue
            
                rat_id = int(rat_id_val)
                #print(f"rat_id: {rat_id} ({type(rat_id)})")
                #print(f"1 - rat_id: {1 - rat_id} ({type(1 - rat_id)})")
                
                if np.isnan(press_time) or i >= len(represses):
                    continue
    
                press_frame = int(press_time * fps)
                if press_frame >= len(inter_mouse_dist):
                    continue
    
                locationOther = pos.returnMouseLocation(1 - rat_id)[press_frame]
                locationRat = pos.returnMouseLocation(rat_id)[press_frame]
                pressLocation = ""
                
                if (locationPresser.iloc[i] == 1):
                    pressLocation = "lev_bottom"
                                        
                elif(locationPresser.iloc[i] == 2):
                    pressLocation = "lev_top"
                else:
                    print("Incorrect LeverNum Data")
                
                if (locationRat != pressLocation):
                    print("Mismatch between locationRat and pressLocation")
                    print("locationRat: ", locationRat)
                    print("pressLocation: ", pressLocation)
                else:
                    print("Correct Location")
                
                if (pressLocation == locationOther):
                    locationOther = "lev_same"
                
                elif (locationOther == "lev_top" or locationOther == "lev_bottom"):
                    locationOther = "lev_other"
                
                elif ((locationOther == "mag_top" and pressLocation == "lev_top") or (locationOther == "mag_bottom" and pressLocation == "lev_bottom")):
                    locationOther = "mag_same"
                
                elif (locationOther == "mag_top" or locationOther == "mag_bottom"):
                    locationOther = "mag_other"
                
                if locationOther in location_dict:
                    location_dict[locationOther].append(represses[i])
                    location_counts[locationOther] += 1
                
                # Collect distance and repressing for scatterplot
                distance_list.append(inter_mouse_dist[press_frame])
                repress_list.append(represses[i])
    
        # === Graph 1: Bar graph of average represses per region ===
        avg_represses_by_region = {
            region: np.mean(vals) if vals else 0
            for region, vals in location_dict.items()
        }
        
        region_counts = {
            region: len(vals)
            for region, vals in location_dict.items()
        }
        
        # Compute standard deviation (or use scipy.stats.sem for standard error)
        '''std_devs = {
            region: np.std(vals) if vals else 0
            for region, vals in location_dict.items()
        }'''
        
        #Plot
        plt.figure(figsize=(10, 5))
        regions = list(avg_represses_by_region.keys())
        means = [avg_represses_by_region[region] for region in regions]
        #errors = [std_devs[region] for region in regions]
        counts = [region_counts[region] for region in regions]
        
        #bars = plt.bar(regions, means, yerr=errors, capsize=5, color='skyblue')
        bars = plt.bar(regions, means, capsize=5, color='skyblue')

        plt.ylabel("Avg Represses")
        plt.title("Average Represses by Region at First Press")
        # Add count annotations on top of each bar
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.05,  # slight offset above bar
                     f'n={count}', ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{self.prefix}BarGraphRepressesPerRegion.png")
        plt.show()
    
        # === Graph 2: Scatter plot of distance vs repressing with trendline ===
        plt.figure(figsize=(10, 5))
        plt.scatter(distance_list, repress_list, alpha=0.6, label="Trial Points")
        plt.xlabel("Inter-Mouse Distance at First Press")
        plt.ylabel("Avg Represses (First Mouse)")
        plt.title("Repressing vs Inter-Mouse Distance")
    
        # --- Trendline calculation ---
        if len(distance_list) > 1:
            dist_np = np.array(distance_list)
            repress_np = np.array(repress_list)
    
            # Fit a linear regression (degree-1 polynomial) to the data
            coeffs = np.polyfit(dist_np, repress_np, 1)
            trendline = np.poly1d(coeffs)
    
            # Generate smooth x and y values for plotting the line
            xs = np.linspace(min(dist_np), max(dist_np), 100)
            ys = trendline(xs)
    
            # Plot the trendline
            plt.plot(xs, ys, color='red', linestyle='--', label='Trendline')
            
            # Calculate R^2
            predicted = trendline(dist_np)
            ss_res = np.sum((repress_np - predicted) ** 2)
            ss_tot = np.sum((repress_np - np.mean(repress_np)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Display R² in the top right of the plot
            plt.text(0.95, 0.05, f'$R^2 = {r_squared:.3f}$',
                     transform=plt.gca().transAxes,
                     fontsize=12,
                     ha='right', va='bottom',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
            
            # Optional: Show slope and intercept
            slope, intercept = coeffs
            print(f"Trendline: y = {slope:.3f}x + {intercept:.3f}")
    
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.prefix}Distance_vs_RepressingBehavior.png")
        plt.show()
        
        # === Graph 3: Pie chart of trial percentages by region ===
        total_trials = sum(location_counts.values())
        if total_trials > 0:
            labels = []
            sizes = []
            for region, count in location_counts.items():
                if count > 0:
                    labels.append(region)
                    sizes.append(count)
        
            plt.figure(figsize=(7, 7))
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.tab20.colors)
            plt.title("Percentage of Trials by Location at First Press")
            plt.tight_layout()
            plt.savefig(f"{self.prefix}TrialPercentagesbyRegion.png")
            plt.show()
        else:
            print("No trial location data to display in pie chart.")
        
    def crossingOverQuantification(self):
        """
        This function generates visual summaries of lever-pressing and reward-collection behavior across all experiments.
    
        It performs two major analyses:
        
        1. **Max vs. Min Lever Presses Pie Chart**:
            - Calculates the total number of lever presses made by the most active lever per trial (Max).
            - Compares this against the number of presses made by the less active lever (Min).
            - Produces a pie chart showing the proportion of Max vs. Min lever usage across all trials.
    
        3. **Crossover Behavior Pie Charts**:
            - Examines each trial to determine if rats collected rewards on the same side as their lever press
              ("Same Side") or the opposite side ("Cross Side").
            - Also accounts for trials with missing or ambiguous reward data:
                - "No Mag Visit": No reward collection detected.
                - "Mag w/ Unknown RatID": Reward collected, but unable to identify which rat collected it.
            - Separate pie charts are generated for successful and failed trials to highlight behavioral patterns
              in different trial outcomes.
         
        """
        
        #Max vs. Min Lever Preference
        numMaxCount = 0
        numMinCountReal = 0
        totalCountPresses = 0
        
        numSwitchCount = 0
        numTrials = 0
        
        for exp in self.experiments:
            lev = exp.lev
            
            numMaxCount += lev.returnMostPressesByLever(0) + lev.returnMostPressesByLever(1)
            numMinCountReal += lev.returnMinPressesByLever(0) + lev.returnMinPressesByLever(1)
            totalCountPresses += lev.returnTotalLeverPressesFiltered()
            
            print("numMaxCount: ", numMaxCount)
            print("numMinCountReal: ", numMinCountReal)
            print("totalCountPresses: ", totalCountPresses)
            
        numMinCount = totalCountPresses - numMaxCount
        if (numMinCount != numMinCountReal):
            print("MISMATCH IN COUNTS")
            print("numMinCount: ", numMinCount)
            print("numMinCountReal: ", numMinCountReal)
        else:
            print("No Mismatch")
        
        labels = ['Preferred', 'Other']
        sizes = [numMaxCount, numMinCount]
        colors = ['green', 'red']
    
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 14})
        plt.title('Lever Preference(%)', fontsize=16)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.tight_layout()
        plt.show()
        if (self.save):
            plt.savefig(f'{self.prefix}LevPreference.png')
        plt.close()
        
        
        #Max vs. Min Mag Preference
        print("Starting Mag Zone Preference Analysis")
        mag_max_count = 0
        mag_total_count = 0

        for exp in self.experiments:
            mag = exp.mag

            mag_max_count += mag.returnMostEntriesbyMag(0) + mag.returnMostEntriesbyMag(1)
            mag_total_count += mag.getTotalMagEventsFiltered()

        mag_min_count = mag_total_count - mag_max_count
        mag_labels = ['Preferred', 'Other']
        mag_sizes = [mag_max_count, mag_min_count]
        mag_colors = ['purple', 'orange']

        plt.figure(figsize=(6, 6))
        plt.pie(mag_sizes, labels=mag_labels, colors=mag_colors, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 14})
        plt.title('Mag Zone Preference (%)', fontsize=16)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
        if (self.save):
            plt.savefig(f'{self.prefix}MagPreference.png')
        plt.close()
        
        
        # Crossing Over Pie Charts
        same_side_success = 0
        cross_side_success = 0
        no_mag_success = 0
        unknown_ratid_mag_success = 0
        
        same_side_fail = 0
        cross_side_fail = 0
        no_mag_fail = 0
        unknown_ratid_mag_fail = 0
        
        for exp in self.experiments:
            # Drop only lever rows with missing RatID (we must know which rat pressed)
            lev_df = exp.lev.data[['TrialNum', 'RatID', 'LeverNum', 'coopSucc']].dropna(subset=['RatID'])
            
            # Keep all mag rows (even those with NaN RatID)
            mag_df = exp.mag.data[['TrialNum', 'RatID', 'MagNum']]
        
            # Get first lever press per trial per mouse
            lev_first = lev_df.drop_duplicates(subset=['TrialNum', 'RatID'], keep='first')
            #print("lev_first: ")
            #print(lev_first)
            
            # Get first mag entry per trial per mouse
            mag_first = mag_df.drop_duplicates(subset=['TrialNum'], keep='first')
            #print("mag_first: ")
            #print(mag_first)
            
            # Merge on TrialNum ONLY — not RatID
            merged = lev_first.merge(mag_first, on='TrialNum', how='left', suffixes=('_lev', '_mag'))

            print("Merged: ")
            print(merged)
            
            print("Merged Successful: ")
            print(merged[merged['coopSucc'] == 1])
            
            
            for _, row in merged.iterrows():
                trialNum = row['TrialNum']
                #absTime = row['AbsTime']
                lever = row['LeverNum']
                mag = row['MagNum'] if not pd.isna(row['MagNum']) else None
                success = row['coopSucc']
                lever_rat = row['RatID_lev']
                mag_rat = row['RatID_mag'] if 'RatID_mag' in row else None  # From mag side
                
                # Skip invalid lever
                if lever not in [1, 2]:
                    continue
        
                if mag is None:
                    # No mag entry recorded at all
                    if success:
                        no_mag_success += 1
                    else:
                        no_mag_fail += 1
                elif pd.isna(mag_rat):
                    # Mag entry exists but with unknown RatID
                    if success:
                        unknown_ratid_mag_success += 1
                    else:
                        unknown_ratid_mag_fail += 1
                else:
                    # Valid known rat and mag entry
                    crossed = (lever == 1 and mag == 2) or (lever == 2 and mag == 1)
                    if success:
                        if crossed:
                            cross_side_success += 1
                        else:
                            same_side_success += 1
                    else:
                        if crossed:
                            cross_side_fail += 1
                        else:
                            same_side_fail += 1
                
                if (success):
                    print("TrialNum: ", trialNum)
                    #print("AbsTime: ", absTime)
                    print("same_side_success: ", same_side_success, "; cross_side_success: ", cross_side_success, "; unknown_ratid_mag_success: ", unknown_ratid_mag_success, "; no_mag_success: ", no_mag_success)
        
        # --- Pie chart for successful trials ---
        labels_success = [
            'Same Side',
            'Cross Side',
            'No Mag Visit',
            'Mag w/ Unknown RatID'
        ]
        sizes_success = [
            same_side_success,
            cross_side_success,
            no_mag_success,
            unknown_ratid_mag_success
        ]
        
        plt.figure(figsize=(6, 6))
        plt.pie(sizes_success, labels=labels_success, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 14})
        plt.title('Crossover Behavior in Successful Trials', fontsize=16)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
        if (self.save):
            plt.savefig(f'{self.prefix}Crossover_Successful.png')
        plt.close()
        
        # --- Pie chart for failed trials ---
        labels_fail = [
            'Same Side',
            'Cross Side',
            'No Mag Visit',
            'Mag w/ Unknown RatID'
        ]
        sizes_fail = [same_side_fail, cross_side_fail, no_mag_fail, unknown_ratid_mag_fail]
        
        plt.figure(figsize=(6, 6))
        plt.pie(sizes_fail, labels=labels_fail, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 14})
        plt.title('Crossover Behavior in Failed Trials', fontsize=16)
        plt.axis('equal')
        plt.tight_layout()
        if (self.save):
            plt.savefig(f'{self.prefix}Crossover_Failed.png')
        plt.show()
        plt.close()
        
    def cooperativeRegionStrategiesQuantification(self):
        """
        This function quantifies and compares the average horizontal distance (in X coordinates) between 
        the head-body markers of two rats during different behavioral trial contexts:
        - Trials that fall within a "cooperative success region" (defined as 4 out of the last 5 trials being successful)
        - Trials that do not fall in such regions.
        - Trials that are successful but not in a success region
    
        For each trial in each experiment:
            - The start and end frame of the trial are calculated based on absolute times and framerate.
            - The average absolute difference in X-position between the two rats is computed over the trial duration.
            - These differences are aggregated separately for trials inside and outside the success regions.
    
        The function then computes the average inter-rat distance per frame for each context and generates a bar plot
        with individual data points overlaid, enabling visual comparison of spatial strategies during cooperative
        vs. non-cooperative behavioral states.
        """
        
        def numSuccessinaRow(successTrials):
            n = len(successTrials)
            res = [0] * n
            count = 0
            for i, succ in enumerate(successTrials):
                if (succ == 1):
                    count += 1
                else:
                    count = 0
                
                res[i] = count
            return res
        
        averageDistance_NoSuccess = 0
        averageDistance_SuccessZone = 0
        averageDistance_Success_NoZone = 0
        
        totFrames_SuccessZone = 0
        totDifference_SuccessZone = 0
        datapoints_SuccessZone = []
        
        totFrames_Success_NoZone = 0 
        totDifference_Success_NoZone = 0
        datapoints_Success_NoZone = []
        
        totFrames_NoSuccess = 0
        totDifference_NoSuccess = 0
        datapoints_NoSuccess = []
        
        successInARowvsDistance = defaultdict(list)
        
        for exp in self.experiments:
            lev = exp.lev
            pos = exp.pos
            fps = exp.fps
            
            listTrials = lev.returnCooperativeSuccessRegionsBool()
            startTimeTrials = lev.returnTimeStartTrials()
            endTimeTrials = lev.returnTimeEndTrials()
            
            successTrial = lev.returnSuccessTrials()
            successInARow = numSuccessinaRow(successTrial)
            
            cat_totFrames_SuccessZone = 0
            cat_totDifference_SuccessZone = 0
            
            cat_totFrames_Success_NoZone = 0 
            cat_totDifference_Success_NoZone = 0
            
            cat_totFrames_NoSuccess = 0
            cat_totDifference_NoSuccess = 0
            
            temp_successInARowvsDistance = defaultdict(list)
            
            print("Lengths:", len(listTrials), len(startTimeTrials), len(endTimeTrials))
            if (len(listTrials) != len(startTimeTrials) or  len(startTimeTrials) != len(endTimeTrials)):
                print("levFiles: ", exp.lev_file)
            
            for i, trialBool in enumerate(listTrials):
                if (startTimeTrials[i] == None or endTimeTrials[i] == None or successInARow[i] == None):
                    continue
                
                startFrame = int(startTimeTrials[i] * fps)
                endFrame = int(endTimeTrials[i] * fps)
                
                #print("startFrame: ", startFrame)
                #print("endFrame: ", endFrame)
                
                numFrames = endFrame - startFrame
                
                rat1_xlocations = pos.data[0, 0, pos.HB_INDEX, startFrame:endFrame]
                rat2_xlocations = pos.data[1, 0, pos.HB_INDEX, startFrame:endFrame]
                
                difference = sum(abs(a - b) for a, b in zip(rat1_xlocations, rat2_xlocations))            
                
                temp_successInARowvsDistance[successInARow[i]].append(difference / numFrames)
                
                if (trialBool):
                    totFrames_SuccessZone += numFrames
                    totDifference_SuccessZone += difference
                    cat_totFrames_SuccessZone += numFrames
                    cat_totDifference_SuccessZone += difference
                    #if (numFrames > 0):
                        #datapoints_SuccessZone.append(difference / numFrames)
                        
                elif(successTrial[i] == 1):
                    totFrames_Success_NoZone += numFrames
                    totDifference_Success_NoZone += difference
                    cat_totFrames_Success_NoZone += numFrames
                    cat_totDifference_Success_NoZone += difference
                    #if (numFrames > 0):
                        #datapoints_Success_NoZone.append(difference / numFrames)
                else:
                    totFrames_NoSuccess += numFrames
                    totDifference_NoSuccess += difference
                    cat_totFrames_NoSuccess += numFrames
                    cat_totDifference_NoSuccess += difference
                    #if (numFrames > 0):
                        #datapoints_NoSuccess.append(difference / numFrames)
                        
            if (cat_totFrames_SuccessZone > 0):
                datapoints_SuccessZone.append(cat_totDifference_SuccessZone / cat_totFrames_SuccessZone)
            
            if (cat_totFrames_Success_NoZone > 0):
                datapoints_Success_NoZone.append(cat_totDifference_Success_NoZone / cat_totFrames_Success_NoZone)
            
            if (cat_totFrames_NoSuccess > 0):
                datapoints_NoSuccess.append(cat_totDifference_NoSuccess / cat_totFrames_NoSuccess)
                
            for key, value in temp_successInARowvsDistance.items():
                avg = np.mean(value)
                successInARowvsDistance[key].append(avg)
        
        print("totFrames_NoSuccess: ", totFrames_NoSuccess)
        print("totFrames_SuccessZone: ", totFrames_SuccessZone)
        print("totFrames_Success_NoZone: ", totFrames_Success_NoZone)
        
        if (totFrames_NoSuccess > 0):
            averageDistance_NoSuccess = totDifference_NoSuccess / totFrames_NoSuccess
            
        if (totFrames_SuccessZone > 0):
            averageDistance_SuccessZone = totDifference_SuccessZone / totFrames_SuccessZone
            
        if (totFrames_Success_NoZone > 0):
            averageDistance_Success_NoZone = totDifference_Success_NoZone / totFrames_Success_NoZone
        
        #Make Graphs: 
        # Labels and values for the bar plot
        labels = ['No Success', 'Success No Zone', 'Success Zone']
        averages = [averageDistance_NoSuccess, averageDistance_Success_NoZone, averageDistance_SuccessZone]
        datapoints = [datapoints_NoSuccess, datapoints_Success_NoZone, datapoints_SuccessZone]
        
        # X locations for the bars and jittered scatter points
        x = np.arange(len(labels))
        width = 0.6
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot bar chart
        bars = ax.bar(x, averages, width, color=['red', 'yellow', 'green'], alpha=0.6, edgecolor='black')
        
        # Overlay individual data points
        for i, points in enumerate(datapoints):
            # Add jitter to the x-position of each point for visibility
            jittered_x = np.random.normal(loc=x[i], scale=0.05, size=len(points))
            ax.scatter(jittered_x, points, alpha=0.8, color='black', s=20)
        
        # Labels and formatting
        ax.set_ylabel('Average Distance', fontsize = 13)
        ax.set_title('Average Head-Body X-Distance per Trial', fontsize = 15)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize = 13)
        ax.tick_params(axis='y', labelsize=13)  # Set y-axis tick font size for consistency
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        
        plt.tight_layout()
        if (self.save):
            plt.savefig(f"{self.prefix}X_Distance_SuccessZonevsNoSuccess.png")
        plt.show()
        
        # Prepare data: calculate averages for each key and collect individual points
        keys = sorted(successInARowvsDistance.keys())
        averages = [np.mean(successInARowvsDistance[k]) for k in keys]
        individual_points = [(k, v) for k in keys for v in successInARowvsDistance[k]]
    
        # Create smoothed line using spline interpolation
        x_smooth = np.linspace(min(keys), max(keys), 300)
        spl = make_interp_spline(keys, averages, k=3)  # Cubic spline
        y_smooth = spl(x_smooth)
    
        # Calculate R² for the smoothed fit (using linear regression on averages for simplicity)
        slope, intercept, r_value, _, _ = linregress(keys, averages)
        r_squared = r_value**2
    
        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot smoothed line
        ax.plot(x_smooth, y_smooth, color='blue', label='Smoothed Average Distance')
        
        # Plot individual data points as faint grey dots
        if individual_points:
            x_points, y_points = zip(*individual_points)
            ax.scatter(x_points, y_points, color='grey', alpha=0.3, s=20, label='Individual Data Points')
        
        # Add R² value to the plot
        ax.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=ax.transAxes, 
                fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
        
        # Labels and formatting
        ax.set_xlabel('Number of Successes in a Row', fontsize = 13)
        ax.set_ylabel('Average Distance', fontsize = 13)
        ax.set_title('Successes in a Row vs. Average Head-Body X-Distance', fontsize = 15)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save and show the plot
        plt.tight_layout()
        if (self.save):
            plt.savefig(f"{self.prefix}Smoothed_X_Distance_vs_SuccessInARow.png")
        plt.show()
        
    def compareAverageVelocityGazevsNot(self):
        '''
        Creates a bar plot comparing the average velocity of the mouse during gazing moments vs. non-gazing
        moments. 
        
        Velocity is calculated by finding the distance moved per frame of the head base
        'flexibilityVar' exists to not count any non-gazing within flexibilityVar frames of a gazing frame.   
        '''
        
        flexibilityVar = 5
        
        totFramesNonGazing = 0
        totMovementNonGazing = 0
        
        totFramesGazing = 0 
        totMovementGazing = 0
        
        for exp in self.experiments:
            pos = exp.pos
            
            x_coordsRat0 = pos.data[0, 0, pos.HB_INDEX, :]
            y_coordsRat0 = pos.data[0, 1, pos.HB_INDEX, :]
            
            x_coordsRat1 = pos.data[1, 0, pos.HB_INDEX, :]
            y_coordsRat1 = pos.data[1, 1, pos.HB_INDEX, :]
            
            arrIsGazingRat0 = pos.returnIsGazing(0)
            arrIsGazingRat1 = pos.returnIsGazing(1)
            
            print("len(arrIsGazingRat0): ", len(arrIsGazingRat0))
            print("len(arrIsGazingRat1): ", len(arrIsGazingRat1))
            print("len(x_coordsRat0): ", len(x_coordsRat0))
            
            
            counterIsGazing = 0
            
            for i, frame in enumerate(arrIsGazingRat0):
                if (i == 0):
                    continue
                
                x = x_coordsRat0[i]
                y = y_coordsRat0[i]
                xp = x_coordsRat0[i-1]
                yp = y_coordsRat0[i-1]
                
                dx = x - xp
                dy = y - yp
                
                dist = np.sqrt(dx ** 2 + dy ** 2)
                
                if (counterIsGazing >= 0 and frame == True):
                    counterIsGazing = flexibilityVar
                    totFramesGazing += 1
                    totMovementGazing += dist
                    
                elif (counterIsGazing > 0 and frame == False):
                    counterIsGazing -= 1
                
                else:
                    totFramesNonGazing += 1
                    totMovementNonGazing += dist
                
            for i, frame in enumerate(arrIsGazingRat1):
                if (i == 0):
                    continue
                
                x = x_coordsRat1[i]
                y = y_coordsRat1[i]
                xp = x_coordsRat1[i-1]
                yp = y_coordsRat1[i-1]
                
                dx = x - xp
                dy = y - yp
                
                dist = np.sqrt(dx ** 2 + dy ** 2)
                
                if (counterIsGazing >= 0 and frame == True):
                    counterIsGazing = flexibilityVar
                    totFramesGazing += 1
                    totMovementGazing += dist
                    
                elif (counterIsGazing > 0 and frame == False):
                    counterIsGazing -= 1
                
                else:
                    totFramesNonGazing += 1
                    totMovementNonGazing += dist
            
        # Compute average velocities
        avgVelGazing = totMovementGazing / totFramesGazing if totFramesGazing > 0 else 0
        avgVelNonGazing = totMovementNonGazing / totFramesNonGazing if totFramesNonGazing > 0 else 0
    
        # Plotting
        labels = ['Gazing', 'Not Gazing']
        values = [avgVelGazing, avgVelNonGazing]
    
        plt.figure(figsize=(6, 5))
        bars = plt.bar(labels, values, color=['blue', 'gray'])
        plt.ylabel('Average Velocity (pixels/frame)')
        plt.title('Average Velocity During Gazing vs Not Gazing')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height * 1.01, f'{height:.2f}', 
                     ha='center', va='bottom', fontsize=10)
    
        plt.tight_layout()
        plt.savefig(f"{self.prefix}compareAverageVelocityGazevsNotGazing.png")
        plt.show()
            
    def makeHeatmapLocation(self):
        '''
        Make a heatmap of all mice and where they spend time by tracking the location of the headbase.
        '''
        bin_size = 5  # Controls resolution of heatmap (larger = coarser)
        height, width = 640, 1392
        heatmap_height = height // bin_size
        heatmap_width = width // bin_size
        heatmap = np.zeros((heatmap_height, heatmap_width))
    
        for exp in self.experiments:
            pos = exp.pos
            data = pos.data  # shape: (2, 2, 5, num_frames)
    
            for mouse in range(2):
                x_coords = data[mouse, 0, pos.HB_INDEX, :]
                y_coords = data[mouse, 1, pos.HB_INDEX, :]
    
                for x, y in zip(x_coords, y_coords):
                    if not np.isnan(x) and not np.isnan(y):
                        x_bin = int(min(max(x // bin_size, 0), heatmap_width - 1))
                        y_bin = int(min(max(y // bin_size, 0), heatmap_height - 1))
                        heatmap[y_bin, x_bin] += 1
        
        # Smooth the heatmap
        heatmap = gaussian_filter(heatmap, sigma=1)
    
        # Optional: Use logarithmic scale for better visibility
        heatmap_log = np.log1p(heatmap)  # log(1 + x) to handle zeroes
    
        plt.figure(figsize=(12, 6))
        plt.imshow(
            heatmap_log,
            cmap='hot',
            interpolation='nearest',
            origin='upper',
            extent=[0, width, height, 0],
            vmin=np.percentile(heatmap_log, 10),  # tune as needed
            vmax=np.percentile(heatmap_log, 99)
        )
        plt.colorbar(label='Log(Time Spent)')
        plt.title('Mouse Location Heatmap (Headbase)')
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        plt.savefig(f"{self.prefix}movementHeatmap.png", bbox_inches='tight')
        plt.show()
                
    def findTotalDistanceMoved(self):
        '''
        For each experiment, compute:
        - Sum of distances moved by both mice
        - Absolute difference in distances moved between mice
        - Minimum Distance Moved by a Rat
        Then, plot both metrics against cooperative success rate, with trendlines.
        
        Additionally, create bucketed smoothed line graphs for standardized total distance moved
        and standardized absolute distance moved vs. cooperative success rate, with 20 buckets.
        '''
        
        distancesSum = []
        distancesDiff = []
        minRatMoved = []
        coop_successes = []
    
        for exp in self.experiments:
            pos = exp.pos
            data = pos.data
            total_distance = [0.0, 0.0]
    
            for rat in range(2):
                total_distance[rat] = pos.returnStandardizedDistanceMoved(rat)
            
            print("Total Distance: ", total_distance)
            
            total_trials = exp.lev.returnNumTotalTrials()
            print("Total Trials: ", total_trials)
            
            if total_trials > 0:
                success_rate = exp.lev.returnNumSuccessfulTrials() / total_trials
                coop_successes.append(success_rate)
                distancesSum.append(total_distance[0] + total_distance[1])
                distancesDiff.append(abs(total_distance[0] - total_distance[1]))
                minRatMoved.append(min(pos.returnStandardizedDistanceMoved(0), pos.returnStandardizedDistanceMoved(1)))
            else:
                print(f"Skipping session {exp} due to zero total trials.")
        
        # Check for sufficient data to make trendlines
        if len(set(distancesSum)) < 2 or len(set(distancesDiff)) < 2 or len(set(minRatMoved)) < 2:
            print("Insufficient variation in distances; cannot compute trendlines.")
            return
        
        print("distancesSum:", distancesSum)
        print("distancesDiff:", distancesDiff)
        print("coop_successes:", coop_successes)
        
        coop_successes_unfiltered = coop_successes
        
        # Remove outliers (top 5% of distancesSum and distancesDiff)
        threshold_sum = np.percentile(distancesSum, 95)  # 95th percentile for distancesSum
        threshold_diff = np.percentile(distancesDiff, 95)  # 95th percentile for distancesDiff
        mask = (np.array(distancesSum) <= threshold_sum) & (np.array(distancesDiff) <= threshold_diff)
        distancesSum_filtered = np.array(distancesSum)[mask].tolist()
        distancesDiff_filtered = np.array(distancesDiff)[mask].tolist()
        coop_successes_filtered = np.array(coop_successes)[mask].tolist()
        
        # Check if enough data remains after filtering
        if len(distancesSum_filtered) < 2 or len(set(distancesSum_filtered)) < 2 or len(set(distancesDiff_filtered)) < 2:
            print("Not enough data after filtering outliers; using original data.")
            distancesSum_filtered = distancesSum
            distancesDiff_filtered = distancesDiff
            coop_successes_filtered = coop_successes
        else:
            print(f"Filtered out {len(distancesSum) - len(distancesSum_filtered)} outliers.")
            distancesSum = distancesSum_filtered
            distancesDiff = distancesDiff_filtered
            coop_successes = coop_successes_filtered
        
        #Scatterplots with Trendlines
        
        # Graph 1: distancesSum linear
        plt.figure(figsize=(8, 6))
        plt.scatter(distancesSum, coop_successes, alpha=0.7, label='Rat', color='blue')
        slope, intercept, r_value, _, _ = linregress(distancesSum, coop_successes)
        r_squared = r_value ** 2
        x_vals = np.linspace(min(distancesSum), max(distancesSum), 100)
        plt.plot(x_vals, slope * x_vals + intercept, color='red', linestyle='--', label='Trendline')
        plt.title('Total Distance Moved vs. Cooperative Success Rate')
        plt.xlabel('Total Distance Moved (pixels)')
        plt.ylabel('Cooperative Success Rate (%)')
        plt.xscale('linear')
        plt.legend()
        plt.grid(True)
        plt.text(0.95, 0.95, f"$R^2$ = {r_squared:.3f}", transform=plt.gca().transAxes,
                 ha='right', va='top', fontsize=12, bbox=dict(facecolor='white', edgecolor='gray'))
        if self.save: 
            plt.savefig(f"{self.prefix}DistMovedSum_vs_CoopSuccessRate_linear.png")
        plt.show()
        plt.close()
        
        # Graph 2: distancesDiff linear
        plt.figure(figsize=(8, 6))
        plt.scatter(distancesDiff, coop_successes, alpha=0.7, label='Rat', color='green')
        slope, intercept, r_value, _, _ = linregress(distancesDiff, coop_successes)
        r_squared = r_value ** 2
        x_vals = np.linspace(min(distancesDiff), max(distancesDiff), 100)
        plt.plot(x_vals, slope * x_vals + intercept, color='red', linestyle='--', label='Trendline')
        plt.title('Abs Diff Distance Moved vs. Cooperative Success Rate')
        plt.xlabel('Diff in Distance Moved (pixels)')
        plt.ylabel('Cooperative Success Rate (%)')
        plt.xscale('linear')
        plt.legend()
        plt.grid(True)
        plt.text(0.95, 0.95, f"$R^2$ = {r_squared:.3f}", transform=plt.gca().transAxes,
                 ha='right', va='top', fontsize=12, bbox=dict(facecolor='white', edgecolor='gray'))
        if self.save:
            plt.savefig(f"{self.prefix}DistMovedDiff_vs_CoopSuccessRate_linear.png")  
        plt.show()
        plt.close()
        
        # Graph 3: minRatMoved vs. success rate
        plt.figure(figsize=(8, 6))
        plt.scatter(minRatMoved, coop_successes_unfiltered, alpha=0.7, label='Rat', color='green')
        slope, intercept, r_value, _, _ = linregress(minRatMoved, coop_successes_unfiltered)
        r_squared = r_value ** 2
        x_vals = np.linspace(min(minRatMoved), max(minRatMoved), 100)
        plt.plot(x_vals, slope * x_vals + intercept, color='red', linestyle='--', label='Trendline')
        plt.title('Min Distance Moved vs. Cooperative Success Rate')
        plt.xlabel('Diff in Distance Moved (pixels)')
        plt.ylabel('Cooperative Success Rate (%)')
        plt.xscale('linear')
        plt.legend()
        plt.grid(True)
        plt.text(0.95, 0.95, f"$R^2$ = {r_squared:.3f}", transform=plt.gca().transAxes,
                 ha='right', va='top', fontsize=12, bbox=dict(facecolor='white', edgecolor='gray'))
        if self.save:
            plt.savefig(f"{self.prefix}MinDistMoved_vs_CoopSuccessRate_linear.png")  
        plt.show()
        plt.close()
        
        #–––––––––-––––––––––––––––––––––––––––––––––––––––––––––––––––––
        
        # New Graphs: Bucketed Smoothed Line Graphs
        trial_distances_sum = []
        trial_distances_diff = []
        trial_successes = []
    
        # Collect trial-level data
        for exp in self.experiments:
            pos = exp.pos
            lev = exp.lev
            fps = exp.fps
            #total_trials = lev.returnNumTotalTrials()
            start_times = lev.returnTimeStartTrials()  # In seconds
            end_times = lev.returnLastPressTime()  # In seconds
            success_trials = lev.returnSuccessTrials()  # Boolean array
    
            for trial_idx in range(len(start_times)):
                start_time = start_times[trial_idx]
                end_time = end_times[trial_idx]
                if np.isnan(start_time) or np.isnan(end_time) or start_time is None or end_time is None:
                    continue
                start_frame = int(start_time * fps)
                end_frame = int(end_time * fps)
                if end_frame <= start_frame or end_frame >= pos.data.shape[-1]:
                    continue
    
                # Calculate distances for each rat
                distances = [0.0, 0.0]
                for rat in range(2):
                    prev_pos = None
                    for t in range(start_frame, end_frame):
                        curr_pos = pos.returnRatHBPosition(rat, t)
                        if curr_pos is None or np.any(np.isnan(curr_pos)):
                            continue
                        if prev_pos is not None:
                            # Euclidean distance between consecutive frames
                            dist = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
                            distances[rat] += dist
                        prev_pos = curr_pos
                
                n = end_frame - start_frame
                trial_sum = (distances[0] + distances[1]) /  n
                trial_diff = (abs(distances[0] - distances[1])) / n
                success = 1 if success_trials[trial_idx] else 0
    
                trial_distances_sum.append(trial_sum)
                trial_distances_diff.append(trial_diff)
                trial_successes.append(success)
        
        # Define buckets
        num_buckets = 20
        sum_bins = np.linspace(0, 75, num_buckets)  # 0 to 75
        diff_bins = np.linspace(0, 40, num_buckets)  # 0 to 40
    
        # Initialize bucket data
        sum_bucket_success = [[] for _ in range(num_buckets)]
        diff_bucket_success = [[] for _ in range(num_buckets)]
    
        # Assign trials to buckets
        for i in range(len(trial_distances_sum)):
            sum_val = trial_distances_sum[i]
            diff_val = trial_distances_diff[i]
            success = trial_successes[i]
            
            # Assign to sum bucket
            sum_bucket_idx = min(np.searchsorted(sum_bins, sum_val, side='right'), num_buckets - 1)
            sum_bucket_success[sum_bucket_idx].append(success)
            
            # Assign to diff bucket
            diff_bucket_idx = min(np.searchsorted(diff_bins, diff_val, side='right'), num_buckets - 1)
            diff_bucket_success[diff_bucket_idx].append(success)
    
        # Compute average success rate and trial counts per bucket
        sum_bucket_avgs = []
        sum_bucket_counts = []
        sum_bucket_centers = []
        for i in range(num_buckets):
            if sum_bucket_success[i]:
                avg = np.mean(sum_bucket_success[i])
                count = len(sum_bucket_success[i])
                sum_bucket_avgs.append(avg * 100)  # Convert to percentage
                sum_bucket_counts.append(count)
                center = sum_bins[i] + 5 if i == num_buckets - 1 else (sum_bins[i] + sum_bins[i+1]) / 2 if i < num_buckets - 1 else sum_bins[i]
                sum_bucket_centers.append(center)
            else:
                sum_bucket_avgs.append(0)
                sum_bucket_counts.append(0)
                sum_bucket_centers.append(sum_bins[i] + (sum_bins[1] - sum_bins[0]) / 2 if i < num_buckets - 1 else sum_bins[i] + 5)
    
        diff_bucket_avgs = []
        diff_bucket_counts = []
        diff_bucket_centers = []
        for i in range(num_buckets):
            if diff_bucket_success[i]:
                avg = np.mean(diff_bucket_success[i])
                count = len(diff_bucket_success[i])
                diff_bucket_avgs.append(avg * 100)  # Convert to percentage
                diff_bucket_counts.append(count)
                center = diff_bins[i] + 5 if i == num_buckets - 1 else (diff_bins[i] + diff_bins[i+1]) / 2 if i < num_buckets - 1 else diff_bins[i]
                diff_bucket_centers.append(center)
            else:
                diff_bucket_avgs.append(0)
                diff_bucket_counts.append(0)
                diff_bucket_centers.append(diff_bins[i] + (diff_bins[1] - diff_bins[0]) / 2 if i < num_buckets - 1 else diff_bins[i] + 5)
    
        # Graph 4: Total Distance Moved (Bucketed)
        plt.figure(figsize=(10, 6))
        plt.plot(sum_bucket_centers, sum_bucket_avgs, color='blue', label='Smoothed Line')
        plt.scatter(sum_bucket_centers, sum_bucket_avgs, color='blue', label='Bucket Averages')
        for i, (x, y, count) in enumerate(zip(sum_bucket_centers, sum_bucket_avgs, sum_bucket_counts)):
            if count > 0:
                plt.text(x, y + 2, f'n={count}', fontsize=8, ha='center', va='bottom')
        plt.title('Bucketed Total Distance Moved vs. Cooperative Success Rate')
        plt.xlabel('Total Distance Moved (pixels)')
        plt.ylabel('Cooperative Success Rate (%)')
        plt.legend()
        plt.grid(True)
        if self.save:
            plt.savefig(f"{self.prefix}Bucketed_TotalDistMoved_vs_CoopSuccessRate.png")
        plt.show()
        plt.close()
    
        # Graph 5: Absolute Difference in Distance Moved (Bucketed)
        plt.figure(figsize=(10, 6))
        plt.plot(diff_bucket_centers, diff_bucket_avgs, color='green', label='Smoothed Line')
        plt.scatter(diff_bucket_centers, diff_bucket_avgs, color='green', label='Bucket Averages')
        for i, (x, y, count) in enumerate(zip(diff_bucket_centers, diff_bucket_avgs, diff_bucket_counts)):
            if count > 0:
                plt.text(x, y + 2, f'n={count}', fontsize=8, ha='center', va='bottom')
        plt.title('Bucketed Abs Diff Distance Moved vs. Cooperative Success Rate')
        plt.xlabel('Abs Diff Distance Moved (pixels)')
        plt.ylabel('Cooperative Success Rate (%)')
        plt.legend()
        plt.grid(True)
        if self.save:
            plt.savefig(f"{self.prefix}Bucketed_AbsDiffDistMoved_vs_CoopSuccessRate.png")
        plt.show()
        plt.close()
        
    def intersectings_vs_percentNaN(self):
        '''
        For each experiment:
        - Get the initial NaN count from `exp.initialNan`
        - Use `exp.pos.checkSelfIntersection(ratID)` for both rats (ratID 0 and 1)
        - Combine both rats' intersection lists and compute the % of intersecting frames
          (i.e., where either rat is self-intersecting)
        - Plot initial NaN count vs % frames intersecting across experiments
        '''
    
        initial_nans = []
        percent_intersecting = []
    
        for exp in self.experiments:
            # Total number of frames
            total_frames = exp.endFrame
    
            # Get list of intersecting frames for each rat
            rat0_intersects = exp.pos.checkSelfIntersection(0)
            rat1_intersects = exp.pos.checkSelfIntersection(1)
    
            # Combine — a frame is intersecting if either rat is intersecting
            combined = [a or b for a, b in zip(rat0_intersects, rat1_intersects)]
    
            # Count how many frames are intersecting
            num_intersecting = sum(combined)
    
            # Append to data lists
            initial_nans.append(exp.initialNan)
            percent = 100 * num_intersecting / total_frames if total_frames > 0 else 0
            percent_intersecting.append(percent)
    
        # Make scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(initial_nans, percent_intersecting, alpha=0.7, color='purple')
        plt.xlabel('Initial NaN Count')
        plt.ylabel('% of Frames Self-Intersecting (Either Rat)')
        plt.title('Initial Missing Data vs. Self-Intersection Rate')
        plt.grid(True)
    
        # Optional: trendline
        if len(initial_nans) >= 2:
            from scipy.stats import linregress
            slope, intercept, r, _, _ = linregress(initial_nans, percent_intersecting)
            x_vals = np.linspace(min(initial_nans), max(initial_nans), 100)
            y_vals = slope * x_vals + intercept
            plt.plot(x_vals, y_vals, linestyle='--', color='red', label=f"Trendline (R²={r**2:.2f})")
            plt.legend()
    
        if self.save:
            plt.savefig(f"{self.prefix}NaN_vs_Intersecting.png")
        
        plt.show()
        plt.close()
        
        print("Done with % Intersecting vs. % Nan")

    def printSummaryStats(self):
        '''
        '''
        print("start")
        total_gaze_events = 0     # Total gaze events (all mice) 
        total_gaze_events_alternate = 0     # Total gaze events (all mice) for alternate definition
        total_frames = 0          # Total number of frames across all sessions
        total_trials = 0          # Total number of trials across sessions
        successful_trials = 0     # Total number of cooperative successful trials
        total_lever_presses = 0   # Total number of lever presses
        total_mag_events = 0      # Total number of magazine entries
        total_gaze_frames = 0     # Total frames where gaze was detected
        total_gaze_frames_alternate = 0

        # Process each experiment within the category
        for i, exp in enumerate(self.experiments):
            print("Round: ", i)
            loader = exp.pos
            print("past r0")
            g0 = loader.returnIsGazing(0, alternateDef=False)
            g1 = loader.returnIsGazing(1, alternateDef=False)
            g2 = loader.returnIsGazing(0)
            g3 = loader.returnIsGazing(1)
            print("past r1")
            
            # Count gaze events and sum up the frames with gazing behavior
            total_gaze_events += loader.returnNumGazeEvents(0, alternateDef=False) + loader.returnNumGazeEvents(1, alternateDef=False)
            total_gaze_frames += np.sum(g0) + np.sum(g1)
            total_frames += g0.shape[0]
            
            total_gaze_events_alternate += loader.returnNumGazeEvents(0) + loader.returnNumGazeEvents(1)
            total_gaze_frames_alternate += np.sum(g2) + np.sum(g3)
            print("past r2")
            
            # Access lever press data and compute trial/success counts
            lev = exp.lev.data
            trials = lev['TrialNum'].nunique()
            succ = lev.groupby('TrialNum').first().query('coopSucc == 1').shape[0]
            total_trials += trials
            successful_trials += succ
            total_lever_presses += lev.shape[0]

            # Count magazine events
            mag = exp.mag.data
            total_mag_events += mag.shape[0]
        
        
        # Print summary statistics for the current category
        print(f"  Number of Files: {len(self.experiments)}")
        print(f"  Total Frames: {total_frames}")
        print(f"  Total Trials: {total_trials}")
        print(f"  Successful Trials: {successful_trials}")
        print(f"  Percent Successful: {successful_trials / total_trials:.2f}")
        print(f"  Frames Gazing: {total_gaze_frames}")
        print(f"  Total Gaze Events: {total_gaze_events}")
        print(f"  Average Gaze Length: {total_gaze_frames / total_gaze_events:.2f}")
        print(f"  Percent Gazing: {100 * total_gaze_frames / total_frames:.2f}%")
        print(f"  Frames Gazing (Alternate): {total_gaze_frames}")
        print(f"  Total Gaze Events (Alternate): {total_gaze_events_alternate}")
        print(f"  Average Gaze Length (Alternate): {total_gaze_frames_alternate / total_gaze_events_alternate:.2f}")
        print(f"  Percent Gazing (Alternate): {100 * total_gaze_frames_alternate / total_frames:.2f}%")
        print(f"  Avg Lever Presses per Trial: {total_lever_presses / total_trials:.2f}")
        print(f"  Total Lever Presses: {total_lever_presses}")
        print(f"  Avg Mag Events per Trial: {total_mag_events / total_trials:.2f}")
        print(f"  Total Mag Events: {total_mag_events}")
        
    def successVsAverageDistance(self):
        """
        Creates a scatterplot of cooperative success probability vs. average inter-mouse distance
        across all experiments. Includes a trendline and R² value.
        """
        success_rates = []
        avg_distances = []
    
        for exp in self.experiments:
            # Calculate success rate
            total_trials = exp.lev.returnNumTotalTrials()
            if total_trials == 0:
                print(f"Skipping experiment {exp.lev.filename} due to zero total trials.")
                continue
            success_rate = exp.lev.returnNumSuccessfulTrials() / total_trials
            success_rates.append(success_rate)
    
            # Calculate average inter-mouse distance
            inter_mouse_dist = exp.pos.returnInterMouseDistance()
            if len(inter_mouse_dist) == 0 or np.all(np.isnan(inter_mouse_dist)):
                print(f"Skipping experiment {exp.pos.filename} due to invalid distance data.")
                continue
            avg_distance = np.nanmean(inter_mouse_dist)  # Ignore NaN values
            avg_distances.append(avg_distance)
    
        # Check for sufficient data
        if len(success_rates) < 2 or len(avg_distances) < 2:
            print("Insufficient data to create scatterplot.")
            return
    
        # Create scatterplot
        plt.figure(figsize=(8, 6))
        plt.scatter(avg_distances, success_rates, alpha=0.7, color='blue', label='Experiments')
    
        # Add trendline and R²
        if len(set(avg_distances)) >= 2:  # Ensure enough variation for regression
            slope, intercept, r_value, _, _ = linregress(avg_distances, success_rates)
            r_squared = r_value ** 2
            x_vals = np.linspace(min(avg_distances), max(avg_distances), 100)
            plt.plot(x_vals, slope * x_vals + intercept, color='red', linestyle='--', label='Trendline')
            plt.text(0.95, 0.05, f"$R^2$ = {r_squared:.3f}", transform=plt.gca().transAxes,
                     ha='right', va='bottom', fontsize=12, bbox=dict(facecolor='white', edgecolor='gray'))
        else:
            print("Insufficient variation in distances for trendline.")
    
        # Plot formatting
        plt.xlabel('Average Inter-Mouse Distance (pixels)')
        plt.ylabel('Cooperative Success Rate')
        plt.title('Success Probability vs. Average Inter-Mouse Distance')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
    
        # Save plot if enabled
        if self.save:
            plt.savefig(f"{self.prefix}Success_vs_AvgDistance.png")
        plt.show()
        plt.close()

    def _calculate_trial_metrics(self, experiment):
        """
        Calculates waiting-related metrics for a single experiment.

        Args:
            experiment: Object containing lever (lev) and position (pos) data loaders.

        Returns:
            tuple: Comprehensive metrics for the experiment including waiting times,
                   latencies, symmetry, and occupancy curves.
        """
        lever_data = experiment.lev
        position_data = experiment.pos
        fps = experiment.fps

        # Initialize counters and accumulators
        trial_count = 0 #Number of trials used
        successful_trials_used = 0 #Number of trials used that were successful
        total_trial_frames = 0 #Number of frames in trials used
        total_waiting_frames = 0 #Number of frames waited (between queue and first lever press) in trials used
        same_lever_frames = 0 #Out of frames in which both rats were in the lever area (between queue and first lever press), the number in which both were in the same lever area
        opposite_lever_frames = 0 #Out of frames in which both rats were in the lever area (between queue and first lever press), the number in which both were in a different lever area
        no_rat_frames = 0 #Out of all frames (between queue and first lever press), number of frames in which no rat was in lever area
        one_rat_frames = 0 #Out of all frames (between queue and first lever press), number of frames in which one rat was in lever area
        both_rats_frames = 0 #Out of all frames (between queue and first lever press), number of frames in which both rats were in lever area
        frames_both_waited_before = 0  # Total number of frames before trial start where both rats were simultaneously in a lever area (lev_top or lev_bottom)
        frames_both_waited_before_success = 0  # Total number of frames before trial start where both rats were simultaneously in a lever area, for successful trials only
        frames_waited_success = 0  # Total number of frames waited (maximum of rat0 or rat1) before trial start in successful trials
        frames_waited_all = 0  # Total number of frames waited (maximum of rat0 or rat1) before trial start across all trials
        max_frames_waited_before = []  # List of maximum waiting frames (max of rat0 or rat1) before trial start for each trial
        
        # Initialize lists for per-trial metrics
        rat0_wait_times = []  # List of frames rat0 spent waiting in a lever area (lev_top or lev_bottom) per trial, before first lever press
        rat1_wait_times = []  # List of frames rat1 spent waiting in a lever area (lev_top or lev_bottom) per trial, before first lever press
        waiting_symmetry = []  # List of absolute differences in waiting frames between rat0 and rat1 per trial, before first lever press
        waiting_symmetry_before = [] # List of absolute differences in waiting frames between rat0 and rat1 per trial, before queue
        synchronous_wait_frames = []  # List of frames per trial where both rats were simultaneously in a lever area, before first lever press
        rat0_latencies = []  # List of frames until rat0 first enters a lever area after trial start, per trial
        rat1_latencies = []  # List of frames until rat1 first enters a lever area after trial start, per trial
        max_wait_before_press = []
        
        # Initialize occupancy curves
        occupancy_curve = np.zeros(self.NUM_BINS)  # Array tracking total frames where at least one rat was in a lever area, across normalized trial time bins
        occupancy_curve_fail = np.zeros(self.NUM_BINS)
        occupancy_curve_success = np.zeros(self.NUM_BINS)
        occupancy_curve_fail_mag = np.zeros(self.NUM_BINS)
        occupancy_curve_success_mag = np.zeros(self.NUM_BINS)
        occupancy_curve_mag = np.zeros(self.NUM_BINS)
        occupancy_curve_both = np.zeros(self.NUM_BINS)  # Array tracking total frames where both rats were in a lever area, across normalized trial time bins
        occupancy_curve_both_mag = np.zeros(self.NUM_BINS)
        trial_counts = np.zeros(self.NUM_BINS)  # Array tracking the number of valid frames contributing to each time bin for normalization
        trial_counts_fail = np.zeros(self.NUM_BINS)
        trial_counts_success = np.zeros(self.NUM_BINS)
        
        # Retrieve trial data
        start_times = lever_data.returnTimeStartTrials()  # Array of trial start times (in seconds) for all trials
        end_times = lever_data.returnTimeEndTrials()  # Array of trial end times (in seconds) for all trials
        press_times = lever_data.returnFirstPressAbsTimes()  # Array of first lever press times (in seconds) for each trial
        press_rat_ids = lever_data.returnRatIDFirstPressTrial()  # Array of rat IDs (0 or 1) that made the first lever press in each trial
        total_trials = lever_data.returnNumTotalTrialswithLeverPress()  # Total number of trials with at least one lever press
        success_trials = lever_data.returnSuccessTrials()  # Array indicating whether each trial was successful (True/False)

        invalid_trial_count = 0

        for trial_idx in range(total_trials):
            start_time = start_times[trial_idx]
            end_time = end_times[trial_idx]
            press_time = press_times[trial_idx]
            rat_id = press_rat_ids[trial_idx]

            # Validate trial data
            if any(np.isnan([start_time, end_time, press_time, rat_id])) or start_time is None or end_time is None:
                invalid_trial_count += 1
                continue

            # Convert times to frames
            start_frame = int(start_time * fps)
            press_frame = int(press_time * fps)
            end_frame = int(end_time * fps)

            if press_frame <= start_frame or press_frame >= position_data.data.shape[-1] or end_frame < start_frame:
                invalid_trial_count += 1
                continue

            trial_count += 1
            trial_frames = press_frame - start_frame
            total_trial_frames += trial_frames

            # Calculate waiting before trial start
            rat0_locations = position_data.returnMouseLocation(0)
            rat1_locations = position_data.returnMouseLocation(1)
            
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

            max_wait_before = max(rat0_waiting, rat1_waiting)
            max_frames_waited_before.append(max_wait_before)
            frames_waited_all += max_wait_before
            frames_both_waited_before += min(rat0_waiting, rat1_waiting)
            
            waiting_symmetry_before.append(abs(rat0_waiting - rat1_waiting))

            if success_trials[trial_idx]:
                frames_both_waited_before_success += min(rat0_waiting, rat1_waiting)
                successful_trials_used += 1
                frames_waited_success += max_wait_before


            #Wait Before Press Analaysis
            t = press_frame - 1
            rat0_waiting_before_press = 0
            rat1_waiting_before_press = 0
            rat0_active = True
            rat1_active = True

            while t >= 0 and t < len(rat0_locations) and t < len(rat1_locations) and rat0_locations[t] is not None:
                if rat0_locations[t] in ['lev_top', 'lev_bottom'] and rat0_active:
                    rat0_waiting_before_press += 1
                else:
                    rat0_active = False

                if rat1_locations[t] in ['lev_top', 'lev_bottom'] and rat1_active:
                    rat1_waiting_before_press += 1
                else:
                    rat1_active = False

                if not (rat0_active or rat1_active):
                    break
                t -= 1
            
            waitBeforePress = max(rat0_waiting_before_press, rat1_waiting_before_press)
            max_wait_before_press.append(waitBeforePress)
            
            # Process trial window
            rat0_trial_locations = rat0_locations[start_frame:press_frame]
            rat1_trial_locations = rat1_locations[start_frame:press_frame]
            rat0_full_locations = rat0_locations[start_frame:end_frame]
            rat1_full_locations = rat1_locations[start_frame:end_frame]

            frame_count_min = min(len(rat0_trial_locations), len(rat1_trial_locations))
            frame_count_full = min(len(rat0_full_locations), len(rat1_full_locations))

            # Calculate latency to lever entry
            def get_latency_to_lever(locations):
                for i, loc in enumerate(locations):
                    if loc in ['lev_top', 'lev_bottom']:
                        return i
                return None

            rat0_latency = get_latency_to_lever(rat0_trial_locations)
            rat1_latency = get_latency_to_lever(rat1_trial_locations)
            
            if rat0_latency is not None:
                rat0_latencies.append(rat0_latency)
            if rat1_latency is not None:
                rat1_latencies.append(rat1_latency)

            # Count lever occupancy
            for i in range(frame_count_min):
                r0 = rat0_trial_locations[i]
                r1 = rat1_trial_locations[i]
                r0_in_lever = r0 in ['lev_top', 'lev_bottom']
                r1_in_lever = r1 in ['lev_top', 'lev_bottom']
                
                if r0_in_lever and r1_in_lever:
                    if r0 == r1:
                        same_lever_frames += 1
                    else:
                        opposite_lever_frames += 1
                    both_rats_frames += 1
                elif r0_in_lever or r1_in_lever:
                    one_rat_frames += 1
                else:
                    no_rat_frames += 1

            # Calculate occupancy curves
            bin_edges = np.linspace(0, frame_count_full, self.NUM_BINS + 1, dtype=int)
            for bin_idx in range(self.NUM_BINS):
                start_bin = bin_edges[bin_idx]
                end_bin = bin_edges[bin_idx + 1]
                
                if start_bin >= frame_count_full:
                    continue

                rat0_bin = rat0_full_locations[start_bin:end_bin]
                rat1_bin = rat1_full_locations[start_bin:end_bin]
                
                mag_zones = {'mag_top', 'mag_bottom'}
                lever_zones = {'lev_top', 'lev_bottom'}
                in_lever = sum(
                    (r0 in lever_zones) or (r1 in lever_zones)
                    for r0, r1 in zip(rat0_bin, rat1_bin)
                )
                in_lever_both = sum(
                    (r0 in lever_zones) and (r1 in lever_zones)
                    for r0, r1 in zip(rat0_bin, rat1_bin)
                )
                
                in_mag = sum(
                    (r0 in mag_zones) or (r1 in mag_zones)
                    for r0, r1 in zip(rat0_bin, rat1_bin)
                )
                in_mag_both = sum(
                    (r0 in mag_zones) and (r1 in mag_zones)
                    for r0, r1 in zip(rat0_bin, rat1_bin)
                )
                
                occupancy_curve[bin_idx] += in_lever
                occupancy_curve_both[bin_idx] += in_lever_both
                occupancy_curve_mag[bin_idx] += in_mag
                occupancy_curve_both_mag[bin_idx] += in_mag_both
                trial_counts[bin_idx] += (end_bin - start_bin)
                
                if (success_trials[trial_idx]):
                    occupancy_curve_success[bin_idx] += in_lever
                    occupancy_curve_success_mag[bin_idx] += in_mag
                    trial_counts_success[bin_idx] += (end_bin - start_bin)
                else:
                    occupancy_curve_fail[bin_idx] += in_lever
                    occupancy_curve_fail_mag[bin_idx] += in_mag
                    trial_counts_fail[bin_idx] += (end_bin - start_bin)
                    

            # Calculate waiting metrics
            waiting_frames = sum(
                (rat0_trial_locations[i] in ['lev_top', 'lev_bottom']) or
                (rat1_trial_locations[i] in ['lev_top', 'lev_bottom'])
                for i in range(frame_count_min)
            )
            total_waiting_frames += waiting_frames

            sync_waiting = sum(
                (rat0_trial_locations[i] in ['lev_top', 'lev_bottom']) and
                (rat1_trial_locations[i] in ['lev_top', 'lev_bottom'])
                for i in range(frame_count_min)
            )
            synchronous_wait_frames.append(sync_waiting)

            rat0_trial_waiting = sum(1 for loc in rat0_trial_locations if loc in ['lev_top', 'lev_bottom'])
            rat1_trial_waiting = sum(1 for loc in rat1_trial_locations if loc in ['lev_top', 'lev_bottom'])
            waiting_symmetry.append(abs(rat0_trial_waiting - rat1_trial_waiting))

        return (
            rat0_wait_times, rat1_wait_times, waiting_symmetry, waiting_symmetry_before, rat0_latencies, rat1_latencies,
            synchronous_wait_frames, total_trial_frames, total_waiting_frames,
            same_lever_frames, opposite_lever_frames, no_rat_frames, one_rat_frames, both_rats_frames,
            occupancy_curve, occupancy_curve_success, occupancy_curve_fail, occupancy_curve_mag, 
            occupancy_curve_success_mag, occupancy_curve_fail_mag, occupancy_curve_both, occupancy_curve_both_mag,
            trial_counts, trial_counts_success, trial_counts_fail, np.mean(max_frames_waited_before) if max_frames_waited_before else 0, 
            np.mean(max_wait_before_press) if max_wait_before_press else 0,
            frames_both_waited_before, frames_waited_success, frames_waited_all,
            trial_count, successful_trials_used, frames_both_waited_before_success
        )

    def waitingStrategy(self):
        """
        Aggregates data across experiments and generates visualizations including:
        - Bar plots comparing waiting times
        - Scatter plots of waiting times vs. success rates
        - Pie charts of lever occupancy
        - Line plots of latency and occupancy over trials
        """
        # Aggregate metrics across experiments
        avg_waiting_times = []
        success_rates = []
        avg_symmetry_values = []
        avg_symmetry_values_before = []
        avg_sync_values = []
        rat0_latencies_per_trial = defaultdict(list)
        rat1_latencies_per_trial = defaultdict(list)
        total_trial_frames = 0
        total_waiting_frames = 0
        total_trials = 0
        total_successful_trials = 0
        max_wait_before_means = []
        max_wait_before_press_means = []
        total_wait_before_success = 0
        total_wait_before_all = 0
        both_wait_before_means = []
        total_both_wait_before = 0
        total_both_wait_before_success = 0
        same_lever_sum = 0
        opposite_lever_sum = 0
        no_rat_sum = 0
        one_rat_sum = 0
        both_rats_sum = 0
        total_occupancy_curve = np.zeros(self.NUM_BINS)
        total_occupancy_curve_succ = np.zeros(self.NUM_BINS)
        total_occupancy_curve_fail = np.zeros(self.NUM_BINS)
        all_occupancy_curves = []
        all_occupancy_curves_succ = []
        all_occupancy_curves_fail = []
        
        total_occupancy_curve_mag = np.zeros(self.NUM_BINS)
        total_occupancy_curve_succ_mag = np.zeros(self.NUM_BINS)
        total_occupancy_curve_fail_mag = np.zeros(self.NUM_BINS)
        all_occupancy_curves_mag = []
        all_occupancy_curves_succ_mag = []
        all_occupancy_curves_fail_mag = []
        
        total_occupancy_curve_both = np.zeros(self.NUM_BINS)
        total_occupancy_curve_both_mag = np.zeros(self.NUM_BINS)
        total_trial_counts = np.zeros(self.NUM_BINS)
        total_trial_counts_succ = np.zeros(self.NUM_BINS)
        total_trial_counts_fail = np.zeros(self.NUM_BINS)

        for exp in self.experiments:
            metrics = self._calculate_trial_metrics(exp)
            (
                rat0_times, rat1_times, symmetry, symmetry_before, rat0_lat, rat1_lat, sync_frames,
                trial_frames, waiting_frames, same_lever, opposite_lever, no_rat,
                one_rat, both_rats, occupancy_curve, occupancy_curve_success, occupancy_curve_fail, occupancy_curve_mag, occupancy_curve_success_mag, occupancy_curve_fail_mag, occupancy_curve_both,
                occupancy_curve_both_mag, trial_counts, trial_counts_success, trial_counts_fail, avg_wait_before, avg_wait_before_press, 
                frames_both_waited_before, frames_wait_before_success, frames_waited_all, num_trials, 
                num_success_trials, frames_both_waited_before_success
            ) = metrics

            total_trials_overall = exp.lev.returnNumTotalTrials()
            if num_trials == 0:
                continue

            total_trial_frames += trial_frames
            total_waiting_frames += waiting_frames
            total_trials += num_trials
            total_successful_trials += num_success_trials
            total_wait_before_all += frames_waited_all
            total_wait_before_success += frames_wait_before_success
            total_both_wait_before += frames_both_waited_before
            total_both_wait_before_success += frames_both_waited_before_success
            max_wait_before_means.append(avg_wait_before)
            max_wait_before_press_means.append(avg_wait_before_press)
            both_wait_before_means.append(frames_both_waited_before / num_trials if num_trials > 0 else 0)

            avg_waiting_times.append(waiting_frames / trial_frames if trial_frames > 0 else 0)
            success_rates.append(exp.lev.returnNumSuccessfulTrials() / total_trials_overall if total_trials_overall > 0 else 0)
            avg_symmetry_values.append(np.sum(symmetry) / num_trials if num_trials > 0 else 0)
            avg_symmetry_values_before.append(np.sum(symmetry_before) / num_trials if num_trials > 0 else 0)
            avg_sync_values.append(np.mean(sync_frames) if sync_frames else 0)

            for trial_idx, (lat0, lat1) in enumerate(zip(rat0_lat, rat1_lat)):
                if lat0 is not None and lat1 is not None:
                    rat0_latencies_per_trial[trial_idx].append(lat0)
                    rat1_latencies_per_trial[trial_idx].append(lat1)

            same_lever_sum += same_lever
            opposite_lever_sum += opposite_lever
            no_rat_sum += no_rat
            one_rat_sum += one_rat
            both_rats_sum += both_rats
            total_occupancy_curve += occupancy_curve
            total_occupancy_curve_succ += occupancy_curve_success
            total_occupancy_curve_fail += occupancy_curve_fail
            
            total_occupancy_curve_mag += occupancy_curve_mag
            total_occupancy_curve_succ_mag += occupancy_curve_success_mag
            total_occupancy_curve_fail_mag += occupancy_curve_fail_mag
            
            if np.any(trial_counts):
                all_occupancy_curves.append(occupancy_curve / trial_counts)
            if np.any(trial_counts_success):
                all_occupancy_curves_succ.append(occupancy_curve_success / trial_counts_success)
            if np.any(trial_counts_fail):
                all_occupancy_curves_fail.append(occupancy_curve_fail / trial_counts_fail)
                
            if np.any(trial_counts):
                all_occupancy_curves_mag.append(occupancy_curve_mag / trial_counts)
            if np.any(trial_counts_success):
                all_occupancy_curves_succ_mag.append(occupancy_curve_success_mag / trial_counts_success)
            if np.any(trial_counts_fail):
                all_occupancy_curves_fail_mag.append(occupancy_curve_fail_mag / trial_counts_fail)
            
            total_occupancy_curve_both += occupancy_curve_both
            total_occupancy_curve_both_mag += occupancy_curve_both_mag
            total_trial_counts += trial_counts
            total_trial_counts_succ += trial_counts_success
            total_trial_counts_fail += trial_counts_fail

        # Calculate average latency per trial
        avg_latency_per_trial = []
        max_trial_index = max(rat0_latencies_per_trial.keys() & rat1_latencies_per_trial.keys(), default=-1)
        for trial_idx in range(max_trial_index + 1):
            if trial_idx in rat0_latencies_per_trial and trial_idx in rat1_latencies_per_trial:
                avg0 = np.mean(rat0_latencies_per_trial[trial_idx])
                avg1 = np.mean(rat1_latencies_per_trial[trial_idx])
                avg_latency_per_trial.append((avg0 + avg1) / 2)
        
        
        # Generate visualizations
        self._plot_bar_wait_times(
            total_both_wait_before_success / total_successful_trials if total_successful_trials > 0 else 0,
            total_both_wait_before / total_trials if total_trials > 0 else 0,
            "both_rats_success_vs_all.png",
            "Average Wait Time by Both Rats: Successful vs. All Trials"
        )

        self._plot_bar_wait_times(
            total_wait_before_success / total_successful_trials if total_successful_trials > 0 else 0,
            total_wait_before_all / total_trials if total_trials > 0 else 0,
            "max_wait_success_vs_all.png",
            "Average Wait Time: Successful vs. All Trials"
        )

        self._plot_scatter(
            both_wait_before_means, success_rates,
            "both_rats_waiting_before_vs_success.png",
            "Avg Waiting Time of Both Rats Before Trial vs. Cooperative Success Rate",
            "Average Waiting Time before Queue (frames)"
        )

        self._plot_scatter(
            avg_waiting_times, success_rates,
            "waiting_during_trial_vs_success.png",
            "Waiting Time during Trial vs. Cooperative Success Rate",
            "Average Waiting Time before Press (frames)"
        )

        self._plot_scatter(
            max_wait_before_means, success_rates,
            "waiting_before_vs_success.png",
            "Waiting Time Before Trial vs. Cooperative Success Rate",
            "Average Waiting Time Before (frames)"
        )
        
        self._plot_scatter(
            max_wait_before_press_means, success_rates,
            "waiting_before_press_vs_success.png",
            "Waiting Time Before Press vs. Cooperative Success Rate",
            "Average Waiting Time Before (frames)"
        )

        self._plot_scatter(
            avg_symmetry_values, success_rates,
            "symmetry_vs_success.png",
            "Success Rate vs. Waiting Symmetry",
            "Average Waiting Symmetry (|rat0 - rat1|)",
            color_data=max_wait_before_press_means
        )
        
        self._plot_scatter(
            avg_symmetry_values_before, success_rates,
            "symmetryBefore_vs_success.png",
            "Success Rate vs. Waiting Symmetry Before Queue",
            "Average Waiting Symmetry (|rat0 - rat1|)",
        )

        self._plot_pie(
            [total_waiting_frames / total_trial_frames * 100 if total_trial_frames > 0 else 0,
             100 - (total_waiting_frames / total_trial_frames * 100 if total_trial_frames > 0 else 0)],
            ['Waiting in Lever Areas', 'Not Waiting'],
            ['green', 'gray'],
            "waiting_time_distribution.png",
            "Percentage of Trial Time Spent Waiting in Lever Areas"
        )

        self._plot_pie(
            [no_rat_sum, one_rat_sum, both_rats_sum],
            ['Neither', 'One', 'Both'],
            ['gray', 'orange', 'green'],
            "lever_zone_occupancy.png",
            "Lever Zone Occupancy (None / One / Both Rats)"
        )

        self._plot_pie(
            [same_lever_sum, opposite_lever_sum],
            ['Same Lever Area', 'Opposite Lever Areas'],
            ['lightgreen', 'salmon'],
            "same_vs_opposite_lever.png",
            "When Both Rats Wait: \nSame vs. Opposite Lever Area"
        )
        
        if len(avg_latency_per_trial) >= 5:
            self._plot_line(
                pd.Series(avg_latency_per_trial).rolling(window=5, min_periods=1, center=True).mean(),
                "smoothed_latency_over_trials.png",
                "Smoothed Lever Entry Latency Over Trials",
                "Trial Index (Across Experiments)",
                "Average Latency to Lever Entry (frames)"
            )
        
        avg_occupancy = total_occupancy_curve / total_trial_counts if np.any(total_trial_counts) else np.zeros(self.NUM_BINS)
        avg_occupancy_succ = total_occupancy_curve_succ / total_trial_counts_succ if np.any(total_trial_counts_succ) else np.zeros(self.NUM_BINS)
        avg_occupancy_fail = total_occupancy_curve_fail / total_trial_counts_fail if np.any(total_trial_counts_fail) else np.zeros(self.NUM_BINS)
        # Calculate standard deviations
        std_occupancy = np.std(all_occupancy_curves, axis=0) if all_occupancy_curves else np.zeros(self.NUM_BINS)
        std_occupancy_succ = np.std(all_occupancy_curves_succ, axis=0) if all_occupancy_curves_succ else np.zeros(self.NUM_BINS)
        std_occupancy_fail = np.std(all_occupancy_curves_fail, axis=0) if all_occupancy_curves_fail else np.zeros(self.NUM_BINS)
        plt.figure(figsize=(8, 5))
        x_vals = np.linspace(0, 100, self.NUM_BINS)
        
        # Plot lines with Gaussian smoothing
        plt.plot(x_vals, gaussian_filter1d(avg_occupancy, sigma=2), color='red', linewidth=2, label='All Trials')
        plt.plot(x_vals, gaussian_filter1d(avg_occupancy_succ, sigma=2), color='green', linewidth=2, label='Successful Trials')
        plt.plot(x_vals, gaussian_filter1d(avg_occupancy_fail, sigma=2), color='purple', linewidth=2, label='Failed Trials')
        
        # Plot standard deviation shaded areas
        plt.fill_between(x_vals, 
                         gaussian_filter1d(avg_occupancy - std_occupancy, sigma=2), 
                         gaussian_filter1d(avg_occupancy + std_occupancy, sigma=2), 
                         color='red', alpha=0.2)
        plt.fill_between(x_vals, 
                         gaussian_filter1d(avg_occupancy_succ - std_occupancy_succ, sigma=2), 
                         gaussian_filter1d(avg_occupancy_succ + std_occupancy_succ, sigma=2), 
                         color='green', alpha=0.2)
        plt.fill_between(x_vals, 
                         gaussian_filter1d(avg_occupancy_fail - std_occupancy_fail, sigma=2), 
                         gaussian_filter1d(avg_occupancy_fail + std_occupancy_fail, sigma=2), 
                         color='purple', alpha=0.2)
        
        plt.xlabel("Trial Time (% of trial)", fontsize=self.labelSize)
        plt.ylabel("Probability of Lever Occupancy", fontsize=self.labelSize)
        plt.title("Lever Zone Occupancy Over Trial Duration", fontsize=self.titleSize)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if self.save:
            plt.savefig(f"{self.prefix}lever_occupancy_over_time.png")
        plt.show()
        plt.close()
        
        '''self._plot_line(
            gaussian_filter1d(avg_occupancy, sigma=2),
            "lever_occupancy_over_time.png",
            "Lever Zone Occupancy Over Trial Duration",
            "Trial Time (% of trial)",
            "Probability of Lever Occupancy",
            x_vals=np.linspace(0, 100, self.NUM_BINS)
        )'''

        avg_occupancy_both = total_occupancy_curve_both / total_trial_counts if np.any(total_trial_counts) else np.zeros(self.NUM_BINS)
        self._plot_line(
            gaussian_filter1d(avg_occupancy_both, sigma=2),
            "dual_lever_occupancy_over_time.png",
            "Dual Lever Zone Occupancy Over Trial Duration",
            "Trial Time (% of trial)",
            "Probability of Lever Occupancy by Both Rats",
            x_vals=np.linspace(0, 100, self.NUM_BINS)
        )
        
        avg_occupancy_mag = total_occupancy_curve_mag / total_trial_counts if np.any(total_trial_counts) else np.zeros(self.NUM_BINS)
        avg_occupancy_succ_mag = total_occupancy_curve_succ_mag / total_trial_counts_succ if np.any(total_trial_counts_succ) else np.zeros(self.NUM_BINS)
        avg_occupancy_fail_mag = total_occupancy_curve_fail_mag / total_trial_counts_fail if np.any(total_trial_counts_fail) else np.zeros(self.NUM_BINS)
        # Calculate standard deviations
        std_occupancy = np.std(all_occupancy_curves_mag, axis=0) if all_occupancy_curves_mag else np.zeros(self.NUM_BINS)
        std_occupancy_succ = np.std(all_occupancy_curves_succ_mag, axis=0) if all_occupancy_curves_succ_mag else np.zeros(self.NUM_BINS)
        std_occupancy_fail = np.std(all_occupancy_curves_fail_mag, axis=0) if all_occupancy_curves_fail_mag else np.zeros(self.NUM_BINS)
        plt.figure(figsize=(8, 5))
        x_vals = np.linspace(0, 100, self.NUM_BINS)
        
        # Plot lines with Gaussian smoothing
        plt.plot(x_vals, gaussian_filter1d(avg_occupancy_mag, sigma=2), color='red', linewidth=2, label='All Trials')
        plt.plot(x_vals, gaussian_filter1d(avg_occupancy_succ_mag, sigma=2), color='green', linewidth=2, label='Successful Trials')
        plt.plot(x_vals, gaussian_filter1d(avg_occupancy_fail_mag, sigma=2), color='purple', linewidth=2, label='Failed Trials')
        
        # Plot standard deviation shaded areas
        plt.fill_between(x_vals, 
                         gaussian_filter1d(avg_occupancy_mag - std_occupancy, sigma=2), 
                         gaussian_filter1d(avg_occupancy_mag + std_occupancy, sigma=2), 
                         color='red', alpha=0.2)
        plt.fill_between(x_vals, 
                         gaussian_filter1d(avg_occupancy_succ_mag - std_occupancy_succ, sigma=2), 
                         gaussian_filter1d(avg_occupancy_succ_mag + std_occupancy_succ, sigma=2), 
                         color='green', alpha=0.2)
        plt.fill_between(x_vals, 
                         gaussian_filter1d(avg_occupancy_fail_mag - std_occupancy_fail, sigma=2), 
                         gaussian_filter1d(avg_occupancy_fail_mag + std_occupancy_fail, sigma=2), 
                         color='purple', alpha=0.2)
        
        plt.xlabel("Trial Time (% of trial)", fontsize=self.labelSize)
        plt.ylabel("Probability of Magazine Occupancy", fontsize=self.labelSize)
        plt.title("Mag Zone Occupancy Over Trial Duration", fontsize=self.titleSize)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if self.save:
            plt.savefig(f"{self.prefix}mag_occupancy_over_time.png")
        plt.show()
        plt.close()
        
        '''
        avg_occupancy = total_occupancy_curve_mag / total_trial_counts if np.any(total_trial_counts) else np.zeros(self.NUM_BINS)
        self._plot_line(
            gaussian_filter1d(avg_occupancy, sigma=2),
            "mag_occupancy_over_time.png",
            "Mag Zone Occupancy Over Trial Duration",
            "Trial Time (% of trial)",
            "Probability of Mag Occupancy",
            x_vals=np.linspace(0, 100, self.NUM_BINS)
        )
        '''

        avg_occupancy_both = total_occupancy_curve_both_mag / total_trial_counts if np.any(total_trial_counts) else np.zeros(self.NUM_BINS)
        self._plot_line(
            gaussian_filter1d(avg_occupancy_both, sigma=2),
            "dual_mag_occupancy_over_time.png",
            "Dual Mag Zone Occupancy Over Trial Duration",
            "Trial Time (% of trial)",
            "Probability of Mag Occupancy by Both Rats",
            x_vals=np.linspace(0, 100, self.NUM_BINS)
        )
        
    def _plot_bar_wait_times(self, success_value, all_value, filename, title):
        """Plots a bar chart comparing wait times for successful vs. all trials."""
        plt.figure(figsize=(6, 5))
        bars = plt.bar(['Successful Trials', 'All Trials'], [success_value, all_value], color=['green', 'gray'])
        plt.ylabel('Average Wait Time (frames)', fontsize = self.labelSize)
        plt.title(title, fontsize = self.titleSize)
        plt.xticks(fontsize=self.labelSize)
        plt.yticks(fontsize=self.labelSize)
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{bar.get_height():.2f}', ha='center', va='bottom')
        plt.tight_layout()
        if self.save:
            plt.savefig(f"{self.prefix}{filename}")
        plt.show()
        plt.close()

    def _plot_scatter(self, x_data, y_data, filename, title, x_label, color_data=None):
        """Plots a scatter plot with trendline and R² value, optionally with gradient coloring."""
        if len(x_data) < 2 or len(y_data) < 2:
            print(f"Insufficient data to create scatterplot for {filename}")
            return
    
        plt.figure(figsize=(8, 6))
        
        if color_data is not None:
            # Use a colormap for gradient coloring based on color_data
            norm = plt.Normalize(min(color_data), max(color_data))
            cmap = plt.cm.viridis
            scatter = plt.scatter(x_data, y_data, c=color_data, cmap=cmap, norm=norm, alpha=0.7, 
                             edgecolors='black', linewidths=1, label='Experiments', s=70)
            plt.colorbar(scatter, label='Average Waiting Time Before Press (frames)')
        else:
            plt.scatter(x_data, y_data, alpha=0.7, color='blue', label='Experiments')
        
        if len(set(x_data)) >= 2:
            if color_data is not None:
                # Define negative exponential decay function: y = a * exp(-b * x) + c
                def exp_decay(x, a, b, c):
                    return a * np.exp(-b * x) + c
                
                # Fit exponential model
                try:
                    popt, _ = curve_fit(exp_decay, x_data, y_data, p0=[max(y_data), 0.1, min(y_data)])
                    x_vals = np.linspace(min(x_data), max(x_data), 100)
                    y_fit = exp_decay(x_vals, *popt)
                    
                    # Calculate pseudo-R²
                    y_mean = np.mean(y_data)
                    ss_tot = sum((y - y_mean) ** 2 for y in y_data)
                    ss_res = sum((y - exp_decay(x, *popt)) ** 2 for x, y in zip(x_data, y_data))
                    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                    
                    # Plot exponential fit
                    plt.plot(x_vals, y_fit, color='red', linestyle='--', label='Exponential Fit')
                    plt.text(0.68, 0.93, f"Pseudo-R² = {r_squared:.3f}", transform=plt.gca().transAxes,
                             ha='right', va='bottom', fontsize=12, bbox=dict(facecolor='white', edgecolor='gray'))
                except RuntimeError:
                    print(f"Exponential fit failed for {filename}, falling back to scatter without fit")
            else:
                # Linear regression for non-colored case
                slope, intercept, r_value, _, _ = linregress(x_data, y_data)
                x_vals = np.linspace(min(x_data), max(x_data), 100)
                plt.plot(x_vals, slope * x_vals + intercept, color='red', linestyle='--', label='Trendline')
                plt.text(0.95, 0.05, f"$R^2$ = {r_value**2:.3f}", transform=plt.gca().transAxes,
                         ha='right', va='bottom', fontsize=12, bbox=dict(facecolor='white', edgecolor='gray'))
        
        plt.xlabel(x_label, fontsize=self.labelSize)
        plt.ylabel('Cooperative Success Rate', fontsize=self.labelSize)
        plt.title(title, fontsize=self.titleSize)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if self.save:
            plt.savefig(f"{self.prefix}{filename}")
        plt.show()
        plt.close()

    def _plot_pie(self, sizes, labels, colors, filename, title):
        """Plots a pie chart with percentage labels."""
        if sum(sizes) == 0:
            print(f"No valid data for pie chart {filename}")
            return

        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, textprops={'fontsize': self.labelSize})
        plt.title(title, fontsize = self.titleSize)
        plt.axis('equal')
        plt.tight_layout()
        if self.save:
            plt.savefig(f"{self.prefix}{filename}")
        plt.show()
        plt.close()

    def _plot_line(self, y_data, filename, title, x_label, y_label, x_vals=None):
        """Plots a line graph with optional custom x-values."""
        plt.figure(figsize=(8, 5))
        if x_vals is None:
            plt.plot(y_data, color='blue', linewidth=2)
        else:
            plt.plot(x_vals, y_data, color='blue', linewidth=2)
        plt.xlabel(x_label, fontsize = self.labelSize)
        plt.ylabel(y_label, fontsize = self.labelSize)
        plt.title(title, fontsize = self.titleSize)
        plt.grid(True)
        plt.tight_layout()
        if self.save:
            plt.savefig(f"{self.prefix}{filename}")
        plt.show()
        plt.close()
            
    def successRateVsThresholdPlot(self):
        """
        Plots the average cooperative success rate for each threshold value.
        Applies smoothing to visualize trends.
        Weights trials equally (not each experiment). 
        Annotates each point with number of sessions (experiments) used.
        """
        threshold_to_rates = defaultdict(list)
    
        # Aggregate success rates by threshold
        for exp in self.experiments:
            lev = exp.lev
            threshold = lev.returnSuccThreshold()
            num_succ = lev.returnNumSuccessfulTrials()
            num_total = lev.returnNumTotalTrials()
    
            if num_total > 0:
                rate = num_succ / num_total
                threshold_to_rates[threshold].append(rate)
    
        # Compute average success rate per threshold
        thresholds = sorted(threshold_to_rates.keys())
        avg_rates = [np.mean(threshold_to_rates[t]) for t in thresholds]
        session_counts = [len(threshold_to_rates[t]) for t in thresholds]
    
        # Smooth using rolling average (pandas)
        df = pd.DataFrame({'Threshold': thresholds, 'AvgSuccessRate': avg_rates, 'NumSessions': session_counts})
        df.set_index('Threshold', inplace=True)
        df['Smoothed'] = df['AvgSuccessRate'].rolling(window=2, min_periods=1, center=True).mean()
    
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(df.index, df['AvgSuccessRate'], 'o-', label='Raw Average', color='gray', alpha=0.6)
        plt.plot(df.index, df['Smoothed'], 'r-', label='Smoothed', linewidth=2)
        
        # Add text annotations above each point
        for x, y, n in zip(df.index, df['AvgSuccessRate'], df['NumSessions']):
            plt.text(x, y + 0.02, f"{n} sess", ha='center', va='bottom', fontsize=9, color='blue')
        
        plt.xlabel('Cooperation Threshold')
        plt.ylabel('Average Success Rate')
        plt.title('Threshold vs. Success Rate')
        plt.xticks(df.index)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
    
        if self.save:
            plt.savefig(f"{self.prefix}threshold_vs_success_rate.png")
        plt.show()
        plt.close()

    def percentSameRatTakesBothRewards(self):
        """
        *Only Considers Sessions where at least 80% of RatID values are non-NaN.
        
        1) Computes the percentage of successful trials in which the same rat
        collects rewards from both magazines. (Out of the trials with data)
        
        2) For each session, identifies the rat that collects both rewards most frequently 
        and computes the average percentage of times the dominant rat collects both 
        rewards across all sessions.
        
        Creates pie charts to visualize the average percentages for both graphs
        """
        
        total_successful_trials = 0
        same_rat_both_rewards = 0
        dominant_count_total = 0
        session_dominant_rat_percentages = []
        session_success_percentages = []
        session_same_rat_percentages = []
        
        sessions_considered = 0

        for exp in self.experiments:
            lev = exp.lev
            mag = exp.mag
            
            # Calculate percentage of non-NaN RatID values
            rat_id_count = mag.data['RatID'].count()  # Non-NaN count
            total_rows = len(mag.data)
            non_nan_percentage = (rat_id_count / total_rows * 100) if total_rows > 0 else 0
            
            if (non_nan_percentage < 80 or total_rows < 50):
                continue
            
            sessions_considered += 1
            
            session_successful_trials = 0
            session_same_rat_both_rewards = 0
            session_same_rat_counts = {}  # Tracks counts per RatID collecting both rewards

            success_trials = lev.returnSuccessTrials()
            print("Success_trials: ", success_trials)
            print("NumSuccessful: ", sum(success_trials))
            print("Total: ", len(success_trials))

            for trial_index, is_success in enumerate(success_trials):
                if is_success != 1:
                    #print("\n Not Success")
                    continue  # Skip unsuccessful trials
                
                #print("\n Success, idx: ", trial_index)
                
                reward_recipients = mag.returnRewardRecipient(trial_index)
                if reward_recipients is None or len(reward_recipients) != 2:
                    #print("None")
                    continue  # Skip malformed trials
                #else: 
                    #print("Rewards: ", reward_recipients)

                session_successful_trials += 1
                total_successful_trials += 1

                if reward_recipients[0] == reward_recipients[1]:
                    session_same_rat_both_rewards += 1
                    same_rat_both_rewards += 1
                    rat_id = reward_recipients[0]
                    session_same_rat_counts[rat_id] = session_same_rat_counts.get(rat_id, 0) + 1

            if session_successful_trials > 0:
                # Calculate session overall same rat percentage
                session_same_rat_percentage = (session_same_rat_both_rewards / session_successful_trials) * 100
                session_same_rat_percentages.append(session_same_rat_percentage)
                
                # Calculate session success percentage
                session_total_trials = lev.returnNumTotalTrials()
                session_success_percentage = (session_successful_trials / session_total_trials) * 100 if session_total_trials > 0 else 0
                session_success_percentages.append(session_success_percentage)
                
                # Find the rat with the most instances of collecting both rewards
                if session_same_rat_counts:
                    print("session_same_rat_counts: ", session_same_rat_counts)
                    dominant_rat = max(session_same_rat_counts, key=session_same_rat_counts.get)
                    print("dominant_rat: ", dominant_rat)
                    dominant_count = session_same_rat_counts[dominant_rat]
                    print("dominant_count: ", dominant_count)
                    dominant_count_total += dominant_count
                    dominant_percentage = (dominant_count / session_same_rat_both_rewards) * 100
                    session_dominant_rat_percentages.append(dominant_percentage)
                else:
                    # No cases where same rat got both rewards, so dominant percentage is 0
                    session_dominant_rat_percentages.append(0)

        if total_successful_trials == 0:
            print("No successful trials found.")
            return

        # Compute overall percentage of same rat collecting both rewards
        overall_same_rat_percentage = (same_rat_both_rewards / total_successful_trials) * 100
        different_rat_percentage = 100 - overall_same_rat_percentage
        
        # Compute average percentage of dominant rat collecting both rewards
        avg_dominant_rat_percentage = dominant_count_total/same_rat_both_rewards * 100 if total_successful_trials > 0 else 0
        other_rat_percentage = 100 - avg_dominant_rat_percentage

        # Create pie chart for average dominant rat percentage
        labels = ['Dominant Rat', 'Submissive Rat']
        sizes = [avg_dominant_rat_percentage, other_rat_percentage]
        colors = ['green', 'lightcoral']
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 14})
        plt.title('Average Percentage of Dominant Rat Stealing Reward', fontsize=16)
        plt.axis('equal')
        plt.tight_layout()
        if self.save:
            plt.savefig(f'{self.prefix}DominantRatBothRewards.png')
        plt.show()
        plt.close()
        
        # Create pie chart for overall percentage of same rat collecting both rewards
        labels = ['Stolen', 'Equity']
        sizes = [overall_same_rat_percentage, different_rat_percentage]
        colors = ['blue', 'orange']
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 14})
        plt.title('Percentage of Successful Trials with Stolen Reward', fontsize=16)
        plt.axis('equal')
        plt.tight_layout()
        if self.save:
            plt.savefig(f'{self.prefix}SameRatBothRewards.png')
        plt.show()
        plt.close()
        
        # Create scatter plot for success percentage vs. session same rat percentage
        print("session_success_percentages: ", session_success_percentages)
        print("session_same_rat_percentages: ", session_same_rat_percentages)
        
        if len(session_success_percentages) > 1 and len(session_same_rat_percentages) > 1:
            success_np = np.array(session_success_percentages)
            same_rat_np = np.array(session_same_rat_percentages)
            
            # Fit a linear regression (degree-1 polynomial) to the data
            coeffs = np.polyfit(success_np, same_rat_np, 1)
            trendline = np.poly1d(coeffs)
            
            # Generate smooth x and y values for plotting the line
            xs = np.linspace(min(success_np), max(success_np), 100)
            ys = trendline(xs)
            
            # Plot the scatter points and trendline
            plt.figure(figsize=(8, 6))
            plt.scatter(session_success_percentages, session_same_rat_percentages, color='purple', alpha=0.6, s=100)
            plt.plot(xs, ys, color='red', linestyle='--', label='Trendline')
            
            # Calculate R^2
            predicted = trendline(success_np)
            ss_res = np.sum((same_rat_np - predicted) ** 2)
            ss_tot = np.sum((same_rat_np - np.mean(same_rat_np)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Display R² in the top right of the plot
            plt.text(0.95, 0.05, f'$R^2 = {r_squared:.3f}$',
                     transform=plt.gca().transAxes,
                     fontsize=12,
                     ha='right', va='bottom',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
            
            # Print slope and intercept
            slope, intercept = coeffs
            print(f"Trendline: y = {slope:.3f}x + {intercept:.3f}")
            
            plt.xlabel('Success Percentage (%)')
            plt.ylabel('Stealing Rewards Percentage (%)')
            plt.title('Success Percentage vs. Stealing Reward Collection Across Sessions')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlim(0, 100)
            plt.ylim(0, 100)
            plt.tight_layout()
            if self.save:
                plt.savefig(f'{self.prefix}SuccessVsStealingScatter.png')
            plt.show()
            plt.close()
        
        print(f"Overall: {same_rat_both_rewards}/{total_successful_trials} successful trials ({overall_same_rat_percentage:.1f}%) had the same rat collecting both rewards.")
        print(f"Average percentage of dominant rat collecting both rewards per session: {avg_dominant_rat_percentage:.1f}%")
        print(f"Sessions Considered: {sessions_considered}")
        
    def stateTransitionModel(self):
        """
        Constructs a behavioral state transition model based on spatial and event data.
        States:
            0 - idle
            1 - approaching lever
            2 - approaching reward
            3 - waiting
            4 - pressed
            5 - reward taken
            6 - exploring
        """
        
        state_names = ["idle", "approaching lever", "approaching reward", "waiting", "pressed", "reward taken", "exploring", "false mag"]
        num_states = len(state_names)
        transition_counts = np.zeros((num_states, num_states))
    
        for exp in self.experiments:
            pos = exp.pos
            lev = exp.lev
            mag = exp.mag
            fps = exp.fps
    
            total_frames = min(pos.returnNumFrames(), lev.endFrame)
            
            for rat_id in [0, 1]:
                pos_data = pos.getHeadBodyTrajectory(rat_id).T  # shape: (num_frames, 2) for x, y
                #print("pos_data: ", pos_data[0])
                velocities = pos.computeVelocity(rat_id)
                lever_zone = pos.getLeverZone(rat_id)
                reward_zone = pos.getRewardZone(rat_id)
                press_frames = lev.getLeverPressFrames(rat_id)
                reward_frames = mag.getRewardReceivedFrames(rat_id)
                false_mag_entry = mag.getEnteredMagFrames(rat_id)
    
                state_sequence = []
                for t in range(total_frames):
                    x, y = pos_data[t]
                    vel = velocities[t]
    
                    # Determine state
                    if (t > 2):
                        vel_before = np.mean(velocities[t - 2:t])
                    else:
                        vel_before = 0
                                    
                    if t in press_frames:
                        state = 4  # pressed
                    elif t in reward_frames:
                        state = 5  # reward taken
                    elif t in false_mag_entry:
                        state = 7 #mag entered but no reward
                    elif lever_zone[t]:
                        state = 3  # waiting
                    elif vel > 8 and pos.approachingMagazine(rat_id, t):
                        state = 2  # approaching reward
                    elif vel < 10 and vel_before < 10:
                        state = 0  # idle
                    elif vel > 8 and pos.approachingLever(rat_id, t):
                        state = 1  # approaching lever
                    else:
                        state = 6  # exploring
    
                    state_sequence.append(state)
    
                # Update transition matrix
                for a, b in zip(state_sequence[:-1], state_sequence[1:]):
                    transition_counts[a][b] += 1
    
        # Normalize to get probabilities
        row_sums = transition_counts.sum(axis=1)
        transition_matrix = np.divide(transition_counts, row_sums[:, np.newaxis], where=row_sums[:, np.newaxis] != 0)
    
        # --- Heatmap ---
        plt.figure(figsize=(8, 6))
        plt.imshow(transition_matrix, cmap='Blues')
        plt.colorbar(label='Transition Probability')
        plt.xticks(range(num_states), state_names, rotation=45)
        plt.yticks(range(num_states), state_names)
        plt.title("State Transition Probability Matrix")
        plt.tight_layout()
        plt.savefig(f"{self.prefix}state_transition_matrix.png")
        plt.show()
        plt.close()
    
        # --- Network Graph ---
        G = nx.DiGraph()
        for i in range(num_states):
            for j in range(num_states):
                prob = transition_matrix[i][j]
                if prob > 0:
                    G.add_edge(state_names[i], state_names[j], weight=prob)
    
        plt.figure(figsize=(10, 8))
        pos_layout = nx.spring_layout(G, seed=42)
        weights = [G[u][v]['weight'] * 10 for u, v in G.edges()]
        nx.draw(G, pos_layout, with_labels=True, node_color='lightblue', node_size=2000,
                arrows=True, width=weights, edge_color='gray', font_size=10)
        plt.title("State Transition Network (Edge Width = Frequency)")
        plt.tight_layout()
        plt.savefig(f"{self.prefix}state_transition_graph.png")
        plt.show()
        plt.close()
        
    def trialStateModel(self):
        '''
        Define 4 Stages to each Trial: 
            trial begin -> first press -> coopPress/lastPress -> first mag -> back to next trialBegin
                - Use lev.returnTimeStartTrials, lev.returnFirstPressAbsTimes, lev.returnCoopTimeorLastPressTime, mag.returnMagStartAbsTimes 
        
        Creates bar plots comparing all the following information across the 4 stages
        
        Classify Different Behaviors: 
            1) Make a heatmap for each category
            2) Classify Interactions
                a) How far apart are the rats? 
                b) How much do they gaze at each other? 
            3) Classify Individual Behavior
                a) How often are they idle? at Lever? at Magazine? 
                    - Create a single bar plot that shows percentage of time idle for each stage, with the bars colored in to show percentage of idle time that was spent at a lever, magazine, or other area
            4) Repeat 1-3 but only for successful trials
                 - Use lev.returnSuccessTrials
        '''
        
        def returnMagStartAbsTimes(lev, mag):
            """
            For each trial number present in lev.data (i.e., trials with lever presses),
            find the first magazine entry in mag.data with the same TrialNum.
            However, there's the additional condition that the abs time has to be greater 
            than that of the last lever press.
        
            Returns:
                list: A list of relative times (AbsTime - TrialTime) for mag entries
                      corresponding to lever press trials. Returns None if no mag entry exists.
                      Length equals lev.returnNumTotalTrialswithLeverPress().
            """
            if lev.data is None or mag.data is None:
                raise ValueError("Lever or magazine data is missing.")
        
            required_cols = {'TrialNum', 'AbsTime'}
            for loader_name, df in [('lev', lev.data), ('mag', mag.data)]:
                if not required_cols.issubset(df.columns):
                    raise ValueError(f"{loader_name}.data missing required columns: {required_cols - set(df.columns)}")
        
            # All trials with lever presses
            coopOrLastPress = lev.returnCoopTimeorLastPressTime()
            lever_trials = sorted(lev.data['TrialNum'].dropna().unique())
            #print("lever_trials: ", lever_trials)
            mag_grouped = mag.data.groupby('TrialNum')
        
            rel_times = []
            for trial_idx, trial in enumerate(lever_trials):
                if trial not in mag_grouped.groups:
                    rel_times.append(None)
                    continue
        
                group = mag_grouped.get_group(trial)
                if group.empty:
                    rel_times.append(None)
                    continue
        
                press_time = coopOrLastPress[trial_idx]
                # Filter for mag entries that occur after the press time
                valid_mags = group[group['AbsTime'] > press_time]
        
                if valid_mags.empty:
                    rel_times.append(None)
                else:
                    first_valid = valid_mags.loc[valid_mags['AbsTime'].idxmin()]
                    rel_times.append(first_valid['AbsTime'])
        
            return rel_times
        
        def filterToLeverPressTrials(original_list, lev):
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
        
        # Initialize data structures to aggregate across experiments
        stages = ['Begin to First Press', 'First Press to Coop/Last Press', 'Coop/Last Press to First Mag', 'First Mag to Next Begin']
        all_trials_data = []
        successful_trials_data = []
        
        # Initialize lists to collect heatmap for averaging per stage
        heatmap_data = {stage_idx: {
            'X_Coords': [],  # Store x coordinates for heatmap
            'Y_Coords': []   # Store y coordinates for heatmap
        } for stage_idx in range(len(stages))}
        heatmap_data_successful = {stage_idx: {
            'X_Coords': [],  # Store x coordinates for heatmap
            'Y_Coords': []   # Store y coordinates for heatmap
        } for stage_idx in range(len(stages))}
        
        
        for exp_idx, exp in enumerate(self.experiments):
            print("exp_idx: ", exp_idx)
            lev = exp.lev
            mag = exp.mag
            pos = exp.pos
            fps = exp.fps
            total_frames = exp.endFrame
            
            # Get trial timings
            trial_starts = lev.returnTimeStartTrials()  # List of trial start times
            first_presses = lev.returnFirstPressAbsTimes()  # List of first press times
            coop_or_last_press = lev.returnCoopTimeorLastPressTime()  # List of coop/last press times
            print("hi")
            first_mags = returnMagStartAbsTimes(lev, mag)  # List of first magazine entry times
            success_status = lev.returnSuccessTrials()  # List of trial success status (1, 0, -1)
            
            print("trial_starts: ", trial_starts)
            print("first_presses: ", first_presses)
            print("coop_or_last_press: ", coop_or_last_press)
            print("first_mags: ", returnMagStartAbsTimes(lev, mag))
            
            success_status = filterToLeverPressTrials(success_status, lev)
            
            print("Got Trial timings")
            
            numTrialsPress = lev.returnNumTotalTrialswithLeverPress()
            numTrialsTot = lev.returnNumTotalTrials()
            print(f"lengths: {len(trial_starts)}, {len(first_presses)}, {len(coop_or_last_press)}, {len(first_mags)}, {len(success_status)}, {numTrialsPress}, {numTrialsTot}")
            
            # Ensure consistent trial lengths
            n_trials = min(len(trial_starts), len(first_presses), len(coop_or_last_press), len(first_mags), len(success_status))
            print("n_trials: ", n_trials)
            
            # Initialize lists to collect data for averaging per stage
            stage_data = {stage_idx: {
                'Durations': [],
                'Distances': [],
                'Gaze_Percents': [],
                'Idle_Percents': [],
                'Idle_Lever_Percents': [],
                'Idle_Mag_Percents': [],
                'Idle_Other_Percents': [],
            } for stage_idx in range(len(stages))}
            stage_data_successful = {stage_idx: {
                'Durations': [],
                'Distances': [],
                'Gaze_Percents': [],
                'Idle_Percents': [],
                'Idle_Lever_Percents': [],
                'Idle_Mag_Percents': [],
                'Idle_Other_Percents': [],
            } for stage_idx in range(len(stages))}
            
            # Compute stage durations and frame indices
            trial_info = []
            for trial_idx in range(n_trials):
                if success_status[trial_idx] == -1:  # Skip missing trials
                    continue
                
                # Define stage boundaries (in seconds)
                t_begin = trial_starts[trial_idx]
                t_first_press = first_presses[trial_idx]
                t_coop_last = coop_or_last_press[trial_idx]
                t_first_mag = first_mags[trial_idx]
                # Next trial begin or end of session
                t_next_begin = trial_starts[trial_idx + 1] if trial_idx + 1 < len(trial_starts) else total_frames / fps
                
                if (t_begin == None or t_first_press == None or t_coop_last == None or t_first_mag == None):
                    continue
                
                # Check for NaN in timings
                if any(np.isnan(t) for t in [t_begin, t_first_press, t_coop_last, t_first_mag]):
                    print(f"[Exp {exp_idx}, Trial {trial_idx}] Skipped: NaN in timings (begin={t_begin}, first_press={t_first_press}, coop_last={t_coop_last}, first_mag={t_first_mag})")
                    continue
                
                # Convert times to frame indices
                f_begin = int(t_begin * fps)
                f_first_press = int(t_first_press * fps)
                f_coop_last = int(t_coop_last * fps)
                f_first_mag = int(t_first_mag * fps)
                f_next_begin = int(t_next_begin * fps)
                
                if (f_coop_last <= f_first_press):
                    print("SKIP")
                    print("f_first_press: ", f_first_press)
                    print("f_coop_last: ", f_coop_last)
                    continue
                
                # Stage durations (in seconds)
                durations = [
                    t_first_press - t_begin,  # Begin to First Press
                    t_coop_last - t_first_press,  # First Press to Coop/Last Press
                    t_first_mag - t_coop_last,  # Coop/Last Press to First Mag
                    t_next_begin - t_first_mag  # First Mag to Next Begin
                ]
                
                #print("durations: ", durations)
                
                # Frame ranges for each stage
                frame_ranges = [
                    (f_begin, f_first_press),
                    (f_first_press, f_coop_last),
                    (f_coop_last, f_first_mag),
                    (f_first_mag, f_next_begin)
                ]
                
                print("frame_ranges: ", frame_ranges)
                
                # Collect behavioral data for each stage
                for stage_idx, (start_frame, end_frame) in enumerate(frame_ranges):
                    if start_frame >= end_frame or end_frame > total_frames:
                        continue  # Skip invalid ranges
                    
                    #print("stage_idx: ", stage_idx)
                    #print("start_frame: ", start_frame)
                    #print("end_frame: ", end_frame)
                    
                    #Heatmap
                    # Collect position data for heatmap
                    x_coords_rat0 = pos.data[0, 0, pos.HB_INDEX, start_frame:end_frame]
                    y_coords_rat0 = pos.data[0, 1, pos.HB_INDEX, start_frame:end_frame]
                    x_coords_rat1 = pos.data[1, 0, pos.HB_INDEX, start_frame:end_frame]
                    y_coords_rat1 = pos.data[1, 1, pos.HB_INDEX, start_frame:end_frame]
                    # Combine coordinates from both rats
                    x_coords = np.concatenate([x_coords_rat0[~np.isnan(x_coords_rat0)], x_coords_rat1[~np.isnan(x_coords_rat1)]])
                    y_coords = np.concatenate([y_coords_rat0[~np.isnan(y_coords_rat0)], y_coords_rat1[~np.isnan(y_coords_rat1)]])
                    heatmap_data[stage_idx]['X_Coords'].extend(x_coords.tolist())
                    heatmap_data[stage_idx]['Y_Coords'].extend(y_coords.tolist())
                    if success_status[trial_idx] == 1:
                        heatmap_data_successful[stage_idx]['X_Coords'].extend(x_coords.tolist())
                        heatmap_data_successful[stage_idx]['Y_Coords'].extend(y_coords.tolist())
                    
                    # Interaction metrics
                    distances = pos.returnInterMouseDistance()[start_frame:end_frame]
                    avg_distance = np.nanmean(distances) if len(distances) > 0 else np.nan
                                        
                    gaze_frames_rat0 = pos.returnIsGazing(mouseID=0)[start_frame:end_frame]
                    #print("gaze_frames_rat0: ", gaze_frames_rat0)
                    gaze_frames_rat1 = pos.returnIsGazing(mouseID=1)[start_frame:end_frame]
                    #print("0.5")
                    gaze_percent_rat0 = np.mean(gaze_frames_rat0) * 100 if len(gaze_frames_rat0) > 0 else 0
                    gaze_percent_rat1 = np.mean(gaze_frames_rat1) * 100 if len(gaze_frames_rat1) > 0 else 0
                    avg_gaze_percent = (gaze_percent_rat0 + gaze_percent_rat1) / 2
                                        
                    # Individual behavior (idle, lever, magazine)
                    locations_rat0 = pos.returnMouseLocation(0)[start_frame:end_frame]
                    locations_rat1 = pos.returnMouseLocation(1)[start_frame:end_frame]
                    
                    # Idle detection using velocity
                    velocities_rat0 = pos.computeVelocity(0)[start_frame:end_frame]
                    velocities_rat1 = pos.computeVelocity(1)[start_frame:end_frame]
                    idle_rat0 = velocities_rat0 < 10  # Velocity threshold for idle
                    idle_rat1 = velocities_rat1 < 10
                    
                    #print("idle_rat0: ", idle_rat0)
                    
                    # Compute idle percentages
                    idle_percent_rat0 = np.mean(idle_rat0) * 100 if len(idle_rat0) > 0 else 0
                    idle_percent_rat1 = np.mean(idle_rat1) * 100 if len(idle_rat1) > 0 else 0
                    avg_idle_percent = (idle_percent_rat0 + idle_percent_rat1) / 2
                    
                    #Breakdown of idle time by location
                    lever_count = 0
                    mag_count = 0
                    other_count = 0
                    total_idle = 0
                    
                    # Breakdown of idle time by location
                    for rat_id, locations, idle in [(0, locations_rat0, idle_rat0), (1, locations_rat1, idle_rat1)]:
                        #print("rat_id: ", rat_id)
                        if not locations or not idle.any():
                            continue
                        idle_locations = np.array(locations)[idle]
                        lever_count += np.sum(np.array([loc.startswith('lev') for loc in idle_locations]))
                        mag_count += np.sum(np.array([loc.startswith('mag') for loc in idle_locations]))
                        other_count += len(idle_locations) - np.sum(np.array([loc.startswith('lev') for loc in idle_locations])) - np.sum(np.array([loc.startswith('mag') for loc in idle_locations]))
                        total_idle += len(idle_locations)
                        #lever_percent = (lever_count / total_idle * 100) if total_idle > 0 else 0
                        #mag_percent = (mag_count / total_idle * 100) if total_idle > 0 else 0
                        #other_percent = (other_count / total_idle * 100) if total_idle > 0 else 0
                        
                        '''# Store trial data
                        trial_data = {
                            'Experiment': exp_idx,
                            'Trial': trial_idx,
                            'Stage': stage_idx,
                            'Stage_Name': stages[stage_idx],
                            'Duration': durations[stage_idx],
                            'Avg_Distance': avg_distance,
                            'Avg_Gaze_Percent': avg_gaze_percent,
                            'Avg_Idle_Percent': avg_idle_percent,
                            'Idle_Lever_Percent': lever_percent,
                            'Idle_Mag_Percent': mag_percent,
                            'Idle_Other_Percent': other_percent,
                            'Is_Successful': success_status[trial_idx] == 1
                        }
                        all_trials_data.append(trial_data)
                        if trial_data['Is_Successful']:
                            successful_trials_data.append(trial_data)'''
                    lever_percent = (lever_count / total_idle * 100) if total_idle > 0 else 0
                    mag_percent = (mag_count / total_idle * 100) if total_idle > 0 else 0
                    other_percent = (other_count / total_idle * 100) if total_idle > 0 else 0
                    
                    # Append to stage data
                    stage_data[stage_idx]['Durations'].append(durations[stage_idx])
                    stage_data[stage_idx]['Distances'].append(avg_distance)
                    stage_data[stage_idx]['Gaze_Percents'].append(avg_gaze_percent)
                    stage_data[stage_idx]['Idle_Percents'].append(avg_idle_percent)
                    stage_data[stage_idx]['Idle_Lever_Percents'].append(lever_percent)
                    stage_data[stage_idx]['Idle_Mag_Percents'].append(mag_percent)
                    stage_data[stage_idx]['Idle_Other_Percents'].append(other_percent)
                    
                    print("stage_idx: ", stage_idx)
                    print("avg_idle_percent: ", avg_idle_percent)
                    print("lever_percent: ", lever_percent)
                    print("mag_percent: ", mag_percent)
                    print("other_percent: ", other_percent)
                    
                    # If trial is successful, append to successful stage data
                    if success_status[trial_idx] == 1:
                        stage_data_successful[stage_idx]['Durations'].append(durations[stage_idx])
                        stage_data_successful[stage_idx]['Distances'].append(avg_distance)
                        stage_data_successful[stage_idx]['Gaze_Percents'].append(avg_gaze_percent)
                        stage_data_successful[stage_idx]['Idle_Percents'].append(avg_idle_percent)
                        stage_data_successful[stage_idx]['Idle_Lever_Percents'].append(lever_percent)
                        stage_data_successful[stage_idx]['Idle_Mag_Percents'].append(mag_percent)
                        stage_data_successful[stage_idx]['Idle_Other_Percents'].append(other_percent)
            
            # Compute averages for each stage and store as single data point per experiment
            for stage_idx, stage_name in enumerate(stages):
                # All trials
                if stage_data[stage_idx]['Durations']:  # Only append if data exists
                    trial_data = {
                        'Experiment': exp_idx,
                        'Trial': -1,  # No specific trial since we're averaging
                        'Stage': stage_idx,
                        'Stage_Name': stage_name,
                        'Duration': np.nanmean(stage_data[stage_idx]['Durations']),
                        'Avg_Distance': np.nanmean(stage_data[stage_idx]['Distances']),
                        'Avg_Gaze_Percent': np.nanmean(stage_data[stage_idx]['Gaze_Percents']),
                        'Avg_Idle_Percent': np.nanmean(stage_data[stage_idx]['Idle_Percents']),
                        'Idle_Lever_Percent': np.nanmean(stage_data[stage_idx]['Idle_Lever_Percents']),
                        'Idle_Mag_Percent': np.nanmean(stage_data[stage_idx]['Idle_Mag_Percents']),
                        'Idle_Other_Percent': np.nanmean(stage_data[stage_idx]['Idle_Other_Percents']),
                        'Is_Successful': False
                    }
                    all_trials_data.append(trial_data)
                
                # Successful trials
                if stage_data_successful[stage_idx]['Durations']:  # Only append if data exists
                    trial_data = {
                        'Experiment': exp_idx,
                        'Trial': -1,  # No specific trial since we're averaging
                        'Stage': stage_idx,
                        'Stage_Name': stage_name,
                        'Duration': np.nanmean(stage_data_successful[stage_idx]['Durations']),
                        'Avg_Distance': np.nanmean(stage_data_successful[stage_idx]['Distances']),
                        'Avg_Gaze_Percent': np.nanmean(stage_data_successful[stage_idx]['Gaze_Percents']),
                        'Avg_Idle_Percent': np.nanmean(stage_data_successful[stage_idx]['Idle_Percents']),
                        'Idle_Lever_Percent': np.nanmean(stage_data_successful[stage_idx]['Idle_Lever_Percents']),
                        'Idle_Mag_Percent': np.nanmean(stage_data_successful[stage_idx]['Idle_Mag_Percents']),
                        'Idle_Other_Percent': np.nanmean(stage_data_successful[stage_idx]['Idle_Other_Percents']),
                        'Is_Successful': True
                    }
                    successful_trials_data.append(trial_data)
            
        # Convert to DataFrames
        all_trials_df = pd.DataFrame(all_trials_data)
        df_copy = all_trials_df.copy()
        df_copy.to_csv("all_trials_data_test.csv", index=False)
        successful_trials_df = pd.DataFrame(successful_trials_data)
        
        
        # Generate heatmaps for each stage
        def generate_heatmaps(df, suffix=''):
            bin_size = 5
            height, width = 640, 1392
            heatmap_height = height // bin_size
            heatmap_width = width // bin_size
            
            stage_names = ['Begin_to_First_Press', 'First_Press_to_Coop_or_Last_Press', 'Coop_or_Last_Press_to_First_Mag', 'First_Mag_to_Next_Begin']
            
            
            for stage_idx, stage_name in enumerate(stage_names):
                heatmap = np.zeros((heatmap_height, heatmap_width))
                if suffix == '_AllTrials':
                    x_coords = heatmap_data[stage_idx]['X_Coords']
                    y_coords = heatmap_data[stage_idx]['Y_Coords']
                else:
                    x_coords = heatmap_data_successful[stage_idx]['X_Coords']
                    y_coords = heatmap_data_successful[stage_idx]['Y_Coords']
                
                if not x_coords or not y_coords:
                    print(f"No data for stage {stage_name} in {suffix}. Skipping heatmap.")
                    continue
                
                for x, y in zip(x_coords, y_coords):
                    if not np.isnan(x) and not np.isnan(y):
                        x_bin = int(min(max(x // bin_size, 0), heatmap_width - 1))
                        y_bin = int(min(max(y // bin_size, 0), heatmap_height - 1))
                        heatmap[y_bin, x_bin] += 1
                
                heatmap = gaussian_filter(heatmap, sigma=1)
                heatmap_log = np.log1p(heatmap)
                
                plt.figure(figsize=(12, 6))
                plt.imshow(
                    heatmap_log,
                    cmap='hot',
                    interpolation='nearest',
                    origin='upper',
                    extent=[0, width, height, 0],
                    vmin=np.percentile(heatmap_log, 4),
                    vmax=np.percentile(heatmap_log, 98)
                )
                print("min/max heatmap:", heatmap.min(), heatmap.max())
                plt.colorbar(label='Log(Time Spent)')
                plt.title(f'Mouse Location Heatmap - {stage_name}{suffix}')
                plt.xlabel('X Position (pixels)')
                plt.ylabel('Y Position (pixels)')
                if self.save:
                    plt.savefig(f'{self.prefix}Heatmap_{stage_name.replace(" ", "_")}_{suffix}.png', bbox_inches='tight')
                plt.show()
                plt.close()
        
        
        
        # Function to create visualizations
        def create_visualizations(df, suffix=''):
            if df.empty:
                print(f"No data available for visualizations{suffix}. Check if trials exist or if success filtering is too restrictive.")
                return
            
            # Ensure all stages are represented
            for stage in stages:
                if stage not in df['Stage_Name'].values:
                    print(f"Warning: Stage '{stage}' missing in {suffix}. Adding placeholder data.")
                    placeholder = {
                        'Experiment': -1,
                        'Trial': -1,
                        'Stage': stages.index(stage),
                        'Stage_Name': stage,
                        'Duration': np.nan,
                        'Avg_Distance': np.nan,
                        'Avg_Gaze_Percent': np.nan,
                        'Avg_Idle_Percent': np.nan,
                        'Idle_Lever_Percent': np.nan,
                        'Idle_Mag_Percent': np.nan,
                        'Idle_Other_Percent': np.nan,
                        'Is_Successful': suffix == '_SuccessfulTrials'
                    }
                    df = pd.concat([df, pd.DataFrame([placeholder])], ignore_index=True)
            
            # Aggregate again across experiments for bar plot
            agg_data = df.groupby('Stage_Name').agg({
                'Duration': ['mean', list],
                'Avg_Distance': ['mean', list],
                'Avg_Gaze_Percent': ['mean', list],
                'Avg_Idle_Percent': ['mean', list],
                'Idle_Lever_Percent': 'mean',
                'Idle_Mag_Percent': 'mean',
                'Idle_Other_Percent': 'mean'
            }).reset_index()
            
            # Flatten multi-index columns
            agg_data.columns = ['Stage_Name', 'Duration_mean', 'Duration_list', 
                                'Avg_Distance_mean', 'Avg_Distance_list',
                                'Avg_Gaze_Percent_mean', 'Avg_Gaze_Percent_list',
                                'Avg_Idle_Percent_mean', 'Avg_Idle_Percent_list',
                                'Idle_Lever_Percent_mean', 'Idle_Mag_Percent_mean', 
                                'Idle_Other_Percent_mean']
            
            # Order stages correctly
            agg_data['Stage_Name'] = pd.Categorical(agg_data['Stage_Name'], categories=stages, ordered=True)
            agg_data = agg_data.sort_values('Stage_Name').reset_index(drop=True)
            
            # Plot individual bar plots for each metric
            metrics = [
                ('Duration', 'Duration (seconds)', f'Duration Across Stages{suffix}'),
                ('Avg_Distance', 'Distance (pixels)', f'Average Inter-Mouse Distance Across Stages{suffix}'),
                ('Avg_Gaze_Percent', 'Gaze Percentage (%)', f'Average Gaze Percentage Across Stages{suffix}'),
                ('Avg_Idle_Percent', 'Idle Percentage (%)', f'Average Idle Percentage Across Stages{suffix}')
            ]
            
            for metric, ylabel, title in metrics:
                plt.figure(figsize=(10, 6))
                # Bar plot for mean values
                sns.barplot(data=agg_data, x='Stage_Name', y=f'{metric}_mean', color='skyblue', label='Mean')
                
                # Scatter individual session data
                for idx, row in agg_data.iterrows():
                    stage_idx = stages.index(row['Stage_Name'])
                    values = [v for v in row[f'{metric}_list'] if not np.isnan(v)]
                    x_positions = [stage_idx + np.random.uniform(-0.2, 0.2) for _ in values]
                    plt.scatter(x_positions, values, color='black', alpha=0.5, label='Individual Sessions' if idx == 0 else None)
                
                plt.title(title)
                plt.xlabel('Stage')
                plt.ylabel(ylabel)
                plt.xticks(rotation=45)
                plt.legend()
                plt.tight_layout()
                if self.save:
                    plt.savefig(f'{self.prefix}{metric}{suffix}.png')
                plt.show()
                plt.close()
            
            # Stacked bar plot for idle time breakdown
            plt.figure(figsize=(10, 6))
            x = np.arange(len(stages))
            idle_means = agg_data['Avg_Idle_Percent_mean']
            
            print("agg_data index:", agg_data.index)
            print("agg_data Stage_Name:\n", agg_data['Stage_Name'].values)
            
            print("\n\nAvg_Idle_Percent_mean: \n",  agg_data['Avg_Idle_Percent_mean'])
            print("Idle_Lever_Percent_means: \n", agg_data['Idle_Lever_Percent_mean'])
            print("Idle_Mag_Percent_mean: \n", agg_data['Idle_Mag_Percent_mean'])
            print("Idle_Other_Percent_mean: \n", agg_data['Idle_Other_Percent_mean'])
            
            lever_proportions = agg_data['Idle_Lever_Percent_mean'] * idle_means / 100
            mag_proportions = agg_data['Idle_Mag_Percent_mean'] * idle_means / 100
            other_proportions = agg_data['Idle_Other_Percent_mean'] * idle_means / 100
            
            # Sanity check: stacked components should match idle mean
            total_stack = lever_proportions + mag_proportions + other_proportions
            if not np.allclose(total_stack, idle_means, atol=1e-2):
                for i, stage in enumerate(stages):
                    print(f"Mismatch at stage {i} ({stage}):")
                    print(f"  Idle mean = {idle_means[i]}")
                    print(f"  Stack sum = {total_stack[i]} "
                          f"(lever {lever_proportions[i]}, mag {mag_proportions[i]}, other {other_proportions[i]})")
            
            for i, stage in enumerate(agg_data['Stage_Name']):
                print(f"Stage {i} ({stage}):")
                print(f"  Avg_Idle_Percent_mean = {idle_means.iloc[i]}")
                print(f"  Idle_Lever_Percent_mean = {agg_data['Idle_Lever_Percent_mean'].iloc[i]}")
                print(f"  Computed lever proportion = {lever_proportions.iloc[i]}")
                print(f"  Idle_Other_Percent = {other_proportions.iloc[i]}")
            
            # Plot stacked bars with total height as Avg_Idle_Percent
            plt.bar(x, lever_proportions, label='Idle at Lever', color='blue')
            plt.bar(x, mag_proportions, bottom=lever_proportions, label='Idle at Magazine', color='green')
            plt.bar(x, other_proportions, bottom=lever_proportions + mag_proportions, label='Idle Elsewhere', color='red')
            
            # Scatter individual session data for total idle percentage
            for idx, row in agg_data.iterrows():
                stage_idx = stages.index(row['Stage_Name'])
                values = [v for v in row['Avg_Idle_Percent_list'] if not np.isnan(v)]
                x_positions = [stage_idx + np.random.uniform(-0.2, 0.2) for _ in values]
                plt.scatter(x_positions, values, color='black', alpha=0.5, label='Individual Sessions' if idx == 0 else None)
            
            plt.ylim(0, 100)  # Set y-axis range from 0 to 100
            plt.xlabel('Stage')
            plt.ylabel('Idle Percentage (%)')
            plt.title(f'Idle Time Breakdown Across Stages{suffix}')
            plt.xticks(x, stages, rotation=45)
            plt.legend()
            plt.tight_layout()
            if self.save:
                plt.savefig(f'{self.prefix}IdleBreakdown{suffix}.png')
            plt.show()
            plt.close()
        
        # Create visualizations for all trials
        create_visualizations(all_trials_df, '_AllTrials')
        generate_heatmaps(all_trials_df, '_AllTrials')
        
        # Create visualizations for successful trials
        if successful_trials_df.empty:
            print("Warning: No successful trials found. Check lev.returnSuccessTrials() for valid success flags.")
        else:
            create_visualizations(successful_trials_df, '_SuccessfulTrials')
            generate_heatmaps(successful_trials_df, '_SuccessfulTrials')
        
        
        
        # Create visualizations for all trials
        create_visualizations(all_trials_df, '_AllTrials')
        
        # Create visualizations for successful trials
        if successful_trials_df.empty:
            print("Warning: No successful trials found. Check lev.returnSuccessTrials() for valid success flags.")
        create_visualizations(successful_trials_df, '_SuccessfulTrials')
            
    def trueCooperationTesting(self):
        
        individual_datapoints_avgDistance = []
        individual_datapoints_timeUntilPress = []
        
        for exp_idx, exp in enumerate(self.experiments): 
            lev = exp.lev
            pos = exp.pos
            fps = exp.fps
            
            trial_starts = lev.returnTimeStartTrials()  # List of trial start times
            first_presses = lev.returnFirstPressAbsTimes()  # List of first press times
            ids_first_press = lev.returnRatIDFirstPressTrial()
            
            if (len(trial_starts) != len(first_presses) or len(first_presses) != len(ids_first_press)):
                print("len(trial_starts): ", len(trial_starts))
                print("len(ids_first_press): ", len(ids_first_press))
                raise ValueError("Inequal Sizes")
             
            sumDistances = 0
            sumNumConsidered = 0
            sumTimeUntilPress = 0
            
            for trial_idx, trialStart in enumerate(trial_starts):
                t_begin = trial_starts[trial_idx]
                t_first_press = first_presses[trial_idx]
                rat_first_press = ids_first_press[trial_idx]
                
                if (t_begin == None or t_first_press == None or rat_first_press == None):
                    continue
                
                # Check for NaN in timings
                if any(np.isnan(t) for t in [t_begin, t_first_press, rat_first_press]):
                    print(f"[Exp {exp_idx}, Trial {trial_idx}] Skipped: NaN in timings (begin={t_begin}, first_press={t_first_press}, rat_first_press={rat_first_press})")
                    continue
                                
                frameStart = int(t_begin * fps)
                frameFirstPress = int(t_first_press * fps)
                
                levAreas = ['lev_top', 'lev_bottom']
                dist = 0
                if (pos.returnRatLocationTime(0, frameStart) in levAreas or pos.returnRatLocationTime(1, frameStart) in levAreas):
                    dist = max(pos.distanceFromLever(0, frameStart), pos.distanceFromLever(1, frameStart))
                    if (dist == pos.distanceFromLever(0, frameStart) and rat_first_press != 0):
                        continue
                    elif (pos.distanceFromLever(1, frameStart) and rat_first_press != 1):
                        continue
                    sumDistances += dist
                    sumNumConsidered += 1
                    sumTimeUntilPress += t_first_press - t_begin
            
            if sumNumConsidered > 0 and sumTimeUntilPress / sumNumConsidered < 40:
                print("\nlev file: ", exp.lev_file)
                print("avgDistance: ", sumDistances / sumNumConsidered)
                print("timeUntilPress: ", sumTimeUntilPress / sumNumConsidered)
                individual_datapoints_avgDistance.append(sumDistances / sumNumConsidered)
                individual_datapoints_timeUntilPress.append(sumTimeUntilPress / sumNumConsidered)

        # Plotting
        if len(individual_datapoints_avgDistance) >= 2 and len(individual_datapoints_timeUntilPress) >= 2:
            plt.figure(figsize=(8, 6))
            plt.scatter(individual_datapoints_avgDistance, individual_datapoints_timeUntilPress,
                        alpha=0.7, color='blue', label='Experiments')
    
            if len(set(individual_datapoints_avgDistance)) >= 2:
                slope, intercept, r_value, _, _ = linregress(individual_datapoints_avgDistance, individual_datapoints_timeUntilPress)
                r_squared = r_value ** 2
    
                x_vals = np.linspace(min(individual_datapoints_avgDistance), max(individual_datapoints_avgDistance), 100)
                plt.plot(x_vals, slope * x_vals + intercept, color='red', linestyle='--', label='Trendline')
                plt.text(0.95, 0.05, f"$R^2$ = {r_squared:.3f}", transform=plt.gca().transAxes,
                         ha='right', va='bottom', fontsize=12, bbox=dict(facecolor='white', edgecolor='gray'))
    
            plt.xlabel('Average Distance from Lever')
            plt.ylabel('Average Time Until First Press (s)')
            plt.title('Lever Distance vs. Time Until First Press')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            if self.save:
                plt.savefig(f"{self.prefix}distance_vs_time_until_press.png")
            plt.show()
            plt.close()
        else:
            print("Insufficient data to create scatterplot.")

    def fiberPhoto(self):
        # Lists to store sums and counts for each wavelength and event type
        sum405List_lev = []
        count405List_lev = []
        sum465List_lev = []
        count465List_lev = []
        sum560List_lev = []
        count560List_lev = []
        
        sum405List_mag = []
        count405List_mag = []
        sum465List_mag = []
        count465List_mag = []
        sum560List_mag = []
        count560List_mag = []
        
        # Lists to store individual data points for scatter overlay
        mean405_lev = []
        mean405_mag = []
        mean465_lev = []
        mean465_mag = []
        mean560_lev = []
        mean560_mag = []
        
        # Lists to store time series data for line plots
        time_series_405_lev = []
        time_series_465_lev = []
        time_series_560_lev = []
        time_series_405_mag = []
        time_series_465_mag = []
        time_series_560_mag = []
        
        for exp in self.experiments:
            if hasattr(exp, 'fp') and exp.fp is not None:
                # Get lever press data
                lev405 = exp.fp.getLev405()
                lev465 = exp.fp.getLev465()
                lev560 = exp.fp.getLev560()
                
                # Get magazine entry data
                mag405 = exp.fp.getMag405()
                mag465 = exp.fp.getMag465()
                mag560 = exp.fp.getMag560()
                
                # Process lever press data
                sum_count_405_lev = exp.fp.getSumandEles(lev405)
                sum405List_lev.append(sum_count_405_lev[0])
                count405List_lev.append(sum_count_405_lev[1])
                
                sum_count_465_lev = exp.fp.getSumandEles(lev465)
                sum465List_lev.append(sum_count_465_lev[0])
                count465List_lev.append(sum_count_465_lev[1])
                
                sum_count_560_lev = exp.fp.getSumandEles(lev560)
                sum560List_lev.append(sum_count_560_lev[0])
                count560List_lev.append(sum_count_560_lev[1])
                
                # Process magazine entry data
                sum_count_405_mag = exp.fp.getSumandEles(mag405)
                sum405List_mag.append(sum_count_405_mag[0])
                count405List_mag.append(sum_count_405_mag[1])
                
                sum_count_465_mag = exp.fp.getSumandEles(mag465)
                sum465List_mag.append(sum_count_465_mag[0])
                count465List_mag.append(sum_count_465_mag[1])
                
                sum_count_560_mag = exp.fp.getSumandEles(mag560)
                sum560List_mag.append(sum_count_560_mag[0])
                count560List_mag.append(sum_count_560_mag[1])
                
                # Compute mean for each event in this experiment for scatter points
                if sum_count_405_lev[1] > 0:
                    mean405_lev.append(sum_count_405_lev[0] / sum_count_405_lev[1])
                if sum_count_465_lev[1] > 0:
                    mean465_lev.append(sum_count_465_lev[0] / sum_count_465_lev[1])
                if sum_count_560_lev[1] > 0:
                    mean560_lev.append(sum_count_560_lev[0] / sum_count_560_lev[1])
                    
                if sum_count_405_mag[1] > 0:
                    mean405_mag.append(sum_count_405_mag[0] / sum_count_405_mag[1])
                if sum_count_465_mag[1] > 0:
                    mean465_mag.append(sum_count_465_mag[0] / sum_count_465_mag[1])
                if sum_count_560_mag[1] > 0:
                    mean560_mag.append(sum_count_560_mag[0] / sum_count_560_mag[1])
                
                # Collect time series data (values from columns 1 to 1526)
                for event in lev405:
                    if len(event) == 1527:  # Ensure correct length
                        time_series_405_lev.append(event[1:])  # Skip 'ts'
                for event in lev465:
                    if len(event) == 1527:
                        time_series_465_lev.append(event[1:])
                for event in lev560:
                    if len(event) == 1527:
                        time_series_560_lev.append(event[1:])
                        
                for event in mag405:
                    if len(event) == 1527:
                        time_series_405_mag.append(event[1:])
                for event in mag465:
                    if len(event) == 1527:
                        time_series_465_mag.append(event[1:])
                for event in mag560:
                    if len(event) == 1527:
                        time_series_560_mag.append(event[1:])
        
        # Calculate overall means for bar plots
        overall_mean_405_lev = sum(sum405List_lev) / sum(count405List_lev) if sum(count405List_lev) > 0 else 0
        overall_mean_465_lev = sum(sum465List_lev) / sum(count465List_lev) if sum(count465List_lev) > 0 else 0
        overall_mean_560_lev = sum(sum560List_lev) / sum(count560List_lev) if sum(count560List_lev) > 0 else 0
        
        overall_mean_405_mag = sum(sum405List_mag) / sum(count405List_mag) if sum(count405List_mag) > 0 else 0
        overall_mean_465_mag = sum(sum465List_mag) / sum(count465List_mag) if sum(count465List_mag) > 0 else 0
        overall_mean_560_mag = sum(sum560List_mag) / sum(count560List_mag) if sum(count560List_mag) > 0 else 0
        
        # Create separate bar plots with scatter overlay
        # 405 nm Bar Plot
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        ax1.bar(['Lever Press', 'Magazine Entry'], [overall_mean_405_lev, overall_mean_405_mag], color=['blue', 'orange'])
        ax1.scatter(['Lever Press'] * len(mean405_lev), mean405_lev, color='black', alpha=0.5, s=20)
        ax1.scatter(['Magazine Entry'] * len(mean405_mag), mean405_mag, color='black', alpha=0.5, s=20)
        ax1.set_title('405 nm Signal')
        ax1.set_ylabel('Average Signal Value')
        plt.tight_layout()
        if hasattr(self, 'save') and self.save:
            plt.savefig('405_nm_bar_plot.png')
        plt.show()
        
        # 465 nm Bar Plot
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.bar(['Lever Press', 'Magazine Entry'], [overall_mean_465_lev, overall_mean_465_mag], color=['blue', 'orange'])
        ax2.scatter(['Lever Press'] * len(mean465_lev), mean465_lev, color='black', alpha=0.5, s=20)
        ax2.scatter(['Magazine Entry'] * len(mean465_mag), mean465_mag, color='black', alpha=0.5, s=20)
        ax2.set_title('465 nm Signal')
        ax2.set_ylabel('Average Signal Value')
        plt.tight_layout()
        if hasattr(self, 'save') and self.save:
            plt.savefig('465_nm_bar_plot.png')
        plt.show()
        
        # 560 nm Bar Plot
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        ax3.bar(['Lever Press', 'Magazine Entry'], [overall_mean_560_lev, overall_mean_560_mag], color=['blue', 'orange'])
        ax3.scatter(['Lever Press'] * len(mean560_lev), mean560_lev, color='black', alpha=0.5, s=20)
        ax3.scatter(['Magazine Entry'] * len(mean560_mag), mean560_mag, color='black', alpha=0.5, s=20)
        ax3.set_title('560 nm Signal')
        ax3.set_ylabel('Average Signal Value')
        plt.tight_layout()
        if hasattr(self, 'save') and self.save:
            plt.savefig('560_nm_bar_plot.png')
        plt.show()
        
        # Calculate average time series for line plots
        mean_ts_405_lev = np.nanmean(time_series_405_lev, axis=0) if time_series_405_lev else np.zeros(1526)
        mean_ts_465_lev = np.nanmean(time_series_465_lev, axis=0) if time_series_465_lev else np.zeros(1526)
        mean_ts_560_lev = np.nanmean(time_series_560_lev, axis=0) if time_series_560_lev else np.zeros(1526)
        
        mean_ts_405_mag = np.nanmean(time_series_405_mag, axis=0) if time_series_405_mag else np.zeros(1526)
        mean_ts_465_mag = np.nanmean(time_series_465_mag, axis=0) if time_series_465_mag else np.zeros(1526)
        mean_ts_560_mag = np.nanmean(time_series_560_mag, axis=0) if time_series_560_mag else np.zeros(1526)
        
        # Create separate line plots
        # Lever Press Line Plot
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        x = range(1, 1527)
        ax4.plot(x, mean_ts_405_lev, label='405 nm', color='purple')
        ax4.plot(x, mean_ts_465_lev, label='465 nm', color='green')
        ax4.plot(x, mean_ts_560_lev, label='560 nm', color='red')
        ax4.set_title('Average Fiber Photometry Signal (Lever Press)')
        ax4.set_xlabel('Time Point')
        ax4.set_ylabel('Signal Value')
        ax4.legend()
        plt.tight_layout()
        if hasattr(self, 'save') and self.save:
            plt.savefig('lever_press_line_plot.png')
        plt.show()
        
        # Magazine Entry Line Plot
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        ax5.plot(x, mean_ts_405_mag, label='405 nm', color='purple')
        ax5.plot(x, mean_ts_465_mag, label='465 nm', color='green')
        ax5.plot(x, mean_ts_560_mag, label='560 nm', color='red')
        ax5.set_title('Average Fiber Photometry Signal (Magazine Entry)')
        ax5.set_xlabel('Time Point')
        ax5.set_ylabel('Signal Value')
        ax5.legend()
        plt.tight_layout()
        if hasattr(self, 'save') and self.save:
            plt.savefig('magazine_entry_line_plot.png')
        plt.show()
    
    def testMotivation(self):
        def filterToLeverPressTrials(original_list, lev):
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
            
            # Compute all indices and those that are kept
            all_indices = set(range(len(original_list)))
            kept_indices = set(trial_num - 1 for trial_num in lever_trials)
            filtered_out_indices = sorted(all_indices - kept_indices)
            
            print("Filtered out indices:", filtered_out_indices)
        
            return filtered_list
        
        '''
        Calculate the percent success rate for each trial number across all experiments
        and create a smoothed line graph with annotations showing the number of experiments
        contributing to each trial's data.
        Create another similar graph, except with distance moved (pixels/frame)
    
        Steps:
        - Loop through experiments to collect success status for each trial number.
        - Compute percent success rate as (# successful trials / # experiments with that trial).
        - Plot a smoothed line graph using a moving average.
        - Annotate each point with the number of experiments.
        - Save the plot if self.save is True.
        '''
        # Collect success rates and experiment counts per trial number
        trial_successes = {}
        trial_distances = {}
        trial_counts_distance = {}
        trial_counts = {}
        
        for exp_idx, exp in enumerate(self.experiments):
            lev = exp.lev
            pos = exp.pos
            fps = exp.fps
            success_status = exp.lev.returnSuccessTrials()  # List of success statuses (1, 0, -1) 
            success_status = filterToLeverPressTrials(success_status, lev)
            
            start_times = lev.returnTimeStartTrials()  # Array of trial start times (in seconds) for all trials
            end_times = lev.returnTimeEndTrials()  # Array of trial end times (in seconds) for all trials
            
            for trial_idx, status in enumerate(success_status):
                start_time = start_times[trial_idx]
                end_time = end_times[trial_idx]

                # Validate trial data
                if any(np.isnan([start_time, end_time])) or start_time is None or end_time is None:
                    continue

                # Convert times to frames
                start_frame = int(start_time * fps)
                end_frame = int(end_time * fps)
                
                
                if trial_idx not in trial_distances:
                    trial_distances[trial_idx] = []
                    trial_counts_distance[trial_idx] = 0
    
                # Get headbase positions for all frames in the trial
                distances_both = []
                
                if status == -1:  # Skip missing trials
                    continue
                
                # Process both rat IDs (0 and 1)
                for ratID in [0, 1]:
                    # Calculate distances for this rat in this trial
                    distances = []
                    #n = end_frame - start_frame
                    for t in range(start_frame, end_frame):
                        if t > start_frame:  # Need at least 2 frames to calculate distance
                            x1, y1 = pos.returnRatHBPosition(ratID, t-1)
                            x2, y2 = pos.returnRatHBPosition(ratID, t)
                            # Calculate Euclidean distance
                            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                            #if (distance > 200):
                                #print("Unrealistic t is: ", t)
                            distances.append(distance)
                    #print("distances: ", distances)
                    #print("mean: ", np.mean(distances))
                    if (np.mean(distances) > 75):
                        print("exp.lev_file: ", exp.lev_file)
                        print("start_frame: ", start_frame)
                        print("end_frame: ", end_frame)
                    else: 
                        distances_both.append(np.mean(distances))
                    
                # Average distance per frame for this rat in this trial
                if distances:  # Only append if distances were calculated
                    trial_distances[trial_idx].append(np.mean(distances_both))
                
                if trial_idx not in trial_successes:
                    trial_successes[trial_idx] = []
                    trial_counts[trial_idx] = 0
                trial_successes[trial_idx].append(status == 1)
                trial_counts[trial_idx] += 1
        
        
        # Calculate percent success rate for each trial number
        trial_numbers = sorted(trial_successes.keys())
        success_rates = [
            np.mean(trial_successes[trial_idx]) * 100 if trial_successes[trial_idx] else 0
            for trial_idx in trial_numbers
        ]
        experiment_counts = [trial_counts[trial_idx] for trial_idx in trial_numbers]
        
        # Apply smoothing (moving average with window size 3)
        if len(success_rates) > 2:  # Need at least 3 points for smoothing
            smoothed_rates = uniform_filter1d(success_rates, size=3, mode='nearest')
        else:
            smoothed_rates = success_rates  # No smoothing if too few points
        
        
        #Success Rate Motivation Plot
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(trial_numbers, smoothed_rates, color='blue', label='Smoothed Success Rate')
        
        # Plot scatter points every 5 trials
        scatter_indices = [i for i in range(len(trial_numbers)) if trial_numbers[i] % 3 == 0]
        scatter_trials = [trial_numbers[i] for i in scatter_indices]
        scatter_rates = [success_rates[i] for i in scatter_indices]
        plt.scatter(scatter_trials, scatter_rates, color='black', alpha=0.5, label='Actual Success Rate')
        
        # Annotate points with number of experiments
        for trial_idx, rate, count in zip(trial_numbers, success_rates, experiment_counts):
            if (trial_idx % 15 == 0):
                plt.text(trial_idx, rate + 2, f'n={count}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.xlabel('Trial Number', fontsize = 13)
        plt.ylabel('Success Rate (%)', fontsize = 13)
        plt.title('Success Rate by Trial Number Across Experiments', fontsize = 15)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if self.save:
            plt.savefig(f'{self.prefix}SuccessRateByTrialNumber.png')
        plt.show()
        plt.close()
        
        #––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        
        #Distance Motivation Plot
        
        # Calculate average distance moved for each trial number
        trial_numbers = sorted(trial_distances.keys())
        avg_distances = [
            np.mean(trial_distances[trial_idx]) if trial_distances[trial_idx] else 0
            for trial_idx in trial_numbers
        ]
        #experiment_counts = [trial_counts_distance[trial_idx] for trial_idx in trial_numbers]
        
        # Apply smoothing (moving average with window size 3)
        if len(avg_distances) > 2:  # Need at least 3 points for smoothing
            smoothed_distances = uniform_filter1d(avg_distances, size=5, mode='nearest')
        else:
            smoothed_distances = avg_distances  # No smoothing if too few points
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(trial_numbers, smoothed_distances, color='blue', label='Smoothed Distance Moved')
        
        # Plot scatter points every 3 trials
        scatter_indices = [i for i in range(len(trial_numbers)) if trial_numbers[i] % 3 == 0]
        scatter_trials = [trial_numbers[i] for i in scatter_indices]
        scatter_distances = [avg_distances[i] for i in scatter_indices]
        plt.scatter(scatter_trials, scatter_distances, color='black', alpha=0.5, label='Actual Distance Moved')
        
        # Annotate points with number of experiments
        for trial_idx, distance, count in zip(trial_numbers, avg_distances, experiment_counts):
            if trial_idx % 15 == 0:
                plt.text(trial_idx, distance + 0.5, f'n={count}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.xlabel('Trial Number', fontsize=13)
        plt.ylabel('Distance Moved (pixels/frame)', fontsize=13)
        plt.title('Average Distance Moved by Trial Number Across Experiments', fontsize=15)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if self.save:
            plt.savefig(f'{self.prefix}DistanceMovedByTrialNumber.png')
        plt.show()
        plt.close()
        
    def gazeHeatmap(self):
        bin_size = 5  # Controls resolution of heatmap (larger = coarser)
        height, width = 640, 1392
        heatmap_height = height // bin_size
        heatmap_width = width // bin_size
        heatmap_gazing = np.zeros((heatmap_height, heatmap_width))
        heatmap_nongazing = np.zeros((heatmap_height, heatmap_width))
        gaze_count = 0
        nongaze_count = 0
        
        for exp in self.experiments:
            pos = exp.pos
            
            isGazing0 = pos.returnIsGazing(0)
            isGazing1 = pos.returnIsGazing(1)
            
            for t in range(exp.endFrame):
                boolGaze0 = isGazing0[t]
                
                if (np.isnan(boolGaze0) or boolGaze0 == None):
                    continue
                
                x, y = pos.returnRatHBPosition(0, t)
                if not np.isnan(x) and not np.isnan(y):
                    x_bin = int(min(max(x // bin_size, 0), heatmap_width - 1))
                    y_bin = int(min(max(y // bin_size, 0), heatmap_height - 1))
                    
                    if (boolGaze0 == True):
                        heatmap_gazing[y_bin, x_bin] += 1
                    else:
                        heatmap_nongazing[y_bin, x_bin] += 1
                
                boolGaze1 = isGazing1[t]
                
                if (np.isnan(boolGaze1) or boolGaze1 == None):
                    continue
                
                x, y = pos.returnRatHBPosition(1, t)
                if not np.isnan(x) and not np.isnan(y):
                    x_bin = int(min(max(x // bin_size, 0), heatmap_width - 1))
                    y_bin = int(min(max(y // bin_size, 0), heatmap_height - 1))
                    
                    if (boolGaze0 == True):
                        heatmap_gazing[y_bin, x_bin] += 1
                        gaze_count += 1
                    else:
                        heatmap_nongazing[y_bin, x_bin] += 1
                        nongaze_count += 1
        # Apply Gaussian filter
        heatmap_gazing_smooth = gaussian_filter(heatmap_gazing, sigma=1)
        heatmap_nongazing_smooth = gaussian_filter(heatmap_nongazing, sigma=1)
        
        if gaze_count > 0:
            heatmap_gazing /= gaze_count
        if nongaze_count > 0:
            heatmap_nongazing /= nongaze_count
        
        # Apply Gaussian Filter
        heatmap_gazing_smooth_standard = gaussian_filter(heatmap_gazing, sigma=1)
        heatmap_nongazing_smooth_standard = gaussian_filter(heatmap_nongazing, sigma=1)
    
        # Plot gazing heatmap
        self._plot_single_heatmap(
            heatmap_gazing_smooth,
            width, height, bin_size,
            title='Gazing Heatmap',
            filename='gazingHeatmap.png'
        )
    
        # Plot non-gazing heatmap
        self._plot_single_heatmap(
            heatmap_nongazing_smooth,
            width, height, bin_size,
            title='Non-Gazing Heatmap',
            filename='nongazingHeatmap.png'
        )
    
        # Difference heatmap (gazing - non-gazing)
        heatmap_diff = heatmap_gazing_smooth_standard - heatmap_nongazing_smooth_standard
    
        plt.figure(figsize=(12, 6))
        vmax = np.percentile(np.abs(heatmap_diff), 99)
        plt.imshow(
            heatmap_diff,
            cmap='seismic',
            interpolation='nearest',
            origin='upper',
            extent=[0, width, height, 0],
            vmin=-vmax,
            vmax=vmax
        )
        plt.colorbar(label='Difference (Gazing - Non-Gazing)')
        plt.title('Difference Heatmap: Gazing vs Non-Gazing')
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        plt.savefig(f"{self.prefix}gazeDifferenceHeatmap.png", bbox_inches='tight')
        plt.show()

    def _plot_single_heatmap(self, heatmap, width, height, bin_size, title, filename):
        heatmap_log = np.log1p(heatmap)
    
        plt.figure(figsize=(12, 6))
        plt.imshow(
            heatmap_log,
            cmap='hot',
            interpolation='nearest',
            origin='upper',
            extent=[0, width, height, 0],
            vmin=np.percentile(heatmap_log, 5),
            vmax=np.percentile(heatmap_log, 99)
        )
        plt.colorbar(label='Log(Time Spent)')
        plt.title(title)
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        plt.savefig(f"{self.prefix}{filename}", bbox_inches='tight')
        plt.show()
                
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
    
    def classifyStrategies(self):
        '''
        Make a plot trying to classify whether the rats choose a 
        strategy or if its more of a continuous spectrum. 
        (Only for Successful Trials)
        
        x-axis (Synchronized Running): avg x-distance between the rats from trial start to trial success
        y-axis (Waiting at Lever): frames both ratswere waiting at the lever
        '''
        
        trial_x = []
        trial_y = []
        
        for exp_idx, exp in enumerate(self.experiments):
            lev = exp.lev
            pos = exp.pos
            fps = exp.fps
            
            # Get trial timings
            trial_starts = lev.returnTimeStartTrials()  # List of trial start times
            coop_or_last_press = lev.returnCoopTimeorLastPressTime()  # List of coop/last press times
            
            success_status = lev.returnSuccessTrials()  # List of trial success status (1, 0, -1)
            success_status = self._filterToLeverPressTrials(success_status, lev)
            
            print("Got Trial timings")
            
            numTrialsPress = lev.returnNumTotalTrialswithLeverPress()
            numTrialsTot = lev.returnNumTotalTrials()
            print(f"lengths: {len(trial_starts)}, {len(coop_or_last_press)}, {len(success_status)}, {numTrialsPress}, {numTrialsTot}")
            
            # Ensure consistent trial lengths
            n_trials = min(len(trial_starts), len(coop_or_last_press), len(success_status))
            print("n_trials: ", n_trials)
            
            for trial_idx in range(n_trials):
                #Preparation: 
                #
                #
                
                if success_status[trial_idx] != 1:  # Only consider successful trials
                    continue
                
                # Define stage boundaries (in seconds)
                t_begin = trial_starts[trial_idx]
                t_coop = coop_or_last_press[trial_idx]
                
                if (t_begin == None or t_coop == None):
                    continue
                
                # Check for NaN in timings
                if any(np.isnan(t) for t in [t_begin, t_coop]):
                    print(f"[Exp {exp_idx}, Trial {trial_idx}] Skipped: NaN in timings (begin={t_begin}, coop={t_coop})")
                    continue
                
                # Convert times to frame indices
                f_begin = int(t_begin * fps)
                f_coop = int(t_coop * fps)
                
                #––––––––––––––––––––––––––––––––––––––––––––––––
                
                #Actual Data Collection
                #
                #
                
                #Synchronized Running
                #
                
                numFrames = f_coop - f_begin
                
                rat1_xlocations = pos.data[0, 0, pos.HB_INDEX, f_begin:f_coop]
                rat2_xlocations = pos.data[1, 0, pos.HB_INDEX, f_begin:f_coop]
                
                difference = sum(abs(a - b) for a, b in zip(rat1_xlocations, rat2_xlocations))            
                x = difference/numFrames                
                
                #Waiting
                #
                
                #Waiting Before Queue Analysis
                t = f_begin - 1
                rat0_waiting = 0
                rat1_waiting = 0
                rat0_active = True
                rat1_active = True
                rat0_locations = pos.returnMouseLocation(0)
                rat1_locations = pos.returnMouseLocation(1)

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
                
                frames_both_waited = min(rat0_waiting, rat1_waiting)
                
                
                if (x < 1000 and frames_both_waited < 350):
                    trial_x.append(difference / numFrames)
                    trial_y.append(frames_both_waited)
            
            # Convert to DataFrame for seaborn
            df = pd.DataFrame({'x_dist': trial_x, 'waiting': trial_y})
            
            # Plot 1: Hexbin
            plt.figure(figsize=(8, 6))
            plt.hexbin(trial_x, trial_y, gridsize=50, cmap='viridis', mincnt=1)
            plt.colorbar(label='Trial Density')
            plt.xlabel('Avg X-Distance (Synchronized Running)')
            plt.ylabel('Frames Both Rats Waited (at Lever)')
            plt.title('Hexbin Plot of Trial Strategies')
            if self.save:
                plt.savefig("hexbin_strategy_plot.png")
            plt.show()
        
            # Plot 2: 2D Histogram
            plt.figure(figsize=(8, 6))
            plt.hist2d(trial_x, trial_y, bins=50, cmap='inferno', vmin=0)
            plt.colorbar(label='Trial Density')
            plt.xlabel('Avg X-Distance (Synchronized Running)')
            plt.ylabel('Frames Both Rats Waited')
            plt.title('2D Histogram of Trial Strategies')
            if self.save:
                plt.savefig("hist2d_strategy_plot.png")
            plt.show()
        
            # Plot 3: Transparent Scatter
            plt.figure(figsize=(8, 6))
            plt.scatter(trial_x, trial_y, alpha=0.35, s=11)
            plt.xlabel('Avg X-Distance (Synchronized Running)')
            plt.ylabel('Frames Both Rats Waited')
            plt.title('Transparent Scatter of Trial Strategies')
            if self.save:
                plt.savefig("scatter_strategy_plot.png")
            plt.show()
        
            # Plot 4: KDE Plot
            plt.figure(figsize=(8, 6))

            # KDE with contours
            kde = sns.kdeplot(
                data=df,
                x='x_dist',
                y='waiting',
                fill=True,
                cmap='coolwarm',
                levels=100,
                thresh=0.05,
                alpha=0.9
            )
        
            # Contour lines
            sns.kdeplot(
                data=df,
                x='x_dist',
                y='waiting',
                color='black',
                levels=6,
                linewidths=1,
            )
        
            # Add a scatter of all points with alpha
            plt.scatter(trial_x, trial_y, s=5, alpha=0.3, color='black')
        
            # Optional: annotate the densest point
            from scipy.stats import gaussian_kde
        
            values = np.vstack([trial_x, trial_y])
            kernel = gaussian_kde(values)
            densities = kernel(values)
            max_density_idx = np.argmax(densities)
            x_peak, y_peak = trial_x[max_density_idx], trial_y[max_density_idx]
                
            plt.xlabel('Avg X-Distance (Synchronized Running)')
            plt.ylabel('Frames Both Rats Waited')
            plt.title('KDE Density of Trial Strategies')
            if self.save:
                plt.savefig("kde_strategy_plot.png")
            plt.show()

    def classifyInteractions(self):
        '''
        Classify dyadic configurations between two rats over time.
        Possible categories:
        - 'facing_each_other'
        - 'side_by_side_same'
        - 'side_by_side_opposite'
        - 'back_to_back'
        - 'other'
        '''
        
        for exp in self.experiments:
            def get_head_vector(rat_id):
                return pos.data[rat_id, :, pos.NOSE_INDEX, :] - pos.data[rat_id, :, pos.HB_INDEX, :]
            
            pos = exp.pos
            
            head_vec0 = get_head_vector(0)
            head_vec1 = get_head_vector(1)
        


#Testing Multi File Graphs
#
#

def getFiltered():
    fe = fileExtractor(filtered)
    fe.data = fe.deleteBadNaN()
    fpsList, totFramesList = fe.returnFPSandTotFrames()
    initial_nan_list = fe.returnNaNPercentage()
    #print("initial_nan_list: ", initial_nan_list)
    return [fe.getLevsDatapath(), fe.getMagsDatapath(), fe.getPosDatapath(), fpsList, totFramesList, initial_nan_list]


fiberPhoto = "/gpfs/radev/home/drb83/project/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/fiber_photo.csv"
def getFiberPhoto():
    fe = fileExtractor(fiberPhoto)
    #fe.data = fe.deleteBadNaN()
    fpsList, totFramesList = fe.returnFPSandTotFrames()
    initial_nan_list = fe.returnNaNPercentage()
    fiberFiles = fe.getFiberPhotoDataPath()
    print("fiberFiles: ", fiberFiles)
    #print("initial_nan_list: ", initial_nan_list)
    return [fe.getLevsDatapath(), fe.getMagsDatapath(), fe.getPosDatapath(), fpsList, totFramesList, initial_nan_list, fiberFiles]


'''
lev_files = ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv"]

mag_files = ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv"] 

pos_files = ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5"]

fpsList = [30, 30]
totFramesList = [15000, 15000]
initialNanList = [0.15, 0.12]
'''



arr = getFiltered()
#arr = getAllTrainingCoop()
#arr = getFiberPhoto()
lev_files = arr[0]
mag_files = arr[1]
pos_files = arr[2]
fpsList = arr[3]
totFramesList = arr[4]
initialNanList = arr[5]
#fiberPhoto = arr[6]



'''
lev_files = ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/4_nanerror_lev.csv"]
mag_files = ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/4_nanerror_mag.csv"]
pos_files = ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/4_nanerror_test.h5"]
fiberPhoto = [["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/090324_Cam1_TrNum14_Coop_KL002B-KL002Y_x405_TTLs.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/090324_Cam1_TrNum14_Coop_KL002B-KL002Y_x465_TTLs.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/090324_Cam1_TrNum14_Coop_KL002B-KL002Y_x560_TTLs.csv"]]
'''

'''#Missing Trial Nums
lev_files = ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/040124_KL005B-KL005Y_lever.csv"]
mag_files = ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/040124_KL005B-KL005Y_mag.csv"]
pos_files = ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/040124_COOPTRAIN_LARGEARENA_KL005B-KL005Y_Camera1.predictions.h5"]
'''

#fpsList = [29]
#totFramesList = [10000]
#initialNanList = [0.3]



print("Start MultiFileGraphs Regular")
experiment = multiFileGraphs(mag_files, lev_files, pos_files, fpsList, totFramesList, initialNanList, prefix = "", save=True)
experiment.classifyStrategies()

#experiment.gazeHeatmap()
#experiment.trialStateModel()
#experiment.testMotivation()
#experiment.waitingStrategy()




# ---------------------------------------------------------------------------------------------------------


#Class to create Graphs with data from a single file (Not really useful and outdated, IGNORE)
#
#


class singleFileGraphs:
    def __init__(self, mag_file, lev_file, pos_file):
        self.experiment = singleExperiment(mag_file, lev_file, pos_file)
    
    
    def magFileDataAvailabilityGraph(self):
        """
        Create a bar graph showing the percentage of non-null data for each category in magData.
        
        The graph is saved as 'mag_data_availability.png'.
        """
        percentageDataPerCategory = []  # Percentage of data values that are not null for each category
        magData = self.experiment.mag 
        totalRows = magData.getNumRows()
        
        if totalRows == 0:
            raise ValueError("No data available in magData.")
        
        for cat in magData.categories:
            numNulls = magData.countNullsinColumn(cat)
            percentageDataPerCategory.append((totalRows - numNulls) / totalRows * 100)
        
        # Create bar graph
        plt.figure(figsize=(10, 6))
        bars = plt.bar(magData.categories, percentageDataPerCategory, color='skyblue')
        
        # Customize the plot
        plt.xlabel('Categories')
        plt.ylabel('Percentage of Non-Null Data (%)')
        plt.title('Data Availability in Mag Data File')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 100)
        
        # Add percentage labels on top of each bar
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval - 5, f'{yval:.1f}%', 
                     ha='center', va='bottom')
        
        # Adjust layout to prevent label cutoff
        #plt.tight_layout()
        
        # Save the plot
        #plt.savefig('mag_data_availability.png')
        plt.show()
        plt.close()
            
    def levFileDataAvailabilityGraph(self):
        """
        Create a bar graph showing the percentage of non-null data for each category in levData.
        
        The graph is saved as 'lev_data_availability.png'.
        """
        percentageDataPerCategory = []  # Percentage of data values that are not null for each category
        levData = self.experiment.lev
        totalRows = levData.getNumRows()
        
        if totalRows == 0:
            raise ValueError("No data available in levData.")
        
        for cat in levData.categories:
            numNulls = levData.countNullsinColumn(cat)
            percentageDataPerCategory.append((totalRows - numNulls) / totalRows * 100)
        
        # Create bar graph
        plt.figure(figsize=(10, 6))
        bars = plt.bar(levData.categories, percentageDataPerCategory, color='skyblue')
        
        # Customize the plot
        plt.xlabel('Categories')
        plt.ylabel('Percentage of Non-Null Data (%)')
        plt.title('Data Availability in Lev Data File')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 100)
        
        # Add percentage labels on top of each bar
        for bar in bars:
            yval = bar.get_height()
            yval = max(yval, 6)
            plt.text(bar.get_x() + bar.get_width()/2, yval - 5, f'{yval:.1f}%', 
                     ha='center', va='bottom')
        
        # Adjust layout to prevent label cutoff
        #plt.tight_layout()
        
        # Save the plot
        #plt.savefig('lev_data_availability.png')
        plt.show()
        plt.close()
    
    def interpressIntervalPlot(self):
        lev_data = self.experiment.lev.data
        #print(lev_data)

        if lev_data is None:
            raise ValueError("Lever press data is not loaded.")

        # Sort by Rat and AbsTime 
        lev_data_sorted = lev_data.sort_values(by=["RatID", "AbsTime"]) 
        
        #Optional: filter out re-presses (Hit = -1 or -2)
        #lev_data_sorted = lev_data_sorted[lev_data_sorted["Hit"] == 1]

        # Calculate Inter-Press Interval (IPI) within each animal group
        lev_data_sorted["IPI"] = lev_data_sorted.groupby("RatID")["AbsTime"].diff()
        
        #Filter outNaNs
        ipi_data = lev_data_sorted[(lev_data_sorted["IPI"].notna())]
        #print(ipi_data)
        
        # Plot IPI histogram
        plt.figure(figsize=(10, 6))
        plt.hist(ipi_data["IPI"], bins=50, edgecolor='black')
        plt.xlabel("Inter-Press Interval (s)")
        plt.ylabel("Frequency")
        plt.title("Histogram of Inter-Press Intervals")
        plt.tight_layout()
        plt.show()
        plt.close()
        
        # Boxplot of IPI per animal
        plt.figure(figsize=(10, 6))
        lev_data_sorted.boxplot(column="IPI", by="RatID")
        plt.title("Inter-Press Interval (IPI) by Rat")
        plt.suptitle("")  # Removes default pandas boxplot title
        plt.xlabel("RatID")
        plt.ylabel("IPI (seconds)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        plt.close()
        
        
        # Time series plot of IPI for each rat
        plt.figure(figsize=(12, 6))
        
        for rat_id, group in lev_data_sorted.groupby("RatID"):
            plt.plot(group["AbsTime"], group["IPI"], label=f"Rat {rat_id}", alpha=0.7)
        
        plt.title("IPI Over Time by Rat")
        plt.xlabel("Absolute Time (s)")
        plt.ylabel("Inter-Press Interval (s)")
        plt.legend(title="RatID", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
            
    def interpressIntervalSuccessPlot(self):
        lev_data = self.experiment.lev.data.copy()

        # Filter only trials with cooperative success
        successful_trials = lev_data[lev_data["coopSucc"] == 1]

        # Group by trial
        grouped = successful_trials.groupby(["TrialNum"])

        ipi_first_to_success = []
        ipi_last_to_success = []
        first_press_rats = []
        success_count_by_rat = {}

        for trial_num, trial_data in grouped:
            trial_data = trial_data.sort_values(by="AbsTime")

            if 1 not in trial_data["Hit"].values:
                continue  # skip if no initial press

            # Find index of cooperative press
            coop_press = trial_data[trial_data["TrialEnd"] == 1]
            if coop_press.empty:
                continue

            coop_press_time = coop_press.iloc[0]["AbsTime"]

            # First press in trial
            first_press = trial_data[trial_data["Hit"] == 1].iloc[0]
            first_press_time = first_press["AbsTime"]

            # Last press before cooperative success
            before_coop = trial_data[trial_data["AbsTime"] < coop_press_time]
            if before_coop.empty:
                continue

            last_press = before_coop.iloc[-1]
            last_press_time = last_press["AbsTime"]

            ipi_first_to_success.append(coop_press_time - first_press_time)
            ipi_last_to_success.append(coop_press_time - last_press_time)

            # Record first press rat
            first_press_rats.append(first_press["RatID"])

            # Count successful coop per rat
            rat_id = coop_press.iloc[0]["RatID"]
            success_count_by_rat[rat_id] = success_count_by_rat.get(rat_id, 0) + 1

        # Plot 1: Histogram - First Press to Success
        plt.figure(figsize=(8, 6))
        plt.hist(ipi_first_to_success, bins=20, color='skyblue')
        plt.title("Histogram: First Press to Success")
        plt.xlabel("IPI (s)")
        plt.ylabel("Count")
        plt.show()
        plt.close()

        # Plot 2: Histogram - Last Press to Success
        plt.figure(figsize=(8, 6))
        plt.hist(ipi_last_to_success, bins=20, color='salmon')
        plt.title("Histogram: Last Press to Success")
        plt.xlabel("IPI (s)")
        plt.ylabel("Count")
        plt.show()
        plt.close()

        # Plot 3: Time Series - First Press to Success
        plt.figure(figsize=(10, 5))
        plt.plot(ipi_first_to_success, marker='o')
        plt.title("Time Series: First Press to Success")
        plt.xlabel("Trial Index")
        plt.ylabel("IPI (s)")
        plt.show()
        plt.close()

        # Plot 4: Time Series - Last Press to Success
        plt.figure(figsize=(10, 5))
        plt.plot(ipi_last_to_success, marker='o', color='orange')
        plt.title("Time Series: Last Press to Success")
        plt.xlabel("Trial Index")
        plt.ylabel("IPI (s)")
        plt.show()
        plt.close()

        # Plot 5: Pie Chart - Who Presses First (by RatID)
        rat_counts = pd.Series(first_press_rats).value_counts()
        plt.figure(figsize=(6, 6))
        plt.pie(rat_counts, labels=rat_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title("Who Presses First (by RatID)")
        plt.show()
        plt.close()


    def percentSuccesfulTrials(self):
        lev_data = self.experiment.lev.data.copy()
        grouped = lev_data.groupby("TrialNum")
        
        
        totalTrials, countSuccess = 0, 0
        for trial_num, trial_data in grouped:
            if (trial_data.iloc[0]['coopSucc'] == 1):
                countSuccess += 1
            
            totalTrials += 1
        
        countFail = totalTrials - countSuccess
        labels = ['Successful', 'Unsuccessful']
        sizes = [countSuccess, countFail]
        colors = ['green', 'red']
    
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 14})
        plt.title('Successful Cooperative Trials (%)', fontsize=16)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.tight_layout()
        plt.show()
        plt.close()
                
            
    
#Testing Single File Graphs
#
#

'''mag_file = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral_Quantification/Example Data Files/magData.csv"
lev_file = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral_Quantification/Example Data Files/leverData.csv"
pos_file = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral_Quantification/Example Data Files/ExampleTrackingCoop.h5"

experiment = singleFileGraphs(mag_file, lev_file, pos_file)
experiment.percentSuccesfulTrials()
experiment.interpressIntervalSuccessPlot()'''




