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
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d

from matplotlib.patches import Patch

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
        FRAME_WINDOW = 1800  # Defines the number of frames corresponding to one minute
    
        # Iterate over each group of experiments (i.e., each category)
        for group in self.allFileGroupExperiments:
            sumEvents = 0     # Total number of gaze events for this category
            sumFrames = 0     # Total number of frames across all experiments in this category
    
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
    
            # Compute average gaze events per 1800 frames if data is available
            if sumFrames > 0:
                avg_events.append(sumEvents / sumFrames * FRAME_WINDOW)
    
        # --- Plot: Bar chart of average gaze events per minute (1800 frames) ---
        plt.figure(figsize=(8, 6))
        plt.bar(range(len(avg_events)), avg_events, color='skyblue')
        plt.xlabel('Category')
        plt.ylabel('Avg. Gaze Events per 1800 Frames')
        plt.title('Average Gaze Events (per Minute) per Category')
        plt.xticks(range(len(avg_events)), self.categoryNames)
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
                    print("\n\nProb: ",  num_succ / num_total)
                    print("Num Trials: ", num_total)
                    print("Lev File: ", self.allFileGroupExperiments[i][j].lev_file)
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
                    
                else:
                    individual_datapoints[i].append(np.nan)
                    datapoint_colors[i].append('gray')
                    print("\n\nTotal Trials was 0")
                    print("Lev File: ", self.allFileGroupExperiments[i][j][1])
    
            # Compute overall success probability for the category
            prob = totalSucc / totalTrials if totalTrials > 0 else 0
            probs.append(prob)
    
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
        plt.xlabel('Category')
        plt.ylabel('Probability of Successful Trials')
        plt.title('Success Probability per Category')
        plt.xticks(bar_positions, self.categoryNames)
        plt.ylim(0, 1)
        legend_patches = [
            Patch(color='red', label='Threshold > 3'),
            Patch(color='orange', label='Threshold > 2'),
            Patch(color='blue', label='Threshold > 1'),
            Patch(color='black', label='Threshold > 0'),
            Patch(color='gray', label='Threshold ≤ 0')
        ]
        plt.legend(handles=legend_patches)
        #plt.legend()
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
            
    def make_bar_plot(self, data, ylabel, title, saveFileName):
        plt.figure(figsize=(8, 5))
        plt.bar(range(len(data)), data, color='skyblue')
        plt.xticks(range(len(data)), self.categoryNames)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        if (self.save):
            plt.savefig(f'{self.path}{saveFileName}{self.endSaveName}')
        plt.show()
        plt.close()
    
    def printSummaryStats(self):
        avg_gaze_lengths = []       # Stores average gaze duration (in frames) per category
        avg_gaze_lengths_alternate = [] # Stores average gaze duration (in frames) per category for alternate definition
        avg_lever_per_trial = []    # Stores average number of lever presses per trial
        avg_mag_per_trial = []      # Stores average number of magazine entries per trial
    
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
                total_gaze_events += loader.returnNumGazeEvents(0, alternateDef=False) + loader.returnNumGazeEvents(1, alternateDef=False)
                total_gaze_frames += np.sum(g0) + np.sum(g1)
                total_frames += g0.shape[0]
                
                total_gaze_events_alternate += loader.returnNumGazeEvents(0) + loader.returnNumGazeEvents(1)
                total_gaze_frames_alternate += np.sum(g2) + np.sum(g3)
    
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
    
            # Calculate averages for the category
            avg_gaze_len = (total_gaze_frames / total_gaze_events) if total_gaze_events > 0 else 0
            avg_gaze_len_alternate = (total_gaze_frames_alternate / total_gaze_events_alternate) if total_gaze_events_alternate > 0 else 0
            avg_lever = (total_lever_presses / total_trials) if total_trials > 0 else 0
            avg_mag = (total_mag_events / total_trials) if total_trials > 0 else 0
    
            # Store for plotting
            avg_gaze_lengths.append(avg_gaze_len)
            avg_gaze_lengths_alternate.append(avg_gaze_len_alternate)
            avg_lever_per_trial.append(avg_lever)
            avg_mag_per_trial.append(avg_mag)
    
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
            print(f"  Percent Gazing: {100 * total_gaze_events / total_frames:.2f}%")
            print(f"  Total Gaze Events (Alternate): {total_gaze_events_alternate}")
            print(f"  Average Gaze Length (Alternate): {total_gaze_frames_alternate / total_gaze_events_alternate:.2f}")
            print(f"  Percent Gazing (Alternate): {100 * total_gaze_frames_alternate / total_frames:.2f}%")
            print(f"  Avg Lever Presses per Trial: {total_lever_presses / total_trials:.2f}")
            print(f"  Total Lever Presses: {total_lever_presses}")
            print(f"  Avg Mag Events per Trial: {total_mag_events / total_trials:.2f}")
            print(f"  Total Mag Events: {total_mag_events}")
    
        # --- Plot results for each metric ---
        self.make_bar_plot(
            avg_gaze_lengths,
            'Avg Gaze Length (frames)',
            'Average Gaze Length per Category',
            "Avg_Gaze_Length"
        )
        
        self.make_bar_plot(
            avg_gaze_lengths_alternate,
            'Avg Gaze Length (frames)',
            'Average Gaze Length per Category (Alternate Def)',
            "Avg_Gaze_Length"
        )
    
        self.make_bar_plot(
            avg_lever_per_trial,
            'Avg Lever Presses per Trial',
            'Lever Presses per Trial per Category',
            "Avg_Lev_Presses_perTrial"
        )
    
        self.make_bar_plot(
            avg_mag_per_trial,
            'Avg Mag Events per Trial',
            'Mag Events per Trial per Category',
            "Avg_Mag_Events_perTrial"
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


levFiles = [["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_lever.csv"], 
            ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv"]]
magFiles = [["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_mag.csv"], 
            ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv"]]
posFiles = [["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G.predictions.h5"], 
            ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5"]]


#categoryExperiments = multiFileGraphsCategories(magFiles, levFiles, posFiles, ["Paired_Testing", "Training_Cooperation"], save=False)
#categoryExperiments.printSummaryStats()

#categoryExperiments.compareSuccesfulTrials()
#categoryExperiments.rePressingBehavior()
#categoryExperiments.gazeAlignmentAngle()

#Paired Testing vs. Training Cooperation


print("Running Paired Testing vs Training Cooperation")
dataPT = getOnlyPairedTesting()
dataTC = getOnlyTrainingCoop()

levFiles = [dataPT[0], dataTC[0]]
magFiles = [dataPT[1], dataTC[1]]
posFiles = [dataPT[2], dataTC[2]]
categoryExperiments = multiFileGraphsCategories(magFiles, levFiles, posFiles, ["Paired_Testing", "Training_Cooperation"])
categoryExperiments.compareSuccesfulTrials()


'''
#Unfamiliar vs. Training Partners
print("Running UF vs TP")
dataUF = getOnlyUnfamiliar() #Unfamiliar
dataTP = getOnlyTrainingPartners() #Training Partners

levFiles = [dataUF[0], dataTP[0]]
magFiles = [dataUF[1], dataTP[1]]
posFiles = [dataUF[2], dataTP[2]]
categoryExperiments = multiFileGraphsCategories(magFiles, levFiles, posFiles, ["Unfamiliar", "Training Partners"])
categoryExperiments.compareSuccesfulTrials()
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
categoryExperiments.compareSuccesfulTrials()
'''


print("0")
#categoryExperiments.compareGazeEventsCategories()
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
    def __init__(self, magFiles: List[str], levFiles: List[str], posFiles: List[str], fpsList: List[int], totFramesList: List[int], initialNanList: List[int], prefix = "", save = True):
        self.experiments = []
        self.prefix = prefix
        self.save = save
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
        
        for i in range(len(magFiles)):
            exp = singleExperiment(magFiles[i], levFiles[i], posFiles[i], fpsList[i], totFramesList[i], initialNanList[i])
            mag_missing = [col for col in exp.mag.categories if col not in exp.mag.data.columns]
            lev_missing = [col for col in exp.lev.categories if col not in exp.lev.data.columns]
            
            print("mag.categories: ", exp.mag.categories)
            print("lev.categories: ", exp.lev.categories)
            
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
            plt.savefig(f'{self.prefix}MaxvsMin.png')
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
            plt.savefig(f'{self.prefix}MagMaxvsMin.png')
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
        
        for exp in self.experiments:
            lev = exp.lev
            pos = exp.pos
            fps = exp.fps
            
            listTrials = lev.returnCooperativeSuccessRegionsBool()
            startTimeTrials = lev.returnTimeStartTrials()
            endTimeTrials = lev.returnTimeEndTrials()
            
            successTrial = lev.returnSuccessTrials()
            
            
            print("Lengths:", len(listTrials), len(startTimeTrials), len(endTimeTrials))
            if (len(listTrials) != len(startTimeTrials) or  len(startTimeTrials) != len(endTimeTrials)):
                print("levFiles: ", exp.lev_file)
            
            for i, trialBool in enumerate(listTrials):
                if (startTimeTrials[i] == None or endTimeTrials[i] == None):
                    continue
                
                startFrame = int(startTimeTrials[i] * fps)
                endFrame = int(endTimeTrials[i] * fps)
                
                #print("startFrame: ", startFrame)
                #print("endFrame: ", endFrame)
                
                numFrames = endFrame - startFrame
                
                rat1_xlocations = pos.data[0, 0, pos.HB_INDEX, startFrame:endFrame]
                rat2_xlocations = pos.data[1, 0, pos.HB_INDEX, startFrame:endFrame]
                
                difference = sum(abs(a - b) for a, b in zip(rat1_xlocations, rat2_xlocations))            
                
                if (trialBool):
                    totFrames_SuccessZone += numFrames
                    totDifference_SuccessZone += difference
                    if (numFrames > 0):
                        datapoints_SuccessZone.append(difference / numFrames)
                        
                elif(successTrial[i] == 1):
                    totFrames_Success_NoZone += numFrames
                    totDifference_Success_NoZone += difference
                    if (numFrames > 0):
                        datapoints_Success_NoZone.append(difference / numFrames)
                else:
                    totFrames_NoSuccess += numFrames
                    totDifference_NoSuccess += difference
                    if (numFrames > 0):
                        datapoints_NoSuccess.append(difference / numFrames)
        
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
        labels = ['No Success', 'Success Zone', 'Success No Zone']
        averages = [averageDistance_NoSuccess, averageDistance_SuccessZone, averageDistance_Success_NoZone]
        datapoints = [datapoints_NoSuccess, datapoints_SuccessZone, datapoints_Success_NoZone]
        
        # X locations for the bars and jittered scatter points
        x = np.arange(len(labels))
        width = 0.6
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot bar chart
        bars = ax.bar(x, averages, width, color=['red', 'green'], alpha=0.6, edgecolor='black')
        
        # Overlay individual data points
        for i, points in enumerate(datapoints):
            # Add jitter to the x-position of each point for visibility
            jittered_x = np.random.normal(loc=x[i], scale=0.05, size=len(points))
            ax.scatter(jittered_x, points, alpha=0.8, color='black', s=20)
        
        # Labels and formatting
        ax.set_ylabel('Average Distance')
        ax.set_title('Average Head-Body X-Distance per Trial')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(f"{self.prefix}X_Distance_SuccessZonevsNoSuccess.png")
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

    def waitingStrategy(self): 
        """
        Creates a scatterplot of average waiting time vs. success rate across experiments,
        and a pie chart showing the percentage of time spent waiting in lever areas.
        """
        NUM_BINS = 30

        def findWaitingTimeTrials(exp):
            """
            For a single experiment, calculate 
                1) the number of frames each rat spends in levTop or 
                   levBot before the first lever press in each trial.
                2) Frame latency to first lever entry 
                   (How long after trial start did it take for a rat to enter a lever area)
                3) How many frames were both rats in a lever area at the same time before the first press?
                4) Change in Waiting Strategy over Time
                5) % of time both rats wait on same vs. opposite levers
                6) Occupancy rate over normalized trial time
                7) % of time 0 / 1 / 2 rats are in lever area
            
            Args:
                exp: singleExperiment object containing lev and pos loaders.
    
            Returns:
                tuple: (rat0_waiting_times, rat1_waiting_times)
                       Lists of waiting times (in frames) for each trial for Rat 0 and Rat 1.
            """
            lev = exp.lev
            pos = exp.pos
            fps = exp.fps
            
            #framesWaitingBeforeTrialStarted
            maxFramesWaitingBeforeTrialStarted = []
            framesWaitedSucc = 0
            
            # Get first press absolute times and rat IDs per trial
            first_press_times = lev.returnFirstPressAbsTimes()
            first_press_rat_ids = lev.returnRatIDFirstPressTrial()
            num_trials = lev.returnNumTotalTrialswithLeverPress()
    
            total_trial_frames = 0  # Total frames considered across all trials in this experiment
            total_waiting_frames = 0 # Total frames where at least one rat is waiting near lever
            
            # Initialize lists to store metrics per trial
            rat0_waiting_times = []      # Frames rat 0 spends waiting near lever
            rat1_waiting_times = []      # Frames rat 1 spends waiting near lever
            waiting_symmetry = []        # Absolute difference in waiting times between rats per trial
            synchronous_waiting_frames = [] # Frames both rats waiting simultaneously per trial
            rat0_latencies = []          # Frames until rat 0 first enters lever zone after trial start
            rat1_latencies = []          # Same for rat 1
            
            same_lever_total = 0 #Number of frames in which both rats are in a lever area and are on the same lever side
            opposite_lever_total = 0 #Number of frames where both rats are in lever areas but on opposite sides.
            none_total = 0 #Frames where neither rat is in a lever area.
            one_total = 0 #Frames where exactly one rat is in a lever area.
            both_total = 0 #Frames where both rats are in lever areas.
    
            #NUM_BINS = 30 #The number of equal-sized time bins into which you divide each trial (e.g. 30 bins = 30 time points from 0% to 100% of the trial duration).
            occupancy_curve = np.zeros(NUM_BINS) #An array of length NUM_BINS that accumulates the number of trials where at least one rat was in a lever area at each time bin
            trial_counts = np.zeros(NUM_BINS) #Parallel to occupancy_curve, this keeps track of how many trials actually had a valid frame in each bin.
            
            # Get trial start times
            start_times = lev.returnTimeStartTrials()
            end_times = lev.returnTimeEndTrials()
            
            idx_counter = 0
            
            succTrials = lev.returnSuccessTrials()
            
            for trial_idx in range(num_trials):
                start_time = start_times[trial_idx]
                end_time = end_times[trial_idx]
                
                if (start_time is None or np.isnan(start_time) or end_time is None or np.isnan(end_time)):
                    idx_counter += 1
                    continue
                press_time = first_press_times[trial_idx - idx_counter]
                rat_id = first_press_rat_ids.iloc[trial_idx - idx_counter]
                
                if np.isnan(press_time) or np.isnan(rat_id) or start_time is None:
                    # Skip trials with invalid data
                    idx_counter += 1
                    #rat0_waiting_times.append(0)
                    #rat1_waiting_times.append(0)
                    continue
                
                
                rat_id = int(rat_id)
                press_frame = int(press_time * fps)
                start_frame = int(start_time * fps)
                end_frame = int(end_time * fps)
                
                print("\nStart Frame: ", start_frame)
                print("\nPress Frame: ", press_frame)
                print("\nEnd Frame: ", end_frame)
                
                numFrames = (press_frame - start_frame)
                if (numFrames == 0):
                    idx_counter += 1
                    continue
    
                if press_frame < start_frame or press_frame >= pos.data.shape[-1]:
                    # Skip invalid frame ranges
                    #rat0_waiting_times.append(0)
                    #rat1_waiting_times.append(0)
                    idx_counter += 1
                    continue
                
                rat0_locations_before = pos.returnMouseLocation(0)
                rat1_locations_before = pos.returnMouseLocation(1)
                
                t = start_frame
                stillIn0 = True
                stillIn1 = True
                countWait0 = 0
                countWait1 = 0
                
                while(t >= 0 and t < len(rat0_locations_before) and t < len(rat1_locations_before) and rat0_locations_before[t] is not None):
                    t -= 1
                    
                    if (rat0_locations_before[t] in ['lev_top', 'lev_bottom'] and stillIn0):
                        countWait0 += 1
                    else:
                        stillIn0 = False
                    
                    if (rat1_locations_before[t] in ['lev_top', 'lev_bottom'] and stillIn1):
                        countWait1 += 1
                    else:
                        stillIn1 = False
                        
                    if (stillIn0 == False and stillIn1 == False):
                        break
                    
                maxFramesWaitingBeforeTrialStarted.append(max(countWait0, countWait1))
                
                if (succTrials[trial_idx]):
                    framesWaitedSucc += max(countWait0, countWait1)
                    
                
                # Get locations for both rats in the trial window
                rat0_locations_min = rat0_locations_before[start_frame:press_frame]
                rat1_locations_min = rat1_locations_before[start_frame:press_frame]
                
                rat0_locations = rat0_locations_before[start_frame:end_frame]
                rat1_locations = rat1_locations_before[start_frame:end_frame]
                
                # Ensure both lists are the same length
                frame_count = min(len(rat0_locations), len(rat1_locations))
                frame_count_min = min(len(rat0_locations_min), len(rat1_locations_min))
                
                total_trial_frames += numFrames
                
                # Latency to lever area entry
                def latency_to_lever(locations):
                    for i, loc in enumerate(locations):
                        if loc in ['lev_top', 'lev_bottom']:
                            return i
                    return None
    
                rat0_lat = latency_to_lever(rat0_locations)
                rat1_lat = latency_to_lever(rat1_locations)
                
                if rat0_lat is not None:
                    rat0_latencies.append(rat0_lat)
                if rat1_lat is not None:
                    rat1_latencies.append(rat1_lat)
                
                print("frame_count is: ", frame_count)
                print("len(rat0_locations_min): ", len(rat0_locations_min))
                for i in range(frame_count_min):
                    r0 = rat0_locations_min[i]
                    r1 = rat1_locations_min[i]
                    r0_in = r0 in ['lev_top', 'lev_bottom']
                    r1_in = r1 in ['lev_top', 'lev_bottom']
                    if r0_in and r1_in:
                        if r0 == r1:
                            same_lever_total += 1
                        else:
                            opposite_lever_total += 1
    
                    if r0_in and r1_in:
                        both_total += 1
                    elif r0_in or r1_in:
                        one_total += 1
                    else:
                        none_total += 1
                
                bin_edges = np.linspace(0, frame_count, NUM_BINS + 1, dtype=int)

                for bin_idx in range(NUM_BINS):
                    start_frame = bin_edges[bin_idx]
                    end_frame = bin_edges[bin_idx + 1]
                
                    # Ensure we stay within frame limits
                    if start_frame >= frame_count:
                        continue
                
                    # Slice the frame window for this bin
                    rat0_bin = rat0_locations[start_frame:end_frame]
                    rat1_bin = rat1_locations[start_frame:end_frame]
                
                    # Count how many frames either rat is in lever zone
                    in_lever = sum(
                        (loc in ['lev_top', 'lev_bottom']) for loc in rat0_bin
                    ) + sum(
                        (loc in ['lev_top', 'lev_bottom']) for loc in rat1_bin
                    )
                
                    occupancy_curve[bin_idx] += in_lever
                    trial_counts[bin_idx] += (end_frame - start_frame)
                
                # Count total waiting frames: any frame where at least one rat is at a lever
                
                waitingFrames = sum(
                    (rat0_locations_min[i] in ['lev_top', 'lev_bottom']) or
                    (rat1_locations_min[i] in ['lev_top', 'lev_bottom'])
                    for i in range(frame_count_min)
                )
                total_waiting_frames += waitingFrames
                print("waitingFrames: ", waitingFrames)
                
                # Per-frame: both rats waiting (synchronous waiting)
                synchronous_waiting = sum(
                    (rat0_locations_min[i] in ['lev_top', 'lev_bottom']) and
                    (rat1_locations_min[i] in ['lev_top', 'lev_bottom'])
                    for i in range(frame_count_min)
                )
                synchronous_waiting_normalized = synchronous_waiting / frame_count if frame_count > 0 else 0 # Standardize by dividing by total frames in the trial
                synchronous_waiting_frames.append(synchronous_waiting)
    
                # Count frames where each rat is in levTop or levBot individually
                rat0_waiting = sum(1 for loc in rat0_locations_min if loc in ['lev_top', 'lev_bottom'])
                rat1_waiting = sum(1 for loc in rat1_locations_min if loc in ['lev_top', 'lev_bottom'])
                
                #Standardize rat0_waiting and rat1_waiting
                rat0_waiting #/= numFrames
                rat1_waiting #/= numFrames
                
                rat0_waiting_times.append(rat0_waiting)
                rat1_waiting_times.append(rat1_waiting)
                waiting_symmetry.append(abs(rat0_waiting - rat1_waiting))
            
            #if (total_trial_frames != exp.endFrame): 
                #print("Inequal Frames, (self, counted): ", exp.endFrame, ", ", total_trial_frames)
            
            avgWaitingBeforeTrialStarted = np.mean(maxFramesWaitingBeforeTrialStarted)
            
            return (rat0_waiting_times, rat1_waiting_times, waiting_symmetry, rat0_latencies, rat1_latencies,
                synchronous_waiting_frames, total_trial_frames, total_waiting_frames,
                same_lever_total, opposite_lever_total, none_total, one_total, both_total,
                occupancy_curve, trial_counts, avgWaitingBeforeTrialStarted, framesWaitedSucc)

        
        # Aggregate data across experiments
        avg_waiting_times = []
        success_rates = []
        avg_symmetry_vals = []
        avg_sync_vals = []
        rat0_lat_per_trial = defaultdict(list)
        rat1_lat_per_trial = defaultdict(list)
        total_waiting_frames = 0
        total_trial_frames = 0
        total_trials_all = 0
        total_succ_trials = 0
        
        maxWaitBeforeMeans = []
        maxWaitBeforeSucc = 0
        maxWaitBeforeAll = 0
        
        same_lever_sum = 0
        opposite_lever_sum = 0
        none_sum = 0
        one_sum = 0
        both_sum = 0
        total_occupancy_curve = np.zeros(NUM_BINS)
        total_trial_counts = np.zeros(NUM_BINS)
    
        for exp in self.experiments:
            lev = exp.lev
            pos = exp.pos
            
            numSucc = lev.returnNumSuccessfulTrials()
            
            (rat0_times, rat1_times, symmetry, rat0_lat, rat1_lat, 
             sync_frames, trial_frames, waiting_frames,
             same_lever, opposite_lever, none, one, both,
             occupancy_curve, trial_counts, avgWaitBeforeSession, totalWaitBeforeSucc) = findWaitingTimeTrials(exp)
            
            maxWaitBeforeMeans.append(avgWaitBeforeSession)
            
            total_trials = exp.lev.returnNumTotalTrials()
            if total_trials == 0:
                continue
            
            maxWaitBeforeAll += (avgWaitBeforeSession * total_trials)
            
            total_trials_all += total_trials
            maxWaitBeforeSucc += totalWaitBeforeSucc
            total_succ_trials += numSucc
            
            # Calculate average waiting time for the experiment (both rats combined)
            valid_times = [t for t in rat0_times + rat1_times if t > 0]
            avg_wait = np.mean(valid_times) if valid_times else 0
            avg_waiting_times.append(avg_wait)
            
            avg_symmetry_vals.append(np.mean(symmetry) if symmetry else 0)
            avg_sync_vals.append(np.mean(sync_frames) if sync_frames else 0)
            for trial_idx, (lat0, lat1) in enumerate(zip(rat0_lat, rat1_lat)):
                if lat0 is not None and lat1 is not None:
                    rat0_lat_per_trial[trial_idx].append(lat0)
                    rat1_lat_per_trial[trial_idx].append(lat1)
            
            # Calculate success rate
            success_rate = exp.lev.returnNumSuccessfulTrials() / total_trials
            success_rates.append(success_rate)
            
            total_trial_frames += trial_frames
            total_waiting_frames += waiting_frames
            
            same_lever_sum += same_lever
            opposite_lever_sum += opposite_lever
            none_sum += none
            one_sum += one
            both_sum += both
            total_occupancy_curve += occupancy_curve
            total_trial_counts += trial_counts
        
        # Compute average latency per trial index
        avg_latency_per_trial = []
        max_trial_index = max(rat0_lat_per_trial.keys() & rat1_lat_per_trial.keys())
        
        for trial_idx in range(max_trial_index + 1):
            if trial_idx in rat0_lat_per_trial and trial_idx in rat1_lat_per_trial:
                # Only use trials where both rats have data
                avg0 = np.mean(rat0_lat_per_trial[trial_idx])
                avg1 = np.mean(rat1_lat_per_trial[trial_idx])
                avg_latency_per_trial.append((avg0 + avg1) / 2)
         
                
        #Graphs
        #
        #
        
        
        #Bar Chart: Avg Wait Before Successful Trials vs. Avg Wait Before All Trials
        # Compute averages
        avg_wait_succ_trials = maxWaitBeforeSucc / total_succ_trials if total_succ_trials > 0 else 0
        avg_wait_all_trials = maxWaitBeforeAll / total_trials_all if total_trials_all > 0 else 0
        
        # Create bar plot
        labels = ['Successful Trials', 'All Trials']
        wait_times = [avg_wait_succ_trials, avg_wait_all_trials]
        
        plt.figure(figsize=(6, 5))
        bars = plt.bar(labels, wait_times, color=['green', 'gray'])
        plt.ylabel('Average Wait Time (frames or seconds)')
        plt.title('Average Wait Time: Successful vs. All Trials')
        
        # Annotate values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        if self.save:
            plt.savefig(f"{self.prefix}barplot_avgWaitBefore_successfulvsAllTrials.png")
        plt.show()
        plt.close()
        
        
        # Scatterplot: Avg Waiting Time Prior to Press vs. Success Rate
        if len(avg_waiting_times) >= 2 and len(success_rates) >= 2:
            plt.figure(figsize=(8, 6))
            plt.scatter(avg_waiting_times, success_rates, alpha=0.7, color='blue', label='Experiments')
    
            # Add trendline and R²
            if len(set(avg_waiting_times)) >= 2:
                slope, intercept, r_value, _, _ = linregress(avg_waiting_times, success_rates)
                r_squared = r_value ** 2
                x_vals = np.linspace(min(avg_waiting_times), max(avg_waiting_times), 100)
                plt.plot(x_vals, slope * x_vals + intercept, color='red', linestyle='--', label='Trendline')
                plt.text(0.95, 0.05, f"$R^2$ = {r_squared:.3f}", transform=plt.gca().transAxes,
                         ha='right', va='bottom', fontsize=12, bbox=dict(facecolor='white', edgecolor='gray'))
    
            plt.xlabel('Average Waiting Time before Press (frames)')
            plt.ylabel('Cooperative Success Rate')
            plt.title('Waiting Time during Trial vs. Cooperative Success Rate')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            if self.save:
                plt.savefig(f"{self.prefix}waiting_time_during_trial_vs_success_rate.png")
            plt.show()
            plt.close()
        else:
            print("Insufficient data to create scatterplot.")
            
        
        # Scatterplot: Avg Waiting Time Before vs. Success Rate
        if len(maxWaitBeforeMeans) >= 2 and len(success_rates) >= 2:
            plt.figure(figsize=(8, 6))
            plt.scatter(maxWaitBeforeMeans, success_rates, alpha=0.7, color='blue', label='Experiments')
    
            # Add trendline and R²
            if len(set(maxWaitBeforeMeans)) >= 2:
                slope, intercept, r_value, _, _ = linregress(maxWaitBeforeMeans, success_rates)
                r_squared = r_value ** 2
                x_vals = np.linspace(min(maxWaitBeforeMeans), max(maxWaitBeforeMeans), 100)
                plt.plot(x_vals, slope * x_vals + intercept, color='red', linestyle='--', label='Trendline')
                plt.text(0.95, 0.05, f"$R^2$ = {r_squared:.3f}", transform=plt.gca().transAxes,
                         ha='right', va='bottom', fontsize=12, bbox=dict(facecolor='white', edgecolor='gray'))
    
            plt.xlabel('Average Waiting Time (frames)')
            plt.ylabel('Cooperative Success Rate')
            plt.title('Waiting Time Before vs. Cooperative Success Rate')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            if self.save:
                plt.savefig(f"{self.prefix}waiting_time_before_vs_success_rate.png")
            plt.show()
            plt.close()
        else:
            print("Insufficient data to create scatterplot.")
        
    
        # Pie Chart: Percentage of Time Spent Waiting
        if total_trial_frames > 0:
            waiting_percent = (total_waiting_frames / total_trial_frames) * 100
            non_waiting_percent = 100 - waiting_percent
            labels = ['Waiting in Lever Areas', 'Not Waiting']
            sizes = [waiting_percent, non_waiting_percent]
            colors = ['green', 'gray']
    
            plt.figure(figsize=(6, 6))
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 14})
            plt.title('Percentage of Trial Time Spent Waiting in Lever Areas')
            plt.axis('equal')
            plt.tight_layout()
            if self.save:
                plt.savefig(f"{self.prefix}waiting_time_pie_chart.png")
            plt.show()
            plt.close()
        else:
            print("No valid trial data for pie chart.")
        
        #Lever Entry Latency Change Over Trials
        if len(avg_latency_per_trial) >= 5:
            smooth_avg_lat = pd.Series(avg_latency_per_trial).rolling(window=5, min_periods=1, center=True).mean()
    
            # Step 4: Plot
            plt.figure(figsize=(8, 5))
            plt.plot(smooth_avg_lat, label="Avg latency (both rats)", color="blue", linewidth=2)
            plt.xlabel("Trial Index (Across Experiments)")
            plt.ylabel("Average Latency to Lever Entry (frames)")
            plt.title("Smoothed Lever Entry Latency Over Trials")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
    
            if self.save:
                plt.savefig(f"{self.prefix}smoothed_avg_latency_over_trials.png")
            plt.show()
            plt.close()
        else:
            print("Not enough valid aligned trials for latency plot.")
    
        # --- Scatterplot: Waiting symmetry vs. success ---
        if len(avg_symmetry_vals) >= 2:
            plt.figure(figsize=(8, 6))
            plt.scatter(avg_symmetry_vals, success_rates, color='purple', alpha=0.7)
            slope, intercept, r_val, _, _ = linregress(avg_symmetry_vals, success_rates)
            r2 = r_val**2
            x = np.linspace(min(avg_symmetry_vals), max(avg_symmetry_vals), 100)
            plt.plot(x, slope * x + intercept, 'r--')
            plt.xlabel("Average Waiting Symmetry (|rat0 - rat1|)")
            plt.ylabel("Cooperative Success Rate")
            plt.title("Success Rate vs. Waiting Symmetry")
            plt.text(0.95, 0.05, f"$R^2$ = {r2:.3f}", transform=plt.gca().transAxes,
                     ha='right', va='bottom', fontsize=12, bbox=dict(facecolor='white', edgecolor='gray'))
            plt.grid(True)
            plt.tight_layout()
            if self.save:
                plt.savefig(f"{self.prefix}symmetry_vs_success.png")
            plt.show()
            plt.close()
            
        # Pie Chart: % time 0 / 1 / 2 rats in lever area
        plt.figure()
        plt.pie([none_sum, one_sum, both_sum],
                labels=['Neither', 'One', 'Both'],
                colors=['gray', 'orange', 'green'],
                autopct='%1.1f%%', startangle=140)
        plt.title("Lever Zone Occupancy (None / One / Both Rats)")
        plt.axis('equal')
        plt.tight_layout()
        if self.save:
            plt.savefig(f"{self.prefix}LeverZoneOccupancyDistribution.png")
        plt.show()
    
        # Pie Chart: Both rats in lever — same vs. opposite lever area
        plt.figure()
        plt.pie([same_lever_sum, opposite_lever_sum],
                labels=['Same Lever Area', 'Opposite Lever Areas'],
                colors=['lightgreen', 'salmon'],
                autopct='%1.1f%%', startangle=140)
        plt.title("When Both Rats Wait: Same vs. Opposite Lever Area")
        plt.axis('equal')
        plt.tight_layout()
        if self.save:
            plt.savefig(f"{self.prefix}SamevsOppositeLeverArea.png")
        plt.show()
    
        # Line Graph: Occupancy across normalized trial time
        avg_occupancy = total_occupancy_curve / total_trial_counts
        smoothed = gaussian_filter1d(avg_occupancy, sigma=2)
        plt.figure()
        plt.plot(np.linspace(0, 100, NUM_BINS), smoothed, color='blue')
        plt.xlabel("Trial Time (% of trial)")
        plt.ylabel("Probability of Lever Occupancy")
        plt.title("Lever Zone Occupancy Over Trial Duration")
        plt.grid(True)
        plt.tight_layout()
        if self.save:
            plt.savefig(f"{self.prefix}LevverZoneOccupancyOverTrialDuration.png")
        plt.show()
            
    def successRateVsThresholdPlot(self):
        """
        Plots the average cooperative success rate for each threshold value.
        Applies smoothing to visualize trends.
        Weights trials equally (not each experiment). 
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
    
        # Smooth using rolling average (pandas)
        df = pd.DataFrame({'Threshold': thresholds, 'AvgSuccessRate': avg_rates})
        df.set_index('Threshold', inplace=True)
        df['Smoothed'] = df['AvgSuccessRate'].rolling(window=2, min_periods=1, center=True).mean()
    
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(df.index, df['AvgSuccessRate'], 'o-', label='Raw Average', color='gray', alpha=0.6)
        plt.plot(df.index, df['Smoothed'], 'r-', label='Smoothed', linewidth=2)
    
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
            plt.ylabel('Same Rat Both Rewards Percentage (%)')
            plt.title('Success Percentage vs. Same Rat Reward Collection Across Sessions')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlim(0, 100)
            plt.ylim(0, 100)
            plt.tight_layout()
            if self.save:
                plt.savefig(f'{self.prefix}SuccessVsSameRatScatter.png')
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


'''lev_files = ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/ExampleLevFile.csv"]

mag_files = ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/ExampleMagFile.csv"] 

pos_files = ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/ExampleTrackingCoop.h5"]

fpsList = [30, 30, 30, 30, 30, 30, 30]
totFramesList = [15000, 26000, 15000, 26000, 15000, 26000, 15000]
initialNanList = [0.15, 0.12, 0.14, 0.16, 0.3, 0.04, 0.2]
'''

'''
arr = getFiltered()
lev_files = arr[0]
mag_files = arr[1]
pos_files = arr[2]
fpsList = arr[3]
totFramesList = arr[4]
initialNanList = arr[5]
'''


'''
lev_files = ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/ExampleLevFile.csv"]

mag_files = ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/ExampleMagFile.csv"]

pos_files = ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/18_nanerror_test.h5"]

fpsList = [28]
totFramesList = [14000]
initialNanList = [0.1]
'''

'''
print("Start MultiFileGraphs Regular")
experiment = multiFileGraphs(mag_files, lev_files, pos_files, fpsList, totFramesList, initialNanList, prefix = "", save=True)
experiment.waitingStrategy()
experiment.stateTransitionModel()

#experiment.percentSameRatTakesBothRewards()
#experiment.successRateVsThresholdPlot()

#experiment.waitingStrategy()
#experiment.successVsAverageDistance()
#experiment.printSummaryStats()
#experiment.compareAverageVelocityGazevsNot()
'''

'''
experiment.rePressingbyDistance()
experiment.percentSuccesfulTrials()
experiment.interpressIntervalPlot()
experiment.quantifyRePressingBehavior()
experiment.crossingOverQuantification()
experiment.cooperativeRegionStrategiesQuantification()
experiment.makeHeatmapLocation()
experiment.intersectings_vs_percentNaN()
experiment.findTotalDistanceMoved()
'''

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




