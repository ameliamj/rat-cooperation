#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 17:57:51 2025

@author: david
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_class import singleExperiment
from typing import List
from file_extractor_class import fileExtractor

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


def getAllValid():
    fe = fileExtractor(all_valid)
    return [fe.getLevsDatapath(), fe.getMagsDatapath(), fe.getPosDatapath()]
    
def getOnlyOpaque():
    fe = fileExtractor(only_opaque)
    return [fe.getLevsDatapath(), fe.getMagsDatapath(), fe.getPosDatapath()]

def getOnlyTranslucent():
    fe = fileExtractor(only_translucent)
    return [fe.getLevsDatapath(), fe.getMagsDatapath(), fe.getPosDatapath()]

def getOnlyTransparent():
    fe = fileExtractor(only_transparent)
    return [fe.getLevsDatapath(), fe.getMagsDatapath(), fe.getPosDatapath()]

def getOnlyUnfamiliar():
    fe = fileExtractor(only_unfamiliar)
    return [fe.getLevsDatapath(), fe.getMagsDatapath(), fe.getPosDatapath()]

def getOnlyTrainingPartners():
    fe = fileExtractor(only_trainingpartners)
    return [fe.getLevsDatapath(), fe.getMagsDatapath(), fe.getPosDatapath()]

def getOnlyPairedTesting():
    fe = fileExtractor(only_PairedTesting)
    return [fe.getLevsDatapath(), fe.getMagsDatapath(), fe.getPosDatapath()]

def getOnlyTrainingCoop():
    fe = fileExtractor(only_TrainingCoop)
    return [fe.getLevsDatapath(), fe.getMagsDatapath(), fe.getPosDatapath()]        
        

# ---------------------------------------------------------------------------------------------------------


##Class to create Graphs with data from multiple files with different categories + Functions to get data from all the different categories
#
#
        
class multiFileGraphsCategories:
    def __init__(self, magFiles: List[List[str]], levFiles: List[List[str]], posFiles: List[List[str]], categoryNames: List[str]):
        self.allFileGroupExperiments = []
        self.categoryNames = categoryNames
        self.numCategories = len(magFiles)

        if not (len(magFiles) == len(levFiles) == len(posFiles) == len(categoryNames)):
            raise ValueError("Mismatch between number of categories and provided file lists or category names.")

        for c in range(self.numCategories):
            file_group = []
            for mag, lev, pos in zip(magFiles[c], levFiles[c], posFiles[c]):
                exp = singleExperiment(mag, lev, pos)
                file_group.append(exp)
            self.allFileGroupExperiments.append(file_group)
        
        self.endSaveName = ""
        for cat in categoryNames:    
            self.endSaveName += f"_{cat}"
        
        #self.path = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Graphs/"
        self.path = ""
        
        if not self.path:
            print("Warning: No save path specified. Saving plots to current directory.")
        
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
        plt.savefig(f'GazeEventsPerMinute{self.endSaveName}')
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
    
        # Iterate through each experimental group (category)
        for i, group in enumerate(self.allFileGroupExperiments):
            individual_datapoints.append([])  # Holds datapoints for this category
            totalSucc = 0
            totalTrials = 0
    
            # Iterate through each experiment in the group
            for exp in group:
                loader = exp.lev
                # Add to totals for computing the average success rate across the category
                num_succ = loader.returnNumSuccessfulTrials()
                num_total = loader.returnNumTotalTrials()
    
                totalSucc += num_succ
                totalTrials += num_total
    
                # Store individual success probability for this experiment
                if num_total > 0:
                    individual_datapoints[i].append(num_succ / num_total)
                else:
                    individual_datapoints[i].append(np.nan)
    
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
            jittered_x = [i + (np.random.rand() - 0.5) * 0.2 for _ in datapoints]  # Add slight x jitter
            if i == 0:
                plt.scatter(jittered_x, datapoints, color='black', alpha=0.7, s=40, label='Individual Data')
            else:
                plt.scatter(jittered_x, datapoints, color='black', alpha=0.7, s=40)
    
        # Formatting
        plt.xlabel('Category')
        plt.ylabel('Probability of Successful Trials')
        plt.title('Success Probability per Category')
        plt.xticks(bar_positions, self.categoryNames)
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
    
        # Save and display the plot
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
        plt.savefig(f'{self.path}{saveFileName}{self.endSaveName}')
        plt.show()
        plt.close()
    
    def printSummaryStats(self):
        avg_gaze_lengths = []       # Stores average gaze duration (in frames) per category
        avg_lever_per_trial = []    # Stores average number of lever presses per trial
        avg_mag_per_trial = []      # Stores average number of magazine entries per trial
    
        # Loop through each experimental group (i.e., category)
        for idx, group in enumerate(self.allFileGroupExperiments):
            total_gaze_events = 0     # Total gaze events (all mice) in the category
            total_frames = 0          # Total number of frames across all sessions
            total_trials = 0          # Total number of trials across sessions
            successful_trials = 0     # Total number of cooperative successful trials
            total_lever_presses = 0   # Total number of lever presses
            total_mag_events = 0      # Total number of magazine entries
            total_gaze_frames = 0     # Total frames where gaze was detected
    
            # Process each experiment within the category
            for exp in group:
                loader = exp.pos
                g0 = loader.returnIsGazing(0)
                g1 = loader.returnIsGazing(1)
    
                # Count gaze events and sum up the frames with gazing behavior
                total_gaze_events += loader.returnNumGazeEvents(0) + loader.returnNumGazeEvents(1)
                total_gaze_frames += np.sum(g0) + np.sum(g1)
                total_frames += g0.shape[0]
    
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
            avg_lever = (total_lever_presses / total_trials) if total_trials > 0 else 0
            avg_mag = (total_mag_events / total_trials) if total_trials > 0 else 0
    
            # Store for plotting
            avg_gaze_lengths.append(avg_gaze_len)
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
        

#magFiles = [["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_lever.csv"], ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv"]]
#levFiles = [["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_mag.csv"], ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv"]]
#posFiles = [["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G.predictions.h5"], ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5"]]                   

'''magFiles = [["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv"],
            ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv"]]
levFiles = [["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv"], 
            ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv"]]
posFiles = [["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5"], 
            ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5"]]'''

#categoryExperiments = multiFileGraphsCategories(levFiles, magFiles, posFiles, ["Paired_Testing", "Training_Cooperation"])



#Paired Testing vs. Training Cooperation
dataPT = getOnlyPairedTesting()
dataTC = getOnlyTrainingCoop()

levFiles = [dataPT[0], dataTC[0]]
magFiles = [dataPT[1], dataTC[1]]
posFiles = [dataPT[2], dataTC[2]]
categoryExperiments = multiFileGraphsCategories(magFiles, levFiles, posFiles, ["Paired_Testing", "Training_Cooperation"])


#Unfamiliar vs. Training Partners
'''dataUF = getOnlyUnfamiliar() #Unfamiliar
dataTP = getOnlyTrainingPartners() #Training Partners

levFiles = [dataUF[0], dataTP[0]]
magFiles = [dataUF[1], dataTP[1]]
posFiles = [dataUF[2], dataTP[2]]
categoryExperiments = multiFileGraphsCategories(magFiles, levFiles, posFiles, ["Unfamiliar", "Training Partners"])'''


#Transparent vs. Translucent vs. Opaque
'''dataTransparent = getOnlyTransparent() #Transparent
dataTranslucent = getOnlyTranslucent() #Translucent
dataOpaque = getOnlyOpaque() #Opaque

levFiles = [dataTransparent[0], dataTranslucent[0], dataOpaque[0]]
magFiles = [dataTransparent[1], dataTranslucent[1], dataOpaque[1]]
posFiles = [dataTransparent[2], dataTranslucent[2], dataOpaque[2]]
categoryExperiments = multiFileGraphsCategories(magFiles, levFiles, posFiles, ["Transparent", "Translucent", "Opaque"])'''


#categoryExperiments.compareGazeEventsCategories()
categoryExperiments.compareSuccesfulTrials()
#categoryExperiments.compareIPI()
#categoryExperiments.printSummaryStats()




# ---------------------------------------------------------------------------------------------------------


##Class to create Graphs with data sorted by mice pairs
#
#


class MicePairGraphs:
    def __init__(self, magGroups, levGroups, posGroups):
        assert len(magGroups) == len(levGroups) == len(posGroups), "Mismatched group lengths."
        self.experimentGroups = []
        for mag_list, lev_list, pos_list in zip(magGroups, levGroups, posGroups):
            group_exps = [singleExperiment(mag, lev, pos) for mag, lev, pos in zip(mag_list, lev_list, pos_list)]
            self.experimentGroups.append(group_exps)

    def _make_boxplot(self, data, ylabel, title, filename):
        plt.figure(figsize=(5, 5))
        plt.boxplot(data, showfliers=False)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks([])  # No x-axis labels since we’re not labeling each pair
        plt.tight_layout()
        plt.savefig(f"{filename}.png")
        plt.show()
        plt.close()

    def _make_histogram(self, data, xlabel, title, filename):
        plt.figure(figsize=(10, 5))
        plt.hist(data, bins=20)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f"{filename}_hist.png")
        plt.show()
        plt.close()

    def boxplot_avg_gaze_length(self):
        all_vals = []
        for group in self.experimentGroups:
            # for each pair, average the two mice's average gaze lengths
            pair_vals = []
            for exp in group:
                l0 = exp.pos.returnAverageGazeLength(0)
                l1 = exp.pos.returnAverageGazeLength(1)
                if l0 is not None and l1 is not None:
                    pair_vals.append((l0 + l1)/2)
            if pair_vals:
                all_vals.append(np.mean(pair_vals))
        
        self._make_boxplot(all_vals, "Frames per Gaze Event", "Avg Gaze Length per Pair", "Box_Gaze_Length")
        self._make_histogram(all_vals, "Frames per Gaze Event", "Gaze Length Distribution", "Hist_Gaze_Length")

    def boxplot_lever_presses_per_trial(self):
        vals = []
        for group in self.experimentGroups:
            pair_rates = []
            for exp in group:
                trials = exp.lev.returnNumTotalTrials()
                presses = exp.lev.returnTotalLeverPresses()
                if trials > 0:
                    pair_rates.append(presses / trials)
            if pair_rates:
                vals.append(np.mean(pair_rates))
        
        self._make_boxplot(vals, "Presses / Trial", "Lever Presses per Trial", "Box_LeverPerTrial")
        self._make_histogram(vals, "Presses / Trial", "Lever Press Distribution", "Hist_LeverPerTrial")

    def boxplot_mag_events_per_trial(self):
        vals = []
        for group in self.experimentGroups:
            pair_rates = []
            for exp in group:
                trials = exp.lev.returnNumTotalTrials()
                mags = exp.mag.getTotalMagEvents()
                if trials > 0:
                    pair_rates.append(mags / trials)
            if pair_rates:
                vals.append(np.mean(pair_rates))
        
        self._make_boxplot(vals, "Mag Events / Trial", "Mag Events per Trial", "Box_MagPerTrial")
        self._make_histogram(vals, "Mag Events / Trial", "Mag Event Distribution", "Hist_MagPerTrial")

    def boxplot_avg_IPI(self):
        vals = []
        # For each pair, compute weighted average IPI across experiments
        for group in self.experimentGroups:
            sum_weighted_ipi = 0.0
            sum_presses = 0
            for exp in group:
                mean_ipi = exp.lev.returnAvgIPI()
                n_presses = exp.lev.returnTotalLeverPresses()
                if mean_ipi and n_presses > 0:
                    sum_weighted_ipi += mean_ipi * n_presses
                    sum_presses += n_presses
            if sum_presses > 0:
                vals.append(sum_weighted_ipi / sum_presses)
                
        self._make_boxplot(vals, "IPI (s)", "Avg Inter-Press Interval", "Box_IPI")
        self._make_histogram(vals, "IPI (s)", "IPI Distribution", "Hist_IPI")

    def boxplot_IPI_first_to_success(self):
        vals = []
        
        # Weighted by number of successful trials
        for group in self.experimentGroups:
            sum_weighted = 0.0
            sum_success = 0
            for exp in group:
                v = exp.lev.returnAvgIPI_FirsttoSuccess()
                n_succ = exp.lev.returnNumSuccessfulTrials()
                if v is not None and n_succ > 0:
                    sum_weighted += v * n_succ
                    sum_success += n_succ
            if sum_success > 0:
                vals.append(sum_weighted / sum_success)
                
        self._make_boxplot(vals, "Time (s)", "IPI: First→Success", "Box_IPI_First")
        self._make_histogram(vals, "Time (s)", "First→Success Distribution", "Hist_IPI_First")

    def boxplot_IPI_last_to_success(self):
        vals = []
        # Weighted by number of successful trials
        for group in self.experimentGroups:
            sum_weighted = 0.0
            sum_success = 0
            for exp in group:
                v = exp.lev.returnAvgIPI_LasttoSuccess()
                n_succ = exp.lev.returnNumSuccessfulTrials()
                if v is not None and n_succ > 0:
                    sum_weighted += v * n_succ
                    sum_success += n_succ
            if sum_success > 0:
                vals.append(sum_weighted / sum_success)
                
        self._make_boxplot(vals, "Time (s)", "IPI: Last→Success", "Box_IPI_Last")
        self._make_histogram(vals, "Time (s)", "Last→Success Distribution", "Hist_IPI_Last")
        
    def boxplot_gaze_events_per_minute(self): 
        #A minute is defined as 1800 Frames
        FRAME_WINDOW = 1800
        
        vals = []
        for group in self.experimentGroups:
            sumEvents = 0
            sumFrames = 0
            
            for exp in group:
                countEvents0 = exp.pos.returnNumGazeEvents(0)
                countEvents1 = exp.pos.returnNumGazeEvents(1)
                numFrames = exp.pos.returnNumFrames()
                
                if countEvents0 is not None and countEvents1 is not None and numFrames is not None:
                    sumEvents += countEvents0 + countEvents1
                    sumFrames += numFrames
            if sumFrames > 0:
                vals.append(sumEvents / numFrames * FRAME_WINDOW)
        
        self._make_boxplot(vals, "Gaze Events / Min", "Gaze Rate per Pair", "Box_GazePerMin")
        self._make_histogram(vals, "Gaze Events / Min", "Gaze Rate Distribution", "Hist_GazePerMin")
     
    def boxplot_percent_successful_trials(self):
        vals = []
        
        # Weighted by number of successful trials
        for group in self.experimentGroups:
            sum_tot = 0
            sum_success = 0
            for exp in group:
                tot = exp.lev.returnNumTotalTrials()
                n_succ = exp.lev.returnNumSuccessfulTrials()
                if n_succ is not None and tot > 0:
                    sum_tot += tot
                    sum_success += n_succ
            if sum_success > 0:
                vals.append(sum_success / sum_tot)
                
        self._make_boxplot(vals, "% Success", "Success Rate per Pair", "Box_Success")
        self._make_histogram(vals, "% Success", "Success Rate Distribution", "Hist_Success")
        
    def difference_last_vs_first(self):
    
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
    
        for group in self.experimentGroups:
            if len(group) < 5:
                continue
            first, last = group[0], group[-1]
            for name, func in metrics.items():
                try:
                    v1, v2 = func(first), func(last)
                    if v1 is not None and v2 is not None:
                        diffs[name].append(v2 - v1)
                except:
                    continue
                
        #Plot individual histograms
        for name, values in diffs.items():
            plt.figure(figsize=(10, 4))
            plt.hist(values, bins=15)
            plt.title(f"Change in {name} (Last Session - 1st Session)")
            plt.xlabel(f"Δ {name}")
            plt.tight_layout()
            plt.savefig(f"Diff_{name.replace(' ', '_')}.png")
            plt.show()
            plt.close()    
        
        # Plot individual bar graphs
        for name, values in diffs.items():
            if not values:
                continue
            avg_diff = np.mean(values)
            error = np.std(values) / np.sqrt(len(values))
        
            plt.figure(figsize=(5, 6))
            
            # Plot the bar
            bar = plt.bar([0], [avg_diff], width=0.3, yerr=[error], capsize=10, 
                          color='lightgreen', edgecolor='black')
            
            # Draw horizontal line at 0
            plt.axhline(0, color='gray', linestyle='--', linewidth=1)
            
            # Set x-ticks to the metric name
            plt.xticks([0], [name], fontsize=10)
            
            # Add vertical padding to y-limits
            ymin = min(0, avg_diff - error)
            ymax = max(0, avg_diff + error)
            yrange = ymax - ymin
            plt.ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
            
            plt.xlim(-0.4, 0.4)
            
            # Clean labels and layout
            plt.ylabel(f"Δ {name} (Last - First)", fontsize=12)
            plt.title(f"Avg Change in {name}", fontsize=14)
            plt.tight_layout()
        
            filename = f"Bar_Change_{name.replace(' ', '_').replace('→', 'to')}.png"
            plt.savefig(filename)
            plt.show()
            plt.close()
        
        #Plot Individual Line Graphs
        
        # Track values across sessions per metric
        max_sessions = max(len(group) for group in self.experimentGroups)
        metric_over_sessions = {name: [[] for _ in range(max_sessions)] for name in metrics}
        
        # Fill values by session
        for group in self.experimentGroups:
            for i, exp in enumerate(group):
                for name, func in metrics.items():
                    try:
                        val = func(exp)
                        if val is not None:
                            metric_over_sessions[name][i].append(val)
                    except Exception as e:
                        print(f"Error computing {name} for session {i} in group: {e}")
                        continue  # Optional: track or log failures more thoroughly
        
        # Average and plot
        for name, session_lists in metric_over_sessions.items():
            averages = [np.mean(vals) if vals else None for vals in session_lists]
            counts = [len(vals) for vals in session_lists]
        
            # Only plot sessions with valid averages
            session_indices = [i+1 for i, v in enumerate(averages) if v is not None]
            y_values = [v for v in averages if v is not None]
            y_counts = [counts[i] for i, v in enumerate(averages) if v is not None]
        
            if not y_values:
                continue
        
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
        
            filename = f"Line_Progression_{name.replace(' ', '_').replace('→', 'to')}.png"
            plt.savefig(filename)
            plt.show()
            plt.close()
    

groupMicePairs = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/group_mice_pairs.csv"

'''def getGroupMicePairs():
    fe = fileExtractor(groupMicePairs)
    return [fe.getLevsDatapath(grouped = True), fe.getMagsDatapath(grouped = True), fe.getPosDatapath(grouped = True)]


data = getGroupMicePairs()

pairGraphs = MicePairGraphs(data[0], data[1], data[2])'''

'''
magFiles = [["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv"],
            ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv"]]
levFiles = [["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv"], 
            ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv"]]
posFiles = [["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5"], 
            ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5", "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5"]]

pairGraphs = MicePairGraphs(levFiles, magFiles, posFiles)'''


'''pairGraphs.boxplot_avg_gaze_length()
pairGraphs.boxplot_lever_presses_per_trial()
pairGraphs.boxplot_mag_events_per_trial()
pairGraphs.boxplot_percent_successful_trials()
pairGraphs.boxplot_gaze_events_per_minute()
pairGraphs.boxplot_avg_IPI()
pairGraphs.boxplot_IPI_first_to_success()
pairGraphs.boxplot_IPI_last_to_success()
pairGraphs.difference_last_vs_first()'''



# ---------------------------------------------------------------------------------------------------------


#Class to create Graphs with data from a single file
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


# ---------------------------------------------------------------------------------------------------------




#Class to create Graphs with data from multiple files + Functions to get data from all the different categories
#
#


class multiFileGraphs:
    def __init__(self, magFiles: List[str], levFiles: List[str], posFiles: List[str]):
        self.experiments = []
        
        if (len(magFiles) != len(levFiles) or len(magFiles) != len(posFiles)):
            raise ValueError("Different number of mag, lev, and pos files")
        
        for i in range(len(magFiles)):
            exp = singleExperiment(magFiles[i], levFiles[i], posFiles[i])
            self.experiments.append(exp)
    
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
        # concat all lev data
        all_lev = pd.concat([exp.lev.data for exp in self.experiments], ignore_index=True)
        if all_lev is None or all_lev.empty:
            raise ValueError("No lever data loaded.")
        
        all_lev = all_lev.sort_values(by=["RatID","AbsTime"])
        all_lev["IPI"] = all_lev.groupby("RatID")["AbsTime"].diff()
        ipi_data = all_lev[all_lev["IPI"].notna()]
        
        # Histogram
        plt.figure(figsize=(10,6))
        plt.hist(ipi_data["IPI"], bins=50, edgecolor='black')
        plt.xlabel("Inter-Press Interval (s)")
        plt.ylabel("Frequency")
        plt.title("Histogram of Inter-Press Intervals (All Rats)")
        plt.tight_layout()
        plt.show()
        plt.savefig('IPI_histogram.png')
        plt.close()
        
        # Boxplot
        plt.figure(figsize=(10,6))
        all_lev.boxplot(column="IPI", by="RatID")
        plt.suptitle("") 
        plt.xlabel("RatID")
        plt.ylabel("IPI (s)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        plt.savefig('IPI_boxplot.png')
        plt.close()
        
        # Time series
        plt.figure(figsize=(12,6))
        for rat_id, grp in all_lev.groupby("RatID"):
            plt.plot(grp["AbsTime"], grp["IPI"], label=str(rat_id), alpha=0.7)
        plt.title("IPI Over Time by Rat (All Files)")
        plt.xlabel("AbsTime (s)")
        plt.ylabel("IPI (s)")
        plt.legend(title="RatID", bbox_to_anchor=(1.05,1), loc='upper left')
        plt.tight_layout()
        plt.show()
        plt.savefig('IPI_timeseries.png')
        plt.close()
      
    def interpressIntervalSuccessPlot(self):
        all_lev = pd.concat([exp.lev.data for exp in self.experiments], ignore_index=True)
        if all_lev is None or all_lev.empty:
            raise ValueError("No lever data loaded.")
        
        # filter successes
        succ = all_lev[all_lev["coopSucc"] == 1]
        grouped = succ.groupby("TrialNum")
        
        ipi_first, ipi_last, first_rats, success_counts = [], [], [], {}
        for _, trial in grouped:
            trial = trial.sort_values("AbsTime")
            if 1 not in trial["Hit"].values: continue
            coop_end = trial[trial["TrialEnd"]==1]
            if coop_end.empty: continue
            t_coop = coop_end.iloc[0]["AbsTime"]
            first = trial[trial["Hit"]==1].iloc[0]
            before = trial[trial["AbsTime"] < t_coop]
            if before.empty: continue
            last = before.iloc[-1]
            
            ipi_first.append(t_coop - first["AbsTime"])
            ipi_last.append(t_coop - last["AbsTime"])
            first_rats.append(first["RatID"])
            rid = coop_end.iloc[0]["RatID"]
            success_counts[rid] = success_counts.get(rid,0) + 1
        
        # Plot 1
        plt.figure(figsize=(8,6))
        plt.hist(ipi_first, bins=20)
        plt.title("First Press → Success IPI")
        plt.xlabel("IPI (s)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()
        plt.savefig('IPI_firstpress_to_success.png')
        plt.close()
        
        # Plot 2
        plt.figure(figsize=(8,6))
        plt.hist(ipi_last, bins=20)
        plt.title("Last Press → Success IPI")
        plt.xlabel("IPI (s)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()
        plt.savefig('IPI_lastpress_to_success.png')
        plt.close()
        
        # Plot 3
        plt.figure(figsize=(10,5))
        plt.plot(ipi_first, marker='o')
        plt.title("Time Series: First Press to Success")
        plt.xlabel("Trial Index")
        plt.ylabel("IPI (s)")
        plt.tight_layout()
        plt.show()
        plt.savefig('Timeseries_IPI_firstpress_to_success.png')
        plt.close()
        
        # Plot 4
        plt.figure(figsize=(10,5))
        plt.plot(ipi_last, marker='o')
        plt.title("Time Series: Last Press to Success")
        plt.xlabel("Trial Index")
        plt.ylabel("IPI (s)")
        plt.tight_layout()
        plt.show()
        plt.savefig('Timeseries_IPI_lastpress_to_success.png')
        plt.close()
        
        # Plot 5
        counts = pd.Series(first_rats).value_counts()
        plt.figure(figsize=(6,6))
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
        plt.title("Who Presses First")
        plt.tight_layout()
        plt.show()
        plt.savefig('Who_Presses_First.png')
        plt.close()
    
    
    def percentSuccesfulTrials(self):
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
        plt.savefig('Percent Successful.png')
        plt.close()

#Testing Multi File Graphs
#
#

#arr = getAllValid()
#lev_files = arr[0]
#mag_files = arr[1]
#pos_files = arr[2]

#mag_files = ["/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral Quantification/Example Data Files/magData.csv", "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral Quantification/Example Data Files/ExampleMagFile.csv"]
#lev_files = ["/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral Quantification/Example Data Files/leverData.csv", "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral Quantification/Example Data Files/ExampleLevFile.csv"]

#experiment = multiFileGraphs(mag_files, lev_files, pos_files)
#experiment.magFileDataAvailabilityGraph()
#experiment.levFileDataAvailabilityGraph()
#experiment.percentSuccesfulTrials()

#experiment.interpressIntervalPlot()
#experiment.interpressIntervalSuccessPlot()




