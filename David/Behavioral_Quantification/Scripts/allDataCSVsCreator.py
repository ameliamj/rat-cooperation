#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 22:41:09 2025

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

'''
Definitions:
    Percent Gazing is defined as (# Gaze Frames Rat 0 + # Gaze Frames Rat 1) / (2 * Total Frames)
'''


class allDataCSVsCreator:
    def __init__(self, metadata_path):
        self.metadata_path = metadata_path
        self.experiments = self._getExps()
    
    def _getExps(self):
        fe = fileExtractor(self.metadata_path)
        fe.data = fe.deleteBadNaN()
        fpsList, totFramesList = fe.returnFPSandTotFrames()
        initialNanList = fe.returnNaNPercentage()
        
        levFiles = fe.getLevsDatapath()
        magFiles = fe.getMagsDatapath()
        posFiles = fe.getPosDatapath()
        familiarity = fe.getFamiliarityList()
        transparency = fe.getBarrierTransparencyList()
        ratPairs = fe.getRatPairList()
        numSessionsBeforeList = fe.getNumSessionsBefore()
        
        deleted_count = 0
        tempExps = []
        
        print("There are ", len(magFiles), " experiments in this data session. ")
        print("")
        
        if (len(magFiles) != len(levFiles) or len(magFiles) != len(posFiles) or len(magFiles) != len(transparency) or len(magFiles) != len(familiarity) or len(magFiles) != len(ratPairs) or len(magFiles) != len(numSessionsBeforeList)):
            raise ValueError("Different number of mag, lev, and pos files")
            
        if ((len(magFiles) != len(fpsList)) or (len(magFiles) != len(totFramesList)) or len(magFiles) != len(initialNanList)):
            print("lenDataFiles: ", len(magFiles))
            print("len(fpsList)", len(fpsList))
            print("len(totFramesList)", len(totFramesList))
            print("len(initialNanList)", len(initialNanList))
            raise ValueError("Different number of fpsList, totFramesList, or initialNanList values")
        
        
        for i in range(len(magFiles)):
            exp = singleExperiment(magFiles[i], levFiles[i], posFiles[i], fpsList[i], totFramesList[i], initialNanList[i], trainingPartner=familiarity[i], transparency=transparency[i], ratPair=ratPairs[i], numSessionsBefore=numSessionsBeforeList[i])
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
            
            tempExps.append(exp)
        
        print(f"Deleted {deleted_count} experiment(s) due to missing categories.")
        
        return tempExps
    
    def createSessionCSV(self):
        """
        Creates a CSV file containing per-session metrics for all experiments.
        Metrics include session info, trial statistics, gaze behavior, interaction, and spatial metrics.
        Saves the output to 'session_metrics.csv'.
        """
        # Initialize a list to store session data
        session_data = []
    
        # Iterate through each experiment
        for idx, exp in enumerate(self.experiments):
            pos = exp.pos
            mag = exp.mag
            lev = exp.lev
            
            # Generate a unique session ID
            session_id = f"exp_{idx:03d}"
    
            # Extract file paths and session data
            lev_file = exp.lev_file
            mag_file = exp.mag_file
            rat_pair = exp.ratPair
            cohort = lev.returnAnimalID()
            familiarity = exp.familiarity
            barrier_transparency = exp.transparency
            times_seen = exp.numSessionsBefore
            success_threshold = lev.returnSuccThreshold()
            fps = exp.fps
    
            # Total number of trials
            total_trials = exp.lev.returnNumTotalTrials()
    
            # Total number of successful trials
            successful_trials = exp.lev.returnNumSuccessfulTrials()
    
            # Success percentage
            success_percentage = exp.lev.returnSuccessPercentage() * 100
    
            # Success percentage in first quarter
            success_percentage_first_quarter = (exp.lev.numSuccFirstQuarter() / exp.lev.numTotalFirstQuarter() * 100) if exp.lev.numTotalFirstQuarter() > 0 else 0
    
            # Gazing percentage (average for both rats, standard definition)
            gaze_frames_rat0 = pos.returnTotalFramesGazing(0)
            gaze_frames_rat1 = pos.returnTotalFramesGazing(1)
            total_frames = exp.pos.returnNumFrames()
            gazing_percentage = ((gaze_frames_rat0 + gaze_frames_rat1) / (2 * total_frames)) * 100 if total_frames > 0 else 0
    
            # Total gaze frames (sum for both rats)
            gaze_frames = (gaze_frames_rat0 + gaze_frames_rat1) / 2
    
            # Average gaze length (average for both rats)
            gaze_events_rat0 = pos.returnNumGazeEvents(0)
            gaze_events_rat1 = pos.returnNumGazeEvents(1)
            total_gaze_events = gaze_events_rat0 + gaze_events_rat1
            average_gaze_length = gaze_frames / total_gaze_events if total_gaze_events > 0 else 0
    
            # Interaction percentage
            interaction_frames = pos.returnTotalFramesInteracting()
            interaction_percentage = (interaction_frames / total_frames) * 100 if total_frames > 0 else 0
    
            # Average wait before cue (both rats or one rat at lever at trial start)
            trial_starts = lev.returnTimeStartTrials()
            end_times = lev.returnTimeEndTrials()
            rat0_locations = pos.returnMouseLocation(0)
            rat1_locations = pos.returnMouseLocation(1)
            
            wait_times_both = []
            wait_times_one = []
            for trial_idx, start_time in enumerate(trial_starts):
                if pd.isna(start_time):
                    continue
                start_frame = int(start_time * fps)
                
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

                min_wait = min(rat0_waiting, rat1_waiting)
                max_wait = max(rat0_waiting, rat1_waiting)
                        
                        
                wait_times_both.append(max_wait / fps)
                wait_times_one.append(min_wait / fps)
                
            avg_wait_before_cue_both = np.mean(wait_times_both)
            avg_wait_before_cue_one = np.mean(wait_times_one)
            
            #avg_wait_before_cue_both = sum(wait_times_both) / len(wait_times_both) if wait_times_both else 0
            #avg_wait_before_cue_one = sum(wait_times_one) / len(wait_times_one) if wait_times_one else 0
    
            # Average distance & X-distance between rats
            distances = exp.pos.returnInterMouseDistance()
            avg_distance = np.nanmean(distances) if len(distances) > 0 else 0
            
            rat1_xlocations = pos.data[0, 0, pos.HB_INDEX]
            rat2_xlocations = pos.data[1, 0, pos.HB_INDEX]
            
            difference = sum(abs(a - b) for a, b in zip(rat1_xlocations, rat2_xlocations))  
            avg_x_distance = difference / total_frames
            
            #Average Success Rate, depending on Rats at Lever
            MIN_AVG_MOVED = 10
            MAX_SYNCHRONIZED = 325
            rat0_locations = pos.returnMouseLocation(0)
            rat1_locations = pos.returnMouseLocation(1)
            
            successes_0rats = 0
            successes_1rat = 0
            successes_2rats = 0
            counts_0rats = 0
            counts_1rat = 0
            counts_2rats = 0
            
            synchronized_successes_0rats = 0
            synchronized_successes_1rat = 0
            synchronized_successes_2rats = 0
            synchronized_counts_0rats = 0
            synchronized_counts_1rat = 0
            synchronized_counts_2rats = 0
            
            successes_filtered = lev.returnSuccessTrialsFiltered()
            
            if (len(successes_filtered) != len(trial_starts)):
                print("NOT EQUAL")
                print("len(trial_starts): ", len(trial_starts))
                print("len(successes_filtered): ", len(successes_filtered))
                
                continue
            
            for trial_idx, start_time in enumerate(trial_starts):
                if pd.isna(start_time) and pd.isna(end_times[trial_idx]):
                    continue
                start_frame = int(start_time * fps)
                end_frame = int(end_times[trial_idx] * fps)
                succ = successes_filtered[trial_idx]
                
                levers = [['lev_top', 'lev_bottom']]
                
                numFrames = end_frame - start_frame
                
                rat1_xlocations = pos.data[0, 0, pos.HB_INDEX, start_frame:end_frame]
                rat2_xlocations = pos.data[1, 0, pos.HB_INDEX, start_frame:end_frame]
                
                avgDifference = sum(abs(a - b) for a, b in zip(rat1_xlocations, rat2_xlocations)) / numFrames
                distanceMoved = np.sum(np.abs(np.diff(rat1_xlocations))) + np.sum(np.abs(np.diff(rat2_xlocations)))
                avgDistanceMoved = distanceMoved / numFrames
                isSynchronized = avgDifference < MAX_SYNCHRONIZED and avgDistanceMoved > MIN_AVG_MOVED
                
                if (rat0_locations[start_frame] in levers and rat1_locations[start_frame] in levers):
                    counts_2rats += 1
                    if (succ):
                        successes_2rats += 1
                        
                    if (isSynchronized):
                        synchronized_counts_2rats += 1
                        if (succ):
                            synchronized_successes_2rats += 1
                            
                elif(rat0_locations[start_frame] in levers or rat1_locations[start_frame] in levers):
                    counts_1rat += 1
                    if (succ):
                        successes_1rat += 1
                    
                    if (isSynchronized):
                        synchronized_counts_2rats += 1
                        if (succ):
                            synchronized_successes_2rats += 1
                            
                else:
                    counts_1rat += 1
                    if (succ):
                        successes_0rats += 1
                    
                    if (isSynchronized):
                        synchronized_counts_2rats += 1
                        if (succ):
                            synchronized_successes_2rats += 1
                
            
            successRate_0rats = successes_0rats / counts_0rats
            successRate_1rats = successes_1rat / counts_1rat
            successRate_2rats = successes_2rats / counts_2rats
            
            synchronized_successRate_0rats = synchronized_successes_0rats / synchronized_counts_0rats
            synchronized_successRate_1rats = synchronized_successes_1rat / synchronized_counts_1rat
            synchronized_successRate_2rats = synchronized_successes_2rats / synchronized_counts_2rats
            
            # Append session data
            session_data.append({
                'session_id': session_id,
                'lev_file': lev_file,
                'mag_file': mag_file,
                'rat_pair': rat_pair,
                'cohort': cohort,
                'familiarity': familiarity,
                'barrier_transparency': barrier_transparency,
                'times_seen': times_seen,
                'success_threshold': success_threshold,
                'fps': fps,
                'total_trials': total_trials,
                'successful_trials': successful_trials,
                'success_percentage': success_percentage,
                'success_percentage_first_quarter': success_percentage_first_quarter,
                'gazing_percentage': gazing_percentage,
                'gaze_frames': gaze_frames,
                'average_gaze_length': average_gaze_length,
                'interaction_percentage': interaction_percentage,
                'total_frames': total_frames,
                'avg_wait_before_cue_both': avg_wait_before_cue_both,
                'avg_wait_before_cue_one': avg_wait_before_cue_one,
                'avg_distance': avg_distance,
                'avg_x_distance': avg_x_distance,
                'successRate_0rats': successRate_0rats,
                'successRate_1rat': successRate_1rats,
                'successRate_2rats': successRate_2rats,
                'synchronized_successRate_0rats': synchronized_successRate_0rats,
                'synchronized_successRate_1rat': synchronized_successRate_1rats,
                'synchronized_successRate_2rats': synchronized_successRate_2rats
                
            })
    
        # Create DataFrame and save to CSV
        df = pd.DataFrame(session_data)
        df.to_csv('session_metrics.csv', index=False)
        
    def createTrialCSV(self):
        '''
        Metrics: 
            Session ID
            Rat Pair 
            Familiarity
            Barrier Transparency 
            Trial #
            Time Begin
            Time First Press (NaN if none)
            Time first Mag Entry (NaN if none)
            Number of Successes in a Row
            Success vs. Fail
            Total Frames in Trial
            Whether a Lever Press Exists for this Trial 
            Percent Gazing
            Percent Interacting
            Time Wait Before Cue
            Time waited to press the Lever if one of the rats is at lever initially (if not NaN)
            Distance of Furthest Rat from Lever
            Average Horizontal Distance
            Average Distance
        '''
        a = 1
        
    def createFrameCSV(self):
        a = 1
        
    
metadata_path = "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/Filtered.csv"

creator = allDataCSVsCreator(metadata_path)
creator.createSessionCSV()
