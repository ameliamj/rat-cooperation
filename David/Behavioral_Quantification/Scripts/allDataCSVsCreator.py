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
        fe.deleteInvalid()
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
    
    def returnMagStartAbsTimes(self, lev, mag):
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
                if pd.isna(start_time) or pd.isna(end_times[trial_idx]):
                    continue
                start_frame = int(start_time * fps)
                end_frame = int(end_times[trial_idx] * fps)
                succ = successes_filtered[trial_idx]
                
                levers = ['lev_top', 'lev_bottom']
                
                numFrames = end_frame - start_frame
                
                rat1_xlocations = pos.data[0, 0, pos.HB_INDEX, start_frame:end_frame]
                rat2_xlocations = pos.data[1, 0, pos.HB_INDEX, start_frame:end_frame]
                
                avgDifference = sum(abs(a - b) for a, b in zip(rat1_xlocations, rat2_xlocations)) / numFrames
                distanceMoved = np.sum(np.abs(np.diff(rat1_xlocations))) + np.sum(np.abs(np.diff(rat2_xlocations)))
                avgDistanceMoved = distanceMoved / numFrames
                isSynchronized = avgDifference < MAX_SYNCHRONIZED and avgDistanceMoved > MIN_AVG_MOVED
                
                print("rat0_locations[start_frame]: ", rat0_locations[start_frame])
                print("rat1_locations[start_frame]: ", rat1_locations[start_frame])
                
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
                        synchronized_counts_1rat += 1
                        if (succ):
                            synchronized_successes_1rat += 1
                            
                else:
                    counts_0rats += 1
                    if (succ):
                        successes_0rats += 1
                    
                    if (isSynchronized):
                        synchronized_counts_0rats += 1
                        if (succ):
                            synchronized_successes_0rats += 1
                
            
            successRate_0rats = np.divide(successes_0rats, counts_0rats, where=counts_0rats != 0)
            successRate_1rats = np.divide(successes_1rat, counts_1rat, where=counts_1rat != 0)
            successRate_2rats = np.divide(successes_2rats, counts_2rats, where=counts_2rats != 0)
            
            synchronized_successRate_0rats = np.divide(synchronized_successes_0rats, synchronized_counts_0rats, where=synchronized_counts_0rats != 0)
            synchronized_successRate_1rats = np.divide(synchronized_successes_1rat, synchronized_counts_1rat, where=synchronized_counts_1rat != 0)
            synchronized_successRate_2rats = np.divide(synchronized_successes_2rats, synchronized_counts_2rats, where=synchronized_counts_2rats != 0)
            
            # Append session data
            session_data.append({
                'session_id': session_id,
                'lev_file': lev_file,
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
        """
        Creates a CSV file containing per-trial metrics for all experiments.
        Metrics include session info, trial timing, success metrics, gaze, interaction, and spatial metrics.
        Saves the output to 'trial_metrics.csv'.
        """
        trial_data = []
    
        for idx, exp in enumerate(self.experiments):
            session_id = f"exp_{idx:03d}"
            pos = exp.pos
            mag = exp.mag
            lev = exp.lev
            fps = exp.fps
            total_frames_session = pos.returnNumFrames()
    
            # Extract session-level data
            rat_pair = exp.ratPair
            familiarity = exp.familiarity  
            barrier_transparency = exp.transparency  
    
            # Get trial data
            trial_starts = lev.returnTimeStartTrials()
            trial_ends = lev.returnTimeEndTrials()
            successes_unfiltered = lev.returnSuccessTrials()
            successes = lev.returnSuccessTrialsFiltered()
            first_presses = lev.returnFirstPressAbsTimes()  
            first_mag_entries = self.returnMagStartAbsTimes(lev, mag)  
            list_trials_no_press = lev.returnListTrialsNoPress()
            trialCounter = 1
    
            # Compute successes in a row
            successes_in_row = 0
            success_counts = []
    
            for succ in successes_unfiltered:
                if succ == 1:
                    successes_in_row += 1
                else:
                    successes_in_row = 0
                    
                if (succ != -1):
                    success_counts.append(successes_in_row)
    
            # Process each trial
            for trial_idx, start_time in enumerate(trial_starts):
                if pd.isna(start_time) or pd.isna(trial_ends[trial_idx]):
                    continue
    
                start_frame = int(start_time * fps)
                end_frame = int(trial_ends[trial_idx] * fps)
                total_frames = end_frame - start_frame if end_frame > start_frame else 0
    
                # Trial metrics
                trial_number = trial_idx + trialCounter
                if (trial_number in list_trials_no_press):
                    print("Trial Number No Press: ", trial_number)
                    trial_data.append({
                        'session_id': session_id,
                        'rat_pair': rat_pair,
                        'familiarity': familiarity,
                        'barrier_transparency': barrier_transparency,
                        'trial_number': trial_number,
                        'time_begin': None,
                        'time_first_press': None,
                        'time_first_mag_entry': None,
                        'successes_in_row': None,
                        'success': 0,
                        'total_frames': total_frames,
                        'lever_press_exists': False,
                        'percent_gazing': None,
                        'percent_interacting': None,
                        'time_wait_before_cue': None,
                        'time_wait_to_press_one': None,
                        'dist_furthest_from_lever': None,
                        'avg_horizontal_distance': None,
                        'avg_distance': None
                    })
                    trial_number += 1
                    trialCounter += 1
                
                time_begin = start_time
                time_first_press = first_presses[trial_idx] if trial_idx < len(first_presses) and pd.notna(first_presses[trial_idx]) else None
                time_first_mag_entry = first_mag_entries[trial_idx] if trial_idx < len(first_mag_entries) and pd.notna(first_mag_entries[trial_idx]) else None
                successes_in_row = success_counts[trial_idx]
                success = successes[trial_idx]
                lever_press_exists = True
    
                # Gazing percentage
                gaze_frames_rat0 = np.sum(pos.returnIsGazing(0)[start_frame:end_frame])
                gaze_frames_rat1 = np.sum(pos.returnIsGazing(1)[start_frame:end_frame])
                percent_gazing = ((gaze_frames_rat0 + gaze_frames_rat1) / (2 * total_frames)) * 100 if total_frames > 0 else 0
    
                # Interaction percentage
                interaction_frames = np.sum(pos.returnIsInteracting()[start_frame:end_frame])
                percent_interacting = (interaction_frames / total_frames) * 100 if total_frames > 0 else 0
    
                # Time wait before cue
                rat0_locations = pos.returnMouseLocation(0)
                rat1_locations = pos.returnMouseLocation(1)
                t = start_frame - 1
                rat0_waiting = 0
                rat1_waiting = 0
                rat0_active = True
                rat1_active = True
                while t >= 0 and t < len(rat0_locations) and t < len(rat1_locations) and (rat0_active or rat1_active):
                    if rat0_locations[t] in ['lev_top', 'lev_bottom'] and rat0_active:
                        rat0_waiting += 1
                    else:
                        rat0_active = False
                    if rat1_locations[t] in ['lev_top', 'lev_bottom'] and rat1_active:
                        rat1_waiting += 1
                    else:
                        rat1_active = False
                    t -= 1
                time_wait_before_cue = min(rat0_waiting, rat1_waiting) / fps if fps > 0 else 0
                
                if (max(rat0_waiting, rat1_waiting) > 0):
                    distance_vs_wait_valid = True
                else:
                    distance_vs_wait_valid = False
    
                # Time waited to press lever if one rat at lever initially
                time_wait_to_press_one = None
                if rat0_locations[start_frame] in ['lev_top', 'lev_bottom'] or rat1_locations[start_frame] in ['lev_top', 'lev_bottom']:
                    if time_first_press is not None:
                        time_wait_to_press_one = time_first_press - start_time if time_first_press >= start_time else 0
    
                # Distance of furthest rat from lever
                dist_rat0 = pos.distanceFromLever(0, start_frame)  # Assumed method
                dist_rat1 = pos.distanceFromLever(1, start_frame)
                dist_furthest_from_lever = max(dist_rat0, dist_rat1) if dist_rat0 is not None and dist_rat1 is not None else 0
    
                # Average horizontal distance
                rat1_xlocations = pos.data[0, 0, pos.HB_INDEX, start_frame:end_frame]
                rat2_xlocations = pos.data[1, 0, pos.HB_INDEX, start_frame:end_frame]
                if len(rat1_xlocations) > 0 and len(rat2_xlocations) > 0:
                    avg_horizontal_distance = np.mean([abs(a - b) for a, b in zip(rat1_xlocations, rat2_xlocations)])
                else:
                    avg_horizontal_distance = 0
    
                # Average distance
                distances = pos.returnInterMouseDistance()[start_frame:end_frame]
                avg_distance = np.nanmean(distances) if len(distances) > 0 else 0
    
                trial_data.append({
                    'session_id': session_id,
                    'rat_pair': rat_pair,
                    'familiarity': familiarity,
                    'barrier_transparency': barrier_transparency,
                    'trial_number': trial_number,
                    'time_begin': time_begin,
                    'time_first_press': time_first_press,
                    'time_first_mag_entry': time_first_mag_entry,
                    'successes_in_row': successes_in_row,
                    'success': success,
                    'trial_frames': total_frames,
                    'lever_press_exists': lever_press_exists,
                    'percent_gazing': percent_gazing,
                    'percent_interacting': percent_interacting,
                    'wait_before_cue_both': time_wait_before_cue,
                    'distance_vs_wait_valid': distance_vs_wait_valid,
                    'time_wait_to_press_one': time_wait_to_press_one,
                    'dist_furthest_from_lever': dist_furthest_from_lever,
                    'avg_horizontal_distance': avg_horizontal_distance,
                    'avg_distance': avg_distance
                })
    
        df = pd.DataFrame(trial_data)
        df.to_csv('trial_metrics.csv', index=False)
        
    def createFrameCSV(self):
        a = 1
        
    
metadata_path = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/dyed_preds_fixed_expanded.csv"

creator = allDataCSVsCreator(metadata_path)
#creator.createSessionCSV()
creator.createTrialCSV()
