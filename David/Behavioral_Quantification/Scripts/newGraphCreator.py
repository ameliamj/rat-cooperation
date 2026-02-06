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
from collections import Counter


NORMAL = 0
LEVER = 1
MAG = 2

class dataAnalysisRegular:
    def __init__(self, magFiles: List[str], levFiles: List[str], posFiles: List[str], fpsList: List[int], totFramesList: List[int], initialNanList: List[int], dates: List[int], sessions: List[int], ratPairs: List[int], familarity: List[str], transparency: List[str], sessionType: List[str], fiberFiles = None, prefix = "", save = True):
        self.mag_files = mag_files
        self.lev_files = lev_files
        self.pos_files = pos_files
        self.fpsList = fpsList
        self.totFramesList = totFramesList
        self.initialNanList = initialNanList
        self.dates = dates
        self.sessions = sessions
        self.ratPairs = ratPairs
        self.familarity = familarity
        self.transparency = transparency
        self.sessionTypes = sessionTypes
        self.prefix = prefix
        self.save = save
        
        self.experiments = []
        self.NUM_BINS = 30 # Number of time bins for trial chunking
        self.labelSize = 17
        self.titleSize = 18
        deleted_count = 0
        
        print("There are ", len(posFiles), " experiments in this data session. ")
        print("")
        
        if (magFiles != None and levFiles != None):
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
                    exp = singleExperiment(magFiles[i], levFiles[i], posFiles[i], fpsList[i], totFramesList[i], initialNanList[i], date = dates[i], sessionID = sessions[i], ratPair=ratPairs[i], trainingPartner=familarity[i], transparency=transparency[i], videoType=sessionType[i])
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
            
        else:
            if ((len(posFiles) != len(fpsList)) or (len(posFiles) != len(totFramesList)) or len(posFiles) != len(initialNanList)):
                raise ValueError("Different number of posFiles, fpsList, totFramesList, or initialNanList values")
            for i in range(len(posFiles)):
                exp = singleExperiment(None, None, posFiles[i], fpsList[i], totFramesList[i], initialNanList[i], date = dates[i], sessionID = sessions[i], ratPair=ratPairs[i])
                self.experiments.append(exp)
            
    def restrict_to_first_n_sessions(self, n=10):
        # sort experiments by date
        self.experiments = sorted(
            self.experiments,
            key=lambda exp: exp.date
        )
        # keep only first n
        self.experiments = self.experiments[:n]
    
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
    
    def generateHeatmap(self, bodypart = 3):
        x = []
        y = []
        
        for i, exp in enumerate(self.experiments):
            pos = exp.pos
        
            x0 = pos.data[0, 0, bodypart, :]
            x1 = pos.data[1, 0, bodypart, :]
            y0 = pos.data[0, 1, bodypart, :]
            y1 = pos.data[1, 1, bodypart, :]
        
            # Extend x with both x0 and x1
            x.extend(x0.tolist())
            x.extend(x1.tolist())
        
            # Extend y with both y0 and y1
            y.extend(y0.tolist())
            y.extend(y1.tolist())
        
        #print("x: ", x)
        
        return x, y
        
    
    def generateGazingAndInteractingHeatmapData(self, bodypart = 3):
        """
        Generate concatenated gaze and intearction location heatmap data.
        """
        gazer_x = [] #All the x positions of the rats doing the gazing
        gazer_y = [] #All the y positions of the rats doing the gazing
        gazee_x = [] #All the x positions of the rats being gazed at
        gazee_y = [] #All the y positions of the rats being gazed at
        
        interaction_x = [] #Same thing but for interactions
        interaction_y = []
        
        freq_x = [] #x-position of every single frame
        freq_y = [] #y-position of every single frame

        for i, exp in enumerate(self.experiments):
            pos = exp.pos
            
            #Get both rats x and y data
            x0 = pos.data[0, 0, bodypart, :]
            x1 = pos.data[1, 0, bodypart, :]
            y0 = pos.data[0, 1, bodypart, :]
            y1 = pos.data[1, 1, bodypart, :]
            
            if (len(x0) != len(x1) or len(x0) != len(y0) or len(x0) != len(y1)):
                print("MISMATCH IN LENGTHS")
            
            freq_x.append(x0)
            freq_x.append(x1)
            freq_y.append(y0)
            freq_y.append(y1)
            
            rat0Gazing = pos.returnIsGazing(0)
            rat1Gazing = pos.returnIsGazing(1)
            interacting = pos.returnIsInteracting()
            
            for j, gazing in enumerate(rat0Gazing):
                if (gazing):
                    gazer_x.append(x0[j])
                    gazer_y.append(y0[j])
                    gazee_x.append(x1[j])
                    gazee_y.append(y1[j])
            
            for j, gazing in enumerate(rat1Gazing):
                if (gazing):
                    gazer_x.append(x1[j])
                    gazer_y.append(y1[j])
                    gazee_x.append(x0[j])
                    gazee_y.append(y0[j])
            
            for j, interacted in enumerate(interacting):
                if (interacted):
                    interaction_x.append(x1[j])
                    interaction_y.append(y1[j])
                    interaction_x.append(x0[j])
                    interaction_y.append(y0[j])
            

        def safe_concat(arr_list):
            # Convert elements to at least 1D arrays and ignore empties
            valid = [np.atleast_1d(a) for a in arr_list if np.size(a) > 0]
            return np.concatenate(valid) if len(valid) > 0 else np.array([])
        
        gazer_x = safe_concat(gazer_x)
        gazer_y = safe_concat(gazer_y)
        gazee_x = safe_concat(gazee_x)
        gazee_y = safe_concat(gazee_y)
        
        interaction_x = safe_concat(interaction_x)
        interaction_y = safe_concat(interaction_y)
        
        freq_x = safe_concat(freq_x)
        freq_y = safe_concat(freq_y)

        return gazer_x, gazer_y, gazee_x, gazee_y, interaction_x, interaction_y, freq_x, freq_y

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
            numFrames = exp.endFrame
            
            levGazingFrames0 = np.sum(pos.returnIsLookingAtObjects(0))
            magGazingFrames0 = np.sum(pos.returnIsLookingAtObjects(0, target="mag"))
            otherGazingFrames0 = np.sum(pos.returnIsGazing(0))
            
            levGazingFrames1 = np.sum(pos.returnIsLookingAtObjects(1))
            magGazingFrames1 = np.sum(pos.returnIsLookingAtObjects(1, target="mag"))
            otherGazingFrames1 = np.sum(pos.returnIsGazing(1))
            
            
            otherGazingPerSession.append(otherGazingFrames0)
            otherGazingPerSession.append(otherGazingFrames0)
            levGazingPerSession.append(levGazingFrames0)
            levGazingPerSession.append(levGazingFrames1)
            magGazingPerSession.append(magGazingFrames0)
            magGazingPerSession.append(magGazingFrames1)
            numFramesPerSession.append(numFrames)
            numFramesPerSession.append(numFrames)
        
        return otherGazingPerSession, levGazingPerSession, magGazingPerSession, numFramesPerSession
    
    def gazingAtOtherVsLevVsMag_SuccVSNonSucc(self):
        """
        Compare percent of time spent gazing at Lever, Magazine, or Other Rat
        between successful vs unsuccessful trials, for each category of experiments.
        """

        results = {
            "Success": {"Lever": [], "Mag": [], "Other": []},
            "Failure": {"Lever": [], "Mag": [], "Other": []}
        }


        for exp_idx, exp in enumerate(self.experiments):
            lev = exp.lev
            pos = exp.pos
            fps= exp.fps

            # Get trial info
            trial_starts = lev.returnTimeStartTrials()
            succ_trials = lev.returnSuccessTrials()
            succ_trials = self._filterToLeverPressTrials(succ_trials, lev)

            if not (len(trial_starts) == len(succ_trials)):
                print(f"[Warning] Mismatched trial counts in experiment {exp_idx}")
                continue

            for trial_idx, t_begin in enumerate(trial_starts):
                if (trial_idx == len(trial_starts) - 1):
                    continue
                
                succ = succ_trials[trial_idx]
                t_end = trial_starts[trial_idx + 1]

                if any(x is None or np.isnan(x) for x in [t_begin, t_end, succ]):
                    continue

                start_frame = int(t_begin * fps)
                end_frame = int(t_end * fps)
                if end_frame <= start_frame:
                    continue

                # Extract gazing booleans for both mice
                for rat_id in [0, 1]:
                    look_lev = np.sum(pos.returnIsLookingAtObjects(rat_id)[start_frame:end_frame])
                    look_mag = np.sum(pos.returnIsLookingAtObjects(rat_id, target="mag")[start_frame:end_frame])
                    look_other = np.sum(pos.returnIsGazing(rat_id)[start_frame:end_frame])
                    num_frames = end_frame - start_frame

                    if num_frames <= 0:
                        continue

                    # Convert to percent time gazing
                    percent_lev = look_lev / num_frames * 100
                    percent_mag = look_mag / num_frames * 100
                    percent_other = look_other / num_frames * 100

                    group_key = "Success" if succ == 1 else "Failure"
                    results[group_key]["Lever"].append(percent_lev)
                    results[group_key]["Mag"].append(percent_mag)
                    results[group_key]["Other"].append(percent_other)

        return results
        
    def gazingData(self, sortByKL=True, save_csv=True, gazingType = 0, csv_path="gazing_data.csv", getInteractions = False):
        isKL = []
        percentGazing = []
        
        #Gazing Type: 0 = Normal, 1 = Lever, 2 = Mag
        
        word = "Gazing"
        if (getInteractions):
            word = "Interacting"
            
        
        print("Entering Gazing Data")
        print("Num Exps is: ", len(self.experiments))
        
        # for CSV
        rows = []
    
        for exp in self.experiments:
            ratPair = exp.ratPair
            pos = exp.pos
            
            if ratPair and ratPair[0:2] == "KL":
                isKLTemp = True
            else:
                isKLTemp = False
    
            isKL.append(isKLTemp)
            numFrames = pos.returnNumFrames()
            
            if (not getInteractions):
                if (gazingType == 0):
                    numFramesGazing = (pos.returnTotalFramesGazing(mouseID=0) + pos.returnTotalFramesGazing(mouseID=1)) / 2
                elif (gazingType == 1):
                    numFramesGazing = (np.sum(pos.returnIsLookingAtObjects(0)) + np.sum(pos.returnIsLookingAtObjects(1))) / 2
                elif (gazingType == 2):
                    numFramesGazing = (np.sum(pos.returnIsLookingAtObjects(0, target="mag")) + np.sum(pos.returnIsLookingAtObjects(1, target="mag"))) / 2
                
                pg = numFramesGazing / numFrames
        
                percentGazing.append(pg)
            
            else:
                numFramesInteracting = pos.returnTotalFramesInteracting()
                pg = numFramesInteracting / numFrames
                percentGazing.append(pg)
    
            rows.append({
                f"percent{word}": pg,
                "isKL": isKLTemp,
                "sessionID": exp.sessionID,
                "levFile": exp.lev_file,
                "ratPair": exp.ratPair,
                "date": exp.date,
                "familiarity": exp.familiarity,
                "transparency": exp.transparency,
                "sessionType": exp.sessionType
            })
    
        if save_csv:
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
    
        if sortByKL:
            return percentGazing, isKL
        else:
            return percentGazing
        

    def allGazingData(self, save_csv=True, csv_path="gazing_data.csv"):
        socialGazing = []
        levGazing = []
        magGazing = []
        socialGazingAtLev_list = []
        socialGazingAtMag_list = []
        socialGazingAtCenter_list = []
        avgLengthSocialGaze = []
        isKL = []
    
        print("Entering Gazing Data")
        print("Num Exps is:", len(self.experiments))
    
        rows = []
    
        for exp in self.experiments:
            ratPair = exp.ratPair
            pos = exp.pos
            
            if ratPair and ratPair[0:2] == "KL":
                isKLTemp = True
            else:
                isKLTemp = False
    
            isKL.append(isKLTemp)
            
            numFrames = pos.returnNumFrames()
    
            gaze0 = pos.returnIsGazing(mouseID=0)
            gaze1 = pos.returnIsGazing(mouseID=1)
    
            loc0 = pos.returnMouseLocation(0)
            loc1 = pos.returnMouseLocation(1)
    
            lev_mask_0 = np.array([l in ("lev_top", "lev_bottom") for l in loc0])
            lev_mask_1 = np.array([l in ("lev_top", "lev_bottom") for l in loc1])
    
            mag_mask_0 = np.array([l in ("mag_top", "mag_bottom") for l in loc0])
            mag_mask_1 = np.array([l in ("mag_top", "mag_bottom") for l in loc1])
    
            mid_mask_0 = np.array([l == "mid" for l in loc0])
            mid_mask_1 = np.array([l == "mid" for l in loc1])
    
            # raw frame counts
            socialGazingFrames = (np.sum(gaze0) + np.sum(gaze1)) / 2
            levGazeFrames = (np.sum(pos.returnIsLookingAtObjects(0)) +
                              np.sum(pos.returnIsLookingAtObjects(1))) / 2
            magGazeFrames = (np.sum(pos.returnIsLookingAtObjects(0, target="mag")) +
                              np.sum(pos.returnIsLookingAtObjects(1, target="mag"))) / 2
    
            socialGazingAtLev = (
                np.sum(gaze0 & lev_mask_0) +
                np.sum(gaze1 & lev_mask_1)
            ) / 2
    
            socialGazingAtMag = (
                np.sum(gaze0 & mag_mask_0) +
                np.sum(gaze1 & mag_mask_1)
            ) / 2
    
            socialGazingAtCenter = (
                np.sum(gaze0 & mid_mask_0) +
                np.sum(gaze1 & mid_mask_1)
            ) / 2
    
            # proportions
            socialGazePG = socialGazingFrames / numFrames
            levGazePG = levGazeFrames / numFrames
            magGazePG = magGazeFrames / numFrames
    
            avgGazeLength = (pos.returnAverageGazeLength(0) + pos.returnAverageGazeLength(1)) / 2
    
            socialGazing.append(socialGazePG)
            levGazing.append(levGazePG)
            magGazing.append(magGazePG)
    
            socialGazingAtLev_list.append(socialGazingAtLev / numFrames)
            socialGazingAtMag_list.append(socialGazingAtMag / numFrames)
            socialGazingAtCenter_list.append(socialGazingAtCenter / numFrames)
            
            avgLengthSocialGaze.append(avgGazeLength)
    
            rows.append({
                "sessionID": exp.sessionID,
                "levFile": exp.lev_file,
                "ratPair": exp.ratPair,
                "date": exp.date,
                "familiarity": exp.familiarity,
                "transparency": exp.transparency,
                "sessionType": exp.sessionType,
                "isKL": isKL,
                "socialGazing": socialGazePG,
                "levGazing": levGazePG,
                "magGazing": magGazePG,
                "socialGazingAtLev": socialGazingAtLev / numFrames,
                "socialGazingAtMag": socialGazingAtMag / numFrames,
                "socialGazingAtCenter": socialGazingAtCenter / numFrames,
                "avgSocialGazeLength": avgGazeLength
            })
    
        if save_csv:
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
    
        return socialGazing, levGazing, magGazing, socialGazingAtLev_list, socialGazingAtMag_list, socialGazingAtCenter_list, avgLengthSocialGaze, isKL


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
    
    def makeRelativeHeatmap(self, event_x, event_y, freq_x, freq_y,
                        bins=150, sigma=2, clip_percent=97, gamma=0.3):
        """Return heatmap weighted by how often they occupy each region."""
        H_event = self.makeHeatmap(event_x, event_y, bins, sigma, clip_percent, gamma)
        H_freq = self.makeHeatmap(freq_x, freq_y, bins, sigma, clip_percent, gamma)
    
        if H_event is None or H_freq is None:
            return None
    
        # Avoid divide-by-zero: mask zeros
        mask = H_freq == 0
        H_rel = np.zeros_like(H_event)
        H_rel[~mask] = H_event[~mask] / H_freq[~mask]
    
        # Normalize to [0,1]
        H_rel = H_rel / H_rel.max() if H_rel.max() > 0 else H_rel
        return H_rel    

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
        
    def saveDifferenceHeatmap(self, H, title, filename):
        if H is None:
            print(f"Skipping {title}, no data")
            return
        
        fig_width = 6
        fig_height = fig_width / self.aspect_ratio
    
        plt.figure(figsize=(fig_width, fig_height))
    
        vmax = np.max(np.abs(H))      # symmetric range
        im = plt.imshow(
            np.flipud(H),
            extent=[0, self.arena_width, 0, self.arena_height],
            origin="upper",
            aspect="auto",
            cmap="bwr",               # diverging: blue→white→red
            vmin=-vmax,
            vmax=vmax
        )
    
        plt.colorbar(im, label="Coop - NonCoop Density")
        plt.xlabel("X position (px)")
        plt.ylabel("Y position (px)")
        plt.title(title)
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
            p_text = f'p = {p_value:.3f}'
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
        
    def plot_gazing_comparison(self, results, ylabel="Percent Time Gazing", title="Gazing Comparison: Success vs Failure", filename="gazing_comparison.png", colors=None, figsize=(8, 6)):
        """
        Create grouped bar plots comparing gaze behavior (Lever, Mag, Other) between
        successful and unsuccessful trials.
    
        Args:
            results (dict): Output from gazingAtOtherVsLevVsMag_SuccVSNonSucc().
            ylabel (str): Label for the y-axis.
            title (str): Plot title.
            filename (str): File name to save the plot.
            colors (list, optional): Custom colors for groups.
            figsize (tuple): Figure size in inches.
        """
        # --- Validate input ---
        if not results or any(k not in results for k in ["Success", "Failure"]):
            print("Invalid results structure.")
            return
    
        categories = ["Lever", "Mag", "Other"]
        success_data = [results["Success"].get(cat, []) for cat in categories]
        failure_data = [results["Failure"].get(cat, []) for cat in categories]
    
        # Filter out empty categories
        valid_cats = [cat for cat, s, f in zip(categories, success_data, failure_data) if len(s) > 0 and len(f) > 0]
        success_data = [s for s in success_data if len(s) > 0]
        failure_data = [f for f in failure_data if len(f) > 0]
    
        if not valid_cats:
            print("No valid data to plot.")
            return
    
        if colors is None:
            colors = plt.cm.Set2(np.arange(2))
    
        # --- Compute stats ---
        success_means = [np.mean(data) for data in success_data]
        failure_means = [np.mean(data) for data in failure_data]
        success_sems = [sem(data) for data in success_data]
        failure_sems = [sem(data) for data in failure_data]
    
        x = np.arange(len(valid_cats))
        width = 0.35
    
        plt.figure(figsize=figsize)
    
        bars1 = plt.bar(x - width/2, success_means, width, yerr=success_sems, label="Success", color=colors[0], capsize=5)
        bars2 = plt.bar(x + width/2, failure_means, width, yerr=failure_sems, label="Failure", color=colors[1], capsize=5)
    
        # --- Add numeric labels ---
        for bars, means in zip([bars1, bars2], [success_means, failure_means]):
            for bar, mean in zip(bars, means):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{mean:.1f}",
                         ha='center', va='bottom', fontsize=9)
    
        # --- Add p-values ---
        for i, (succ, fail) in enumerate(zip(success_data, failure_data)):
            _, p = ttest_ind(succ, fail, equal_var=False)
            plt.text(i, max(success_means[i], failure_means[i]) + 3, f"p={p:.3f}", ha="center", fontsize=9)
    
        plt.xticks(x, valid_cats)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
    
    def plot_bar_comparison(self, data_list1, data_list2, category_labels, group_labels,
                            ylabel, title, filename, colors=None, figsize=(7, 5)):
        """
        Create a grouped bar chart comparing two datasets across multiple categories.

        Args:
            data_list1 (List[List[float]]): First dataset (e.g., Cooperative sessions), list of lists (one per category).
            data_list2 (List[List[float]]): Second dataset (e.g., Non-Cooperative sessions), same shape as data_list1.
            category_labels (List[str]): Names of the categories (e.g., ["Other Rat", "Lever", "Magazine"]).
            group_labels (List[str]): Names of the two groups being compared (e.g., ["Coop", "NonCoop"]).
            ylabel (str): Y-axis label.
            title (str): Plot title.
            filename (str): Path to save figure.
            colors (List[str], optional): Colors for bars. Defaults to tab10 colors.
            figsize (tuple): Figure size.
        """
        num_categories = len(category_labels)
        if colors is None:
            colors = plt.cm.tab10(np.arange(2))

        # Compute means and SEMs
        means1 = [np.mean(d) for d in data_list1]
        means2 = [np.mean(d) for d in data_list2]
        sems1 = [sem(d) for d in data_list1]
        sems2 = [sem(d) for d in data_list2]

        # Compute p-values for each category
        p_values = []
        for d1, d2 in zip(data_list1, data_list2):
            if len(d1) > 1 and len(d2) > 1:
                _, p = ttest_ind(d1, d2, equal_var=False)
                p_values.append(p)
            else:
                p_values.append(np.nan)

        # Bar positions
        x = np.arange(num_categories)
        width = 0.35

        plt.figure(figsize=figsize)
        bars1 = plt.bar(x - width/2, means1, width, yerr=sems1, capsize=5, color=colors[0], label=group_labels[0])
        bars2 = plt.bar(x + width/2, means2, width, yerr=sems2, capsize=5, color=colors[1], label=group_labels[1])

        # Add mean labels above bars
        for bars, means in zip([bars1, bars2], [means1, means2]):
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, height + max(sems1 + sems2)/3, f'{mean:.2f}',
                         ha='center', va='bottom', fontsize=9)

        # Add p-values above paired bars
        for i, p in enumerate(p_values):
            if not np.isnan(p):
                plt.text(x[i], max(means1[i], means2[i]) + max(sems1[i], sems2[i]) * 2,
                         f'p={p:.3f}', ha='center', va='bottom', fontsize=9,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

        plt.xticks(x, category_labels)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
    
    def barGraphGazing(self,
                       gazeCoop,
                       gazeNonCoop,
                       isKL,
                       save_path,
                       title="Gazing Per Session",
                       ylabel="Percent Gazing"):

        gazeCoop = np.asarray(gazeCoop, dtype=float)
        gazeNonCoop = np.asarray(gazeNonCoop, dtype=float)
        
        isKL = np.asarray(isKL).astype(bool)
        
        if len(isKL) != len(gazeCoop):
            print("isKL: ", isKL)
            print("gazeCoop: ", gazeCoop)
            raise ValueError(
                f"isKL length ({len(isKL)}) must match gazeCoop length ({len(gazeCoop)})"
            )
        
        gazeKL = gazeCoop[isKL]
        gazeEB = gazeCoop[~isKL]

        mean_coop = np.mean(gazeCoop)
        mean_non = np.mean(gazeNonCoop)

        x_coop = 0
        x_non = 1
        bar_width = 0.4
        jitter = 0.06

        fig, ax = plt.subplots(figsize=(6, 6))

        # ---- Bars (overall means) ----
        ax.bar(
            x_coop,
            mean_coop,
            width=bar_width,
            color="lightgray",
            edgecolor="black",
            zorder=1
        )

        ax.bar(
            x_non,
            mean_non,
            width=bar_width,
            color="gray",
            edgecolor="black",
            zorder=1
        )

        # ---- Scatter points ----
        kl_scatter = None
        eb_scatter = None

        if len(gazeKL) > 0:
            kl_scatter = ax.scatter(
                np.random.normal(x_coop, jitter, size=len(gazeKL)),
                gazeKL,
                color="tab:blue",
                alpha=0.8,
                label="KL (Coop)",
                zorder=2
            )

        if len(gazeEB) > 0:
            eb_scatter = ax.scatter(
                np.random.normal(x_coop, jitter, size=len(gazeEB)),
                gazeEB,
                color="tab:orange",
                alpha=0.8,
                label="EB (Coop)",
                zorder=2
            )

        other_scatter = ax.scatter(
            np.random.normal(x_non, jitter, size=len(gazeNonCoop)),
            gazeNonCoop,
            color="black",
            alpha=0.7,
            label="Other (Non-Coop)",
            zorder=2
        )

        # ---- Dashed subgroup means on Coop bar ----
        if len(gazeKL) > 0:
            ax.hlines(
                np.mean(gazeKL),
                x_coop - bar_width / 2,
                x_coop + bar_width / 2,
                colors="tab:blue",
                linestyles="dashed",
                linewidth=2
            )

        if len(gazeEB) > 0:
            ax.hlines(
                np.mean(gazeEB),
                x_coop - bar_width / 2,
                x_coop + bar_width / 2,
                colors="tab:orange",
                linestyles="dashed",
                linewidth=2
            )

        # ---- Axes formatting ----
        ax.set_xticks([x_coop, x_non])
        ax.set_xticklabels(["Coop", "Non-Coop"])
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # ---- Legends ----
        coop_handles = []
        if kl_scatter is not None:
            coop_handles.append(kl_scatter)
        if eb_scatter is not None:
            coop_handles.append(eb_scatter)

        if coop_handles:
            legend_coop = ax.legend(
                handles=coop_handles,
                loc="upper left",
                frameon=False,
                title="Cooperative"
            )
            ax.add_artist(legend_coop)

        ax.legend(
            handles=[other_scatter],
            loc="upper right",
            frameon=False,
            title="Non-Cooperative"
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
    def barGraphGazingAll(self,
                      gazeCoopDict,
                      gazeNonCoopDict,
                      isKL,
                      save_path,
                      title="Amount of gaze at object",
                      ylabel="Percent Gazing"):

        categories = ["other", "lever", "mag"]
        labels = ["Other animal", "Lever", "Mag"]
    
        isKL = np.asarray(isKL).astype(bool)
    
        # ---- Compute means ----
        coop_means = [np.mean(gazeCoopDict[c]) for c in categories]
        non_means  = [np.mean(gazeNonCoopDict[c]) for c in categories]
    
        x = np.arange(len(categories))
        bar_width = 0.35
        jitter = 0.05
    
        fig, ax = plt.subplots(figsize=(7, 6))
    
        # ---- Bars ----
        coop_bars = ax.bar(
            x - bar_width / 2,
            coop_means,
            width=bar_width,
            color="dimgray",
            edgecolor="black",
            label="Success periods",
            zorder=1
        )
    
        non_bars = ax.bar(
            x + bar_width / 2,
            non_means,
            width=bar_width,
            color="lightgray",
            edgecolor="black",
            label="Not success periods",
            zorder=1
        )
    
        # ---- Scatter (Coop: KL vs EB) ----
        for i, c in enumerate(categories):
            gaze = np.asarray(gazeCoopDict[c], dtype=float)
    
            gazeKL = gaze[isKL]
            gazeEB = gaze[~isKL]
    
            if len(gazeKL) > 0:
                ax.scatter(
                    np.random.normal(x[i] - bar_width / 2, jitter, size=len(gazeKL)),
                    gazeKL,
                    color="tab:blue",
                    alpha=0.8,
                    zorder=2,
                    label="KL (Coop)" if i == 0 else None
                )
    
            if len(gazeEB) > 0:
                ax.scatter(
                    np.random.normal(x[i] - bar_width / 2, jitter, size=len(gazeEB)),
                    gazeEB,
                    color="tab:orange",
                    alpha=0.8,
                    zorder=2,
                    label="EB (Coop)" if i == 0 else None
                )
    
        # ---- Scatter (Non-Coop) ----
        for i, c in enumerate(categories):
            gaze = np.asarray(gazeNonCoopDict[c], dtype=float)
    
            ax.scatter(
                np.random.normal(x[i] + bar_width / 2, jitter, size=len(gaze)),
                gaze,
                color="black",
                alpha=0.6,
                zorder=2,
                label="Other (Non-Coop)" if i == 0 else None
            )
    
        # ---- Axes formatting ----
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    
        # ---- Legends (match example layout) ----
        handles, labels_ = ax.get_legend_handles_labels()
    
        coop_handles = []
        noncoop_handles = []
    
        for h, l in zip(handles, labels_):
            if "Coop" in l or "KL" in l or "EB" in l:
                coop_handles.append((h, l))
            else:
                noncoop_handles.append((h, l))
    
        if coop_handles:
            ax.legend(
                [h for h, _ in coop_handles],
                [l for _, l in coop_handles],
                loc="upper left",
                frameon=False
            )
    
        if noncoop_handles:
            ax.legend(
                [h for h, _ in noncoop_handles],
                [l for _, l in noncoop_handles],
                loc="upper right",
                frameon=False
            )
    
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    
    
#DATA ANALYSIS GENERATION
#
#

#Real

filtered = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/Filtered.csv"
minReq = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/dyed_preds_min_requirements_valid.csv"
minReqTesting = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_PairedTesting_filtered_partiallyValid.csv"
minReqTraining = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_TrainingCooperation_filtered_partiallyValid.csv" 
minReqComp = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/comp_minReq_valid.csv"
minReqIneq = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/ineq_minReq_valid.csv"
minReqNonCoop = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/nonCoop_minReq_valid.csv"

def getFiltered():
    print("TESTING\n")
    fe = fileExtractor(minReqTesting)
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
    fe.data = fe.keepOnlyPred()
    #fe.deleteOnlyFullyInvalid()
    #fe.filterOutBadNums()x
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
    sessionTypes = fe.getSessionType()
    #print("initial_nan_list: ", initial_nan_list)
    return [fe.getLevsDatapath(), fe.getMagsDatapath(), fe.getPosDatapath(), fpsList, totFramesList, initial_nan_list, dates, sessions, ratPairs, familiarity, transparency, sessionTypes]

def minRequirementsComp():
    fe = fileExtractor(minReqComp)
    #fe.data = fe.deleteBadNaN()
    #fe.deleteOnlyFullyInvalid()
    #fe.filterOutBadNums()
    fpsList, totFramesList = fe.returnFPSandTotFrames()
    initial_nan_list = fe.returnNaNPercentage()
    dates = fe.getDatesList()
    sessions = fe.getSessionIDList()
    #dates = dates.tolist()
    #sessions = sessions.tolist()
    ratPairs = fe.getRatPairList()
    familiarity = fe.getFamiliarityList()
    transparency = fe.getBarrierTransparencyList()
    sessionTypes = fe.getSessionType()
    #print("initial_nan_list: ", initial_nan_list)
    return [None, None, fe.getPosDatapath(), fpsList, totFramesList, initial_nan_list, dates, sessions, ratPairs, familiarity, transparency, sessionTypes]

def minRequirementsIneq():
    fe = fileExtractor(minReqIneq)
    #fe.data = fe.deleteBadNaN()
    #fe.deleteOnlyFullyInvalid()
    #fe.filterOutBadNums()
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
    sessionTypes = fe.getSessionType()
    #print("initial_nan_list: ", initial_nan_list)
    return [None, None, fe.getPosDatapath(), fpsList, totFramesList, initial_nan_list, dates, sessions, ratPairs, familiarity, transparency, sessionTypes]

def minRequirementsNonCoop():
    fe = fileExtractor(minReqNonCoop)
    fe.data = fe.deleteBadNaN()
    fe.data = fe.keepOnlyPred()
    #fe.deleteOnlyFullyInvalid()
    #fe.filterOutBadNums()
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
    sessionTypes = fe.getSessionType()
    #print("initial_nan_list: ", initial_nan_list)
    return [None, None, fe.getPosDatapath(), fpsList, totFramesList, initial_nan_list, dates, sessions, ratPairs, familiarity, transparency, sessionTypes]



#arr = getFiltered()
#arr = minRequirements()


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
familarity = arr[9]
transparency = arr[10]
sessionTypes = arr[11]


arr = minRequirementsComp()
lev_files_comp = arr[0]
mag_files_comp = arr[1]
pos_files_comp = arr[2]
fpsList_comp = arr[3]
totFramesList_comp = arr[4]
initialNanList_comp = arr[5]
dates_comp = arr[6]
sessions_comp = arr[7]
ratPairs_comp = arr[8]    
familarity_comp = arr[9]
transparency_comp = arr[10]
sessionTypes_comp = arr[11]


arr = minRequirementsIneq()
lev_files_ineq = arr[0]
mag_files_ineq = arr[1]
pos_files_ineq = arr[2]
fpsList_ineq = arr[3]
totFramesList_ineq = arr[4]
initialNanList_ineq = arr[5]
dates_ineq = arr[6]
sessions_ineq = arr[7]
ratPairs_ineq = arr[8]    
familarity_ineq = arr[9]
transparency_ineq = arr[10]
sessionTypes_ineq = arr[11]


arr = minRequirementsNonCoop()
lev_files2 = arr[0]
mag_files2 = arr[1]
pos_files2 = arr[2]
fpsList2 = arr[3]
totFramesList2 = arr[4]
initialNanList2 = arr[5]
dates2 = arr[6]
sessions2 = arr[7]
ratPairs2 = arr[8]
familarity2 = arr[9]
transparency2 = arr[10]
sessionTypes2 = arr[11]



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

print("pos_files: ", pos_files)
print("pos_files2: ", pos_files2)


# Create the data generation object
data = dataAnalysisRegular(
    mag_files, lev_files, pos_files,
    fpsList, totFramesList, initialNanList,
    dates, sessions, ratPairs,
    familarity, transparency, sessionTypes,
    prefix="", save=True
)

data_comp = dataAnalysisRegular(
    mag_files_comp, lev_files_comp, pos_files_comp, fpsList_comp,
    totFramesList_comp, initialNanList_comp, dates_comp, sessions_comp,
    ratPairs_comp, familarity_comp, transparency_comp, sessionTypes_comp,
    prefix="", save=True
)

data_ineq = dataAnalysisRegular(
    mag_files_ineq, lev_files_ineq, pos_files_ineq, fpsList_ineq,
    totFramesList_ineq, initialNanList_ineq, dates_ineq, sessions_ineq,
    ratPairs_ineq, familarity_ineq, transparency_ineq, sessionTypes_ineq,
    prefix="", save=True
)

'''
data2 = dataAnalysisRegular(mag_files2, lev_files2, pos_files2, fpsList2, totFramesList2, initialNanList2, dates2, sessions2, ratPairs2, familarity2, transparency2, sessionTypes2, prefix = "", save=True)
''''''

# Create the graph object
graphs = createGraphs()


#
# Gaze Graphs: Coop vs NonCoop
#

#
# Set Data
#
prefix = "gazing" #interactions or gazing
interactions = False  #True for interactions, false for gaze

############################################################
# ------------------ EXTRACT GAZE / INTERACTIONS ------------
############################################################

#Extract Gazing Data
#gaze_coop, isKL_coop = data.gazingData(sortByKL=True, save_csv=False, getInteractions=interactions)
#gaze_comp, isKL_comp = data_comp.gazingData(sortByKL=True, save_csv=False, getInteractions=interactions)
#gaze_ineq, isKL_ineq = data_ineq.gazingData(sortByKL=True, save_csv=False, getInteractions=interactions)
'''

socialGazing, levGazing, magGazing, socialGazingAtLev_list, socialGazingAtMag_list, socialGazingAtCenter_list, avgLengthSocialGaze, isKL = data_comp.allGazingData()

def save_session_level_gaze_csv(
    data_obj,
    socialGaze,
    leverGaze,
    magGaze,
    socialGazeAtLev,
    socialGazeAtMag,
    socialGazeAtCenter,
    avgLengthSocialGaze,
    isKL,
    label,
    csv_path,
    indices=None
):
    rows = []

    if indices is None:
        indices = range(len(socialGaze))

    for row_i, data_i in enumerate(indices):
        rows.append({
            "label": label,
            "date": data_obj.dates[data_i],
            "lev_file": data_obj.lev_files[data_i],
            "session": data_obj.sessions[data_i],
            "ratPair": data_obj.ratPairs[data_i],
            "familiarity": data_obj.familarity[data_i],
            "transparency": data_obj.transparency[data_i],
            "social_gaze": socialGaze[row_i],
            "lever_gaze": leverGaze[row_i],
            "mag_gaze": magGaze[row_i],
            "social_gaze_at_lev": socialGazeAtLev[row_i],
            "social_gaze_at_mag": socialGazeAtMag[row_i],
            "social_gaze_at_center": socialGazeAtCenter[row_i],
            "avg_social_gaze_length": avgLengthSocialGaze,
            "isKL": bool(isKL[row_i]) if isKL is not None else np.nan
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

save_session_level_gaze_csv(
    data_comp,
    socialGazing,
    levGazing, magGazing, socialGazingAtLev_list, socialGazingAtMag_list, socialGazingAtCenter_list, avgLengthSocialGaze, isKL,
    label="comp",
    csv_path=f"comp_sessions_allGazeData.csv"
)

'''
save_session_level_gaze_csv(
    data_comp,
    gaze_comp,
    isKL_comp,
    label="comp",
    csv_path=f"{prefix}_sessions_comp.csv"
)

save_session_level_gaze_csv(
    data_ineq,
    gaze_ineq,
    isKL_ineq,
    label="ineq",
    csv_path=f"{prefix}_sessions_ineq.csv"
)'''

exit()

############################################################
# --------- BAR PLOT: COOP vs COMP vs INEQ -----------------
############################################################

def barGraphAvgGazeThreeTypes(
    gaze_lists,
    isKL_lists,
    labels,
    bar_colors,
    save_path
):
    means = [np.mean(g) for g in gaze_lists]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(6, 6))

    # bars
    ax.bar(x, means, color=bar_colors, edgecolor="black", zorder=1)

    # scatter individual points
    for i, (gaze, isKL) in enumerate(zip(gaze_lists, isKL_lists)):
        jitter = np.random.normal(0, 0.04, size=len(gaze))
        colors = ["black" if v else "red" for v in isKL]

        ax.scatter(
            np.full(len(gaze), x[i]) + jitter,
            gaze,
            c=colors,
            s=35,
            zorder=2
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Percent Interacting")
    ax.set_title("Average Interacting (First 10 Sessions per Rat Pair)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


barGraphAvgGazeThreeTypes(
    gaze_lists=[gaze_coop, gaze_comp, gaze_ineq],
    isKL_lists=[isKL_coop, isKL_comp, isKL_ineq],
    labels=["Coop", "Comp", "Ineq"],
    bar_colors=["navy", "gray", "dimgray"],
    save_path=f"avg_{prefix}_first10_per_ratpair.png"
)



############################################################
# --------- LINE PLOT: AVERAGED BY SESSION INDEX -----------
############################################################

def average_gaze_by_session_index(data_obj, gaze, n_sessions=10):
    ratpair_to_gaze = {}
    for rp, g in zip(data_obj.ratPairs, gaze):
        ratpair_to_gaze.setdefault(rp, []).append(g)

    avg_by_session = []
    for i in range(n_sessions):
        vals = [
            g_list[i]
            for g_list in ratpair_to_gaze.values()
            if len(g_list) > i
        ]
        avg_by_session.append(np.mean(vals))

    return avg_by_session

#gaze_coop_avg = average_gaze_by_session_index(data, gaze_coop)
#gaze_comp_avg = average_gaze_by_session_index(data_comp, gaze_comp)
#gaze_ineq_avg = average_gaze_by_session_index(data_ineq, gaze_ineq)


def lineGraphGazeOverTime(gaze_lists, labels, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))

    for gaze, label in zip(gaze_lists, labels):
        x = np.arange(len(gaze))
        line, = ax.plot(x, gaze, marker="o", label=label)

        slope, intercept, *_ = linregress(x, gaze)
        ax.plot(x, intercept + slope * x, linestyle="--", color=line.get_color())

    ax.set_xlabel("Session Number")
    ax.set_ylabel("Percent Interacting")
    ax.set_title("Interacting Over Time (First 10 Sessions per Rat Pair)")
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


'''
lineGraphGazeOverTime(
    [gaze_coop_avg, gaze_comp_avg, gaze_ineq_avg],
    ["Coop", "Comp", "Ineq"],
    f"{prefix}_over_time_first10_per_ratpair.png"
)
'''


'''def lineGraphGazeOverTime(
    gaze_lists,
    labels,
    isKL_lists,
    save_path,
    csv_path):
    
    fig, ax = plt.subplots(figsize=(7, 6))

    word = "Interacting"

    for gaze, label, isKL in zip(gaze_lists, labels, isKL_lists):
        x = np.arange(len(gaze))
        gaze_array = np.array(gaze)

        # Plot data
        if (label == "Coop EB" or label == "Coop KL") and isKL is not None:
            isKL = np.array(isKL, dtype=bool) 
            
            kl_label = "Coop (KL)" if label == "Coop KL" else None
            eb_label = "Coop (EB)" if label == "Coop EB" else None
            
            ax.scatter(x[isKL], gaze_array[isKL], color="black", label=kl_label)
            ax.scatter(x[~isKL], gaze_array[~isKL], color="red", label=eb_label)
            base_color = "black" if label == "Coop KL" else "red"
                    
        else:
            line, = ax.plot(x, gaze, marker="o", label=label)
            base_color = line.get_color()

        # Trendline (same color as condition)
        slope, intercept, *_ = linregress(x, gaze)
        ax.plot(
            x,
            intercept + slope * x,
            linestyle="--",
            color=base_color
        )

    ax.set_xlabel("Session (Chronological)")
    ax.set_ylabel(f"Percent {word}")
    ax.set_title(f"{word} Over Time (First 10 Sessions)")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), frameon=False, loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()



lineGraphGazeOverTime(
    [gaze_coop, gaze_eb, gaze_comp, gaze_ineq],
    ["Coop KL", "Coop EB", "Comp", "Ineq"],
    [isKL_coop, isKL_eb, None, None],  # or isEB_coop
    f"{prefixGazeOrInteractions}_over_time_first10_coop_vs_nonCoop.png",
    f"{prefixGazeOrInteractions}_over_time_first10_coop_vs_nonCoop.csv"
)'''







'''gazingType = LEVER
suffix = "normal"

print("SUFFIX is: ", suffix)


# ---- Coop ----
gazePerSessionCoop, isKL = data.gazingData(
    csv_path= f"coop_gazing_{suffix}.csv"
)

gazePerSessionCoopLever, isKL = data.gazingData(gazingType=LEVER,
    csv_path= f"coop_gazing_lever_{suffix}.csv"
)

gazePerSessionCoopMag, isKL = data.gazingData(gazingType=MAG,
    csv_path= f"coop_gazing_mag_{suffix}.csv"
)

# ---- Non-Coop ----
gazePerSessionNonCoop = data2.gazingData(
    sortByKL=False,
    csv_path = f"noncoop_gazing_{suffix}.csv"
)

gazePerSessionNonCoopLever = data2.gazingData(
    sortByKL=False, gazingType=LEVER,
    csv_path = f"noncoop_gazing_{suffix}.csv"
)

gazePerSessionNonCoopMag = data2.gazingData(
    sortByKL=False, gazingType=MAG,
    csv_path = f"noncoop_gazing_{suffix}.csv"
)

gazeCoop = {
    "other": gazePerSessionCoop,
    "lever": gazePerSessionCoopLever,
    "mag":   gazePerSessionCoopMag
}

gazeNonCoop = {
    "other": gazePerSessionNonCoop,
    "lever": gazePerSessionNonCoopLever,
    "mag":   gazePerSessionNonCoopMag
}


graphs.barGraphGazingAll(
    gazeCoopDict=gazeCoop,
    gazeNonCoopDict=gazeNonCoop,
    isKL=isKL,
    save_path=f"gazing_coop_vs_noncoop_{suffix}.png",
    title="Amount of Gaze at Object",
    ylabel="Percent Gazing"
)'''


'''
graphs.barGraphGazing(
    gazePerSessionCoop,
    gazePerSessionNonCoop,
    isKL,
    save_path = f"gazing_coop_vs_nonCoop_{suffix}.png",
    title = "Gazing Behavior by Pair Type"
)
'''


'''
#
# Generate Comparison of Coop vs. Non Coop Heatmap
#
xCoop, yCoop = data.generateHeatmap()
xNonCoop, yNonCoop = data2.generateHeatmap()

# Make heatmaps
H_coop = graphs.makeHeatmap(xCoop, yCoop)
H_nonCoop = graphs.makeHeatmap(xNonCoop, yNonCoop)

# Compute difference (Coop - NonCoop)
H_diff = None
if H_coop is not None and H_nonCoop is not None:
    # Ensure same shape
    if H_coop.shape == H_nonCoop.shape:
        H_diff = H_coop - H_nonCoop
    else:
        print("Heatmap shapes differ, cannot compute difference.")

# Save heatmaps
graphs.saveHeatmap(H_coop, "Cooperative Rat Heatmap", "cooperative_rat_heatmap.png")
graphs.saveHeatmap(H_nonCoop, "Non-Cooperative Rat Heatmap", "non_cooperative_rat_heatmap.png")
graphs.saveDifferenceHeatmap(
    H_diff,
    "Difference Heatmap (Coop - NonCoop)",
    "difference_heatmap.png"
)


print("Finished Saving Heatmaps")
'''



'''
#
# Gazing at Locations: Coop vs Non-Coop
#

otherGazingPerSessionCoop, levGazingPerSessionCoop, magGazingPerSessionCoop, numFramesPerSessionCoop = data.gazingAtOtherVsLevVsMag()
otherGazingPerSessionNonCoop, levGazingPerSessionNonCoop, magGazingPerSessionNonCoop, numFramesPerSessionNonCoop = data2.gazingAtOtherVsLevVsMag()
graphs.plot_bar_comparison(
    data_list1=[otherGazingPerSessionCoop, levGazingPerSessionCoop, magGazingPerSessionCoop],
    data_list2=[otherGazingPerSessionNonCoop, levGazingPerSessionNonCoop, magGazingPerSessionNonCoop],
    category_labels=["Other Rat", "Lever", "Magazine"],
    group_labels=["Coop", "NonCoop"],
    ylabel="Gazing Frames",
    title="Gazing Comparison: Cooperative vs Non-Cooperative Sessions",
    filename="gazing_comparison_barplot.png"
)'''


'''
#
# Gazing at Locations: Succ vs Non-Succ
#

results = data.gazingAtOtherVsLevVsMag_SuccVSNonSucc()
graphs.plot_gazing_comparison(results, title="Gazing: Success vs Failure", filename="gazing_success_vs_failure.png")()
'''

'''
#
# Gazing and Interaction Location Heatmaps
#

(gazer_x, gazer_y,
 gazee_x, gazee_y,
 interaction_x, interaction_y,
 freq_x, freq_y) = data.generateGazingAndInteractingHeatmapData()

# --- EVENT HEATMAPS ---
H_gazer = graphs.makeHeatmap(gazer_x, gazer_y)
H_gazee = graphs.makeHeatmap(gazee_x, gazee_y)
H_interact = graphs.makeHeatmap(interaction_x, interaction_y)

# Save them
graphs.saveHeatmap(H_gazer, "Gazer Position Heatmap", "gazer_heatmap.png")
graphs.saveHeatmap(H_gazee, "Gazee Position Heatmap", "gazee_heatmap.png")
graphs.saveHeatmap(H_interact, "Interaction Location Heatmap", "interaction_heatmap.png")

# --- RELATIVE EVENT HEATMAPS ---
H_gazer_rel = graphs.makeRelativeHeatmap(gazer_x, gazer_y, freq_x, freq_y)
H_gazee_rel = graphs.makeRelativeHeatmap(gazee_x, gazee_y, freq_x, freq_y)
H_interact_rel = graphs.makeRelativeHeatmap(interaction_x, interaction_y, freq_x, freq_y)

graphs.saveHeatmap(H_gazer_rel, "Relative Gazer Heatmap", "gazer_relative_heatmap.png")
graphs.saveHeatmap(H_gazee_rel, "Relative Gazee Heatmap", "gazee_relative_heatmap.png")
graphs.saveHeatmap(H_interact_rel, "Relative Interaction Heatmap", "interaction_relative_heatmap.png")
'''


'''
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
'''


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
  




# Generate left/right position data
'''
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
        
