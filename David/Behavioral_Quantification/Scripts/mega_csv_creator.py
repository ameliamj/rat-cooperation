#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 10:46:04 2025

@author: david
"""

import pandas as pd
import numpy as np
from collections import Counter
from experiment_class import singleExperiment

class megaCSVCreator:
    def __init__(self, experiment):
        """
        Initialize with a singleExperiment instance.
        
        Args:
            experiment (singleExperiment): Instance containing mag, lev, and pos data
        """
        self.exp = experiment
    
    def _get_trial_times(self, trial_num):
        """
        Get trial start and end times for a specific trial.
        
        Args:
            trial_num (int): Trial number (1-based)
        
        Returns:
            tuple: (start_time, end_time) or (NaN, NaN) if not found
        """
        start_times = self.exp.lev.returnTimeStartTrials()
        end_times = self.exp.lev.returnTimeEndTrials()
        idx = int(trial_num) - 1
        return (start_times[idx] if idx < len(start_times) else np.nan,
                end_times[idx] if idx < len(end_times) else np.nan)
    
    def _get_ipi_rat_specific(self, row_idx, rat_id):
        """
        Compute IPI for a specific lever press by a rat.
        Returns NaN if the current row's RatID does not match rat_id or for the first press by this rat.
        
        Args:
            row_idx (int): Index of the lever press in lev.data
            rat_id (int): Rat ID (0 or 1)
        
        Returns:
            float: IPI or NaN if not the specified rat, first press, or no prior press
        """
        df = self.exp.lev.data
        curr_row = df.iloc[row_idx]
        if curr_row['RatID'] != rat_id:
            return np.nan
        curr_time = curr_row['AbsTime']
        prev_presses = df[(df['RatID'] == rat_id) & (df['AbsTime'] < curr_time)]
        if prev_presses.empty:
            return np.nan
        prev_time = prev_presses['AbsTime'].max()
        return curr_time - prev_time
    
    def _get_ipi_general(self, row_idx):
        """
        Compute general IPI for a specific lever press (time since last press by any rat).
        Returns NaN for the first press in the session.
        
        Args:
            row_idx (int): Index of the lever press in lev.data
        
        Returns:
            float: IPI or NaN if first press
        """
        df = self.exp.lev.data
        if row_idx == 0:
            return np.nan
        prev_time = df.iloc[row_idx - 1]['AbsTime']
        curr_time = df.iloc[row_idx]['AbsTime']
        return curr_time - prev_time
    
    def _get_ipi_first_to_success(self, row_idx):
        """
        Compute IPI from first press to success for a lever press in a successful trial.
        
        Args:
            row_idx (int): Index of the lever press in lev.data
        
        Returns:
            float: IPI or NaN if not a successful trial or not the success press
        """
        df = self.exp.lev.data
        row = df.iloc[row_idx]
        if row['coopSucc'] != 1:
            return np.nan
        
        trial_num = row['TrialNum']
        trial_data = df[df['TrialNum'] == trial_num].sort_values(by='AbsTime')
        rats_seen = set()
        for _, press in trial_data.iterrows():
            rat = press['RatID']
            if rat not in rats_seen:
                rats_seen.add(rat)
                if len(rats_seen) == 2:
                    if press['AbsTime'] == row['AbsTime']:
                        first_press = trial_data.iloc[0]
                        return row['AbsTime'] - first_press['AbsTime']
                    break
        return np.nan
    
    def _get_ipi_last_to_success(self, row_idx):
        """
        Compute IPI from last press by first rat to success for a lever press.
        
        Args:
            row_idx (int): Index of the lever press in lev.data
        
        Returns:
            float: IPI or NaN if not a successful trial or not the success press
        """
        df = self.exp.lev.data
        row = df.iloc[row_idx]
        if row['coopSucc'] != 1:
            return np.nan
        
        trial_num = row['TrialNum']
        trial_data = df[df['TrialNum'] == trial_num].sort_values(by='AbsTime')
        rats_seen = set()
        presses = []
        for _, press in trial_data.iterrows():
            rat = press['RatID']
            presses.append((rat, press['AbsTime']))
            if rat not in rats_seen:
                rats_seen.add(rat)
                if len(rats_seen) == 2:
                    if press['AbsTime'] == row['AbsTime']:
                        first_rat = presses[0][0]
                        first_rat_presses = [t for r, t in presses[:-1] if r == first_rat]
                        if not first_rat_presses:
                            return np.nan
                        last_first_rat_time = max(first_rat_presses)
                        return row['AbsTime'] - last_first_rat_time
                    break
        return np.nan
    
    def _get_coop_success_zone(self, trial_num):
        """
        Get cooperative success zone status for a trial.
        
        Args:
            trial_num (int): Trial number (1-based)
        
        Returns:
            bool: True if trial is in a cooperative success zone
        """
        coop_zones = self.exp.lev.returnCooperativeSuccessRegionsBool()
        idx = int(trial_num) - 1
        return coop_zones[idx] if idx < len(coop_zones) else False
    
    def _get_rat_location(self, rat_id, abs_time):
        """
        Get the rat's location at the time of a lever press.
        
        Args:
            rat_id (int): Rat ID (0 or 1)
            abs_time (float): Absolute time of the lever press (seconds)
        
        Returns:
            str: Location ('lev_top', 'lev_bottom', 'mag_top', 'mag_bottom', 'mid', 'other')
        """
        locations = self.exp.pos.returnMouseLocation(rat_id)
        frame_idx = min(int(abs_time * self.exp.fps), len(locations) - 1)
        return locations[frame_idx] if frame_idx >= 0 else 'other'
    
    def createExpandedCSV(self, output_file):
        """
        Generate an expanded CSV with metrics for each lever press.
        
        Args:
            output_file (str): Path to save the output CSV
        """
        df = self.exp.lev.data.copy()
        total_rows = len(df)
        
        # Initialize lists for new columns
        start_times = []
        end_times = []
        ipi_rat0 = []
        ipi_rat1 = []
        ipi_general = []
        ipi_first_success = []
        ipi_last_success = []
        coop_success_zones = []
        locations_rat0 = []
        locations_rat1 = []
        
        
        #start_times = self.exp.lev.returnTimeStartTrials()
        #end_times = self.exp.lev.returnTimeEndTrials()
        #print("start_times: ", start_times)
        #print("end_times: ", end_times)
        
        # Compute metrics for each lever press
        for idx in range(total_rows):
            row = df.iloc[idx]
            trial_num = row['TrialNum']
            abs_time = row['AbsTime']
            
            # Trial times
            start_time, end_time = self._get_trial_times(trial_num)
            #print("start_time, end_time: ", start_time, end_time)
            start_times.append(start_time)
            end_times.append(end_time)
            
            # IPI metrics
            ipi_rat0.append(self._get_ipi_rat_specific(idx, 0))
            ipi_rat1.append(self._get_ipi_rat_specific(idx, 1))
            ipi_general.append(self._get_ipi_general(idx))
            ipi_first_success.append(self._get_ipi_first_to_success(idx))
            ipi_last_success.append(self._get_ipi_last_to_success(idx))
            
            # Cooperative success zone
            coop_success_zones.append(self._get_coop_success_zone(trial_num))
            
            # Locations
            locations_rat0.append(self._get_rat_location(0, abs_time))
            locations_rat1.append(self._get_rat_location(1, abs_time))
        
        # Add new columns to DataFrame
        df['TrialStartTime'] = start_times
        df['TrialEndTime'] = end_times
        df['IPI_Rat0'] = ipi_rat0
        df['IPI_Rat1'] = ipi_rat1
        df['IPI_General'] = ipi_general
        df['IPI_FirstToSuccess'] = ipi_first_success
        df['IPI_LastToSuccess'] = ipi_last_success
        df['CoopSuccessZone'] = coop_success_zones
        df['Location_Rat0'] = locations_rat0
        df['Location_Rat1'] = locations_rat1
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Expanded CSV saved to {output_file}")
        

mag_file = "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/4_nanerror_mag.csv"
lev_file = "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/4_nanerror_lev.csv"
pos_file = "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/4_nanerror_test.h5"

experiment = singleExperiment(mag_file, lev_file, pos_file)

megaCSV = megaCSVCreator(experiment)
megaCSV.createExpandedCSV("testMega.csv")












