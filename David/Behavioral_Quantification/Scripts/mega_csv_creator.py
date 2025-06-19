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
    
    def _get_trial_times(self):
        """
        Get trial start and end times from levLoader.
        
        Returns:
            tuple: (list of start times, list of end times)
        """
        return self.exp.lev.returnTimeStartTrials(), self.exp.lev.returnTimeEndTrials()
    
    def _get_ipi_rat_specific(self, rat_id):
        """
        Compute rat-specific IPI (time between last two presses by the same rat).
        
        Args:
            rat_id (int): Rat ID (0 or 1)
        
        Returns:
            list: IPI values per trial
        """
        df = self.exp.lev.data[self.exp.lev.data['RatID'] == rat_id][['TrialNum', 'AbsTime']].sort_values(by='AbsTime')
        ipis = []
        for trial_num, group in df.groupby('TrialNum'):
            times = group['AbsTime'].values
            ipi = times[-1] - times[-2] if len(times) >= 2 else np.nan
            ipis.append(ipi)
        total_trials = self.exp.lev.returnNumTotalTrials()
        result = [np.nan] * total_trials
        for i, trial_num in enumerate(df['TrialNum'].unique()):
            if i < len(ipis):
                result[int(trial_num) - 1] = ipis[i]
        return result
    
    def _get_ipi_general(self):
        """
        Compute general IPI (time between last two presses by any rat).
        
        Returns:
            list: General IPI values per trial
        """
        df = self.exp.lev.data[['TrialNum', 'AbsTime']].sort_values(by='AbsTime')
        ipis = []
        for trial_num, group in df.groupby('TrialNum'):
            times = group['AbsTime'].values
            ipi = times[-1] - times[-2] if len(times) >= 2 else np.nan
            ipis.append(ipi)
        total_trials = self.exp.lev.returnNumTotalTrials()
        result = [np.nan] * total_trials
        for i, trial_num in enumerate(df['TrialNum'].unique()):
            if i < len(ipis):
                result[int(trial_num) - 1] = ipis[i]
        return result
    
    def _get_ipi_first_to_success(self):
        """
        Get IPI from first press to success for successful trials.
        
        Returns:
            list: IPI values per trial (NaN for non-successful trials)
        """
        ipis = self.exp.lev.returnAvgIPI_FirsttoSuccess(returnList=True)
        total_trials = self.exp.lev.returnNumTotalTrials()
        result = [np.nan] * total_trials
        success_trials = self.exp.lev.data[self.exp.lev.data['coopSucc'] == 1]['TrialNum'].unique()
        for i, trial_num in enumerate(success_trials):
            if i < len(ipis):
                result[int(trial_num) - 1] = ipis[i]
        return result
    
    def _get_ipi_last_to_success(self):
        """
        Get IPI from last press to success for successful trials.
        
        Returns:
            list: IPI values per trial (NaN for non-successful trials)
        """
        ipis = self.exp.lev.returnAvgIPI_LasttoSuccess(returnList=True)
        total_trials = self.exp.lev.returnNumTotalTrials()
        result = [np.nan] * total_trials
        success_trials = self.exp.lev.data[self.exp.lev.data['coopSucc'] == 1]['TrialNum'].unique()
        for i, trial_num in enumerate(success_trials):
            if i < len(ipis):
                result[int(trial_num) - 1] = ipis[i]
        return result
    
    def _get_coop_success_zone(self):
        """
        Get boolean list of cooperative success zones.
        
        Returns:
            list: Boolean values per trial
        """
        return self.exp.lev.returnCooperativeSuccessRegionsBool()
    
    def _get_rat_location(self, rat_id):
        """
        Get the most frequent location of the rat during each trial.
        
        Args:
            rat_id (int): Rat ID (0 or 1)
        
        Returns:
            list: Most frequent location per trial
        """
        locations = self.exp.pos.returnMouseLocation(rat_id)
        trial_starts = self.exp.lev.returnTimeStartTrials()
        trial_ends = self.exp.lev.returnTimeEndTrials()
        total_trials = self.exp.lev.returnNumTotalTrials()
        result = []
        
        for trial in range(total_trials):
            start_time = trial_starts[trial]
            end_time = trial_ends[trial]
            if start_time is None or end_time is None:
                result.append('other')
                continue
            start_frame = int(start_time * self.exp.fps) if start_time is not None else 0
            end_frame = min(int(end_time * self.exp.fps), len(locations)) if end_time is not None else len(locations)
            trial_locations = locations[start_frame:end_frame]
            most_common = Counter(trial_locations).most_common(1)[0][0] if trial_locations else 'other'
            result.append(most_common)
        
        return result
    
    def createExpandedCSV(self, output_file):
        """
        Generate an expanded CSV with additional trial-level metrics.
        
        Args:
            output_file (str): Path to save the output CSV
        """
        total_trials = self.exp.lev.returnNumTotalTrials()
        trial_nums = list(range(1, total_trials + 1))
        start_times, end_times = self._get_trial_times()
        ipi_rat0 = self._get_ipi_rat_specific(0)
        ipi_rat1 = self._get_ipi_rat_specific(1)
        ipi_general = self._get_ipi_general()
        ipi_first_success = self._get_ipi_first_to_success()
        ipi_last_success = self._get_ipi_last_to_success()
        coop_success_zone = self._get_coop_success_zone()
        locations_rat0 = self._get_rat_location(0)
        locations_rat1 = self._get_rat_location(1)
        
        # Create trial-level DataFrame
        data = {
            'TrialNum': trial_nums,
            'TrialStartTime': start_times,
            'TrialEndTime': end_times,
            'IPI_Rat0': ipi_rat0,
            'IPI_Rat1': ipi_rat1,
            'IPI_General': ipi_general,
            'IPI_FirstToSuccess': ipi_first_success,
            'IPI_LastToSuccess': ipi_last_success,
            'CoopSuccessZone': coop_success_zone,
            'Location_Rat0': locations_rat0,
            'Location_Rat1': locations_rat1
        }
        
        trial_df = pd.DataFrame(data)
        
        # Merge with original lever data
        merged_df = self.exp.lev.data.merge(trial_df, on='TrialNum', how='left')
        
        # Save to CSV
        merged_df.to_csv(output_file, index=False)
        print(f"Expanded CSV saved to {output_file}")