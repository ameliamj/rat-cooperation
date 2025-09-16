#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 13:10:02 2025

@author: david
"""

import pandas as pd

class allDataTrials:
    def __init__(self, csv_path: str):
        """
        Load CSV into a pandas DataFrame.
        """
        self.data = pd.read_csv(csv_path)
        
        
    #SORTING Functions: Get the data you want. 
    #
    #
    
    def keepOnlyRangeTrials(self, start: int, end: int):
        """Keep only trials in the given trial_number range (inclusive)."""
        self.data = self.data[(self.data['trial_number'] >= start) & 
                              (self.data['trial_number'] <= end)]
        return self

    def keepOnlyPairedTesting(self):
        """Keep only trials that have a rat pair recorded (non-null)."""
        self.data = self.data[self.data['rat_pair'].notna()]
        return self

    def keepOnlyTrials(self, trial_list):
        """Keep only the trials whose numbers are in the given list."""
        self.data = self.data[self.data['trial_number'].isin(trial_list)]
        return self

    def keepOnlyTransparent(self):
        self.data = self.data[self.data['barrier_transparency'] == 'transparent']
        return self

    def keepOnlyTranslucent(self):
        self.data = self.data[self.data['barrier_transparency'] == 'translucent']
        return self

    def keepOnlyOpaque(self):
        self.data = self.data[self.data['barrier_transparency'] == 'opaque']
        return self

    def keepOnlyTrainingPartners(self):
        self.data = self.data[self.data['familiarity'] == 'Training Partner']
        return self

    def keepOnlyUnfamiliar(self):
        self.data = self.data[self.data['familiarity'] == 'Unfamiliar']
        return self

    def keepOnlyValidLeverPress(self):
        self.data = self.data[self.data['lever_press_exists'] == True]
        return self

    def keepOnlyAllStages(self):
        self.data = self.data[self.data['all_time_sections_valid'] == True]
        return self

    def keepOnlyRat(self, ratPair: str):
        """Keep only trials belonging to a specific rat pair."""
        self.data = self.data[self.data['rat_pair'] == ratPair]
        return self

    def keepOnlyCooperativeTrials(self):
        self.data = self.data[self.data['success'] == 1]
        return self

    def keepOnlyNonCooperativeTrials(self):
        self.data = self.data[self.data['success'] == 0]
        return self

    def get(self):
        """Return the filtered DataFrame."""
        return self.data.copy()

    def reset(self, csv_path: str):
        """Reset the data back to original CSV."""
        self.data = pd.read_csv(csv_path)
        return self
    
    
    
    # General Data Retrieval Functions
    #
    #
    def getColumn(self, col: str):
        """Return a single column as a pandas Series."""
        if col not in self.data.columns:
            raise ValueError(f"Column '{col}' not found in data.")
        return self.data[col].dropna()

    def getColumns(self, cols: list):
        """Return multiple columns as a DataFrame."""
        missing = [c for c in cols if c not in self.data.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}")
        return self.data[cols].dropna(how="all")
    
    
    
    
    #Graph Functions
    #
    #
    
    
    
    #Special Data Retrieval Functions
    #
    #
    
    
#Class Use
#
#

megaCSV = "/Users/david/Downloads/trial_metrics_standard_full.csv"

trialData = allDataTrials(megaCSV)

    