#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 13:10:02 2025

@author: david
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import linregress, ttest_ind

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
        """Return a single column as a pandas Series for all not nan values."""
        if col not in self.data.columns:
            raise ValueError(f"Column '{col}' not found in data.")
        return self.data[col].dropna()

    def getSessionAverages(self, column: list, weigh_by_frames: bool = False):
        """
        Compute session-level averages for one or more trial-average columns.
        
        Args:
            column (list): Name of the column to average (e.g., 'percent_gazing').
            weigh_by_frames (bool): If True, weight each trial by the number of frames.
                                     If False, weight each trial equally.
        
        Returns:
            float: Session-level average
        """
        
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in data.")
        
        df = self.data.copy()
        if weigh_by_frames:
            if 'trial_frames' not in df.columns:
                raise ValueError("Column 'trial_frames' is required for frame-weighted averaging.")
            weighted_sum = (df[column] * df['trial_frames']).sum()
            total_frames = df['trial_frames'].sum()
            return weighted_sum / total_frames if total_frames > 0 else np.nan
        else:
            return df[column].mean()
    
    def getData(self, sessionID: str, trialNumber: int):
        """
        Retrieve the row of data for a specific sessionID and trialNumber.
        
        Args:
            sessionID (str): The session ID to look up (e.g., 'exp_001').
            trialNumber (int): The trial number within that session.
            
        Returns:
            pandas.Series: Row of trial data, or None if not found.
        """
        df = self.data.copy()
        row = df[(df['session_id'] == sessionID) & (df['trial_number'] == trialNumber)]
        if row.empty:
            return None
        return row.iloc[0]

    def addDerivedColumn(self, new_col_name: str, func):
        """
        Adds a new column to self.data based on a function applied to each row.
        
        Args:
            new_col_name (str): Name of the new column to create.
            func (callable): A function that takes a row (pd.Series) or row index
                             and returns the value for the new column.
                             Can use rowNum with iloc if you want.
                             
        Example:
            def findDiffGazingAndInteracting(row):
                return row['percent_gazing'] - row['percent_interacting']
            
            trialData.addDerivedColumn('gaze_minus_interact', findDiffGazingAndInteracting)
        """
        # Apply function to each row
        self.data[new_col_name] = self.data.apply(func, axis=1)
        print(f"Added derived column: '{new_col_name}'")
        return self
    
    #Graph Functions
    #
    #
    def scatterplot(self, x_col, y_col, save_csv=False, csv_path="scatterplot_data.csv", show_fit=True):
        """
        Scatterplot with linear regression, R², p-value, N, and optional CSV.
        """
        df = self.data[[x_col, y_col, 'date', 'rat_pair', 'session_id', 'trial_number']].dropna()
        x = df[x_col].values
        y = df[y_col].values
        N = len(x)
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        formula = f"y = {slope:.4f}x + {intercept:.4f}"
        R2 = r_value**2
        
        # Plot
        plt.figure(figsize=(8,6))
        sns.scatterplot(x=x, y=y)
        if show_fit:
            plt.plot(x, intercept + slope*x, 'r', label=formula)
            # Annotate R² and p-value directly on the plot
            plt.text(0.05, 0.95, f"R² = {R2:.3f}\np = {p_value:.3g}\nN = {N}", 
                     transform=plt.gca().transAxes,
                     verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
            plt.legend()
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"{y_col} vs {x_col}")
        plt.show()
        
        # Save CSV
        if save_csv:
            df_out = df.copy()
            df_out['predicted_y'] = intercept + slope*df_out[x_col]
            df_out['residual'] = df_out[y_col] - df_out['predicted_y']
            stats = pd.DataFrame({
                'stat': ['slope','intercept','R2','p_value','N'],
                'value': [slope, intercept, R2, p_value, N]
            })
            # Save as CSV with two sheets: main data + stats
            df_out.to_csv(csv_path.replace(".csv","_data.csv"), index=False)
            stats.to_csv(csv_path.replace(".csv","_stats.csv"), index=False)
            print(f"Scatterplot data saved to {csv_path.replace('.csv','_data.csv')}")
            print(f"Scatterplot stats saved to {csv_path.replace('.csv','_stats.csv')}")
        
        return slope, intercept, R2, p_value, N
    
    
        
    
    
    #Special Data Retrieval Functions
    #
    #
    
    
#Class Use
#
#

megaCSV = "/Users/david/Downloads/trial_metrics_standard_full.csv"

trialData = allDataTrials(megaCSV)

    