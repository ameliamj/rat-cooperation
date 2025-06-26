#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 15:13:09 2025

@author: david
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

class fiberPhotoLoader:
    def __init__(self, x405_path, x465_path, x560_path):
        '''
        Reads and Stores the x405, x465, and x560 data around 4 events.
        
        Events: 
            1) ttlLabels('1') = 'Session Start'
            2) ttlLabels('2') = 'Left Lever Press'
            3) ttlLabels('4') = 'Right Lever Press'
            4) ttlLabels('8') = 'Left Magazine Entry'
            5) ttlLabels('16') = 'Right Magazine Entry'
        
        Column Labels in CSV: ['code', 'ts', '1', '2', ..., '1526']
        '''
        
        self.x405_path = x405_path
        self.x465_path = x465_path
        self.x560_path = x560_path
        
        self.x405 = None
        self.x465 = None
        self.x560 = None
        
        self._load_data()
        self.subtractMeanFromData()
        
    def _load_data(self): 
        """
        Load each CSV file into a pandas DataFrame.
        Handles file not found or malformed CSV errors.
        Converts numeric columns where possible.
        """
        
        # Load x405
        try:
            self.x405 = pd.read_csv(self.x405_path, sep=',', na_values=[''])
            # Convert object columns to numeric if possible
            for col in self.x405.columns:
                if self.x405[col].dtype == 'object':
                    try:
                        self.x405[col] = pd.to_numeric(self.x405[col], errors='coerce')
                    except Exception:
                        pass
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at: {self.x405_path}")
        except pd.errors.ParserError:
            raise ValueError("Error parsing x405 CSV file. Ensure it is properly formatted.")
            
        # Load x465
        try:
            self.x465 = pd.read_csv(self.x465_path, sep=',', na_values=[''])
            for col in self.x465.columns:
                if self.x465[col].dtype == 'object':
                    try:
                        self.x465[col] = pd.to_numeric(self.x465[col], errors='coerce')
                    except Exception:
                        pass
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at: {self.x465_path}")
        except pd.errors.ParserError:
            raise ValueError("Error parsing x465 CSV file. Ensure it is properly formatted.")
        
        # Load x560
        try:
            self.x560 = pd.read_csv(self.x560_path, sep=',', na_values=[''])
            for col in self.x560.columns:
                if self.x560[col].dtype == 'object':
                    try:
                        self.x560[col] = pd.to_numeric(self.x560[col], errors='coerce')
                    except Exception:
                        pass
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at: {self.x560_path}")
        except pd.errors.ParserError:
            raise ValueError("Error parsing x560 CSV file. Ensure it is properly formatted.")
    
    def _extract_event_data(self, df, event_codes):
        """
        Helper function to filter the dataframe based on event codes and then return a nested list.
        Each row in the returned list is a list with the first element the value of 'ts' and the 
        subsequent 1526 elements the values from columns '1' to '1526'.
        """
        # Filter rows where the 'code' column is one of the event codes
        df_event = df[df['code'].isin(event_codes)]
        
        # Define the column order we want: first 'ts', then columns '1' ... '1526'
        # (ignoring 'code', which was only used for filtering)
        numeric_cols = list(map(str, range(1, 1527)))  # creates ['1', '2', ..., '1526']
        col_order = ['ts'] + numeric_cols
        
        # Check that all expected columns are present
        missing_cols = set(col_order) - set(df_event.columns)
        if missing_cols:
            raise ValueError(f"The following expected columns are missing: {missing_cols}")
        
        # Extract the desired columns and convert to a nested list of lists.
        # Each row: [ts, value1, value2, ..., value1526]
        data_list = df_event[col_order].values.tolist()
        return data_list
    
    def subtractMeanFromData(self):
        '''
        Computes the mean of non-NaN values in columns '1' to '1526' for each DataFrame
        (x405, x465, x560) and subtracts these means from the respective columns to normalize
        the data in place.
        '''
        # Define the columns to process
        numeric_cols = [str(i) for i in range(1, 1527)]
        
        # Process x405
        if self.x405 is not None:
            # Check if expected columns exist
            available_cols = [col for col in numeric_cols if col in self.x405.columns]
            if available_cols:
                # Compute mean of all non-NaN values in numeric columns
                mean_405 = self.x405[available_cols].values.flatten()
                mean_405 = np.nanmean(mean_405) if len(mean_405) > 0 else 0
                # Subtract mean from each numeric column
                for col in available_cols:
                    self.x405[col] = self.x405[col] - mean_405
            else:
                print("Warning: No numeric columns (1 to 1526) found in x405 DataFrame")
        
        # Process x465
        if self.x465 is not None:
            available_cols = [col for col in numeric_cols if col in self.x465.columns]
            if available_cols:
                mean_465 = self.x465[available_cols].values.flatten()
                mean_465 = np.nanmean(mean_465) if len(mean_465) > 0 else 0
                for col in available_cols:
                    self.x465[col] = self.x465[col] - mean_465
            else:
                print("Warning: No numeric columns (1 to 1526) found in x465 DataFrame")
        
        # Process x560
        if self.x560 is not None:
            available_cols = [col for col in numeric_cols if col in self.x560.columns]
            if available_cols:
                mean_560 = self.x560[available_cols].values.flatten()
                mean_560 = np.nanmean(mean_560) if len(mean_560) > 0 else 0
                for col in available_cols:
                    self.x560[col] = self.x560[col] - mean_560
            else:
                print("Warning: No numeric columns (1 to 1526) found in x560 DataFrame")
    
    def getLev405(self):
        '''
        Uses self.x405, filters by rows where 'code' == 2, -2, 4, or -4, and returns a nested 
        list (one list per row) with the first element being the 'ts' value and the next 1526 values 
        from columns '1' to '1526'.
        '''
        lever_codes = [2, -2, 4, -4]
        return self._extract_event_data(self.x405, lever_codes)
    
    def getLev465(self):
        '''
        Uses self.x465, filters by rows where 'code' == 2, -2, 4, or -4, and returns a nested 
        list with the first element being the 'ts' value and the next 1526 values from columns '1' 
        to '1526'.
        '''
        lever_codes = [2, -2, 4, -4]
        return self._extract_event_data(self.x465, lever_codes)
        
    def getLev560(self):
        '''
        Uses self.x560, filters by rows where 'code' == 2, -2, 4, or -4, and returns a nested 
        list with the first element being the 'ts' value and the next 1526 values from columns '1' 
        to '1526'.
        '''
        lever_codes = [2, -2, 4, -4]
        return self._extract_event_data(self.x560, lever_codes)
    
    def getMag405(self):
        '''
        Uses self.x405, filters by rows where 'code' == 8, -8, 16, or -16, and returns a nested 
        list with the first element being the 'ts' value and the next 1526 values from columns '1' 
        to '1526'.
        '''
        magazine_codes = [8, -8, 16, -16]
        return self._extract_event_data(self.x405, magazine_codes)
     
    def getMag465(self):
        '''
        Uses self.x465, filters by rows where 'code' == 8, -8, 16, or -16, and returns a nested 
        list with the first element being the 'ts' value and the next 1526 values from columns '1' 
        to '1526'.
        '''
        magazine_codes = [8, -8, 16, -16]
        return self._extract_event_data(self.x465, magazine_codes)
         
    def getMag560(self):
        '''
        Uses self.x560, filters by rows where 'code' == 8, -8, 16, or -16, and returns a nested 
        list with the first element being the 'ts' value and the next 1526 values from columns '1' 
        to '1526'.
        '''
        magazine_codes = [8, -8, 16, -16]
        return self._extract_event_data(self.x560, magazine_codes)
    
    def getSumandEles(self, arr):
        '''
        Gets mean of nestedArr ignoring the first index of each nested list
        '''
        sumNums = 0
        numEles = 0
        
        for i in range(len(arr)):
            for j in range(1, 1527):
                if (not math.isnan(arr[i][j])):
                    sumNums += arr[i][j]
                    numEles += 1
        
        return [sumNums, numEles]

x405 = "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/090324_Cam1_TrNum14_Coop_KL002B-KL002Y_x405_TTLs.csv"
x465 = "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/090324_Cam1_TrNum14_Coop_KL002B-KL002Y_x465_TTLs.csv"
x560 = "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/090324_Cam1_TrNum14_Coop_KL002B-KL002Y_x560_TTLs.csv"

loader = fiberPhotoLoader(x405, x465, x560)



    