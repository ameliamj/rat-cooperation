#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 15:06:01 2025

@author: david
"""

import pandas as pd
#import uuid


#Class
#
#

class magLoader: 
    
    def __init__(self, filename, fps = 30): #Constructor
        self.filename = filename
        self.data = None
        self._load_data()
        #print("Data Before: ")
        #print(self.data)
        self._delete_bad_data()
        #print("Data after: ")
        #print(self.data)
        self.categories = ["TrialNum", "MagNum", "AbsTime", "Duration", "TrialCond", "DispTime", "TrialTime", "Hit", "TrialEnd", "RatID"]
        self.fps = fps
        
    def _load_data(self): #Uses pandas to read csv file and store in a pandas datastructure
        """
        Load the CSV file into a pandas DataFrame.
        Handles file not found or malformed CSV errors.
        """
        
        try:
            self.data = pd.read_csv(self.filename, sep=',', na_values=[''])
            # Ensure numeric columns are properly typed
            for col in self.data.columns:
                if self.data[col].dtype == 'object':
                    # Try converting to numeric, but preserve strings if not possible
                    try:
                        self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                    except:
                        pass
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at: {self.filename}")
        except pd.errors.ParserError:
            raise ValueError("Error parsing CSV file. Ensure it is properly formatted.")
            
    def _delete_bad_data(self):    
        """
        Delete rows where MagNum or AbsTime are null, 
        but only if those columns exist.
        """
        if self.data is not None:
            required_columns = ['MagNum', 'AbsTime']
            existing_columns = [col for col in required_columns if col in self.data.columns]
            
            if existing_columns:
                #print(f"Dropping rows with NaN in columns: {existing_columns}")
                self.data = self.data.dropna(subset=existing_columns)
            else:
                print("Warning: Neither 'MagNum' nor 'AbsTime' columns found in mag data. Skipping dropna.")
            
    def get_value(self, row_idx, col): #Gets a value in any row/col of the data
        """
        Access the value at the specified row index and column (index or name).
        
        Args:
            row_idx (int): The 0-based row index.
            col (int or str): The 0-based column index or column name.
        
        Returns:
            The value at the specified position.
        
        Raises:
            ValueError: If the row index or column is invalid or data is not loaded.
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        if not isinstance(row_idx, int) or row_idx < 0 or row_idx >= len(self.data):
            raise ValueError(f"Invalid row index: {row_idx}. Must be between 0 and {len(self.data)-1}.")
        
        if isinstance(col, str):
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in data.")
            return self.data.at[row_idx, col]
        elif isinstance(col, int):
            if col < 0 or col >= len(self.data.columns):
                raise ValueError(f"Invalid column index: {col}. Must be between 0 and {len(self.data.columns)-1}.")
            return self.data.iloc[row_idx, col]
        else:
            raise ValueError("Column must be an integer index or string name.")
    
    def getNumRows(self): #Gets number of rows in the data
        """
        Return the number of rows in the DataFrame.
        
        Returns:
            int: The number of rows.
        
        Raises:
            ValueError: If no data is loaded.
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        return len(self.data)
    
    def getColumnData(self, column): #Input Column Name (eg: "TrialNum") and return the entire column
        """
        Return data for a specific column.
        
        Args:
            column (str): The column name.
        
        Returns:
            pandas.Series: The column data.
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in data.")
        return self.data[column]
    
    def countNullsinColumn(self, column): #Input Column Name (eg: "TrialNum") and return the number of nulls in the column
            """
            Count the number of null entries in the specified column.
            
            Args:
                column (str): The column name.
            
            Returns:
                int: The number of null entries in the column.
            
            Raises:
                ValueError: If the column is invalid or data is not loaded.
            """
            if self.data is None:
                raise ValueError("No data loaded.")
            if column not in self.data.columns:
                raise ValueError(f"Column '{column}' not found in mag data from file '{self.filename}'.")
            return self.data[column].isna().sum()

    def getTrialNum(self, row_idx): #returns the trial number for row_idx of the data
        return self.get_value(row_idx, 0)
    
    def getMagNum(self, row_idx): #returns the mag num for row_idx of the data
        return self.get_value(row_idx, 1)

    #Graph Stuff: 
    
    def getTotalMagEvents(self):
        total_mag_events = self.data.shape[0]
        return total_mag_events
    
    def getTotalMagEventsFiltered(self):
        '''
        Same as getTotalMagEvents but you delete any rows in which there's missing RatID'
        '''
        df = self.data.copy()
        
        if df is not None:
            required_columns = ['RatID', 'AbsTime']
            existing_columns = [col for col in required_columns if col in df.columns]
            
            if existing_columns:
                print(f"Dropping rows with NaN in columns: {existing_columns}")
                df = df.dropna(subset=existing_columns)
            else:
                print("Warning: Neither 'RatID' nor 'AbsTime' columns found in mag data. Skipping dropna.")
        
        total_mag_events_filtered = df.shape[0]
        return total_mag_events_filtered
    
    def returnMostEntriesbyMag(self, ratID):
        """
        Finds the number of entries by the specified rat on Mag 1 and Mag 2. Then returns the higher of the 2
        
        Args:
            ratID (int): The ID of the rat to filter presses for.
        
        Returns:
            max(num_mag1_entries, num_mag2_entries)
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        
        if 'RatID' not in self.data.columns or 'MagNum' not in self.data.columns:
            raise ValueError("Required columns 'RatID' or 'MagNum' are missing from data.")
        
        rat_data = self.data[self.data['RatID'] == ratID]
        
        num_mag1_entries = (rat_data['MagNum'] == 1).sum()
        num_mag2_entries = (rat_data['MagNum'] == 2).sum()
        return max(num_mag1_entries, num_mag2_entries)

    def returnRewardRecipient(self, trial_index):
        """
        Returns the RatIDs of the rats that collected rewards from Mag 1 and Mag 2 for a given trial,
        using the 'Hit' category to confirm reward collection.

        Args:
            trial_index (int): The 0-based index of the trial (TrialNum = trial_index + 1).

        Returns:
            list: A list of two RatIDs [RatID_Mag1, RatID_Mag2], or None if the trial is invalid.

        Raises:
            ValueError: If no data is loaded or the trial_index is invalid.
        """
        if self.data is None:
            raise ValueError("No data loaded.")

        trial_num = trial_index + 1  # Convert 0-based index to 1-based TrialNum

        # Filter data for the specified trial
        trial_data = self.data[self.data['TrialNum'] == trial_num]

        if trial_data.empty:
            return None

        # Check for required columns
        if not {'MagNum', 'RatID', 'Hit'}.issubset(trial_data.columns):
            return None

        # Filter for events where Hit == 1 (reward collected)
        hit_data = trial_data[trial_data['Hit'] == 1]

        if hit_data.empty:
            return None

        # Get events for Mag 1 and Mag 2 with Hit == 1
        mag1_data = hit_data[hit_data['MagNum'] == 1]
        mag2_data = hit_data[hit_data['MagNum'] == 2]

        # Ensure exactly one event per magazine and non-null RatID
        if mag1_data.empty or mag2_data.empty:
            return None

        mag1_rat = mag1_data['RatID'].iloc[0]  # Take first event if multiple
        mag2_rat = mag2_data['RatID'].iloc[0]  # Take first event if multiple

        if pd.isna(mag1_rat) or pd.isna(mag2_rat):
            return None

        return [mag1_rat, mag2_rat]

    def getRewardReceivedFrames(self, rat_id):
        """
        Returns a set of frame indices where the specified rat received a reward.
        Assumes reward is received at AbsTime when RatID matches and Hit == 1.
        """
        if self.data is None or self.data.empty:
            return set()
        reward_data = self.data[(self.data['Hit'] == 1) & (self.data['RatID'] == rat_id)]
        frames = (reward_data['AbsTime'] * self.fps).astype(int)
        return set(frames)
    
    def getEnteredMagFrames(self, rat_id):
        if self.data is None or self.data.empty:
            return set()
        reward_data = self.data[(self.data['Hit'] == 0) & (self.data['RatID'] == rat_id)]
        frames = (reward_data['AbsTime'] * self.fps).astype(int)
        return set(frames)


#Testing
#
#

#file1 = "/Users/david/Documents/Research/Saxena Lab/Behavioral_Quantification/Example Data Files/ExampleMagFile.csv"
#mag1 = magLoader(file1)

#print(mag1.getTrialNum(0))
#print(mag1.getTrialNum(20))

