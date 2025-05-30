#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 15:17:08 2025

@author: david
"""

import pandas as pd
import numpy as np

class fileExtractor:
    def __init__(self, information_path):
        self.filename = information_path
        self.data = None
        self._load_data()
        
        
    def _load_data(self):
        """
        Load the CSV file in information_path into a pandas DataFrame.
        Handles file not found or malformed CSV errors.
        """
        try:
            self.data = pd.read_csv(self.filename, sep=',', na_values=[''])
            
            # If you want to attempt type inference, use this instead:
            self.data = self.data.convert_dtypes()
    
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at: {self.filename}")
        except pd.errors.ParserError:
            raise ValueError("Error parsing CSV file. Ensure it is properly formatted.")

    def writeNewFixedFile(self): #Write a new csv and save it working directory where it's the same except that for any row where tial type = coop and familiarity is a NaN, change familiarity to TP
        """
        Write a new CSV in the working directory where:
        - If trial type == 'coop' and familiarity is NaN, change familiarity to 'TP'
        - For trial type == 'coop', set color pair to last 13 chars of 'vid'
        - Add a new column 'dividers' for coop trials only, with values:
            'transparent', 'translucent', or 'opaque' based on 'session'
        - Save result to 'fixed_file.csv'
        """
        df_copy = self.data.copy()
    
        # Ensure familiarity is an object column
        df_copy['familiarity'] = df_copy['familiarity'].astype('object')
    
        # Condition for 'coop' trial type
        coop_condition = df_copy['trial type'] == 'coop'
    
        # Fix missing familiarity in coop rows
        df_copy.loc[coop_condition & df_copy['familiarity'].isna(), 'familiarity'] = 'TP'
    
        # Set 'color pair' from last 13 characters of 'vid' for coop rows
        df_copy.loc[coop_condition, 'color pair'] = df_copy.loc[coop_condition, 'vid'].str[-13:]
    
        # Initialize 'dividers' as blank (object type)
        df_copy['dividers'] = np.nan
    
        # Safely cast 'session' to string
        session_col = df_copy['session'].astype(str)
    
        # Assign divider types only to coop rows
        df_copy.loc[coop_condition & session_col.str.endswith('opaque'), 'dividers'] = 'opaque'
        df_copy.loc[coop_condition & session_col.str.endswith('translucent'), 'dividers'] = 'translucent'
        df_copy.loc[coop_condition & ~session_col.str.endswith(('opaque', 'translucent')), 'dividers'] = 'transparent'
    
        # Save to file
        df_copy.to_csv("dyed_preds_fixed_expanded.csv", index=False)
    
    def getCategories(self): # returns a List with the names of all the categories
        """
        Return a list of all the column/category names in the dataset.
        """
        return list(self.data.columns)
    
    def deleteInvalid(self): # gets rid of all rows where trial type != coop, or levers = false, or mags = false
        """
        Remove rows where:
        - trial type is not 'coop'
        - OR levers is not TRUE
        - OR mags is not TRUE
        """
        self.data = self.data[
            (self.data['trial type'] == 'coop') &
            (self.data['levers'] == True) &
            (self.data['mags'] == True)
        ]
        
        df_copy = self.data.copy()
        df_copy.to_csv("dyed_preds_all_valid.csv", index=False)

    def getTrainingPartner(self): # gets rid of all rows where familiarity != TP
        """
        Keep only rows where familiarity == 'TP'
        """
        self.data = self.data[self.data['familiarity'] == 'TP']
        df_copy = self.data.copy()
        df_copy.to_csv("only_training_partners.csv", index=False)
        
    def getUnfamiliarPartners(self): # gets rid of all rows where familiarity != UF
        """
        Keep only rows where familiarity == 'UF'
        """
        self.data = self.data[self.data['familiarity'] == 'UF']
        df_copy = self.data.copy()
        df_copy.to_csv("only_unfamiliar_partners.csv", index=False)
    
    def getTransparentSessions(self):
        """
        Remove all rows where 'session' ends with 'opaque' or 'translucent'
        (i.e., retain only transparent divider sessions).
        """
        self.data = self.data[
            ~self.data['session'].str.endswith(('opaque', 'translucent'), na=False)
        ]
        df_copy = self.data.copy()
        df_copy.to_csv("only_transparent_sessions.csv", index=False)

    def getTranslucentSessions(self):
        """
        Keep only rows where 'session' ends with 'translucent'.
        """
        self.data = self.data[
            self.data['session'].str.endswith('translucent', na=False)
        ]
        df_copy = self.data.copy()
        df_copy.to_csv("only_translucent_sessions.csv", index=False)

    def getOpaqueSessions(self):
        """
        Keep only rows where 'session' ends with 'opaque'.
        """
        self.data = self.data[
            self.data['session'].str.endswith('opaque', na=False)
        ]
        df_copy = self.data.copy()
        df_copy.to_csv("only_opaque_sessions.csv", index=False)

    def sortByMicePairs(self):
        """
        Groups rows by mice pair (the last 13 characters of the 'vid' string).
        Returns a list of DataFrames, each containing rows with the same pair.
        """
        # Extract mice pair key from 'vid' column
        self.data['mice_pair'] = self.data['vid'].str[-13:]

        # Group by this key and return a list of DataFrames (or lists, if preferred)
        grouped = [group_df for _, group_df in self.data.groupby('mice_pair')]

        # Drop the temporary 'mice_pair' column
        self.data.drop(columns='mice_pair', inplace=True)
        df_copy = self.data.copy()
        df_copy.to_csv("group_mice_pairs.csv", index=False)

        return grouped
    
    def getCorrectFileNames(self):
        def correct_name(row):
            if row['test/train'] == 'test':
                return row['vid']
            elif row['test/train'] == 'train':
                original = row['vid']
                if len(original) < 41:
                    return original  # fallback for malformed or short vids
                # First 7 chars
                part1 = original[:7]
                # Skip 21 characters (from index 7 to 28), take next 13 (index 28 to 41)
                part2 = original[28:41]
                return part1 + part2
            else:
                return None  # or raise an error/log

        self.data['correct_name'] = self.data.apply(correct_name, axis=1)
    
    def getMagsDatapath(self, grouped = False):
        """
        Return a list of datapaths to mag files for each row in self.data.
        
        - If grouped=False: return a flat list.
        - If grouped=True: use sortBySameMicePairs to return a nested list where
          each sublist corresponds to all magfiles for a given mice pair.
    
        Path format:
        /gpfs/radev/pi/saxena/aj764/{folderName}/{sessionName}/Behavioral/Processed/mag/{vidName}_mag.csv
        """
        base_path = "/gpfs/radev/pi/saxena/aj764"
        
        self.getCorrectFileNames()
        
        def construct_path(row):
            folder = "PairedTestingSessions" if row["test/train"] == "test" else "Training_COOPERATION"
            processed = "/processed" if row["test/train"] == "test" else ""
            return f"{base_path}/{folder}/{row['session']}/Behavioral{processed}/mag/{row['correct_name']}_mag.csv" if pd.isna(row["vid"]) == False and pd.isna(row["session"]) == False else None
    
        if not grouped:
            return [
                construct_path(row)
                for _, row in self.data.iterrows()
            ]
        else:
            grouped_rows = self.sortByMicePairs()
            grouped_paths = []
            for group in grouped_rows:
                group_paths = []
                for _, row in group.iterrows():
                    group_paths.append(construct_path(row))
                grouped_paths.append(group_paths)
            return grouped_paths
            
    def getLevsDatapath(self, grouped = False):
        """
        Return a list of datapaths to lev files for each row in self.data.
        
        - If grouped=False: return a flat list.
        - If grouped=True: use sortBySameMicePairs to return a nested list where
          each sublist corresponds to all levFiles for a given mice pair.
    
        Path format:
        /gpfs/radev/pi/saxena/aj764/{folderName}/{sessionName}/Behavioral/Processed/lever/{vidName}_lever.csv
        """
        base_path = "/gpfs/radev/pi/saxena/aj764"
        
        self.getCorrectFileNames()
        
        def construct_path(row):
            folder = "PairedTestingSessions" if row["test/train"] == "test" else "Training_COOPERATION"
            processed = "/processed" if row["test/train"] == "test" else ""
            return f"{base_path}/{folder}/{row['session']}/Behavioral{processed}/lever/{row['correct_name']}_lever.csv" if pd.isna(row["vid"]) == False and pd.isna(row["session"]) == False else None
    
        if not grouped:
            return [
                construct_path(row)
                for _, row in self.data.iterrows()
            ]
        else:
            grouped_rows = self.sortByMicePairs()
            grouped_paths = []
            for group in grouped_rows:
                group_paths = []
                for _, row in group.iterrows():
                    group_paths.append(construct_path(row))
                grouped_paths.append(group_paths)
            return grouped_paths
    
    
#information_path = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/scripts/notebooks/dyed_preds_df.csv"
#information_path = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral Quantification/Sorted Data Files/dyed_preds_df_fixed.csv"
information_path = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral Quantification/Sorted Data Files/dyed_preds_fixed_expanded.csv"

all_valid = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral Quantification/Sorted Data Files/dyed_preds_all_valid.csv"
only_opaque = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral Quantification/Sorted Data Files/only_opaque_sessions.csv"
only_training = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral Quantification/Sorted Data Files/only_training_partners.csv"
only_translucent = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral Quantification/Sorted Data Files/only_translucent_sessions.csv"
only_transparent = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral Quantification/Sorted Data Files/only_transparent_sessions.csv"
only_unfamiliar = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral Quantification/Sorted Data Files/only_unfamiliar_partners.csv"
only_trainingpartners = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral Quantification/Sorted Data Files/only_training_partners.csv"

#fe = fileExtractor(all_valid)
#fe.deleteInvalid()
#print(fe.getMagsDatapath())






