#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 15:17:08 2025

@author: david
"""

import pandas as pd
import cv2
import numpy as np
import re

class fileExtractor:
    def __init__(self, information_path):
        self.filename = information_path
        self.data = None
        self._load_data()
        self.preFix = "_filtered"
         
    def _load_data(self):
        """
        Load the CSV file in information_path into a pandas DataFrame.
        Handles file not found or malformed CSV errors.
        """
        try:
            self.data = pd.read_csv(self.filename, sep=',', na_values=[''], dtype = {'session' : str, 'vid' : str, 'single/multi' : str, 'test/train' : str})
            
            #self.data = pd.read_csv(self.filename, sep=',', na_values=[''])
            #self.data = self.data.convert_dtypes()
    
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
        - Save result to 'dyed_preds_fixed_expanded.csv'
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
    
        # Initialize 'dividers' as an empty object (string-compatible) column
        df_copy['dividers'] = pd.Series(dtype='object')
    
        # Safely cast 'session' to string
        session_col = df_copy['session'].astype(str)
    
        # Assign divider types only to coop rows
        df_copy.loc[coop_condition & session_col.str.endswith('Opaque'), 'dividers'] = 'opaque'
        df_copy.loc[coop_condition & session_col.str.endswith('Translucent'), 'dividers'] = 'translucent'
        df_copy.loc[coop_condition & ~session_col.str.endswith(('Opaque', 'Translucent')), 'dividers'] = 'transparent'
    
        # Save to file
        df_copy.to_csv("dyed_preds_fixed_expanded.csv", index=False)
    
    def getCategories(self): # returns a List with the names of all the categories
        """
        Return a list of all the column/category names in the dataset.
        """
        return list(self.data.columns)
    
    def deleteBadNaN(self):
        self.data = self.data[self.data['initial nan'] <= 0.2]
        
        return self.data
    
    def deleteInvalid(self, saveFile = False): # gets rid of all rows where trial type != coop, or levers = false, or mags = false
        """
        Remove rows where:
        - trial type is not 'coop'
        - OR levers is not TRUE
        - OR mags is not TRUE
        """
        self.data = self.data[
            (self.data['trial type'] == 'coop') &
            (self.data['levers'] == True) &
            (self.data['mags'] == True) &
            (self.data['correct'] == True)
        ]
        
        self.data = self.deleteBadNaN()
        
        df_copy = self.data.copy()
        if (saveFile):
            df_copy.to_csv("dyed_preds_all_valid.csv", index=False)

    def deleteAllButFirstPTvsTC(self, saveFile = False):
        """
        For each unique mice pair, retain only:
        - The first row (by date and TrNum) where 'test/train' == 'test'
        - The first row (by date and TrNum) where 'test/train' == 'train'
        Delete all other rows.
        """
        grouped = self.sortByMicePairs()
        first_rows = []
    
        for group in grouped:
            # Find first test session
            test_rows = group[group['test/train'] == 'test']
            if not test_rows.empty:
                first_rows.append(test_rows.iloc[0])
    
            # Find first train session
            train_rows = group[group['test/train'] == 'train']
            if not train_rows.empty:
                first_rows.append(train_rows.iloc[0])
    
        self.data = pd.DataFrame(first_rows).reset_index(drop=True)
        if (saveFile):
            df_copy = self.data.copy()
            df_copy.to_csv("dyed_preds_all_valid.csv", index=False)
            
    
    def deleteAllButFirstTransparency(self, saveFile = False):
        """
        For each unique mice pair, retain only:
        - The first row (by date and TrNum) where 'test/train' == 'test'
        - The first row (by date and TrNum) where 'test/train' == 'train'
        Delete all other rows.
        """
        grouped = self.sortByMicePairs()
        first_rows = []
    
        for group in grouped:
            # Find first transparent session
            test_rows = group[group['dividers'] == 'transparent']
            if not test_rows.empty:
                first_rows.append(test_rows.iloc[0])
    
            # Find first translucent session
            train_rows = group[group['dividers'] == 'translucent']
            if not train_rows.empty:
                first_rows.append(train_rows.iloc[0])
                
            # Find first opaque session
            train_rows = group[group['dividers'] == 'opaque']
            if not train_rows.empty:
                first_rows.append(train_rows.iloc[0])
    
        self.data = pd.DataFrame(first_rows).reset_index(drop=True)
        if (saveFile):
            df_copy = self.data.copy()
            df_copy.to_csv("dyed_preds_all_valid.csv", index=False)
            
    def deleteAllButFirstFamiliarity(self, saveFile = False):
        """
        For each unique mice pair, retain only:
        - The first row (by date and TrNum) where 'test/train' == 'test'
        - The first row (by date and TrNum) where 'test/train' == 'train'
        Delete all other rows.
        """
        grouped = self.sortByMicePairs()
        first_rows = []
    
        for group in grouped:
            # Find first unfamiliar session
            test_rows = group[group['familiarity'] == 'UF']
            if not test_rows.empty:
                first_rows.append(test_rows.iloc[0])
    
            # Find first training partner session
            train_rows = group[group['familiarity'] == 'TP']
            if not train_rows.empty:
                first_rows.append(train_rows.iloc[0])
    
        self.data = pd.DataFrame(first_rows).reset_index(drop=True)
        if (saveFile):
            df_copy = self.data.copy()
            df_copy.to_csv("dyed_preds_all_valid.csv", index=False)
    

    def getFirstSessionPerMicePair(self):
        """
        Uses sortByMicePairs to get the first session (earliest date + TrNum)
        for each unique mice pair.
        Returns a DataFrame containing only the first session per pair.
        """
        grouped = self.sortByMicePairs()
        
        # Extract the first row of each group
        first_sessions = [group.iloc[0] for group in grouped if not group.empty]
        
        
        # Combine them into a single DataFrame and assign back to self.data
        self.data = pd.DataFrame(first_sessions).reset_index(drop=True)
        # Convert list of Series to a DataFrame
        #return pd.DataFrame(first_sessions)
        
        #Save
        #df_copy = self.data.copy()
        #df_copy.to_csv("onlyFirstSession.csv", index=False)

    def getTrainingCoopSessions(self, sortOut = True, saveFile = False):
        """
        Keep only rows where test/train == 'train'
        """
        if (sortOut):
            #self.deleteAllButFirstPTvsTC()
            self.getTrainingPartner(sortOut=False)
            self.getTransparentSessions(sortOut=False)
            
        self.data = self.data[self.data['test/train'] == 'train']
        
        if (saveFile):
            df_copy = self.data.copy()
            name = ""
            if (sortOut):
               name = self.preFix
            df_copy.to_csv(f"only_TrainingCooperation{name}.csv", index=False)
        
    def getPairedTestingSessions(self, sortOut = True, saveFile = False):
        """
        Keep only rows where test/train == 'test'
        """
        if (sortOut):
            #self.deleteAllButFirstPTvsTC()
            self.getTrainingPartner(sortOut=False)
            self.getTransparentSessions(sortOut=False)
        
        self.data = self.data[self.data['test/train'] == 'test']
        
        if (saveFile):
            df_copy = self.data.copy()
            name = ""
            if (sortOut):
                name = self.preFix
            df_copy.to_csv(f"only_PairedTesting{name}.csv", index=False)
    
        
    def getTrainingPartner(self, sortOut = True, saveFile = False): # gets rid of all rows where familiarity != TP
        """
        Keep only rows where familiarity == 'TP'
        """
        if (sortOut):
            self.getPairedTestingSessions(sortOut=False)
            self.getTransparentSessions(sortOut=False)
            #self.deleteAllButFirstFamiliarity()
            #print("deleted all but first")
        
        self.data = self.data[self.data['familiarity'] == 'TP']
        
        if (saveFile):
            df_copy = self.data.copy()
            name = ""
            if (sortOut):
                name = self.preFix
            df_copy.to_csv(f"only_training_partners{name}.csv", index=False)
        
    def getUnfamiliarPartners(self, sortOut = True, saveFile = False): # gets rid of all rows where familiarity != UF
        """
        Keep only rows where familiarity == 'UF'
        """
        if (sortOut):
            self.getPairedTestingSessions(sortOut=False)
            self.getTransparentSessions(sortOut=False)
            #self.deleteAllButFirstFamiliarity()
        
        self.data = self.data[self.data['familiarity'] == 'UF']
        
        
        if (saveFile):
            df_copy = self.data.copy()
            name = ""
            if (sortOut):
                name = self.preFix
            df_copy.to_csv(f"only_unfamiliar_partners{name}.csv", index=False)
    
    def getTransparentSessions(self, sortOut = True, saveFile = False):
        """
        Remove all rows where 'session' ends with 'opaque' or 'translucent'
        (i.e., retain only transparent divider sessions).
        """
        if (sortOut):
            self.getPairedTestingSessions(sortOut=False)
            self.getTrainingPartner(sortOut=False)
            #self.deleteAllButFirstTransparency()
        
        self.data = self.data[
            ~self.data['session'].str.endswith(('Opaque', 'Translucent'), na=False)
        ]
        df_copy = self.data.copy()
        
        if (saveFile):
            name = ""
            if (sortOut):
                name = self.preFix
            df_copy.to_csv(f"only_transparent_sessions{name}.csv", index=False)

    def getTranslucentSessions(self, sortOut = True, saveFile = False):
        """
        Keep only rows where 'session' ends with 'translucent'.
        """
        if (sortOut):
            self.getPairedTestingSessions(sortOut=False)
            #self.deleteAllButFirstTransparency()
            self.getTrainingPartner(sortOut=False)
        
        self.data = self.data[
            self.data['session'].str.endswith('Translucent', na=False)
        ]
        
        
        if (saveFile):
            df_copy = self.data.copy()
            name = ""
            if (sortOut):
                name = self.preFix
            df_copy.to_csv(f"only_translucent_sessions{name}.csv", index=False)

    def getOpaqueSessions(self, sortOut = True, saveFile = False):
        """
        Keep only rows where 'session' ends with 'opaque'.
        """
        if (sortOut):
            self.getPairedTestingSessions(sortOut=False)
            #self.deleteAllButFirstTransparency()
            self.getTrainingPartner(sortOut=False)
        
        self.data = self.data[
            self.data['session'].str.endswith('Opaque', na=False)
        ]
        
        
        if (saveFile):
            df_copy = self.data.copy()
            name = ""
            if (sortOut):
                name = self.preFix
            df_copy.to_csv(f"only_opaque_sessions{name}.csv", index=False)

    def sortByMicePairs(self, saveFile=False):
        """
        Groups rows by mice pair using format-dependent parsing of 'vid',
        and sorts primarily by date (first 6 characters of 'vid'), then TrNum (if present).
        """
        import re
    
        df = self.data.copy()
    
        def extract_mice_pair(row):
            vid = row['vid']
            category = row['test/train']
            # Step 1: Extract the full 13-character ID
            if category == 'test':
                full_pair = vid[-13:]
            elif category == 'train':
                full_pair = vid[:-8][-13:]
            else:
                return "UNKNOWN"
            
            # Step 2: Split into two mouse IDs
            mouse1 = full_pair[:6]
            #print("Mouse 1: ", mouse1)
            mouse2 = full_pair[-6:]
            #print("Mouse 2: ", mouse2)
            
            
            # Step 3: Alphabetically order
            if mouse1 <= mouse2:
                return mouse1 + "-" + mouse2
            else:
                return mouse2 + "-" + mouse1
    
        def extract_date(vid):
            return int(vid[:6]) if vid[:6].isdigit() else float('inf')
    
        def extract_trnum(vid):
            match = re.search(r'TrNum(\d+)', vid)
            return int(match.group(1)) if match else float('inf')
    
        # Add helper columns
        df['mice_pair'] = df.apply(extract_mice_pair, axis=1)
        df['date'] = df['vid'].apply(extract_date)
        df['trnum'] = df['vid'].apply(extract_trnum)
    
        grouped = []
        for _, group_df in df.groupby('mice_pair'):
            sorted_group = group_df.sort_values(by=['date', 'trnum'], ascending=True)
            grouped.append(sorted_group.drop(columns=['date', 'trnum']))
    
        # Optional CSV save
        if saveFile:
            pd.concat(grouped).to_csv("group_mice_pairs.csv", index=False)
    
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
            #zero = "" if row["test/train"] == "test" else "0"
            zero = ""
            return f"{base_path}/{folder}/{zero}{row['session']}/Behavioral{processed}/mag/{row['correct_name']}_mag.csv" if pd.isna(row["vid"]) == False and pd.isna(row["session"]) == False else None
    
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
            #zero = "" if row["test/train"] == "test" else "0"
            zero = ""
            return f"{base_path}/{folder}/{zero}{row['session']}/Behavioral{processed}/lever/{row['correct_name']}_lever.csv" if pd.isna(row["vid"]) == False and pd.isna(row["session"]) == False else None
    
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
        
    def getPosDatapath(self, grouped = False):
        """
        Return a list of datapaths to pos files for each row in self.data.
        
        - If grouped=False: return a flat list.
        - If grouped=True: use sortBySameMicePairs to return a nested list where
          each sublist corresponds to all levFiles for a given mice pair.
    
        Path format:
        /gpfs/radev/pi/saxena/aj764/{folderName}/{sessionName}/
        """
        base_path = "/gpfs/radev/pi/saxena/aj764"
                
        def construct_path(row):
            folder = "PairedTestingSessions" if row["test/train"] == "test" else "Training_COOPERATION"
            #zero = "" if row["test/train"] == "test" else "0" 
            zero = ""
            return f"{base_path}/{folder}/{zero}{row['session']}/Tracking/h5/{row['vid']}.predictions.h5" if pd.isna(row["vid"]) == False and pd.isna(row["session"]) == False else None
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
    
    def returnFPSandTotFrames(self, grouped = False):
        '''
        Return a list of numbers, representing the fps for each file
        '''
        
        base_path = "/gpfs/radev/pi/saxena/aj764"
        
        def construct_path(row):
            folder = "PairedTestingSessions" if row["test/train"] == "test" else "Training_COOPERATION"
            videoName = row["vid"] + ".mp4"
            #zero = "" if row["test/train"] == "test" else "0" 
            zero = ""
            videos = "" if row["test/train"] == "train" else "Videos/"
            return f"{base_path}/{folder}/{zero}{row['session']}/{videos}{videoName}" if pd.isna(row["vid"]) == False and pd.isna(row["session"]) == False else None
        
        def getFPSandTotFrames(videoPath):
            cap = cv2.VideoCapture(videoPath)
            if not cap.isOpened():
                raise IOError("Could not open video file.")

            fps = cap.get(cv2.CAP_PROP_FPS)
            totFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            return fps, totFrames
                
        fpsList = []
        totFramesList = []
        
        if not grouped:
            for _, row in self.data.iterrows():
                path = construct_path(row)
                #print("Path is: ", path)
                fps, totFrames = getFPSandTotFrames(path)
                fpsList.append(fps)
                totFramesList.append(totFrames)
        
        else:
            grouped_rows = self.sortByMicePairs()
            for group in grouped_rows:
                fpsTempList = []
                totFramesTempList = []
                
                for _, row in group.iterrows():
                    path = construct_path(row)
                    fps, totFrames = getFPSandTotFrames(path)
                    fpsTempList.append(fps)
                    totFramesTempList.append(totFrames)
                
                fpsList.append(fpsTempList)
                totFramesList.append(totFramesTempList)
        
        return fpsList, totFramesList
    
    
    def returnNaNPercentage(self):
        df = pd.DataFrame(self.data)

        # Return the column named 'initial nan' as a list
        initial_nan_column = df['initial nan'].tolist()
        
        return initial_nan_column
    
#information_path = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/scripts/notebooks/dyed_preds_df.csv"
#information_path = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral Quantification/Sorted Data Files/dyed_preds_df_fixed.csv"
information_path = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/originalFile.csv"

fixedExpanded = "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/dyed_preds_fixed_expanded.csv"

all_valid = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/dyed_preds_all_valid.csv"
only_opaque = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral_Quantification/Sorted Data Files/only_opaque_sessions.csv"
only_training = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral_Quantification/Sorted Data Files/only_training_partners.csv"
only_translucent = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral_Quantification/Sorted Data Files/only_translucent_sessions.csv"
only_transparent = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral_Quantification/Sorted Data Files/only_transparent_sessions.csv"
only_unfamiliar = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral_Quantification/Sorted Data Files/only_unfamiliar_partners.csv"
only_trainingpartners = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral_Quantification/Sorted Data Files/only_training_partners.csv"

'''fe = fileExtractor(fixedExpanded)
fe.deleteInvalid()
fe.getTransparentSessions(sortOut=False)
fe.getPairedTestingSessions(sortOut=False)
fe.getTrainingPartner(sortOut=False,saveFile=True)'''

#print("Res: ", fe.returnFPSandTotFrames())
#fe.getFirstSessionPerMicePair()
#fe.getPairedTestingSessions()
#fe.sortByMicePairs(True)

#fe.getOpaqueSessions()
#fe.getTrainingCoopSessions()
#fe.getPairedTestingSessions()
#print(fe.getLevsDatapath())


def saveAllCSVs():
    methods_to_call = [
        #"writeNewFixedFile",
        "getTrainingCoopSessions",
        "getPairedTestingSessions",
        "getTrainingPartner",
        "getUnfamiliarPartners",
        "getTransparentSessions",
        "getTranslucentSessions",
        "getOpaqueSessions",
    ]
    
    for method_name in methods_to_call:
        if (method_name == "writeNewFixedFile"):
            fe = fileExtractor(fixedExpanded)
            fe.writeNewFixedFile()
        else:
            fe = fileExtractor(fixedExpanded)
            fe.deleteInvalid()  # ensure you start from the cleaned data
            method = getattr(fe, method_name)
            method(sortOut=True, saveFile=True)  # call the method

#saveAllCSVs()







