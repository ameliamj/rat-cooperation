#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 11:23:52 2025

@author: david
"""
import pandas as pd
import numpy as np

class levLoader: 
    #Class to read, store, and access a csv file with data from each lever press for a single coop experimental session
    
    def __init__(self, filename, endFrame = 0, fps = 30): #Constructor
        self.filename = filename
        self.data = None
        self._load_data()
        self.categories = ["TrialNum", "LeverNum", "AbsTime", "TrialCond", "DispTime", "TrialTime", "coopTS", "coopSucc", "Hit", "TrialEnd", "AnimalID", "RatID"]
        self.endFrame = endFrame
        self.fps = fps
        #print("Columns in DataFrame:", self.data.columns.tolist())
        
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
                raise ValueError(f"Column '{column}' not found in data.")
            return self.data[column].isna().sum()

    def getTrialNum(self, row_idx): #returns the trial number for row_idx of the data
        return self.get_value(row_idx, 0)
    
    def getLevNum(self, row_idx): #returns the mag num for row_idx of the data
        return self.get_value(row_idx, 1)
    
    
    #Graph Stuff: 
    def returnSuccThreshold(self):
        if ('CoopTimeLimit' in self.data.columns.tolist()):
            return self.data['CoopTimeLimit'].iloc[0]
        else:
            return 1
        
    def remove100msTrials(self):
        """
        Remove all lever presses in trials where the first lever press has a TrialTime <= 0.1.
        Modifies the DataFrame in place.
        
        Raises:
            ValueError: If no data is loaded or required columns are missing.
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        if not all(col in self.data.columns for col in ["TrialNum", "TrialTime"]):
            raise ValueError("Required columns 'TrialNum' or 'TrialTime' not found in data.")
        
        # Identify first lever press in each trial
        first_presses = self.data.groupby("TrialNum").first().reset_index()
        print("first_presses: ", first_presses)
        
        # Find trials where first press has TrialTime <= 0.1
        trials_to_remove = first_presses[first_presses["TrialTime"] <= 0.1]["TrialNum"]
        print("trials_to_remove: ", trials_to_remove)
        
        
        print("original data: ", self.data)
        # Remove all rows corresponding to these trials
        self.data = self.data[~self.data["TrialNum"].isin(trials_to_remove)].reset_index(drop=True)
        print("new data: ", self.data)
        
        return len(trials_to_remove)
        
    def returnSuccessTrials(self):
        """
        For each possible trial number from 1 to N:
        - If a trial with that TrialNum exists and coopSucc == 1 → 1
        - If it exists and coopSucc != 1 → 0
        - If no such trial exists → -1
    
        Returns:
            List[int]: array of 1s (success), 0s (failure), or -1s (missing trial)
        """
        n = self.returnNumTotalTrials()
        arr = [0] * n
        
        for i in range(n):
            trial_df = self.data[self.data['TrialNum'] == i + 1]
            
            if trial_df.empty:
                arr[i] = -1
            else:
                # Take first row (if there are multiple)
                trial = trial_df.iloc[0]
                if trial['coopSucc'] == 1:
                    arr[i] = 1
                else:
                    arr[i] = 0
        
        return arr
        
    def returnTimeStartTrials(self):  
        """
        Returns a list of absolute times (in seconds) each trial started at.
        The function groups the data by 'TrialNum' and finds the smallest 'AbsTime' per trial, 
        and then subtracts the 'TrialTime' to get the result.
        Length is equal to the number of trials with a press
        
        Returns:
            list: A list of floats, each being the absolute time of the trial starting
        """
        if self.data is None:
            raise ValueError("No data loaded.")
    
        if not {'TrialNum', 'AbsTime', 'TrialTime'}.issubset(self.data.columns):
            raise ValueError("Required columns 'TrialNum', 'AbsTime', or 'TrialTime' are missing from data.")
    
        #print("self.data: ", self.data)
    
        # Find the index of the row with the minimum AbsTime for each TrialNum
        min_time_idx = self.data.groupby('TrialNum')['AbsTime'].idxmin()
        #print("min_time_idx: ", min_time_idx)
        
        # Get the corresponding rows
        trial_start_rows = self.data.loc[min_time_idx]
        #print("trial_start_rows: ", trial_start_rows)
    
        # Subtract TrialTime from AbsTime to get the true start time
        trial_starts_dict = {
            row['TrialNum']: row['AbsTime'] - row['TrialTime']
            for _, row in trial_start_rows.iterrows()
        }
        #print("trial_starts_dict: ", trial_starts_dict)
        
        # Construct a list of length total_trials, using None for missing trials
        total_trials = self.returnNumTotalTrials()
        trial_starts = [
            float(trial_starts_dict[i + 1])
            for i in range(total_trials)
            if (i + 1) in trial_starts_dict and trial_starts_dict[i + 1] is not None
        ]
        
        #print("trial_starts: ", trial_starts)
        
        #print("length: ", len(trial_starts))
        
        return trial_starts
    
    def returnTimeEndTrials(self):
        trial_starts = self.returnTimeStartTrials()
        trial_ends = []
        
        for i, start in enumerate(trial_starts):
            if (i < len(trial_starts) - 1):
                trial_ends.append(trial_starts[i + 1])
            else:
                trial_ends.append((self.endFrame - 2) / self.fps)
        
        return trial_ends
    
    def returnRatIDFirstPressTrial(self):
        '''
        Returns a list of the RatID for the rat that pressed the lever 
        first for each trial where there was a lever press.
        '''
        if self.data is None:
            raise ValueError("No data loaded.")
    
        if not {'TrialNum', 'RatID'}.issubset(self.data.columns):
            raise ValueError("Required columns 'TrialNum'or 'RatID' are missing from data.")
        
        first_ratids = self.data.groupby('TrialNum')['RatID'].first()
        
        return first_ratids
    
    def returnRatLocationFirstPressTrial(self): 
        '''
        Returns a list of the leverNum for the rat that presses the lever first
        for each trial where there was a lever press. 
        '''
        
        if self.data is None:
            raise ValueError("No data loaded.")
    
        if not {'TrialNum', 'LeverNum'}.issubset(self.data.columns):
            raise ValueError("Required columns 'TrialNum'or 'LeverNum' are missing from data.")
        
        leverNums = self.data.groupby('TrialNum')['LeverNum'].first()
        
        return leverNums
    
    def returnFirstPressAbsTimes(self):
        """
        Returns a list of absolute times (in seconds) for the first press in each trial.
        The function groups the data by 'TrialNum' and finds the smallest 'AbsTime' per trial.
        
        Returns:
            list: A list of floats, each being the absolute time of the first lever press in a trial.
        """
        if self.data is None:
            raise ValueError("No data loaded.")
    
        if 'TrialNum' not in self.data.columns or 'AbsTime' not in self.data.columns:
            raise ValueError("Required columns 'TrialNum' or 'AbsTime' are missing from data.")
        
        # Group by TrialNum and find the first (earliest) absolute press time in each trial
        first_presses = self.data.groupby('TrialNum')['AbsTime'].min().dropna()
        #print(first_presses.tolist())
        
        return first_presses.tolist()
    
    def returnMostPressesByLever(self, ratID):
        """
        Returns the number of presses by the specified rat on Lever 1 and Lever 2.
        
        Args:
            ratID (int): The ID of the rat to filter presses for.
        
        Returns:
            max(num_lever1_presses, num_lever2_presses)
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        
        if 'RatID' not in self.data.columns or 'LeverNum' not in self.data.columns:
            raise ValueError("Required columns 'RatID' or 'LeverNum' are missing from data.")
        
        rat_data = self.data[self.data['RatID'] == ratID]
        
        lever1_count = (rat_data['LeverNum'] == 1).sum()
        lever2_count = (rat_data['LeverNum'] == 2).sum()
        return max(lever1_count, lever2_count)
    
    def returnMinPressesByLever(self, ratID):
        if self.data is None:
            raise ValueError("No data loaded.")
        
        if 'RatID' not in self.data.columns or 'LeverNum' not in self.data.columns:
            raise ValueError("Required columns 'RatID' or 'LeverNum' are missing from data.")
        
        rat_data = self.data[self.data['RatID'] == ratID]
        
        lever1_count = (rat_data['LeverNum'] == 1).sum()
        lever2_count = (rat_data['LeverNum'] == 2).sum()
        return min(lever1_count, lever2_count)
    
    def returnNumTotalTrials(self):
        trials = self.data["TrialNum"].iloc[-1]
        #print("Num Trials: ", trials)
        
        num = 0
        #num = self.remove100msTrials()
        #print("Num Trials After: ", trials - num)
        
        return trials - num
    
    def returnNumTotalTrialswithLeverPress(self):
        num_unique_trials = self.data['TrialNum'].nunique()
        return num_unique_trials
    
    def returnTotalLeverPresses(self):
        res = self.data.shape[0]
        return res
    
    def returnTotalLeverPressesFiltered(self):
        '''
        Same as returnTotalLeverPresses but you delete any rows in which there's missing RatID or AbsTime'
        '''
        df = self.data.copy()
        
        if df is not None:
            required_columns = ['RatID', 'AbsTime']
            existing_columns = [col for col in required_columns if col in df.columns]
            
            if existing_columns:
                print(f"Dropping rows with NaN in columns: {existing_columns}")
                df = df.dropna(subset=existing_columns)
            else:
                print("Warning: Neither 'RatID' nor 'AbsTime' columns found in lev data. Skipping dropna.")
        
        total_lev_events_filtered = df.shape[0]
        return total_lev_events_filtered
    
    def returnLevPressesPerTrial(self):
       return self.returnTotalLeverPresses() / self.returnNumTotalTrials()
    
    def returnNumSuccessfulTrials(self):
        succ = self.data.groupby('TrialNum').first().query('coopSucc == 1').shape[0]
        #print("Original Succ: ", succ)
        
        #self.remove100msTrials()
        #succ = self.data.groupby('TrialNum').first().query('coopSucc == 1').shape[0]
        #print("New Succ: ", succ)
        
        return succ
    
        
    def returnAvgIPI(self, test = False, returnList = False, returnLen = False):
        """
        Compute IPI (Inter-Press Interval) for each rat separately (based on RatID),
        then average across all IPIs from both rats.
    
        Parameters:
        - test: If True and enough data, return only the first 5 IPIs for inspection.
        - returnList: If True, return the full list of IPIs instead of their average.
    
        Returns:
        - List of IPIs or their average, depending on returnList.
        """
        
        df = self.data.copy()
    
        # Filter necessary columns and drop rows with missing AbsTime or RatID
        df = df[["RatID", "AbsTime"]].dropna()
    
        if df.empty:
            return [] if returnList or test else 0
    
        ipis = []
    
        # Group by RatID and compute IPIs for each rat
        for rat_id, group in df.groupby("RatID"):
            group = group.sort_values(by="AbsTime")
            times = group["AbsTime"].values
    
            if len(times) < 2:
                continue  # Can't compute IPI with fewer than 2 presses
    
            rat_ipis = [t2 - t1 for t1, t2 in zip(times[:-1], times[1:])]
            ipis.extend(rat_ipis)
    
        if len(ipis) == 0:
            print("\nNo IPIs computed")
            print("Lev File: ", self.filename)
            return [] if returnList or test else 0
        
        if (returnLen):
            return len(ipis)
        
        if test and len(ipis) >= 5:
            return ipis[:5]
    
        if returnList:
            return ipis
    
        return sum(ipis) / len(ipis)
    
    def returnAvgIPI_FirsttoSuccess(self, test = False, returnList = False):
        '''Time between first press and successful press (second rat press) in successful trials.'''
        
        df = self.data.copy()
        df = df.sort_values(by=["coopSucc", "TrialNum", "AbsTime"])
        
        ipis = []

        for trial_num, trial_data in df[df["coopSucc"] == 1].groupby("TrialNum"):
            rats_pressed = set()
            presses = []
            #print("TrialNum: ", trial_num)
            #print("trial_data: \n", trial_data)

            for _, row in trial_data.iterrows():
                rat = row["RatID"]
                presses.append((rat, row["AbsTime"]))

                if rat not in rats_pressed:
                    rats_pressed.add(rat)
        
                if len(rats_pressed) == 2:
                    # Second rat has just pressed — trial becomes successful
                    first_press_time = presses[0][1]
                    success_press_time = row["AbsTime"]
                    #print("Res: ", success_press_time - first_press_time)
                    ipis.append(success_press_time - first_press_time)
                    break  # Done with this trial

        #ipis is a List of all the times: 
        sumTimes = sum(ipis)
        numSuccessfulTrials = len(ipis)
        if (test == True and len(ipis) >= 5):
            return ipis[:5]
        if (returnList == True):
            return ipis
        if (numSuccessfulTrials > 0): 
            return sumTimes/numSuccessfulTrials
        else:
            print("\n numSuccessfulTrials is 0")
            print("Lev File: ", self.filename)
            print("ipis", ipis)
            return 0
    
    def returnAvgIPI_LasttoSuccess(self, test = False, returnList = False):
        """Time between last press (by same rat as first) and success press (by second rat)."""
        df = self.data.copy()
        df = df.sort_values(by=["coopSucc", "TrialNum", "AbsTime"])
        ipis = []

        for trial_num, trial_data in df[df["coopSucc"] == 1].groupby("TrialNum"):
            rats_pressed = set()
            presses = []

            for _, row in trial_data.iterrows():
                rat = row["RatID"]
                presses.append((rat, row["AbsTime"]))

                if rat not in rats_pressed:
                    rats_pressed.add(rat)

                if len(rats_pressed) == 2:
                    # Find last press before success from first rat
                    first_rat = presses[0][0]
                    first_rat_presses = [t for r, t in presses[:-1] if r == first_rat]
                    if not first_rat_presses:
                        continue  # skip this trial — no valid prior press from first rat
                    last_first_rat_time = max(first_rat_presses)
                    success_press_time = row["AbsTime"]
                    #print("TrialNum: ", trial_num)
                    #print("Success Press Time: ", success_press_time)
                    #print("Last FirstRat Time: ", last_first_rat_time)
                    #print("Res: ", success_press_time - last_first_rat_time, "\n")
                    ipis.append(success_press_time - last_first_rat_time)
                    break

        #ipis is a List of all the times: 
        sumTimes = sum(ipis)
        numSuccessfulTrials = len(ipis)
        if (test == True):
            return ipis[:5]
        if (returnList == True):
            return ipis
        if (numSuccessfulTrials > 0): 
            return sumTimes/numSuccessfulTrials
        else:
            print("\nnumSuccessfulTrials is 0")
            print("Lev File: ", self.filename)
            print("ipis", ipis)
            return 0

    def returnRatFirstPress(self):
        res = [0, 0] #res[0] = numPressesRat0, res[1] = numPressesRat1
        
        #print("self.data")
        #print(self.data)
        
        grouped = self.data.groupby("TrialNum")
        
        #print("\n\nGrouped")
        #print(grouped)
        
        for trial_num, trial_data in grouped:
            #print("Start:")
            #print(trial_num)
            #print(trial_data)
            
            if (trial_data.iloc[0]['RatID'] == 0):
                res[0] += 1
            elif(trial_data.iloc[0]['RatID'] == 1):
                res[1] += 1
        
        return res

    def returnAvgRepresses_FirstMouse(self, returnArr = False):
        """
        For each trial, find the first mouse to press a lever and count how many total times
        that mouse pressed the lever during the entire trial. Return the average of these counts.
        """
        df = self.data.copy()
        first_mouse_repress_counts = []
    
        for trial_num, trial_data in df.groupby("TrialNum"):
            trial_data_sorted = trial_data.sort_values(by="AbsTime")
            first_mouse = trial_data_sorted.iloc[0]["RatID"]
            count_first_mouse_presses = (trial_data["RatID"] == first_mouse).sum()
            first_mouse_repress_counts.append(count_first_mouse_presses)
        
        if (returnArr):
            return first_mouse_repress_counts
    
        return sum(first_mouse_repress_counts) / len(first_mouse_repress_counts) if first_mouse_repress_counts else 0

    def returnAvgRepresses_SecondMouse_Success(self):
        """
        For all successful trials, identify the second mouse to press and calculate the average number
        of presses made by the second mouse in each of those trials.
        """
        df = self.data.copy()
        df = df[df["coopSucc"] == 1]
        second_mouse_repress_counts = []
    
        for trial_num, trial_data in df.groupby("TrialNum"):
            trial_data_sorted = trial_data.sort_values(by="AbsTime")
            rats_seen = set()
            second_mouse = None
            for _, row in trial_data_sorted.iterrows():
                rat = row["RatID"]
                if rat not in rats_seen:
                    rats_seen.add(rat)
                    if len(rats_seen) == 2:
                        second_mouse = rat
                        break
            if second_mouse is not None:
                count_second_mouse_presses = (trial_data["RatID"] == second_mouse).sum()
                second_mouse_repress_counts.append(count_second_mouse_presses)
    
        return sum(second_mouse_repress_counts) / len(second_mouse_repress_counts) if second_mouse_repress_counts else 0

    def returnAvgRepresses_FirstMouse_SuccessVsNon(self):
        """
        For all trials, find the first mouse to press and count how many times they pressed the lever
        during the entire trial. Return two averages:
        - Average for successful trials
        - Average for unsuccessful trials
        """
        df = self.data.copy()
        success_counts = []
        nonsuccess_counts = []
    
        for trial_num, trial_data in df.groupby("TrialNum"):
            trial_data_sorted = trial_data.sort_values(by="AbsTime")
            first_mouse = trial_data_sorted.iloc[0]["RatID"]
            count_first_mouse_presses = (trial_data["RatID"] == first_mouse).sum()
            
            if "coopSucc" not in df.columns:
                return np.nan  # or 0 or continue, depending on your logic
            is_success = trial_data_sorted.iloc[0]["coopSucc"] == 1
            if is_success:
                success_counts.append(count_first_mouse_presses)
            else:
                nonsuccess_counts.append(count_first_mouse_presses)
    
        avg_success = sum(success_counts) / len(success_counts) if success_counts else 0
        avg_non_success = sum(nonsuccess_counts) / len(nonsuccess_counts) if nonsuccess_counts else 0
    
        return avg_success, avg_non_success

    def returnCooperativeSuccessRegionsBool(self):
        '''
        Return a boolean list of all trials that are in a region of cooperative success.
        A trial is considered in a "region of cooperative success" if 4 out of the last 5
        trials (including itself) were successful.
    
        Returns:
            List[bool]: A list where each index corresponds to a TrialNum and is True
                        if that trial is in a region of cooperative success.
        '''
        thresholdZone = 3
        
        if self.data is None:
            raise ValueError("No data loaded.")
    
        if 'TrialNum' not in self.data.columns or 'coopSucc' not in self.data.columns:
            raise ValueError("Required columns 'TrialNum' or 'coopSucc' are missing from data.")
    
        # Get success status for each unique trial
        trial_success = self.data.groupby('TrialNum').first()['coopSucc'].fillna(0).astype(int)
    
        # Total number of trials
        total_trials = self.returnNumTotalTrials()
    
        # Initialize the result list
        region_bool = [False] * total_trials
    
        # Sliding window over the last 5 trials including current
        for i in range(total_trials):
            window_start = max(0, i - 4)
            window = trial_success.iloc[window_start:i+1]
            if window.sum() >= 4:
                region_bool[i] = True
        
        cooperationZone = False
        counter = 0
        for i in range(total_trials):
            if (not cooperationZone and region_bool[i] == True):
                cooperationZone = True
            elif (cooperationZone and region_bool[i] == True):
                counter = 0
            elif(region_bool[i] == False and cooperationZone):
                counter += 1
                if (counter >= 3):
                    counter = 0
                    cooperationZone = False
                else:
                    region_bool[i] = True
        
        return region_bool
    
    def returnNumCooperationSuccessZones(self):
        '''
        Return the number of cooperative success zones where a zone is a span of the trials that are in a region cooperative success until 3 trials (make this a variable) in a row are not
        '''
        res = 0
        
        arr = self.returnCooperativeSuccessRegionsBool()
        for i in range(1, len(arr)):
            if (arr[i] == True and arr[i-1] == False):
                res += 1
                
        return res
        
    def returnAvgLengthCooperativeSuccessZones(self):
        '''
        number of trials in a region of cooperative success/Num cooperation success zones
        '''
        region_bool = self.returnCooperativeSuccessRegionsBool()
        num_true = sum(region_bool)
        num_zones = self.returnNumCooperationSuccessZones()
        return num_true/num_zones
    
    def getLeverPressFrames(self, rat_id):
        """
        Returns a set of frame indices where the specified rat pressed the lever.
        Assumes lever press occurs at AbsTime when RatID matches and coopSucc == 1.
        """
        if self.data is None or self.data.empty:
            return set()
        press_data = self.data[(self.data['RatID'] == rat_id)]
        frames = (press_data['AbsTime'] * self.fps).astype(int)
        return set(frames)
    
    def returnCoopTimeorLastPressTime(self):
        """
        Returns a list of AbsTime - TrialTime values per trial:
        - For successful trials (coopSucc == 1), returns the second lever press where Hit == 1.
        - For unsuccessful trials (coopSucc == 0), returns the last lever press in the trial.
        - If the second Hit == 1 does not exist, appends None.
    
        Returns:
            list: A list of floats or None per trial.
        """
        if self.data is None or self.data.empty:
            return []
    
        times = []
        grouped = self.data.groupby('TrialNum')
    
        for trial_num, group in grouped:
            is_successful = group['coopSucc'].iloc[0] == 1
    
            if is_successful:
                hits = group[group['Hit'] == 1]
                if len(hits) >= 2:
                    second_hit_row = hits.iloc[1]
                    if pd.notna(second_hit_row['AbsTime']):
                        rel_time = second_hit_row['AbsTime']
                        times.append(rel_time)
                    else:
                        times.append(None)
                else:
                    times.append(None)
            else:
                last_row = group.iloc[-1]
                if pd.notna(last_row['AbsTime']):
                    rel_time = last_row['AbsTime']
                    times.append(rel_time)
                else:
                    times.append(None)
    
        return times
        
        
#Testing
#
#

file1 = "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/leverData.csv"
#lev = levLoader(file1)

#print(lev.returnCooperativeSuccessRegionsBool())
#print(lev.returnNumCooperationSuccessZones())
#print(lev.returnAvgLengthCooperativeSuccessZones())

