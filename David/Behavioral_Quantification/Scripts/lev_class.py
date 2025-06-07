#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 11:23:52 2025

@author: david
"""
import pandas as pd

class levLoader: 
    #Class to read, store, and access a csv file with data from each lever press for a single coop experimental session
    
    def __init__(self, filename): #Constructor
        self.filename = filename
        self.data = None
        self._load_data()
        self.categories = ["TrialNum", "LeverNum", "AbsTime", "TrialCond", "DispTime", "TrialTime", "coopTS", "coopSucc", "Hit", "TrialEnd", "AnimalID", "RatID"]
        
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
    
    def returnNumTotalTrials(self):
        trials = self.data["TrialNum"].iloc[-1]
        return trials
    
    def returnTotalLeverPresses(self):
        res = self.data.shape[0]
        return res
    
    def returnLevPressesPerTrial(self):
       return self.returnTotalLeverPresses() / self.returnNumTotalTrials()
    
    def returnNumSuccessfulTrials(self):
        succ = self.data.groupby('TrialNum').first().query('coopSucc == 1').shape[0]
        return succ
    
        
    def returnAvgIPI(self, test = False, returnList = False):
        """Compute IPI (time between all consecutive lever presses)."""
        
        df = self.data.copy()
        df = df.sort_values(by="AbsTime")  # Sort all presses chronologically
        times = df["AbsTime"].values
    
        if len(times) < 2:
            return []
    
        ipis = [t2 - t1 for t1, t2 in zip(times[:-1], times[1:])]
        
        if (test == True and len(ipis) >= 5):
            return ipis[:5]
        
        if (returnList == True):
            return ipis
        
        #ipis is a List of all the times: 
        sumTimes = sum(ipis)
        numSuccessfulTrials = len(ipis)
        if (numSuccessfulTrials > 0): 
            return sumTimes/numSuccessfulTrials
        else:
            print("\n numSuccessfulTrials is 0")
            print("Lev File: ", self.filename)
            print("ipis", ipis)
            return 0
    
    
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
    
    def returnAvgIPI_LasttoSuccess(self, test = False):
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

    def returnAvgRepresses_FirstMouse(self):
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
            is_success = trial_data_sorted.iloc[0]["coopSucc"] == 1
            if is_success:
                success_counts.append(count_first_mouse_presses)
            else:
                nonsuccess_counts.append(count_first_mouse_presses)
    
        avg_success = sum(success_counts) / len(success_counts) if success_counts else 0
        avg_non_success = sum(nonsuccess_counts) / len(nonsuccess_counts) if nonsuccess_counts else 0
    
        return avg_success, avg_non_success

#Testing
#
#

#file1 = "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/leverData.csv"
#lev1 = levLoader(file1)
#print(lev1.returnAvgIPI_LasttoSuccess())

