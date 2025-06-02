#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 17:57:51 2025

@author: david
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_class import singleExperiment
from typing import List
from file_extractor_class import fileExtractor

import sys
sys.stdout.flush()
sys.stderr.flush()

#Class to create Graphs with data from a single file
#
#


class singleFileGraphs:
    def __init__(self, mag_file, lev_file, pos_file):
        self.experiment = singleExperiment(mag_file, lev_file, pos_file)
    
    
    def magFileDataAvailabilityGraph(self):
        """
        Create a bar graph showing the percentage of non-null data for each category in magData.
        
        The graph is saved as 'mag_data_availability.png'.
        """
        percentageDataPerCategory = []  # Percentage of data values that are not null for each category
        magData = self.experiment.mag 
        totalRows = magData.getNumRows()
        
        if totalRows == 0:
            raise ValueError("No data available in magData.")
        
        for cat in magData.categories:
            numNulls = magData.countNullsinColumn(cat)
            percentageDataPerCategory.append((totalRows - numNulls) / totalRows * 100)
        
        # Create bar graph
        plt.figure(figsize=(10, 6))
        bars = plt.bar(magData.categories, percentageDataPerCategory, color='skyblue')
        
        # Customize the plot
        plt.xlabel('Categories')
        plt.ylabel('Percentage of Non-Null Data (%)')
        plt.title('Data Availability in Mag Data File')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 100)
        
        # Add percentage labels on top of each bar
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval - 5, f'{yval:.1f}%', 
                     ha='center', va='bottom')
        
        # Adjust layout to prevent label cutoff
        #plt.tight_layout()
        
        # Save the plot
        #plt.savefig('mag_data_availability.png')
        plt.show()
        plt.close()
            
    def levFileDataAvailabilityGraph(self):
        """
        Create a bar graph showing the percentage of non-null data for each category in levData.
        
        The graph is saved as 'lev_data_availability.png'.
        """
        percentageDataPerCategory = []  # Percentage of data values that are not null for each category
        levData = self.experiment.lev
        totalRows = levData.getNumRows()
        
        if totalRows == 0:
            raise ValueError("No data available in levData.")
        
        for cat in levData.categories:
            numNulls = levData.countNullsinColumn(cat)
            percentageDataPerCategory.append((totalRows - numNulls) / totalRows * 100)
        
        # Create bar graph
        plt.figure(figsize=(10, 6))
        bars = plt.bar(levData.categories, percentageDataPerCategory, color='skyblue')
        
        # Customize the plot
        plt.xlabel('Categories')
        plt.ylabel('Percentage of Non-Null Data (%)')
        plt.title('Data Availability in Lev Data File')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 100)
        
        # Add percentage labels on top of each bar
        for bar in bars:
            yval = bar.get_height()
            yval = max(yval, 6)
            plt.text(bar.get_x() + bar.get_width()/2, yval - 5, f'{yval:.1f}%', 
                     ha='center', va='bottom')
        
        # Adjust layout to prevent label cutoff
        #plt.tight_layout()
        
        # Save the plot
        #plt.savefig('lev_data_availability.png')
        plt.show()
        plt.close()
    
    def interpressIntervalPlot(self):
        lev_data = self.experiment.lev.data
        #print(lev_data)

        if lev_data is None:
            raise ValueError("Lever press data is not loaded.")

        # Sort by Rat and AbsTime 
        lev_data_sorted = lev_data.sort_values(by=["RatID", "AbsTime"]) 
        
        #Optional: filter out re-presses (Hit = -1 or -2)
        #lev_data_sorted = lev_data_sorted[lev_data_sorted["Hit"] == 1]

        # Calculate Inter-Press Interval (IPI) within each animal group
        lev_data_sorted["IPI"] = lev_data_sorted.groupby("RatID")["AbsTime"].diff()
        
        #Filter outNaNs
        ipi_data = lev_data_sorted[(lev_data_sorted["IPI"].notna())]
        #print(ipi_data)
        
        # Plot IPI histogram
        plt.figure(figsize=(10, 6))
        plt.hist(ipi_data["IPI"], bins=50, edgecolor='black')
        plt.xlabel("Inter-Press Interval (s)")
        plt.ylabel("Frequency")
        plt.title("Histogram of Inter-Press Intervals")
        plt.tight_layout()
        plt.show()
        
        # Boxplot of IPI per animal
        plt.figure(figsize=(10, 6))
        lev_data_sorted.boxplot(column="IPI", by="RatID")
        plt.title("Inter-Press Interval (IPI) by Rat")
        plt.suptitle("")  # Removes default pandas boxplot title
        plt.xlabel("RatID")
        plt.ylabel("IPI (seconds)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        
        # Time series plot of IPI for each rat
        plt.figure(figsize=(12, 6))
        
        for rat_id, group in lev_data_sorted.groupby("RatID"):
            plt.plot(group["AbsTime"], group["IPI"], label=f"Rat {rat_id}", alpha=0.7)
        
        plt.title("IPI Over Time by Rat")
        plt.xlabel("Absolute Time (s)")
        plt.ylabel("Inter-Press Interval (s)")
        plt.legend(title="RatID", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
            
    def interpressIntervalSuccessPlot(self):
        lev_data = self.experiment.lev.data.copy()

        # Filter only trials with cooperative success
        successful_trials = lev_data[lev_data["coopSucc"] == 1]

        # Group by trial
        grouped = successful_trials.groupby(["TrialNum"])

        ipi_first_to_success = []
        ipi_last_to_success = []
        first_press_rats = []
        success_count_by_rat = {}

        for trial_num, trial_data in grouped:
            trial_data = trial_data.sort_values(by="AbsTime")

            if 1 not in trial_data["Hit"].values:
                continue  # skip if no initial press

            # Find index of cooperative press
            coop_press = trial_data[trial_data["TrialEnd"] == 1]
            if coop_press.empty:
                continue

            coop_press_time = coop_press.iloc[0]["AbsTime"]

            # First press in trial
            first_press = trial_data[trial_data["Hit"] == 1].iloc[0]
            first_press_time = first_press["AbsTime"]

            # Last press before cooperative success
            before_coop = trial_data[trial_data["AbsTime"] < coop_press_time]
            if before_coop.empty:
                continue

            last_press = before_coop.iloc[-1]
            last_press_time = last_press["AbsTime"]

            ipi_first_to_success.append(coop_press_time - first_press_time)
            ipi_last_to_success.append(coop_press_time - last_press_time)

            # Record first press rat
            first_press_rats.append(first_press["RatID"])

            # Count successful coop per rat
            rat_id = coop_press.iloc[0]["RatID"]
            success_count_by_rat[rat_id] = success_count_by_rat.get(rat_id, 0) + 1

        # Plot 1: Histogram - First Press to Success
        plt.figure(figsize=(8, 6))
        plt.hist(ipi_first_to_success, bins=20, color='skyblue')
        plt.title("Histogram: First Press to Success")
        plt.xlabel("IPI (s)")
        plt.ylabel("Count")
        plt.show()

        # Plot 2: Histogram - Last Press to Success
        plt.figure(figsize=(8, 6))
        plt.hist(ipi_last_to_success, bins=20, color='salmon')
        plt.title("Histogram: Last Press to Success")
        plt.xlabel("IPI (s)")
        plt.ylabel("Count")
        plt.show()

        # Plot 3: Time Series - First Press to Success
        plt.figure(figsize=(10, 5))
        plt.plot(ipi_first_to_success, marker='o')
        plt.title("Time Series: First Press to Success")
        plt.xlabel("Trial Index")
        plt.ylabel("IPI (s)")
        plt.show()

        # Plot 4: Time Series - Last Press to Success
        plt.figure(figsize=(10, 5))
        plt.plot(ipi_last_to_success, marker='o', color='orange')
        plt.title("Time Series: Last Press to Success")
        plt.xlabel("Trial Index")
        plt.ylabel("IPI (s)")
        plt.show()

        # Plot 5: Pie Chart - Who Presses First (by RatID)
        rat_counts = pd.Series(first_press_rats).value_counts()
        plt.figure(figsize=(6, 6))
        plt.pie(rat_counts, labels=rat_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title("Who Presses First (by RatID)")
        plt.show()


    def percentSuccesfulTrials(self):
        lev_data = self.experiment.lev.data.copy()
        grouped = lev_data.groupby("TrialNum")
        
        
        totalTrials, countSuccess = 0, 0
        for trial_num, trial_data in grouped:
            if (trial_data.iloc[0]['coopSucc'] == 1):
                countSuccess += 1
            
            totalTrials += 1
        
        countFail = totalTrials - countSuccess
        labels = ['Successful', 'Unsuccessful']
        sizes = [countSuccess, countFail]
        colors = ['green', 'red']
    
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 14})
        plt.title('Successful Cooperative Trials (%)', fontsize=16)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.tight_layout()
        plt.show()
                
            
        
    
#Testing Single File Graphs
#
#

'''mag_file = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral_Quantification/Example Data Files/magData.csv"
lev_file = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral_Quantification/Example Data Files/leverData.csv"
pos_file = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral_Quantification/Example Data Files/ExampleTrackingCoop.h5"

experiment = singleFileGraphs(mag_file, lev_file, pos_file)
experiment.percentSuccesfulTrials()
experiment.interpressIntervalSuccessPlot()'''


# ---------------------------------------------------------------------------------------------------------


#Class to create Graphs with data from multiple files + Functions to get data from all the different categories
#
#

#all_valid = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral Quantification/Sorted Data Files/dyed_preds_all_valid.csv"
all_valid = "/gpfs/radev/home/drb83/project/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/dyed_preds_all_valid.csv"

only_opaque = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_opaque_sessions.csv"
only_translucent = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_translucent_sessions.csv"
only_transparent = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_transparent_sessions.csv"

only_unfamiliar = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_unfamiliar_partners.csv"
only_trainingpartners = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_training_partners.csv"

only_PairedTesting = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_PairedTesting.csv"
only_TrainingCoop = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/only_TrainingCooperation.csv"


def getAllValid():
    fe = fileExtractor(all_valid)
    return [fe.getLevsDatapath(), fe.getMagsDatapath(), fe.getPosDatapath()]
    
def getOnlyOpaque():
    fe = fileExtractor(all_valid)
    return [fe.getLevsDatapath(), fe.getMagsDatapath(), fe.getPosDatapath()]

def getonlyTranslucent():
    fe = fileExtractor(all_valid)
    return [fe.getLevsDatapath(), fe.getMagsDatapath(), fe.getPosDatapath()]

def getOnlyTransparent():
    fe = fileExtractor(all_valid)
    return [fe.getLevsDatapath(), fe.getMagsDatapath(), fe.getPosDatapath()]

def getOnlyUnfamiliar():
    fe = fileExtractor(all_valid)
    return [fe.getLevsDatapath(), fe.getMagsDatapath(), fe.getPosDatapath()]

def getOnlyTrainingPartners():
    fe = fileExtractor(all_valid)
    return [fe.getLevsDatapath(), fe.getMagsDatapath(), fe.getPosDatapath()]

def getOnlyPairedTesting():
    fe = fileExtractor(only_PairedTesting)
    return [fe.getLevsDatapath(), fe.getMagsDatapath(), fe.getPosDatapath()]

def getOnlyTrainingCoop():
    fe = fileExtractor(only_TrainingCoop)
    return [fe.getLevsDatapath(), fe.getMagsDatapath(), fe.getPosDatapath()]



class multiFileGraphs:
    def __init__(self, magFiles: List[str], levFiles: List[str], posFiles: List[str]):
        self.experiments = []
        
        if (len(magFiles) != len(levFiles) or len(magFiles) != len(posFiles)):
            raise ValueError("Different number of mag, lev, and pos files")
        
        for i in range(len(magFiles)):
            exp = singleExperiment(magFiles[i], levFiles[i], posFiles[i])
            self.experiments.append(exp)
    
    def magFileDataAvailabilityGraph(self):
        # Expected column structure
        expected_cats = self.experiments[0].mag.categories
    
        # Filter only experiments with correct columns
        filtered_experiments = []
        for exp in self.experiments:
            actual_cats = list(exp.mag.data.columns)
            if actual_cats == expected_cats:
                filtered_experiments.append(exp)
            #else:
                #print(f"Excluding {exp.mag.filename} due to mismatched columns.\nExpected: {expected_cats}\nGot: {actual_cats}")
    
        if not filtered_experiments:
            raise ValueError("No experiments with matching mag columns were found.")
    
        # Optional: update self.experiments to only valid ones
        self.experiments = filtered_experiments
    
        # Compute total rows and initialize null counter
        total_rows = sum(exp.mag.getNumRows() for exp in filtered_experiments)
        nulls_per_cat = {cat: 0 for cat in expected_cats}
    
        # Count nulls
        for exp in filtered_experiments:
            for cat in expected_cats:
                nulls_per_cat[cat] += exp.mag.countNullsinColumn(cat)
    
        # Compute non-null percentages
        pct = [(total_rows - nulls_per_cat[cat]) / total_rows * 100 for cat in expected_cats]
    
        # Plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(expected_cats, pct, color='steelblue')
        plt.xlabel('Categories')
        plt.ylabel('Percentage of Non-Null Data (%)')
        plt.title('Aggregated Data Availability in Mag Files')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 100)
        for bar in bars:
            y = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, y - 5, f'{y:.1f}%', ha='center', va='bottom')
        plt.tight_layout()
        plt.show()
        plt.savefig('mag_data_availability.png')
        
    def levFileDataAvailabilityGraph(self):
        # Expected column structure
        expected_cats = self.experiments[0].lev.categories
    
        # Filter only experiments with correct columns
        filtered_experiments = []
        for exp in self.experiments:
            actual_cats = list(exp.lev.data.columns)
            if actual_cats == expected_cats:
                filtered_experiments.append(exp)
            #else:
                #print(f"Excluding {exp.lev.filename} due to mismatched columns.\nExpected: {expected_cats}\nGot: {actual_cats}")
    
        if not filtered_experiments:
            raise ValueError("No experiments with matching mag columns were found.")
    
        # Optional: update self.experiments to only valid ones
        self.experiments = filtered_experiments
    
        # Compute total rows and initialize null counter
        total_rows = sum(exp.lev.getNumRows() for exp in filtered_experiments)
        nulls_per_cat = {cat: 0 for cat in expected_cats}
    
        # Count nulls
        for exp in filtered_experiments:
            for cat in expected_cats:
                nulls_per_cat[cat] += exp.lev.countNullsinColumn(cat)
    
        # Compute non-null percentages
        pct = [(total_rows - nulls_per_cat[cat]) / total_rows * 100 for cat in expected_cats]
    
        # Plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(expected_cats, pct, color='steelblue')
        plt.xlabel('Categories')
        plt.ylabel('Percentage of Non-Null Data (%)')
        plt.title('Aggregated Data Availability in Lev Files')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0,100)
        for bar in bars:
            y = max(bar.get_height(), 6)
            plt.text(bar.get_x()+bar.get_width()/2, y-5, f'{y:.1f}%', ha='center')
        plt.tight_layout()
        plt.show()
        plt.savefig('lev_data_availability.png')
    
    def interpressIntervalPlot(self):
        # concat all lev data
        all_lev = pd.concat([exp.lev.data for exp in self.experiments], ignore_index=True)
        if all_lev is None or all_lev.empty:
            raise ValueError("No lever data loaded.")
        
        all_lev = all_lev.sort_values(by=["RatID","AbsTime"])
        all_lev["IPI"] = all_lev.groupby("RatID")["AbsTime"].diff()
        ipi_data = all_lev[all_lev["IPI"].notna()]
        
        # Histogram
        plt.figure(figsize=(10,6))
        plt.hist(ipi_data["IPI"], bins=50, edgecolor='black')
        plt.xlabel("Inter-Press Interval (s)")
        plt.ylabel("Frequency")
        plt.title("Histogram of Inter-Press Intervals (All Rats)")
        plt.tight_layout()
        plt.show()
        plt.savefig('IPI_histogram.png')
        
        # Boxplot
        plt.figure(figsize=(10,6))
        all_lev.boxplot(column="IPI", by="RatID")
        plt.suptitle("") 
        plt.xlabel("RatID")
        plt.ylabel("IPI (s)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        plt.savefig('IPI_boxplot.png')
        
        # Time series
        plt.figure(figsize=(12,6))
        for rat_id, grp in all_lev.groupby("RatID"):
            plt.plot(grp["AbsTime"], grp["IPI"], label=str(rat_id), alpha=0.7)
        plt.title("IPI Over Time by Rat (All Files)")
        plt.xlabel("AbsTime (s)")
        plt.ylabel("IPI (s)")
        plt.legend(title="RatID", bbox_to_anchor=(1.05,1), loc='upper left')
        plt.tight_layout()
        plt.show()
        plt.savefig('IPI_timeseries.png')
    
    
    def interpressIntervalSuccessPlot(self):
        all_lev = pd.concat([exp.lev.data for exp in self.experiments], ignore_index=True)
        if all_lev is None or all_lev.empty:
            raise ValueError("No lever data loaded.")
        
        # filter successes
        succ = all_lev[all_lev["coopSucc"] == 1]
        grouped = succ.groupby("TrialNum")
        
        ipi_first, ipi_last, first_rats, success_counts = [], [], [], {}
        for _, trial in grouped:
            trial = trial.sort_values("AbsTime")
            if 1 not in trial["Hit"].values: continue
            coop_end = trial[trial["TrialEnd"]==1]
            if coop_end.empty: continue
            t_coop = coop_end.iloc[0]["AbsTime"]
            first = trial[trial["Hit"]==1].iloc[0]
            before = trial[trial["AbsTime"] < t_coop]
            if before.empty: continue
            last = before.iloc[-1]
            
            ipi_first.append(t_coop - first["AbsTime"])
            ipi_last.append(t_coop - last["AbsTime"])
            first_rats.append(first["RatID"])
            rid = coop_end.iloc[0]["RatID"]
            success_counts[rid] = success_counts.get(rid,0) + 1
        
        # Plot 1
        plt.figure(figsize=(8,6))
        plt.hist(ipi_first, bins=20)
        plt.title("First Press → Success IPI")
        plt.xlabel("IPI (s)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()
        plt.savefig('IPI_firstpress_to_success.png')
        
        # Plot 2
        plt.figure(figsize=(8,6))
        plt.hist(ipi_last, bins=20)
        plt.title("Last Press → Success IPI")
        plt.xlabel("IPI (s)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()
        plt.savefig('IPI_lastpress_to_success.png')
        
        # Plot 3
        plt.figure(figsize=(10,5))
        plt.plot(ipi_first, marker='o')
        plt.title("Time Series: First Press to Success")
        plt.xlabel("Trial Index")
        plt.ylabel("IPI (s)")
        plt.tight_layout()
        plt.show()
        plt.savefig('Timeseries_IPI_firstpress_to_success.png')
        
        # Plot 4
        plt.figure(figsize=(10,5))
        plt.plot(ipi_last, marker='o')
        plt.title("Time Series: Last Press to Success")
        plt.xlabel("Trial Index")
        plt.ylabel("IPI (s)")
        plt.tight_layout()
        plt.show()
        plt.savefig('Timeseries_IPI_lastpress_to_success.png')
        
        # Plot 5
        counts = pd.Series(first_rats).value_counts()
        plt.figure(figsize=(6,6))
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
        plt.title("Who Presses First")
        plt.tight_layout()
        plt.show()
        plt.savefig('Who_Presses_First.png')
    
    
    def percentSuccesfulTrials(self):
        all_lev = pd.concat([exp.lev.data for exp in self.experiments], ignore_index=True)
        
        grouped = all_lev.groupby("TrialNum")
        
        totalTrials, countSuccess = 0, 0
        for trial_num, trial_data in grouped:
            if (trial_data.iloc[0]['coopSucc'] == 1):
                countSuccess += 1
            
            totalTrials += 1
        
        countFail = totalTrials - countSuccess
        labels = ['Successful', 'Unsuccessful']
        sizes = [countSuccess, countFail]
        colors = ['green', 'red']
    
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 14})
        plt.title('Successful Cooperative Trials (%)', fontsize=16)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.tight_layout()
        plt.show()
        plt.savefig('Percent Successful.png')

#Testing Multi File Graphs
#
#

#arr = getAllValid()
#lev_files = arr[0]
#mag_files = arr[1]

#mag_files = ["/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral Quantification/Example Data Files/magData.csv", "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral Quantification/Example Data Files/ExampleMagFile.csv"]
#lev_files = ["/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral Quantification/Example Data Files/leverData.csv", "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral Quantification/Example Data Files/ExampleLevFile.csv"]

#experiment = multiFileGraphs(mag_files, lev_files, mag_files)
#experiment.magFileDataAvailabilityGraph()
#experiment.levFileDataAvailabilityGraph()
#experiment.interpressIntervalPlot()
#experiment.interpressIntervalSuccessPlot()
#experiment.percentSuccesfulTrials()
        
        

# ---------------------------------------------------------------------------------------------------------


##Class to create Graphs with data from multiple files with different categories + Functions to get data from all the different categories
#
#
        
class multiFileGraphsCategories:
    def __init__(self, magFiles: List[List[str]], levFiles: List[List[str]], posFiles: List[List[str]], categoryNames: List[str]):
        self.allFileGroupExperiments = []
        self.categoryNames = categoryNames
        self.numCategories = len(magFiles)

        if not (len(magFiles) == len(levFiles) == len(posFiles) == len(categoryNames)):
            raise ValueError("Mismatch between number of categories and provided file lists or category names.")

        for c in range(self.numCategories):
            file_group = []
            for mag, lev, pos in zip(magFiles[c], levFiles[c], posFiles[c]):
                exp = singleExperiment(mag, lev, pos)
                file_group.append(exp)
            self.allFileGroupExperiments.append(file_group)
        
        self.endSaveName = ""
        for cat in categoryNames:    
            self.endSaveName += f"_{cat}"
        
        self.endSaveName += ".png"
        #self.path = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Graphs/"
        self.path = ""
        
    def compareGazeEventsCategories(self):
        avg_events = []
        ts_data = []
        cum_event_data = []
        FRAME_WINDOW = 1800
    
        for group in self.allFileGroupExperiments:
            normalized_event_counts = []
            smoothed_gaze_arrays = []
            cumulative_event_arrays = []
    
            for exp in group:
                loader = exp.pos
                total_event_count = 0
                combined_gaze = None
                combined_cumulative = None
    
                for rat in [0, 1]:
                    is_gazing = loader.returnIsGazing(rat).astype(bool)
                    last_gaze = -5
                    event_count = 0
                    cumulative = np.zeros_like(is_gazing, dtype=int)
    
                    for i, gazing in enumerate(is_gazing):
                        if gazing:
                            if i - last_gaze >= 5:
                                event_count += 1
                            last_gaze = i
                        cumulative[i] = event_count
    
                    total_event_count += event_count
    
                    if combined_gaze is None:
                        combined_gaze = is_gazing.astype(float)
                        combined_cumulative = cumulative.astype(float)
                    else:
                        combined_gaze += is_gazing.astype(float)
                        combined_gaze = np.clip(combined_gaze, 0, 1)
                        combined_cumulative += cumulative.astype(float)
    
                norm_count = total_event_count / len(combined_gaze) * FRAME_WINDOW
                normalized_event_counts.append(norm_count)
                smoothed_gaze_arrays.append(combined_gaze)
                cumulative_event_arrays.append(combined_cumulative)
    
            avg_events.append(np.mean(normalized_event_counts))
    
            max_len = max(arr.shape[0] for arr in smoothed_gaze_arrays)
            padded_smoothed = np.array([
                np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan) for arr in smoothed_gaze_arrays
            ])
            padded_cumulative = np.array([
                np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan) for arr in cumulative_event_arrays
            ])
            ts_data.append(np.nanmean(padded_smoothed, axis=0))
            cum_event_data.append(np.nanmean(padded_cumulative, axis=0))
    
        # --- Plot 1: Bar chart of average gaze events per 1800 frames ---
        plt.figure(figsize=(8, 6))
        plt.bar(range(len(avg_events)), avg_events, color='skyblue')
        plt.xlabel('Category')
        plt.ylabel('Avg. Gaze Events per 1800 Frames')
        plt.title('Average Gaze Events (per Minute) per Category')
        plt.xticks(range(len(avg_events)), self.categoryNames)
        plt.tight_layout()
        plt.savefig(f'GazeEventsPerMinute{self.endSaveName}')
        plt.show()
    
        # --- Plot 2: Smoothed gaze fraction over time ---
        '''plt.figure(figsize=(10, 6))
        for idx, series in enumerate(ts_data):
            smooth = np.convolve(series, np.ones(100)/100, mode='same')
            plt.plot(smooth, label=self.categoryNames[idx])
        plt.xlabel('Frame')
        plt.ylabel('Fraction of Experiments Gazing')
        plt.title('Gaze Events Time Series per Category')
        plt.legend()
        plt.tight_layout()
        plt.show()'''
    
        # --- Plot 3: Cumulative gaze events over time ---
        plt.figure(figsize=(10, 6))
        for idx, cum_series in enumerate(cum_event_data):
            plt.plot(cum_series, label=self.categoryNames[idx])
        plt.xlabel('Frame')
        plt.ylabel('Cumulative Gaze Events')
        plt.title('Cumulative Gaze Events Over Time per Category')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.path}GazeEventsoverTime{self.endSaveName}')
        plt.show()


    def compareSuccesfulTrials(self):
        probs = []
        for group in self.allFileGroupExperiments:
            cat_probs = []
            for exp in group:
                lev = exp.lev.data
                total = lev['TrialNum'].nunique()

                print("lev.columns:", lev.columns.tolist())
                print("First few rows:\n", lev.head())
                succ = lev[lev['coopSucc'] == 1]['TrialNum'].nunique()
                
                cat_probs.append(succ / total if total > 0 else 0)
            probs.append(np.mean(cat_probs))

        plt.figure(figsize=(8, 6))
        plt.bar(range(len(probs)), probs, color='green')
        plt.xlabel('Category')
        plt.ylabel('Probability of Successful Trials')
        plt.title('Success Probability per Category')
        plt.xticks(range(len(probs)), self.categoryNames)
        plt.ylim(0, 1)
        plt.savefig(f'{self.path}ProbofSuccesfulTrial_{self.endSaveName}')
        plt.show()

    def compareIPI(self):
        avg_ipi = []
        avg_first_to = []
        avg_last_to = []

        for group in self.allFileGroupExperiments:
            ipis, firsts, lasts = [], [], []
            for exp in group:
                lev = exp.lev.data.copy()
                df = lev.sort_values(['RatID', 'AbsTime'])
                df['IPI'] = df.groupby('RatID')['AbsTime'].diff()
                ipis.extend(df['IPI'].dropna().tolist())

                for _, trial in lev.groupby('TrialNum'):
                    trial = trial.sort_values('AbsTime')
                    if trial['coopSucc'].iloc[0] != 1:
                        continue
                    coop = trial.query('TrialEnd==1')
                    if coop.empty:
                        continue
                    t_coop = coop['AbsTime'].iloc[0]
                    first = trial.query('Hit==1')
                    if not first.empty:
                        t_first = first['AbsTime'].iloc[0]
                        firsts.append(t_coop - t_first)
                    before = trial[trial['AbsTime'] < t_coop]
                    if not before.empty:
                        t_last = before['AbsTime'].iloc[-1]
                        lasts.append(t_coop - t_last)
            avg_ipi.append(np.mean(ipis) if ipis else np.nan)
            avg_first_to.append(np.mean(firsts) if firsts else np.nan)
            avg_last_to.append(np.mean(lasts) if lasts else np.nan)

        for title, data, color in zip(
            ['Avg IPI per Category', 'Avg First->Success per Category', 'Avg Last->Success per Category'],
            [avg_ipi, avg_first_to, avg_last_to],
            ['blue', 'skyblue', 'salmon']):
            plt.figure(figsize=(8, 6))
            plt.bar(range(len(data)), data, color=color)
            plt.xticks(range(len(data)), self.categoryNames)
            plt.xlabel('Category')
            plt.ylabel('Time (s)')
            plt.title(title)
            plt.savefig(f'{self.path}{title}{self.endSaveName}')
            plt.show()
            
    def make_bar_plot(self, data, ylabel, title, saveFileName):
        plt.figure(figsize=(8, 5))
        plt.bar(range(len(data)), data, color='skyblue')
        plt.xticks(range(len(data)), self.categoryNames)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f'{self.path}{saveFileName}{self.endSaveName}')
        plt.show()
    
    def printSummaryStats(self):
        avg_gaze_lengths = []
        avg_lever_per_trial = []
        avg_mag_per_trial = []
        
        for idx, group in enumerate(self.allFileGroupExperiments):
            total_gaze_events = 0
            total_frames = 0
            total_trials = 0
            successful_trials = 0
            total_lever_presses = 0
            total_mag_events = 0
            total_gaze_frames = 0

            for exp in group:
                loader = exp.pos
                g0 = loader.returnIsGazing(0)
                g1 = loader.returnIsGazing(1)
                combined = g0 | g1
                total_gaze_events += loader.returnNumGazeEvents(0) + loader.returnNumGazeEvents(1)
                total_gaze_frames += np.sum(combined) #Frames where at least 1 mouse was gazing
                total_frames += combined.shape[0]
                
                lev = exp.lev.data
                trials = lev['TrialNum'].nunique()
                succ = lev.groupby('TrialNum').first().query('coopSucc == 1').shape[0]
                total_trials += trials
                successful_trials += succ
                total_lever_presses += lev.shape[0]

                mag = exp.mag.data
                total_mag_events += mag.shape[0]
            
            avg_gaze_len = (total_gaze_frames / total_gaze_events) if total_gaze_events > 0 else 0
            avg_lever = (total_lever_presses / total_trials) if total_trials > 0 else 0
            avg_mag = (total_mag_events / total_trials) if total_trials > 0 else 0
    
            avg_gaze_lengths.append(avg_gaze_len)
            avg_lever_per_trial.append(avg_lever)
            avg_mag_per_trial.append(avg_mag)
            
            print(f"\nCategory: {self.categoryNames[idx]}")
            print(f"  Total Frames: {total_frames}")
            print(f"  Total Trials: {total_trials}")
            print(f"  Successful Trials: {successful_trials}")
            print(f"  Percent Successful: {successful_trials / total_trials:.2f}")
            print(f"  Frames Gazing: {total_gaze_frames}")
            print(f"  Total Gaze Events: {total_gaze_events}")
            print(f"  Average Gaze Length: {total_gaze_frames / total_gaze_events:.2f}")
            print(f"  Percent Gazing: {100 * total_gaze_events / total_frames:.2f}%")
            print(f"  Avg Lever Presses per Trial: {total_lever_presses / total_trials:.2f}")
            print(f"  Total Lever Presses: {total_lever_presses}")
            print(f"  Avg Mag Events per Trial: {total_mag_events / total_trials:.2f}")
            print(f"  Total Mag Events: {total_mag_events}")
           
        self.make_bar_plot(avg_gaze_lengths, 'Avg Gaze Length (frames)', 'Average Gaze Length per Category', "Avg_Gaze_Length")
        self.make_bar_plot(avg_lever_per_trial, 'Avg Lever Presses per Trial', 'Lever Presses per Trial per Category', "Avg_Lev_Presses_perTrial")
        self.make_bar_plot(avg_mag_per_trial, 'Avg Mag Events per Trial', 'Mag Events per Trial per Category', "Avg_Mag_Events_perTrial")
        

'''magFiles = [["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_lever.csv"], ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_lever.csv"]]
levFiles = [["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_mag.csv"], ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G_mag.csv"]]
posFiles = [["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G.predictions.h5"], ["/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum11_Coop_KL007Y-KL007G.predictions.h5"]]                   
categoryExperiments = multiFileGraphsCategories(levFiles, magFiles, posFiles, ["Paired_Testing", "Training_Cooperation"])'''



#Paired Testing vs. Training Cooperation
dataPT = getOnlyPairedTesting()
dataTC = getOnlyTrainingCoop()

magFiles = [dataPT[0], dataTC[0]]
levFiles = [dataPT[1], dataTC[1]]
posFiles = [dataPT[2], dataTC[2]]
categoryExperiments = multiFileGraphsCategories(levFiles, magFiles, posFiles, ["Paired_Testing", "Training_Cooperation"])



''' #Unfamiliar vs. Training Partners
dataUF = getOnlyUnfamiliar() #Unfamiliar
dataTP = getOnlyTrainingPartners() #Training Partners

magFiles = [dataUF[0], dataTP[0]]
levFiles = [dataUF[1], dataTP[1]]
posFiles = [dataUF[2], dataTP[2]]
categoryExperiments = multiFileGraphsCategories(magFiles, levFiles, posFiles, ["Unfamiliar", "Training Partners"])
'''


''' #Transparent vs. Translucent vs. Opaque
dataTransparent = getOnlyTransparent() #Transparent
dataTranslucent = getOnlyTranslucent() #Translucent
dataOpaque = getOnlyOpaque() #Opaque

magFiles = [dataTransparent[0], dataTranslucent[0], dataOpaque[0]]
levFiles = [dataTransparent[1], dataTranslucent[1], dataOpaque[1]]
posFiles = [dataTransparent[2], dataTranslucent[2], dataOpaque[2]]
categoryExperiments = multiFileGraphsCategories(magFiles, levFiles, posFiles, ["Transparent", "Translucent", "Opaque"])
'''


#Transparent vs. Translucent vs. Opaque
#dataTransparent = getOnlyPairedTesting() #Transparent
'''dataTranslucent = getonlyTranslucent() #Translucent
dataOpaque = getOnlyOpaque() #Opaque

magFiles = [dataTranslucent[0], dataOpaque[0]]
levFiles = [dataTranslucent[1], dataOpaque[1]]
posFiles = [dataTranslucent[2], dataOpaque[2]]
categoryExperiments = multiFileGraphsCategories(magFiles, levFiles, posFiles, ["Translucent", "Opaque"])'''



categoryExperiments.compareGazeEventsCategories()
categoryExperiments.compareSuccesfulTrials()
categoryExperiments.compareIPI()
categoryExperiments.printSummaryStats()
        