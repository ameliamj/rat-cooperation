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

        # Plot 6: Pie Chart - Number of Trials with Successful Cooperation (by RatID)
        success_counts = pd.Series(success_count_by_rat)
        plt.figure(figsize=(6, 6))
        plt.pie(success_counts, labels=success_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title("Number of Successful Cooperative Trials (by RatID)")
        plt.show()

        
        
    
#Testing Single File Graphs
#
#

'''mag_file = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral Quantification/Example Data Files/magData.csv"
lev_file = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral Quantification/Example Data Files/leverData.csv"
pos_file = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral Quantification/Example Data Files/ExampleTrackingCoop.h5"

experiment = singleFileGraphs(mag_file, lev_file, pos_file)
experiment.magFileDataAvailabilityGraph()
experiment.levFileDataAvailabilityGraph()
experiment.interpressIntervalPlot()
experiment.interpressIntervalSuccessPlot()'''


# ---------------------------------------------------------------------------------------------------------


#Class to create Graphs with data from multiple files + Functions to get data from all the different categories
#
#

#all_valid = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral Quantification/Sorted Data Files/dyed_preds_all_valid.csv"
all_valid = "/gpfs/radev/home/drb83/project/rat-cooperation/David/Behavioral Quantification/Sorted Data Files/dyed_preds_all_valid.csv"
only_opaque = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral Quantification/Sorted Data Files/only_opaque_sessions.csv"
only_translucent = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral Quantification/Sorted Data Files/only_translucent_sessions.csv"
only_transparent = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral Quantification/Sorted Data Files/only_transparent_sessions.csv"
only_unfamiliar = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral Quantification/Sorted Data Files/only_unfamiliar_partners.csv"
only_trainingpartners = "/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral Quantification/Sorted Data Files/only_training_partners.csv"

def getAllValid():
    fe = fileExtractor(all_valid)
    return [fe.getLevsDatapath(), fe.getMagsDatapath()]
    
def getOnlyOpaque():
    fe = fileExtractor(all_valid)
    return [fe.getLevsDatapath(), fe.getMagsDatapath()]

def getonlyTranslucent():
    fe = fileExtractor(all_valid)
    return [fe.getLevsDatapath(), fe.getMagsDatapath()]

def getOnlyTransparent():
    fe = fileExtractor(all_valid)
    return [fe.getLevsDatapath(), fe.getMagsDatapath()]

def getOnlyUnfamiliar():
    fe = fileExtractor(all_valid)
    return [fe.getLevsDatapath(), fe.getMagsDatapath()]

def getOnlyTrainingPartners():
    fe = fileExtractor(all_valid)
    return [fe.getLevsDatapath(), fe.getMagsDatapath()]


class multiFileGraphs:
    def __init__(self, magFiles: List[str], levFiles: List[str], posFiles: List[str]):
        self.experiments = []
        
        if (len(magFiles) != len(levFiles) or len(magFiles) != len(posFiles)):
            raise ValueError("Different number of mag, lev, and pos files")
        
        for i in range(len(magFiles)):
            exp = singleExperiment(magFiles[i], levFiles[i], posFiles[i])
            self.experiments.append(exp)
    
    def magFileDataAvailabilityGraph(self):
        # Gather totals
        cats_init = self.experiments[0].mag.categories
        total_rows = sum(exp.mag.getNumRows() for exp in self.experiments)
        nulls_per_cat = {cat: 0 for cat in cats_init}
        
        # Sum nulls across exps
        for i, exp in enumerate(self.experiments):
            cats = exp.mag.categories
            if (len(cats) != len(cats_init)):
                self.experiments.pop(i)
                continue
                
            for cat in cats:
                nulls_per_cat[cat] += exp.mag.countNullsinColumn(cat)
        
        # Compute percentages
        pct = [ (total_rows * len(cats) - nulls_per_cat[cat]) / total_rows * 100
                for cat in cats ]
        
        # Plot
        plt.figure(figsize=(10,6))
        bars = plt.bar(cats, pct)
        plt.xlabel('Categories')
        plt.ylabel('Percentage of Non-Null Data (%)')
        plt.title('Aggregated Data Availability in Mag Files')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0,100)
        for bar in bars:
            y = bar.get_height()
            plt.text(bar.get_x()+bar.get_width()/2, y-5, f'{y:.1f}%', ha='center')
        plt.tight_layout()
        plt.show()
        
    def levFileDataAvailabilityGraph(self):
        cats_init = self.experiments[0].lev.categories
        total_rows = sum(exp.lev.getNumRows() for exp in self.experiments)
        nulls_per_cat = {cat: 0 for cat in cats_init}
        
        for i, exp in enumerate(self.experiments):
            cats = exp.lev.categories
            
            if (len(cats) != len(cats_init)):
                self.experiments.pop(i)
                continue
        
            for cat in cats:
                nulls_per_cat[cat] += exp.lev.countNullsinColumn(cat)
        
        pct = [ (total_rows * len(cats) - nulls_per_cat[cat]) / total_rows * 100
                for cat in cats ]
        
        plt.figure(figsize=(10,6))
        bars = plt.bar(cats, pct)
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
        
        # Boxplot
        plt.figure(figsize=(10,6))
        all_lev.boxplot(column="IPI", by="RatID")
        plt.suptitle("") 
        plt.xlabel("RatID")
        plt.ylabel("IPI (s)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
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
        
        # Plot 2
        plt.figure(figsize=(8,6))
        plt.hist(ipi_last, bins=20)
        plt.title("Last Press → Success IPI")
        plt.xlabel("IPI (s)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()
        
        # Plot 3
        plt.figure(figsize=(10,5))
        plt.plot(ipi_first, marker='o')
        plt.title("Time Series: First Press to Success")
        plt.xlabel("Trial Index")
        plt.ylabel("IPI (s)")
        plt.tight_layout()
        plt.show()
        
        # Plot 4
        plt.figure(figsize=(10,5))
        plt.plot(ipi_last, marker='o')
        plt.title("Time Series: Last Press to Success")
        plt.xlabel("Trial Index")
        plt.ylabel("IPI (s)")
        plt.tight_layout()
        plt.show()
        
        # Plot 5
        counts = pd.Series(first_rats).value_counts()
        plt.figure(figsize=(6,6))
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
        plt.title("Who Presses First")
        plt.tight_layout()
        plt.show()
        
        # Plot 6
        sc = pd.Series(success_counts)
        plt.figure(figsize=(6,6))
        plt.pie(sc, labels=sc.index, autopct='%1.1f%%', startangle=140)
        plt.title("Successful Cooperative Trials by Rat")
        plt.tight_layout()
        plt.show()
    
            
#Testing Multi File Graphs
#
#

arr = getAllValid()
lev_files = arr[0]
mag_files = arr[1]

experiment = multiFileGraphs(mag_files, lev_files, mag_files)
experiment.magFileDataAvailabilityGraph()
experiment.levFileDataAvailabilityGraph()
experiment.interpressIntervalPlot()
experiment.interpressIntervalSuccessPlot()
        
        

# ---------------------------------------------------------------------------------------------------------


##Class to create Graphs with data from multiple files with different categories + Functions to get data from all the different categories
#
#
        
class multiFileGraphsCategories:
    def __init__(self, magFiles: List[[str]], levFiles: List[[str]], posFiles: List[[str]], numCategories: int):
        self.allFileGroupExperiments = []
        
        if (len(magFiles) != len(levFiles) or len(magFiles) != len(posFiles)):
            raise ValueError("Different number of mag, lev, and pos files")
        
        length = len(magFiles[0])
        for i in numCategories:
            if (length != magFiles[i] or length != levFiles[i] or length != posFiles[i]):
                raise ValueError("Different number of mag, lev, and pos files at index ", i)
        
        for i in range(numCategories):
            singleFileGroupExperiments = []
            for j in range(len(magFiles)):
                experiment = singleExperiment(magFiles[j], levFiles[j], posFiles[j])
                singleFileGroupExperiments.append(experiment)
            
            self.allFileGroupExperiments.append(singleFileGroupExperiments)
        
    
        
        
        