#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 11:54:14 2025

@author: david
"""

import pandas as pd

from mag_class import magLoader
from lev_class import levLoader
from pos_class import posLoader
from fiberPhotoClass import fiberPhotoLoader

class singleExperiment:
    def __init__(self, mag_file, lev_file, pos_file, fps = 30, endFrame = 25182, initialNan = 0.1, fp_files = None, fiberPhoto = "False", trainingPartner = "Training Partner", transparency = "Transparent", ratPair = "", numSessionsBefore = 0, sessionID = "BLANK", rat1 = "rat1", rat2 = "rat2", date = "01/01/2000"):
        self.mag_file = mag_file
        self.lev_file = lev_file
        self.pos_file = pos_file
        
        #Session Data
        self.familiarity = trainingPartner
        self.transparency = transparency
        self.ratPair = ratPair
        self.numSessionsBefore = numSessionsBefore
        self.date = date
        self.sessionID = sessionID
        self.fiberPhoto = fiberPhoto
        
        #self.pos = posLoader(pos_file, endFrame)
        #self.endFrame = self.pos.returnNumFrames()
        self.endFrame = endFrame
        self.rat1 = rat1
        self.rat2 = rat2
        
        if (mag_file != None and lev_file != None):
            self.mag = magLoader(mag_file, fps=fps)
            self.lev = levLoader(lev_file, self.endFrame, fps)
        if (not fp_files is None and len(fp_files) == 3):
            self.fp = fiberPhotoLoader(fp_files[0], fp_files[1], fp_files[2])
        
        self.fps = fps
        self.initialNan = initialNan
    
    def deleteSub100msTrials(self):
        '''
        Deletes all rows in lev.data and pos.data associated with trials where any
        TrialTime is below 0.1 seconds.
        '''
        # Find all trial numbers with any TrialTime under 0.1
        bad_trials = self.lev.data.loc[self.lev.data["TrialTime"] < 0.1, "TrialNum"].unique()
        
        # Filter out rows from lev.data where TrialNum is in bad_trials
        self.lev.data = self.lev.data[~self.lev.data["TrialNum"].isin(bad_trials)]
    
        # Filter out rows from pos.data where TrialNum is in bad_trials
        self.pos.data = self.pos.data[~self.pos.data["TrialNum"].isin(bad_trials)]
    
    def unitTest(self):
        print("Mag Unit Tests")
        print("    Total Mag Events: ", self.mag.getTotalMagEvents())
        
        print("\n\n Lev Unit Tests")
        print("    Successful Trials: ", self.lev.returnNumSuccessfulTrials())
        print("    Total Trials: ", self.lev.returnNumTotalTrials())
        print("    Total Lever Presses: ", self.lev.returnTotalLeverPresses())
        print("    First 5 IPIs: ", self.lev.returnAvgIPI(test = True))
        print("    Len (IPIs): ", len(self.lev.returnAvgIPI(returnList = True)))
        print("    First 5 IPIs First->Success: ", self.lev.returnAvgIPI_FirsttoSuccess(True))
        print("    Len (IPIs First->Success): ", len(self.lev.returnAvgIPI_FirsttoSuccess(returnList = True)))
        print("    First 5 IPIs Last->Success: ", self.lev.returnAvgIPI_LasttoSuccess(True))
        
        
        print("\n\n Pos Unit Tests")
        print("    When is Gazing: ", self.pos.returnIsGazing(0, True))
    
    
#mag = "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_mag.csv"
#lev = "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G_lever.csv"
#pos = "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Example_Data_Files/041824_Cam3_TrNum5_Coop_KL007Y-KL007G.predictions.h5"

#lev = "/Users/david/Downloads/041024_EB001R-EB003B_lever.csv"

#test = singleExperiment(mag, lev, pos)

#test.unitTest()








