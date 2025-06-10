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

class singleExperiment:
    def __init__(self, mag_file, lev_file, pos_file, fps = 30, endFrame = 5000):
        self.mag_file = mag_file
        self.lev_file = lev_file
        self.pos_file = pos_file
        
        self.mag = magLoader(mag_file)
        self.lev = levLoader(lev_file, endFrame, fps)
        self.pos = posLoader(pos_file, endFrame)
        self.fps = fps
        self.endFrame = endFrame
    
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