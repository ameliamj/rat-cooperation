#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 17:57:51 2025

@author: david
"""

import matplotlib.pyplot as plt
import numpy as np

from experiment import singleExperiment
from typing import List

#Class to create Graphs with data from a single file
#
#


class singleFileGraphs:
    def __init__(self, mag_file, lev_file, pos_file):
        self.experiment = singleExperiment(mag_file, lev_file, pos_file)
        
    def interpressIntervalPlot(self):
        leverNums = self.experiment.lev.getColumn("LeverNum")
        absTimes = self.experiment.lev.getAbsTime("AbsTime")
        
        
#Testing Single File Graphs
#
#

mag_file = ""
lev_file = ""
pos_file = ""

experiment = singleFileGraphs(mag_file, lev_file, pos_file)


# ---------------------------------------------------------------------------------------------------------


#Class to create Graphs with data from multiple files
#
#


class multiFileGraphs:
    def __init__(self, magFiles: List[str], levFiles: List[str], posFiles: List[str]):
        self.experiments = []
        
        if (len(magFiles) != len(levFiles) or len(magFiles) != len(posFiles)):
            raise ValueError("different number of mag, lev, and pos files")
        
        for i in range(len(magFiles)):
            experiment = singleExperiment(magFiles[i], levFiles[i], posFiles[i])
            self.experiments.append(experiment)
            

#Testing Multi File Graphs
#
#


            
        
        
        
        
    
        
        
        