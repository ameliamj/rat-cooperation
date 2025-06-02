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
    def __init__(self, mag_file, lev_file, pos_file):
        self.mag_file = mag_file
        self.lev_file = lev_file
        self.pos_file = pos_file
        self.mag = magLoader(mag_file)
        self.lev = levLoader(lev_file)
        self.pos = posLoader(pos_file)
    