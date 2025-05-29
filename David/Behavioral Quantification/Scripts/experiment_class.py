#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 11:54:14 2025

@author: david
"""

import pandas as pd

from mag_class import magLoader
from lev_class import levLoader

class singleExperiment:
    def __init__(self, mag_file, lev_file, pos_file):
        self.mag = magLoader(mag_file)
        self.lev = levLoader(lev_file)
    