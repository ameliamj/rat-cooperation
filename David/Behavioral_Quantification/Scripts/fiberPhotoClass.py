#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 15:13:09 2025

@author: david
"""

import panda as pd
import numpy as np

class fiberPhotoLoader:
    def __init__(self, x405_path, x465_path, x560_path):
        '''
        Reads and Stores the x405, x465, and x560 data around 4 events
        
        Events: 
            1) ttlLabels('1') = 'Session Start'
            2) ttlLabels('2') = 'Left Lever Press'
            3) ttlLabels('4') = 'Right Lever Press'
            4) ttlLabels('8') = 'Left Magazine Entry'
            5) ttlLabels('16') = 'Right Magazine Entry'
        '''
        
        self.x405_path = x405_path
        self.x465_path = x465_path
        self.x560_path = x560_path
        
        self.x405 = None
        self.x465 = None
        self.x560 = None
        self._load_data()
        
    def _load_data(self): #Uses pandas to read csv file and store in a pandas datastructure
        """
        Load the CSV file into a pandas DataFrame.
        Handles file not found or malformed CSV errors.
        """
        
        try:
            self.x405 = pd.read_csv(self.x405, sep=',', na_values=[''])
            # Ensure numeric columns are properly typed
            for col in self.data.columns:
                if self.data[col].dtype == 'object':
                    # Try converting to numeric, but preserve strings if not possible
                    try:
                        self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                    except:
                        pass
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at: {self.x405_path}")
        except pd.errors.ParserError:
            raise ValueError("Error parsing CSV file. Ensure it is properly formatted.")
            
        try:
            self.x465 = pd.read_csv(self.x465, sep=',', na_values=[''])
            # Ensure numeric columns are properly typed
            for col in self.data.columns:
                if self.data[col].dtype == 'object':
                    # Try converting to numeric, but preserve strings if not possible
                    try:
                        self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                    except:
                        pass
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at: {self.x465_path}")
        except pd.errors.ParserError:
            raise ValueError("Error parsing CSV file. Ensure it is properly formatted.")
        
        try:
            self.x560 = pd.read_csv(self.x560, sep=',', na_values=[''])
            # Ensure numeric columns are properly typed
            for col in self.data.columns:
                if self.data[col].dtype == 'object':
                    # Try converting to numeric, but preserve strings if not possible
                    try:
                        self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                    except:
                        pass
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at: {self.x560_path}")
        except pd.errors.ParserError:
            raise ValueError("Error parsing CSV file. Ensure it is properly formatted.")
    
    def getLevAverage405(self):
        '''
        '''
        a = 1
    
    def getLevAverage465(self):
        '''
        '''
        a = 1
        
    def getLevAverage560(self):
        '''
        '''
        a = 1

            
    