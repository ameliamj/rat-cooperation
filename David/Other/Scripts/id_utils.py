#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 15:33:35 2025

@author: david
"""

import numpy as np
import pandas as pd
import cv2 
from .error_utils import load_file
from .global_utils import levx, loc2y, loc1y, magx
from .global_utils import ROOTDIR, TESTDIR, TRAINDIR

def get_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps

def construct_path():
    video_path = ROOTDIR #Finish
    return video_path

# given a row of the data frame from PredLoader, will add the rat id to each lever and mag
# event or return that a lever / mag file doesn't exist for this trial
def get_lever_mag(row):
    tt = TESTDIR if row['test/train'] == 'test' else TRAINDIR
    behav = '/Behavioral/processed/' if row['test/train'] == 'test' else '/Behavioral/'
    session = row['session']
        
    if row['test/train'] == 'test': 
        vid = row['vid']
    else:
        vid_temp = row['vid'].split('_')
        vid = vid_temp[0] + '_'  + vid_temp[3]
    if row['pred'] == True:
        locations = load_file(row)
        
    lever_exists, mag_exists = True, True
    try:
        lever = pd.read_csv(ROOTDIR + tt + session + behav + 'lever/' +  vid + '_lever.csv')
        if row['pred'] == True:
            lever = get_rat_id(lever, locations, 'lever')
            lever.to_csv(ROOTDIR + tt + session + behav + 'lever/' +  vid + '_lever.csv', index=False)
    except FileNotFoundError:
        lever_exists = False
        
    try:
        mag = pd.read_csv(ROOTDIR + tt + session + behav + 'mag/' + vid + '_mag.csv')
        if row['pred'] == True:
            mag = get_rat_id(mag, locations, 'mag')
            mag.to_csv(ROOTDIR + tt + session + behav + 'mag/' +  vid + '_mag.csv', index=False)
    except FileNotFoundError:
        mag_exists = False 
    
    return lever_exists, mag_exists
    
# for a given list of events and locations and event type (mag/lever), will add an 
# additional column to events that has the identity of which rat particapted in the 
# event given the rat locations
def get_rat_id(events, locations, event_type):
    ratID = []
    
    videoPath = construct_path()
    fps = get_fps(videoPath)

    for row in events.itertuples(index=False):
        # Calculate the frame for every lever press
        if np.isnan(row.AbsTime):
            ratNum = np.nan
            ratID.append(ratNum)
            continue
        else:
            frame = int(row.AbsTime * fps)
            if frame > locations.shape[0]:
                ratNum = np.nan
                ratID.append(ratNum)
                continue
            else:
                # Get coordinates of both mice for the said frame
                ratpos1 = locations[frame, 0, :, 0]
                ratpos2 = locations[frame, 0, :, 1]
                
                # Calculate the distances
                if event_type == 'lever':
                    if row.LeverNum == 1:
                        distance1 = np.sqrt((ratpos1[0] - levx)**2 + (ratpos1[1] - loc1y)**2)
                        distance2 = np.sqrt((ratpos2[0] - levx)**2 + (ratpos2[1] - loc1y)**2)
                
                    elif row.LeverNum == 2:
                        distance1 = np.sqrt((ratpos1[0] - levx)**2 + (ratpos1[1] - loc2y)**2)
                        distance2 = np.sqrt((ratpos2[0] - levx)**2 + (ratpos2[1] - loc2y)**2)
                    else:
                        ratNum = np.nan #assign number for Nan values
                        ratID.append(ratNum)
                        continue  # Skip if LeverNum is not 1 or 2
                elif event_type == 'mag':
                    if row.MagNum == 1:
                        distance1 = np.sqrt((ratpos1[0] - magx)**2 + (ratpos1[1] - loc1y)**2)
                        distance2 = np.sqrt((ratpos2[0] - magx)**2 + (ratpos2[1] - loc1y)**2)
            
                    elif row.MagNum == 2:
                        distance1 = np.sqrt((ratpos1[0] - magx)**2 + (ratpos1[1] - loc2y)**2)
                        distance2 = np.sqrt((ratpos2[0] - magx)**2 + (ratpos2[1] - loc2y)**2)
                
                    else:
                        ratNum = np.nan #assign number for Nan values
                        ratID.append(ratNum)
                        continue  # Skip if MagNum is not 1 or 2
                else:
                    raise Exception("not a valid event type (yikes)")
            
                # Assign the action to one of the mice
                ratNum = 0 if distance1 < distance2 else 1
            
                # Add new element to the list
                ratID.append(ratNum)
    
    # Add new column to the dataframe
    events["RatID"] = ratID
    return events