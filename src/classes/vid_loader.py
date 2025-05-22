import sys
sys.path.append('../../')

import os
import numpy as np
from src.utils.error_utils import get_color

class VidLoader:
    # given cohorts and file path will create dictionaries of all videos in test and train directories
    # and split these videos into single and multi animal and into respective color pairs
    def __init__(self, cohorts = ['KL', 'EB', 'HF'], filepath="/gpfs/radev/pi/saxena/aj764/", out=True):
        self.cohorts = cohorts # TODO: will need to add other cohorts to this
        self.rootdir = filepath
        self.test_dir = 'PairedTestingSessions/'
        self.train_dir = 'Training_COOPERATION/'

        self.pts_single_vids, self.pts_multi_vids, self.tc_multi_vids = self.get_vids()
        self.pts_color_vids = self.get_color_vids(self.pts_multi_vids, 'test')        
        self.tc_color_vids = self.get_color_vids(self.tc_multi_vids, 'train')

        if out:
            self.print_output()

    # creates dictionaries that have all of the videos from PairedTestingSessions and 
    # Training_COOPERATION
    # NOTE: SPECIFICALLY ONLY FOR APRIL 2024 THROUGH NOVEMBER 2024, because these are the 
    # only vides with dye
    def get_vids(self):
        # get vids from paired testing session
        vid_subdirs = []
        for subdir, dirs, files in os.walk(self.rootdir + self.test_dir):
            if subdir.endswith("Videos"):
                vid_subdirs.append(subdir)
        vid_subdirs = sorted(vid_subdirs)
        
        pts_single_vids = {}
        pts_multi_vids = {}
        for vids in vid_subdirs:
            files = os.listdir(vids)
            cut_vids = vids[len(self.rootdir):]
            pts_single_vids[cut_vids] = []
            pts_multi_vids[cut_vids] = []
            for file in files:
                if file.endswith('.mp4') and self.in_date_range(file):
                    coh_count = 0
                    for coh in self.cohorts:
                        coh_count += file.count(coh) 
                    if coh_count == 2:
                        pts_multi_vids[cut_vids].append(file)
                    else:
                        pts_single_vids[cut_vids].append(file)
            if len(pts_single_vids[cut_vids]) == 0:
                pts_single_vids.pop(cut_vids)
            if len(pts_multi_vids[cut_vids]) == 0:
                pts_multi_vids.pop(cut_vids)

        # get vids from training cooperation
        vid_subdirs = []
        for subdir, dirs, files in os.walk(self.rootdir + self.train_dir):
            vid_subdirs.append(subdir)
        vid_subdirs = sorted(vid_subdirs)
        
        tc_multi_vids = {}
        for vids in vid_subdirs:
            files = os.listdir(vids)
            cut_vids = vids[len(self.rootdir):]
            tc_multi_vids[cut_vids] = []
            for file in files:
                if file.endswith('.mp4') and self.in_date_range(file):
                    tc_multi_vids[cut_vids].append(file)
            if len(tc_multi_vids[cut_vids]) == 0:
                tc_multi_vids.pop(cut_vids) 

        return pts_single_vids, pts_multi_vids, tc_multi_vids

    # reorgs all of the multi vids into a dictionary by the color pair in the multi vid
    def get_color_vids(self, multi_vids, trial_type):
        color_vids = {}
        for key, value in multi_vids.items():
            for vid in value:
                trial_key = get_color(vid, trial_type)
                if trial_key not in color_vids.keys():
                    color_vids[trial_key] = []
                color_vids[trial_key].append(vid)
        return color_vids

    # prints out how many of each type of video there are
    def print_output(self):
        print(f'these are how many videos there are in {self.test_dir}')
        print(f'There are {self.get_num_vids(self.pts_single_vids.items())} single instance videos')
        print(f'There are {self.get_num_vids(self.pts_multi_vids.items())} multi instance videos')
        color_len = self.get_num_vids(self.pts_color_vids.items(), color=True)
        print(f'There are {color_len} multi instance videos')

        print()
        
        print(f'these are how many videos there are in {self.train_dir}')
        print(f'There are {self.get_num_vids(self.tc_multi_vids.items())} multi instance videos')
        color_len = self.get_num_vids(self.tc_color_vids.items(), color=True)
        print(f'There are {color_len} multi instance videos')

    # makes sure the video is in the date range from April 2024 and November 2024 (non-inclusive)
    # because these are the dyed color pairs
    def in_date_range(self, file):
        month = int(file[:2])
        year = int(file[4:6])
        return (month >= 4 and month < 11 and year == 24)

    # gets the nubmer of videos that are stored in the given vid dict
    @staticmethod
    def get_num_vids(vid_dict, color=False):
        if color:
            tot_len = 0
            for key, value in vid_dict.items():
                print(f'There are {len(value)} videos from {key} color pair')
                tot_len += len(value)
        else:
            tot_len = 0
            for key, value in vid_dict.items():
                tot_len += len(value)
        return tot_len        

    