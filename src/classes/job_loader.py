# need to define centroid and topdown, should be dicts where (color_type, color_pair) is the key!
import sys
sys.path.append('../../')

import os
import pandas as pd
from src.utils.global_utils import CENTROID, TOPDOWN, SINGLE
from src.utils.global_utils import ROOTDIR, TESTDIR, TRAINDIR, JOBDIR

class JobLoader:

    def __init__(self, filename, color_type):
        self.df = pd.read_csv(filename)
        self.color_type = color_type

    def get_undone_vids(self, inst, color_pair=None):
        subset = self.df[(self.df['single/multi'] == inst)]
        if inst == 'multi' and color_pair is not None:
            subset = subset[(subset['color pair'] == color_pair)]
        run = subset[subset['pred'] == False]
        return run, subset

    def get_job_script(self, inst, color_pair=None, write=False):
        run, _ = self.get_undone_vids(inst, color_pair)

        start_command = f'module load miniconda; conda activate sleap; cd {ROOTDIR};'
        command_lines = ''
        for index, row in run.iterrows():
            tt = TESTDIR if row['test/train'] == 'test' else TRAINDIR
            output_path = ROOTDIR + tt + row['session'] + '/Tracking'
            # makes directory for tracking output if not already made
            if not os.path.isdir(output_path):
                os.mkdir(output_path)
            if not os.path.isdir(output_path + '/slp'):
                os.mkdir(output_path + '/slp')
            if not os.path.isdir(output_path + '/h5'):
                os.mkdir(output_path + '/h5')

            if row['test/train'] == 'test':
                video_path = ROOTDIR + tt + row['session'] + '/Video/' + row['vid'] + '.mp4'
            else:
                video_path = ROOTDIR + tt + row['session'] + '/' + row['vid'] + '.mp4'
            output_file = row['vid'] + 'predictions.'

            if inst == 'single':
                model = SINGLE
                track_command = f'sleap-track "{video_path}" --first-gpu -o "{output_path + '/slp/' + output_file + 'slp'}" -m "{model}"/'
            else:
                centroid_model = CENTROID[(self.color_type, color_pair)]
                topdown_model = TOPDOWN[(self.color_type, color_pair)]
    
                track_command = f'sleap-track "{video_path}" --first-gpu -o "{output_path + '/slp/' + output_file + 'slp'}" -m "{centroid_model}" -m "{topdown_model}"'
            convert_command = f'; sleap-convert --format analysis -o "{output_path + '/h5/' + output_file + 'h5'}" "{output_path + '/slp/' + output_file + 'slp'}"'
            command_lines += (start_command + track_command + convert_command + '\n')

        if write:
            if not os.path.isdir(JOBDIR): 
                os.mkdir(JOBDIR)
            with open(f"{JOBDIR}/{color_pair}_vids_job.txt", "w") as file:
                file.write(command_lines) 
        return command_lines

    def get_progress(self, inst, color_pair=None):
        undone, subset = self.get_undone_vids(inst, color_pair)
        per_done = round(((subset.shape[0] - undone.shape[0]) / subset.shape[0]) * 100, 2)
        print(f'{per_done}% of videos from {color_pair} have been tracked ({subset.shape[0] - undone.shape[0]} tracked videos, {undone.shape[0]} untracked videos)')
        
        