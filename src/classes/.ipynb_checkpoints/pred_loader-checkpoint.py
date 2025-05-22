import sys
sys.path.append('../../')

import pandas as pd
from src.utils.error_utils import get_vs, get_color, load_file, get_nan_prec, nan_good, high_vel_nan, lowess_fill, save_file
from src.utils.id_utils import get_lever_mag
from tqdm import tqdm

class PredLoader:

    def __init__(self, filename):
        self.df = pd.read_csv(filename)
        self.filename = filename

    # creates a pred loader object from a vid loader object, takes a while!
    @classmethod
    def from_vids(cls, vid_loader, filename='preds_df.csv'):
        df = pd.DataFrame(columns=['vid', 'session', 'single/multi', 'test/train', 'pred',
                                      'color pair', 'initial nan',  'vel nan', 'correct', 'final nan'])
        # gathers all of the pred files from a vid loader object
        i = 0
        for key, value in vid_loader.pts_single_vids.items():
            for vid in value:
                v,s = get_vs(key, vid)
                df.loc[i] = [v, s, 'single','test', False, None, -1, -1, False, -1]
                i += 1
        for key, value in vid_loader.pts_multi_vids.items():
            for vid in value:
                v,s = get_vs(key, vid)
                cp = get_color(vid, 'test')
                df.loc[i] = [v, s, 'multi','test', False, cp, -1, -1, False, -1]
                i += 1
        for key, value in vid_loader.tc_multi_vids.items():
            for vid in value:
                v,s = get_vs(key, vid)
                cp = get_color(vid, 'train')
                df.loc[i] = [v, s, 'multi','train', False, cp, -1, -1, False, -1]
                i += 1

        # fill in initial nan, vel nan, correction true/false
        for index, row in df.iterrows():
            locations = load_file(row)
            print(row)
            if locations is not None:
                df.at[index, 'pred'] = True
                initial_nan = get_nan_prec(locations)
                df.at[index, 'initial nan'] = initial_nan
                locations = high_vel_nan(locations)
                vel_nan = get_nan_prec(locations)
                df.at[index, 'vel nan'] = vel_nan

                df.at[index, 'correct'] = nan_good(initial_nan, vel_nan)
        df.to_csv(filename, index=False)
        return cls(filename)

    # correct files and get final nan
    def correct(self):
        correct_df = self.df.loc[self.df['correct'] == True]
        for index, row in tqdm(correct_df.iterrows()):
            locations = load_file(row)

            if locations is not None and row['correct']: # just double checking for no good reason
                locations = high_vel_nan(locations)
                locations = lowess_fill(locations)
                self.df.at[index, 'final nan'] = get_nan_prec(locations)
                
                save_file(row, locations)
        self.df.to_csv(self.filename, index=False)

    # fill in the rat id for each lever and mag event
    def get_event_ids(self):
        levers = []
        mags = []
        for index, row in tqdm(self.df.iterrows()):
            lever, mag = get_lever_mag(row)
            levers.append(lever)
            mags.append(mag)
        self.df['levers'] = levers
        self.df['mags'] = mags
        self.df.to_csv(self.filename, index=False)

    # get the trial type (coop, ineq, comp) for multi animal test vids
    def get_trial_types(self):
        for index, row in self.df.iterrows():
            if row['single/multi'] == 'single':
                self.df.at[index, 'trial type'] = 'single'
            elif row['test/train'] == 'train' or 'Coop' in row['vid']:
                self.df.at[index, 'trial type'] = 'coop'
            elif 'Comp' in row['vid']:
                self.df.at[index, 'trial type'] = 'comp'
            elif 'Ineq' in row['vid']:
                self.df.at[index, 'trial type'] = 'ineq'
        self.df.to_csv(self.filename, index=False)

    # get familiaritiy level (TP, UF, CM) for multi animal test vids
    def get_familiar(self):
        test_df = self.df[(self.df['test/train'] == 'test') & (self.df['single/multi'] == 'multi')]
        sessions = test_df['session'].unique()
        total_coop = 0
        coop_tp = 0
        coop_uf = 0
        coop_cm = 0 
        
        missing_count = 0
        for sesh in sessions:
            files = os.listdir(rootdir + sesh)
            for file in files:
                if file.endswith('.xlsx'):
                    excel_df = pd.read_excel(rootdir + sesh + '/' + file, engine='openpyxl')
                    excel_df = excel_df[excel_df['Cond'] == 'Coop']
        
                    # count number of different cooperative pairs
                    total_coop += excel_df.shape[0]
                    try:
                        coop_tp += excel_df[excel_df['SubTwoFam'] == 'TP'].shape[0]
                        coop_uf += excel_df[excel_df['SubTwoFam'] == 'UF'].shape[0]
                        coop_cm += excel_df[excel_df['SubTwoFam'] == 'CM'].shape[0]
                    except:
                        continue
                    
                    sesh_df = test_df[test_df['session'] == sesh]
                    for index, row in excel_df.iterrows():
                        vid_index = sesh_df.index[(sesh_df['vid'].str.contains(row['SubOneID'])) & 
                                                  (sesh_df['vid'].str.contains(row['SubTwoID'])) & 
                                                  (sesh_df['vid'].str.contains('TrNum' + str(int(row['TrialNum']))))]                
                        if vid_index.shape[0] == 1:
                            preds.df.at[vid_index[0], 'familiarity'] = row['SubTwoFam']
                        else:
                            # print(sesh, row)
                            missing_count += 1

        print(f'there are {total_coop} total cooperation trials')
        print(f'{coop_tp} are with training partners and {coop_uf} are with unfamilar rats')
        print(f'and {total_coop - coop_tp - coop_uf} others are probably also with training partners')
        print()
        print(f'there are {preds.df[(preds.df['familiarity'] == 'TP')].shape[0]} training partner trials that have videos') 
        print(f'there are {preds.df[(preds.df['familiarity'] == 'UF')].shape[0]} unfamiliar pair trials that have videos') 

        self.df.to_csv(self.filename, index=False)
