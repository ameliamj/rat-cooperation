import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import h5py

# MAGIC NUBMERS
MAX_VEL = 200 # pixels per frames that is max rat velocity
SMOOTHING = 10 # number of frames before and after nan value to use for smoothing
BAD_NAN = .30 # percent of initial nan at which point, don't consider correcting

# gets rid of any predictions of nodes that travel faster than max rat velocity
def high_vel_nan(locations):
    frames, nodes, _, rats = locations.shape
    for r in range(rats):
        for n in range(nodes):
            node_locs = locations[:, n, :, r]
            j = 0
            while np.sum(np.isnan(node_locs[j])) != 0 and j < (frames - 1): # make sure first node isn't nan
                j += 1
            k = j + 1
            while k < frames:
                if np.sum(np.isnan(node_locs[k])) == 0:
                    vel = np.sqrt(np.sum(np.square(node_locs[j] - node_locs[k]))) / (k - j)
                    if np.isnan(vel): 
                        print(j, node_locs[j])
                        print(k, node_locs[k])
                        print()
                    if (vel) > 200:
                        node_locs[k] = [np.nan, np.nan]
                        k += 1
                    else: 
                        j += 1 
                        while np.sum(np.isnan(node_locs[j])) != 0 and j < (frames - 1): # make sure j isn't nan
                            j += 1
                        k = j + 1
                else:
                    k += 1
                
    return locations        

# fills in nan values in locations with lowess smoothing
def lowess_fill(locations):
    frames, nodes, points, rats = locations.shape
    lowess = sm.nonparametric.lowess
    for r in range(rats):
        for n in range(nodes):
            node_locs = locations[:, n, :, r]
            for f in range(frames):
                for p in range(points):
                    if np.isnan(node_locs[f, p]):
                        if (f - SMOOTHING >= 0 and f + (SMOOTHING + 1) < frames): 
                            
                            endog = node_locs[f - SMOOTHING: f + (SMOOTHING + 1), p]
    
                            endog = np.concatenate((endog[:SMOOTHING], endog[SMOOTHING + 1:]))
                            exog = np.arange(0, (SMOOTHING * 2) + 1, 1) + (f - SMOOTHING)
                            exog = np.concatenate((exog[:SMOOTHING], exog[SMOOTHING + 1:]))
                            node_locs[f, p] = lowess(endog, exog, xvals=[f])  
    return locations        

# graphs the x and y position of the rats separately for the given node,
# from start frame to the end frame
def graph_locations(locations, node, start, end):
    fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
    ax[0].plot(locations[start:end, node, 0, ])
    ax[0].set_title("x pos over time")
    ax[0].set_xlabel("frames")
    ax[0].set_ylabel("x pos")
    ax[1].plot(locations[start:end, node, 1, ])
    ax[1].set_title("y pos over time")
    ax[1].set_xlabel("frames")
    ax[1].set_ylabel("y pos")

# gets the percentage of predicted node locations that are nan
def get_nan_prec(locations):
    return np.sum(np.isnan(locations)) / np.prod(locations.shape)

# returns if the precentage of nan locations are over the abitrary threshold
# where we won't bother correcting them
def nan_good(initial_nan, vel_nan):
    return ((initial_nan < BAD_NAN) and (vel_nan - initial_nan < BAD_NAN))

# gets video and session name from a key/vid from VidLoader object
def get_vs(key, vid):
    sesh_split = key.split('/')
    v = vid[:-4]
    s = sesh_split[1]
    return v,s

# gets the color pairing of a multi-anmial video given the video name
# and whether the video is in PairedTestingSessions or Training_COOPERATION
def get_color(vid, trial_type):
    if trial_type == 'test':
        trial_color = [vid[-12], vid[-5]]
    elif trial_type == 'train':
        parsed = vid.split('-')
        trial_color = [parsed[0][-1], parsed[1][5]]
    else:
        raise Exception("didn't specify valid trial type boooo")
    
    trial_key = ''
    if 'R' in trial_color:
        trial_key += 'R'
    if 'G' in trial_color:
        trial_key += 'G'
    if 'Y' in trial_color:
        trial_key += 'Y'
    if 'B' in trial_color:
        trial_key += 'B'
    return trial_key

# given a row of the data frame from PredLoader, will load the h5 files and return 
# the predicted locations
def load_file(row):
    t = 'PairedTestingSessions' if row['test/train'] == 'test' else 'Training_COOPERATION'
    session = row['session']
    vid = row['vid']
    try: 
        with h5py.File(f'/gpfs/radev/pi/saxena/aj764/{t}/' + session + '/Tracking/h5/' + vid + '.predictions.h5','r') as f:
            locations = f["tracks"][:].T
        return locations
    except FileNotFoundError:
        return None

# given a row of the data frame from PredLoader and the corrected locations, will save
# the corrected locations over the old predicted locations
def save_file(row, locations):
    t = 'PairedTestingSessions' if row['test/train'] == 'test' else 'Training_COOPERATION'
    session = row['session']
    vid = row['vid']
    f = h5py.File(f'/gpfs/radev/pi/saxena/aj764/{t}/' + session + '/Tracking/h5/' + vid + '.predictions.h5','r+')
    f["tracks"][:] = locations.T
    f.close()