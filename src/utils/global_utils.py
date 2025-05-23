# cohorts
DYED_COHORTS = ['KL', 'EB', 'HF']
COLLAR_COHORTS = ['KL', 'EB', 'HF', 'KM', 'KF', 'NM', 'NF']

# directories
ROOTDIR = '/gpfs/radev/pi/saxena/aj764/'
TESTDIR = 'PairedTestingSessions/'
TRAINDIR = 'Training_COOPERATION/'
JOBDIR = '/gpfs/radev/project/saxena/aj764/ood/projects/default/sleap_tracking'

# models (needs to be the WHOLE file path!!)
SINGLE = 'Tracking/DLC_SingleAnimal/SingleAnimal-V1-2024-07-16/SLEAP/models/240808_075503.single_instance.n=720'
CENTROID = {
    ('dyed', 'RG'): '',
    ('dyed', 'RB'): '',
    ('dyed', 'RY'): '',
    ('dyed', 'YB'): '',
    ('dyed', 'GY'): '',
    ('dyed', 'GB'): '',
    ('collar', 'RG'): '',
    ('collar', 'RB'): '',
    ('collar', 'RY'): '',
    ('collar', 'YB'): '',
    ('collar', 'GY'): '',
    ('collar', 'GB'): ''
}
TOPDOWN = {
    ('dyed', 'RG'): '',
    ('dyed', 'RB'): '',
    ('dyed', 'RY'): '',
    ('dyed', 'YB'): '',
    ('dyed', 'GY'): '',
    ('dyed', 'GB'): '',
    ('collar', 'RG'): '',
    ('collar', 'RB'): '',
    ('collar', 'RY'): '',
    ('collar', 'YB'): '',
    ('collar', 'GY'): '',
    ('collar', 'GB'): ''
}

# MAGIC NUBMERS for error utils
MAX_VEL = 200 # pixels per frames that is max rat velocity
SMOOTHING = 10 # number of frames before and after nan value to use for smoothing
BAD_NAN = .30 # percent of initial nan at which point, don't consider correcting


# MAGIC NUMBERS for id utils
fps = 30 # frames per second the video was recorded at
# somewhat guessed numbers for the location of lever / mag
levx = 135
loc2y = 200
loc1y = 440
magx = 1260