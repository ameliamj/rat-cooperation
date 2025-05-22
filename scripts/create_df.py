import sys
sys.path.append('../')

from src.classes.vid_loader import VidLoader 
from src.classes.pred_loader import PredLoader

# create vid loader and pred loader
vids = VidLoader(out=False)
# preds = PredLoader.from_vids(vids)
preds = PredLoader('preds_df.csv')

# correct the predictions
# print("STARTING CORRECTIONS")
# preds.correct()

# get the event ids for lever/mag events
print("STARTING EVENT MATCHING")
preds.get_event_ids()

# get the trial types for the test multi trials
# preds.get_trial_types()

# the the familiarity level for test multi trials
# preds.get_familiar()


