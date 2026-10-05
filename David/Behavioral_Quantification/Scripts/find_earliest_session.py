#!/usr/bin/env python3
"""Find the earliest session in the coop and non-coop spreadsheets."""

from file_extractor_class import fileExtractor

minReq = "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/dyed_preds_min_requirements_valid.csv"
minReqNonCoop = "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/nonCoop_minReq_valid.csv"

for label, path in [("Coop (minReq)", minReq), ("Non-Coop (minReqNonCoop)", minReqNonCoop)]:
    fe = fileExtractor(path)
    dates = fe.getDatesList()
    earliest_idx = dates.index(min(dates))
    earliest_date = dates[earliest_idx]
    vid = fe.data.iloc[earliest_idx]['vid']
    session = fe.data.iloc[earliest_idx]['session']
    print(f"{label}:")
    print(f"  Earliest date: {earliest_date.strftime('%m/%d/%Y')}")
    print(f"  vid: {vid}")
    print(f"  session: {session}")
    print(f"  row index: {earliest_idx}")
    print()



