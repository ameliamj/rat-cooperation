"""
filter_out_cno_sessions.py

Removes all rows whose 'session' column contains the substring 'CNO' from
the *_uncorrectedData.csv files (interacting + gaze, coop/ineq/comp).
Overwrites files in place.
"""

import pandas as pd

PATHS = [
    "../DataStorage/InteractionAndOtherDataNew/comp_allInteractingAndOtherData_uncorrectedData.csv",
    "../DataStorage/InteractionAndOtherDataNew/coop_allInteractingAndOtherData_uncorrectedData.csv",
    "../DataStorage/InteractionAndOtherDataNew/ineq_allInteractingAndOtherData_uncorrectedData.csv",
    "../DataStorage/AllGazeDataNew/ineq_sessions_allGazeData_uncorrectedData.csv",
    "../DataStorage/AllGazeDataNew/coop_sessions_allGazeData_uncorrectedData.csv",
    "../DataStorage/AllGazeDataNew/comp_sessions_allGazeData_uncorrectedData.csv",
]


def main():
    for path in PATHS:
        df = pd.read_csv(path)
        is_cno = df["session"].str.contains("CNO", na=False)
        n_removed = int(is_cno.sum())
        df = df[~is_cno]
        df.to_csv(path, index=False)
        print(f"{path}: removed {n_removed} CNO rows, {len(df)} rows remain")


if __name__ == "__main__":
    main()
