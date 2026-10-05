"""
fix_dividers_and_transparency.py

Fixes two related data bugs in the ineq/comp CSVs:

1. In the *_minReq_valid.csv files, the 'dividers' column is blank for every
   row. It should be derived from the 'session' column: sessions ending in
   'Opaque' -> 'Opaque', ending in 'Translucent' -> 'Translucent', otherwise
   -> 'Transparent'.

2. In the *_allInteractingAndOtherData_uncorrectedData.csv and
   *_sessions_allGazeData_uncorrectedData.csv files, the 'transparency'
   column is a placeholder (always 0). The correct value is looked up from
   the now-fixed minReq_valid 'dividers' column by matching rows between the
   two files:
       minReq 'vid'     = "{date}_Cam{n}_TrNum{trnum}_{TrialType}_{ratPair}"
       minReq 'session' = "{date}_{label}_TimeOut[_Opaque|_Translucent]"
       other  'session' = "{date}_{label}_TimeOut[_Opaque|_Translucent]_TrNum{trnum}"
   i.e. other['session'] == minReq['session'] + '_TrNum' + trnum_from(minReq['vid'])

Run from the Scripts directory. Overwrites files in place.
"""

import re

import pandas as pd

TRNUM_RE = re.compile(r"_TrNum(\d+)_")


def divider_from_session(session):
    if session.endswith("Opaque"):
        return "Opaque"
    if session.endswith("Translucent"):
        return "Translucent"
    return "Transparent"


def fix_minreq_dividers(paths):
    """Recompute 'dividers' from 'session' and save to every path given (duplicates kept in sync)."""
    fixed = {}
    for group, group_paths in paths.items():
        df = pd.read_csv(group_paths[0])
        df["dividers"] = df["session"].apply(divider_from_session)
        for p in group_paths:
            df.to_csv(p, index=False)
        print(f"[{group}] dividers fixed -> {df['dividers'].value_counts().to_dict()}  "
              f"(saved to {len(group_paths)} file(s))")
        fixed[group] = df
    return fixed


def build_transparency_lookup(minreq_df):
    """Map other-file 'session' key -> lowercase divider string."""
    lookup = {}
    for _, row in minreq_df.iterrows():
        match = TRNUM_RE.search(row["vid"])
        if not match:
            continue
        key = f"{row['session']}_TrNum{match.group(1)}"
        lookup[key] = row["dividers"].lower()
    return lookup


def fix_transparency_column(path, lookup, group):
    df = pd.read_csv(path)
    mapped = df["session"].map(lookup)
    n_missing = mapped.isna().sum()
    if n_missing:
        print(f"  [WARN] {path}: {n_missing}/{len(df)} sessions had no matching minReq row; left unchanged")
        df["transparency"] = mapped.where(mapped.notna(), df["transparency"])
    else:
        df["transparency"] = mapped
    df.to_csv(path, index=False)
    print(f"  [{group}] transparency fixed -> {df['transparency'].value_counts().to_dict()}  ({path})")


def main():
    minreq_paths = {
        "ineq": [
            "../Sorted_Data_Files/ineq_minReq_valid.csv",
            "./ineq_minReq_valid.csv",
        ],
        "comp": [
            "../Sorted_Data_Files/comp_minReq_valid.csv",
            "./comp_minReq_valid.csv",
        ],
    }
    inter_paths = {
        "ineq": "../DataStorage/InteractionAndOtherDataNew/ineq_allInteractingAndOtherData_uncorrectedData.csv",
        "comp": "../DataStorage/InteractionAndOtherDataNew/comp_allInteractingAndOtherData_uncorrectedData.csv",
    }
    gaze_paths = {
        "ineq": "../DataStorage/AllGazeDataNew/ineq_sessions_allGazeData_uncorrectedData.csv",
        "comp": "../DataStorage/AllGazeDataNew/comp_sessions_allGazeData_uncorrectedData.csv",
    }

    print("=== Step 1: fixing 'dividers' in minReq_valid CSVs ===")
    fixed_minreq = fix_minreq_dividers(minreq_paths)

    print("\n=== Step 2: fixing 'transparency' in interacting/gaze CSVs ===")
    for group in ("ineq", "comp"):
        lookup = build_transparency_lookup(fixed_minreq[group])
        fix_transparency_column(inter_paths[group], lookup, group)
        fix_transparency_column(gaze_paths[group], lookup, group)


if __name__ == "__main__":
    main()
