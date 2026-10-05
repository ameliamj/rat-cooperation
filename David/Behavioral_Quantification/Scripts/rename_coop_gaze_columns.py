"""
rename_coop_gaze_columns.py — Rename coop gaze CSV columns to match non-coop schema.

The coop gaze CSV uses camelCase / different names; the ineq + comp gaze CSVs
use snake_case. This script rewrites the coop CSV (or a copy of it) so all three
have the same column names and analyzeSessionCSVs.py can treat them uniformly.

Mapping:
    sessionID            -> session
    sessionType          -> label
    socialGazing         -> social_gaze
    levGazing            -> lever_gaze
    magGazing            -> mag_gaze
    socialGazingAtLev    -> social_gaze_at_lev
    socialGazingAtMag    -> social_gaze_at_mag
    socialGazingAtCenter -> social_gaze_at_center
    avgSocialGazeLength  -> avg_social_gaze_length

Columns not in the mapping (e.g. levFile, ratPair, date, familiarity,
transparency, isKL) are left as-is.

Edit the SETTINGS block and run:
    python rename_coop_gaze_columns.py
"""

import os
import pandas as pd


# ══════════════════════════════════════════════════════════════
# SETTINGS
# ══════════════════════════════════════════════════════════════

INPUT_CSV  = "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/DataStorage/AllGazeData/coop_sessions_allGazeData_updated.csv"
OUTPUT_CSV = "/Users/david/Documents/Research/Saxena_Lab/rat-cooperation/David/Behavioral_Quantification/DataStorage/AllGazeData/coop_sessions_allGazeData_updated_renamed.csv"

# If True, overwrite INPUT_CSV in place (ignores OUTPUT_CSV).
OVERWRITE_IN_PLACE = False


# ══════════════════════════════════════════════════════════════
# Renaming
# ══════════════════════════════════════════════════════════════

COLUMN_MAP = {
    "sessionID":            "session",
    "sessionType":          "label",
    "socialGazing":         "social_gaze",
    "levGazing":            "lever_gaze",
    "magGazing":            "mag_gaze",
    "socialGazingAtLev":    "social_gaze_at_lev",
    "socialGazingAtMag":    "social_gaze_at_mag",
    "socialGazingAtCenter": "social_gaze_at_center",
    "avgSocialGazeLength":  "avg_social_gaze_length",
}


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows, {len(df.columns)} cols from {INPUT_CSV}")
    print(f"Original columns: {list(df.columns)}")

    present  = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    missing  = [k for k in COLUMN_MAP if k not in df.columns]
    conflict = [v for v in present.values() if v in df.columns and v not in present]

    if conflict:
        raise ValueError(
            f"Refusing to rename — these target names already exist in the CSV "
            f"and would collide: {conflict}"
        )

    df = df.rename(columns=present)

    print(f"\nRenamed {len(present)} columns:")
    for old, new in present.items():
        print(f"  {old:<22} -> {new}")
    if missing:
        print(f"\nNot present in source (skipped): {missing}")

    out_path = INPUT_CSV if OVERWRITE_IN_PLACE else OUTPUT_CSV
    df.to_csv(out_path, index=False)
    print(f"\nWrote renamed CSV to: {out_path}")
    print(f"New columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
