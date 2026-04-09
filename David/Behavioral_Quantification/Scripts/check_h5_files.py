#!/usr/bin/env python3
"""
Check which .h5 files loaded by multiFileGraphs (getFiltered path) are corrupted.
Prints a summary of good vs bad files.
"""

import sys
import h5py
import numpy as np
from file_extractor_class import fileExtractor

filtered = "/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/Filtered.csv"


def check_h5(path):
    """Try to open and read from the .h5 file. Return (ok, error_msg)."""
    try:
        with h5py.File(path, "r") as f:
            # Try to access the main data — SLEAP stores tracks in 'tracks'
            if "tracks" in f:
                data = f["tracks"][:]
                if np.all(np.isnan(data)):
                    return False, "all NaN data"
                return True, None
            else:
                keys = list(f.keys())
                return False, f"no 'tracks' key; keys found: {keys}"
    except FileNotFoundError:
        return False, "file not found"
    except Exception as e:
        return False, str(e)


def main():
    fe = fileExtractor(filtered)
    fe.data = fe.deleteBadNaN()
    pos_files = fe.getPosDatapath()

    print(f"Total .h5 files to check: {len(pos_files)}\n")

    good = []
    bad = []

    for i, path in enumerate(pos_files):
        if path is None:
            bad.append((i, path, "path is None"))
            continue

        ok, err = check_h5(path)
        if ok:
            good.append((i, path))
        else:
            bad.append((i, path, err))

        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  checked {i + 1}/{len(pos_files)}...", file=sys.stderr)

    print(f"\n{'='*60}")
    print(f"GOOD files: {len(good)}")
    print(f"BAD files:  {len(bad)}")
    print(f"{'='*60}\n")

    if bad:
        print("CORRUPTED / PROBLEMATIC FILES:")
        for idx, path, err in bad:
            print(f"  [{idx}] {path}")
            print(f"       Error: {err}\n")


if __name__ == "__main__":
    main()
