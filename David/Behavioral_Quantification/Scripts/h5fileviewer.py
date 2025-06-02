#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 15:43:01 2025

@author: david
"""

import h5py
import pandas as pd
import numpy as np

# 🔧 SET YOUR FILE PATH HERE
file_path = r"/Users/david/Documents/Research/Saxena Lab/rat-cooperation/David/Behavioral Quantification/Example Data Files/041624_Cam4_TrNum10_Coop_KL001B-KL001Y.predictions.h5" 

dataset_path = '/point_scores'          # <- Set the dataset path here

def count_nans_in_tracks(file_path):
    with h5py.File(file_path, 'r') as f:
        if 'tracks' not in f:
            print("❌ Dataset 'tracks' not found in the file.")
            return

        # Load dataset into memory
        data = f['tracks'][:]

        # Check for NaNs
        if not np.issubdtype(data.dtype, np.floating):
            print("⚠️ Dataset is not floating-point. NaNs are unlikely.")
            return

        nan_count = np.isnan(data).sum()
        total_elements = data.size
        print(f"🔍 NaN count in 'tracks': {nan_count}")
        print(f"📊 Total elements: {total_elements}")
        print(f"📉 Percentage NaN: {100 * nan_count / total_elements:.4f}%")

def try_display_dataset(dataset, max_rows=10):
    """Display dataset if possible, show shape and some slices."""
    data = dataset[()]  # Load full array
    print(f"\nDataset '{dataset.name}'")
    print(f"Shape: {data.shape}, Dtype: {dataset.dtype}")

    if data.ndim == 1 or data.ndim == 2:
        df = pd.DataFrame(data)
        print(df.head(max_rows))
    elif data.ndim == 3:
        print("3D dataset. Showing slice [0, :, :]:\n")
        df = pd.DataFrame(data[:, :, 0])
        print(df.head(max_rows))
    else:
        print("Dataset has more than 3 dimensions — not displaying.")

def view_specific_dataset(file_path, dataset_path):
    with h5py.File(file_path, 'r') as f:
        if dataset_path in f:
            dataset = f[dataset_path]
            try_display_dataset(dataset)
        else:
            print(f"Dataset '{dataset_path}' not found in file.")

# 🔍 Run the viewer
view_specific_dataset(file_path, dataset_path)
count_nans_in_tracks(file_path)

def explore_h5(file_path):
    def visit(name, obj):
        indent = '  ' * name.count('/')
        if isinstance(obj, h5py.Dataset):
            print(f"{indent} Dataset: {name}")
            print(f"{indent}  Shape: {obj.shape}, Dtype: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"{indent} Group: {name}")
        for key, value in obj.attrs.items():
            print(f"{indent} Attribute - {key}: {value}")

    with h5py.File(file_path, 'r') as f:
        print(f"Exploring HDF5 file: {file_path}")
        f.visititems(visit)

explore_h5(file_path)
