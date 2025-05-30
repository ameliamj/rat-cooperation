#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 15:43:01 2025

@author: david
"""

import h5py
import pandas as pd

# 🔧 SET YOUR FILE PATH HERE
file_path = r'/Users/david/Documents/Research/Saxena Lab/Behavioral Quantification/Example Data Files/ExampleTrackingCoop.h5'  

dataset_path = '/point_scores'          # <- Set the dataset path here

def try_display_dataset(dataset, max_rows=10):
    """Display dataset if possible, show shape and some slices."""
    data = dataset[()]  # Load full array
    print(f"\n📊 Dataset '{dataset.name}'")
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
            print(f"❌ Dataset '{dataset_path}' not found in file.")

# 🔍 Run the viewer
view_specific_dataset(file_path, dataset_path)

'''def display_structure(h5file, path='/'):
    """Recursively print the structure of the HDF5 file."""
    for key in h5file[path]:
        item = h5file[path + key]
        if isinstance(item, h5py.Dataset):
            print(f"[Dataset] {path + key} - shape: {item.shape}, dtype: {item.dtype}")
        elif isinstance(item, h5py.Group):
            print(f"[Group]   {path + key}/")
            display_structure(h5file, path + key + '/')

def try_display_dataset(dataset, max_rows=10):
    """Attempt to display the dataset as a DataFrame if possible."""
    try:
        data = dataset[()]
        if data.ndim == 1 or data.ndim == 2:
            df = pd.DataFrame(data)
            print(df.head(max_rows))
        else:
            print(f"Cannot display data with shape {data.shape}")
    except Exception as e:
        print(f"Could not load dataset: {e}")

def first_dataset(group):
    for key in group:
        item = group[key]
        if isinstance(item, h5py.Dataset):
            return item
        elif isinstance(item, h5py.Group):
            ds = first_dataset(item)
            if ds:
                return ds
    return None

def view_h5_file(file_path):
    with h5py.File(file_path, 'r') as f:
        print(f"\nStructure of '{file_path}':\n")
        display_structure(f)

        print("\nAttempting to display first dataset:\n")
        dataset = first_dataset(f)
        if dataset:
            print(f"Displaying: {dataset.name}")
            try_display_dataset(dataset)
        else:
            print("No datasets found.")

# 🔍 Run the viewer
view_h5_file(file_path)'''
