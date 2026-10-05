#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 11:09:21 2026

@author: david
"""

import pandas as pd
import numpy as np
import re


# Load CSV
df = pd.read_csv("/Users/david/Downloads/ineq_sessions_allGazeData.csv")

# 1) Set all values in familiarity to NaN
df["familiarity"] = np.nan

# 2) Set transparency based on session
# Default
df["transparency"] = "transparent"

# If session contains 'Opaque'
df.loc[df["session"].str.contains("Opaque", na=False), "transparency"] = "opaque"

# If session contains 'Translucent'
df.loc[df["session"].str.contains("Translucent", na=False), "transparency"] = "translucent"

# Save back to CSV if desired
#df.to_csv("comp_sessions_allGazeData_updated.csv", index=False)




#df = pd.read_csv("/Users/david/Downloads/comp_sessions_allGazeData.csv")

# Get the raw string from the first row
raw = df.loc[0, "avg_social_gaze_length"]

# Remove 'np.float64(' and ')'
cleaned = re.sub(r"np\.float64\(|\)", "", raw)

# Convert string to list
values = eval(cleaned, {"nan": np.nan})

# Convert to numpy floats (handles nan correctly)
values = [float(x) if x == x else np.nan for x in values]

# Assign sequentially down the column
df["avg_social_gaze_length"] = values[:len(df)]

df.to_csv("ineq_sessions_allGazeData_updated.csv", index=False)
