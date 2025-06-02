"""
Created on Mon Jun  2 12:49:50 2025

@author: david
"""

#!/bin/bash
#SBATCH --job-name=rat_analysis
#SBATCH --output=rat_analysis_%j.out
#SBATCH --mem=16G       # Adjust memory here
#SBATCH --time=01:00:00 # Adjust time here
#SBATCH --cpus-per-task=2

# Load the Python module (make sure 3.8 exists — check with `module avail python`)
module load python/3.8

# Run your script
python3 graph_creator.py