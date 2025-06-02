#!/bin/bash
#SBATCH --job-name=rat_analysis
#SBATCH --output=rat_analysis_%j.out
#SBATCH --mem=16G       # Adjust memory here
#SBATCH --time=01:00:00 # Adjust time here
#SBATCH --cpus-per-task=2

# Load the Python module (make sure 3.8 exists — check with `module avail python`)
module load Python/3.10.8-GCCcore-12.2.0
module load matplotlib/3.7.0-gfbf-2022b
module load h5py/3.8.0-foss-2022b

# Run your script
python3 graph_creator.py