#!/bin/bash
#SBATCH --job-name=rat_analysis
#SBATCH --output=rat_analysis_%j.out
#SBATCH --mem=400G
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2

source ~/.bashrc
conda activate /gpfs/radev/home/drb83/.conda/envs/myenv

which python
python --version

python3 graph_creator.py