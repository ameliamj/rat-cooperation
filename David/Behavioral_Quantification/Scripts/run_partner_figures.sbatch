#!/bin/bash
#SBATCH --partition=day
#SBATCH --job-name=partner_figures
#SBATCH --ntasks=1 --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=23:00:00

set -eo pipefail
module load StdEnv
export SLURM_EXPORT_ENV=ALL
module load miniconda
conda activate myenv

cd /gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification
python3 -c 'import cv2, h5py, matplotlib, numpy, pandas, shapely, sklearn; assert int(shapely.__version__.split(".")[0]) >= 2, "Shapely 2 or newer is required"'
python3 Scripts/figure_b_e.py \
  --analysis both \
  --index Sorted_Data_Files/only_PairedTesting.csv \
  --data-root /gpfs/radev/pi/saxena/aj764 \
  --output Graphs/PartnerSpecific/revised
