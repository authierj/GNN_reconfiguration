#!/bin/bash

# Slurm sbatch options
#SBATCH -o parameter_test.sh.log-%j
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20

# Loading Modules
source /etc/profile
module load anaconda/2023a
export LD_LIBRARY_PATH=/state/partition1/llgrid/pkg/anaconda/anaconda3-2023a/lib:$LD_LIBRARY_PATH


python3 experiments.py --model "GatedSwitchGNN_globalMLP" --param "lr"
python3 experiments.py --model "GatedSwitchGNN_globalMLP" --param "hiddenFeatures"
python3 experiments.py --model "GatedSwitchGNN_globalMLP" --param "numLayers"