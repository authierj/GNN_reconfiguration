#!/bin/bash

# Slurm sbatch options
#SBATCH -o topo_cost_test.sh.log-%j
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20

# Loading Modules
source /etc/profile
module load anaconda/2023a
export LD_LIBRARY_PATH=/state/partition1/llgrid/pkg/anaconda/anaconda3-2023a/lib:$LD_LIBRARY_PATH


python3 experiments.py --model "GatedSwitchGNN"
python3 experiments.py --model "GCN_local_MLP"
python3 experiments.py --model "GatedSwitchGNN_globalMLP"
python3 experiments.py --model "GCN_Global_MLP_reduced_model"