#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --output=./logs/refined-sweep/refined-sweep-%j.log
#SBATCH --job-name=drqn-search
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=short
#SBATCH --open-mode=append
#SBATCH --clusters=all

# Load necessary applications
module load Anaconda3
module load CUDA/11.3.1
module load cuDNN/8.2.1.32-CUDA-11.3.1

# Load conda environment
source activate $HOME/.conda/envs/evotorch

# Start sweep
# wandb sweep drqn_random_sweep.yml

# Start agent
wandb agent hericks/drqn-hyperparameter-search/6e8ji29h