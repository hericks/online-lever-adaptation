#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --output=./logs/drqn-all-length3-%j.log
#SBATCH --job-name=drqn-all-length3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=short
#SBATCH --open-mode=append
#SBATCH --clusters=all

# Load necessary applications
module load Anaconda3
module load CUDA/11.3.1
module load cuDNN/8.2.1.32-CUDA-11.3.1

# Load conda environment
source activate $HOME/.conda/envs/evotorch

# Start agent
wandb agent hericks/drqn-all-length3/bcq2vc2j