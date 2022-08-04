#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --output=./logs/log_%j.log
#SBATCH --job-name=single_core
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=short

# Load necessary applications
module load Anaconda3
module load CUDA/11.3.1
module load cuDNN/8.2.1.32-CUDA-11.3.1

# Load conda environment
source activate $HOME/.conda/envs/torch

python3 -u online_qlearner_all_length3.py
