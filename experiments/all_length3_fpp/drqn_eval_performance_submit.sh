#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --output=./logs/drqn_eval_%j.log
#SBATCH --job-name=drqn_eval
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=short
#SBATCH --clusters=all

# Load necessary applications
module load Anaconda3
module load CUDA/11.3.1
module load cuDNN/8.2.1.32-CUDA-11.3.1

# Load conda environment
source activate $HOME/.conda/envs/evotorch

python3 -u odql_eval_performance.py