#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --output=./logs/history_representation_%j.log
#SBATCH --job-name=hist_rep
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
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

# Done.
# python3 -u hist_rep.py --seed=0 --save --log_interval=0

# Running.
python3 -u hist_rep.py --seed=1 --save --log_interval=0 --n_train_evals=50

# TODO.
#