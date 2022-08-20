#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --output=./logs/odql_%j.log
#SBATCH --job-name=odql
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=medium

# Load necessary applications
module load Anaconda3
module load CUDA/11.3.1
module load cuDNN/8.2.1.32-CUDA-11.3.1

# Load conda environment
source activate $HOME/.conda/envs/evotorch

python3 -u odql.py --seed=0 --train_id_start=0 --n_train_evals=5 --save=True --log_interval=0
