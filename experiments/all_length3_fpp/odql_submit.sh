#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --output=./logs/odql-48cpus_%j.log
#SBATCH --job-name=odql
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=medium
#SBATCH --open-mode=append
#SBATCH --clusters=all

# Load necessary applications
module load Anaconda3
module load CUDA/11.3.1
module load cuDNN/8.2.1.32-CUDA-11.3.1

# Load conda environment
source activate $HOME/.conda/envs/evotorch

# Online Deep Q-Learner experiments
# python3 -u odql.py --seed=0 --train_id_start=0 --n_train_evals=1 --save --log_interval=0
# python3 -u odql.py --seed=1 --train_id_start=1 --n_train_evals=1 --save --log_interval=0
# python3 -u odql.py --seed=2 --train_id_start=2 --n_train_evals=1 --save --log_interval=0
# python3 -u odql.py --seed=3 --train_id_start=3 --n_train_evals=1 --save --log_interval=0
# python3 -u odql.py --seed=4 --train_id_start=4 --n_train_evals=1 --save --log_interval=0
# python3 -u odql.py --seed=5 --train_id_start=5 --n_train_evals=5 --save --log_interval=0
# python3 -u odql.py --seed=6 --train_id_start=10 --n_train_evals=5 --save --log_interval=0
# python3 -u odql.py --seed=7 --train_id_start=15 --n_train_evals=5 --save --log_interval=0
# python3 -u odql.py --seed=8 --train_id_start=20 --n_train_evals=5 --save --log_interval=0

# Running.
#

# TODO.
#
