#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --output=./logs/drqn_%j.log
#SBATCH --job-name=drqn
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=medium

# Load necessary applications
module load Anaconda3
module load CUDA/11.3.1
module load cuDNN/8.2.1.32-CUDA-11.3.1

# Load conda environment
source activate $HOME/.conda/envs/evotorch

# Online Deep Q-Learner experiments
# python3 -u odql.py --seed=0 --train_id_start=0 --n_train_evals=5 --save --log_interval=0
# python3 -u odql.py --seed=1 --train_id_start=5 --n_train_evals=5 --save --log_interval=0

# Deep Recurrent Q-Learner experiments
# python3 -u drqn.py --seed=0 --train_id_start=0 --n_train_evals=5 --save
python3 -u drqn.py --seed=1 --train_id_start=5 --n_train_evals=5 --save