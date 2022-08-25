#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --output=./logs/drqn_%j.log
#SBATCH --job-name=drqn
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=medium
#SBATCH --open-mode=append
#SBATCH --clusters=all

# Load necessary applications
module load Anaconda3
module load CUDA/11.3.1
module load cuDNN/8.2.1.32-CUDA-11.3.1

# Load conda environment
source activate $HOME/.conda/envs/evotorch

# Deep Recurrent Q-Learner experiments
# python3 -u drqn.py --seed=0 --train_id_start=0 --n_train_evals=5 --save
# python3 -u drqn.py --seed=1 --train_id_start=5 --n_train_evals=5 --save
# python3 -u drqn.py --seed=2 --train_id_start=10 --n_train_evals=5 --save
# python3 -u drqn.py --seed=3 --train_id_start=15 --n_train_evals=5 --save
# python3 -u drqn.py --seed=4 --train_id_start=20 --n_train_evals=5 --save
# python3 -u drqn.py --seed=5 --train_id_start=25 --n_train_evals=5 --save
# python3 -u drqn.py --seed=6 --train_id_start=30 --n_train_evals=5 --save
# python3 -u drqn.py --seed=7 --train_id_start=35 --n_train_evals=5 --save
# python3 -u drqn.py --seed=8 --train_id_start=40 --n_train_evals=5 --save
# python3 -u drqn.py --seed=9 --train_id_start=45 --n_train_evals=5 --save

# Running.
#

# TODO.
#