#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --output=./logs/log_%j.log
#SBATCH --ntasks=14
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=short

# Load necessary applications
module load Anaconda3
module load CUDA/11.3.1
module load cuDNN/8.2.1.32-CUDA-11.3.1

# Load conda environment
source activate $HOME/.conda/envs/torch

srun -n1 --exclusive python3 -u run.py --patterns-start-idx  0 --n-patterns 5 --evals-start-idx 3 --n-evals 1 > logs/00_05_eid_3.log &
srun -n1 --exclusive python3 -u run.py --patterns-start-idx  5 --n-patterns 5 --evals-start-idx 3 --n-evals 1 > logs/05_10_eid_3.log &
srun -n1 --exclusive python3 -u run.py --patterns-start-idx 10 --n-patterns 5 --evals-start-idx 3 --n-evals 1 > logs/10_15_eid_3.log &
srun -n1 --exclusive python3 -u run.py --patterns-start-idx 15 --n-patterns 5 --evals-start-idx 3 --n-evals 1 > logs/15_20_eid_3.log &
srun -n1 --exclusive python3 -u run.py --patterns-start-idx 20 --n-patterns 5 --evals-start-idx 3 --n-evals 1 > logs/20_25_eid_3.log &
srun -n1 --exclusive python3 -u run.py --patterns-start-idx 25 --n-patterns 5 --evals-start-idx 3 --n-evals 1 > logs/25_30_eid_3.log &
srun -n1 --exclusive python3 -u run.py --patterns-start-idx 30 --n-patterns 5 --evals-start-idx 3 --n-evals 1 > logs/30_35_eid_3.log &
srun -n1 --exclusive python3 -u run.py --patterns-start-idx 35 --n-patterns 5 --evals-start-idx 3 --n-evals 1 > logs/35_40_eid_3.log &
srun -n1 --exclusive python3 -u run.py --patterns-start-idx 40 --n-patterns 5 --evals-start-idx 3 --n-evals 1 > logs/40_45_eid_3.log &
srun -n1 --exclusive python3 -u run.py --patterns-start-idx 45 --n-patterns 5 --evals-start-idx 3 --n-evals 1 > logs/45_50_eid_3.log &
srun -n1 --exclusive python3 -u run.py --patterns-start-idx 50 --n-patterns 5 --evals-start-idx 3 --n-evals 1 > logs/50_55_eid_3.log &
srun -n1 --exclusive python3 -u run.py --patterns-start-idx 55 --n-patterns 5 --evals-start-idx 3 --n-evals 1 > logs/55_60_eid_3.log &
srun -n1 --exclusive python3 -u run.py --patterns-start-idx 60 --n-patterns 5 --evals-start-idx 3 --n-evals 1 > logs/60_65_eid_3.log &
srun -n1 --exclusive python3 -u run.py --patterns-start-idx 65 --n-patterns 5 --evals-start-idx 3 --n-evals 1 > logs/65_70_eid_3.log &

# Important to make sure the batch job won't exit before all the
# simultaneous runs are completed.
wait
