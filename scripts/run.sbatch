#!/bin/bash
#
#SBATCH --job-name=sweep_test_%a
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --array=1-10 # this creates an array!
#SBATCH --mem=50GB
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=error_logs/sweep__%A_%a.err
#SBATCH --gres=gpu:mi50:1




source /scratch/${USER}/overlay/env.sh
conda activate sweep_testing_pod

source /scratch/${USER}/overlay/sweep_testing_pod/env.sh
cd ${PROJECT_ROOT}

python main.py experiments=exp_1 +mode=controllable +username=ms12010 +domain=zelda