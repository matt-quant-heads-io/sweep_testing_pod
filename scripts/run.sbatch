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




source /scratch/ms12010/overlay/env.sh
conda activate sweep_testing_pod

/scratch/ms12010/overlay/miniconda3/envs/sweep_testing_pod/bin/python /scratch/ms12010/overlay/sweep_testing_pod/main.py experiments=exp_$SLURM_ARRAY_TASK_ID +mode=controllable +username=ms12010 +domain=zelda