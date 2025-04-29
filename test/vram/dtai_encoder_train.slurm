#!/bin/bash
#SBATCH --job-name=train_on_repoeval
#SBATCH --account=becw-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120g
#SBATCH --constraint=projects
#SBATCH --output=output.log
#SBATCH --error=error.log



export PYTHONPATH=.
python training-scripts/train_on_repoeval.py
