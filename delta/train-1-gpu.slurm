#!/bin/bash
#SBATCH --account=becw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=100g
#SBATCH --job-name=train_on_the_stack_v2
#SBATCH --output=job_output_%j.txt

# Load your environment modules or activate conda environment here
# module load ...
# source activate ...

# Run your actual job command here
conda env list

export PYTHONPATH=.
python training-scripts/train_on_the_stack_v2.py