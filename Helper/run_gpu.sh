#!/bin/bash

#SBATCH --job-name=NsPOD   # Job name
#SBATCH --ntasks=1         # Total number of tasks
#SBATCH --cpus-per-task=1  # Number of CPUs per task
#SBATCH --gres=gpu:1       # Total number of GPUs
#SBATCH --time=150:00:00   # Time limit
#SBATCH --output=/work/burela/NsPOD_%j.log  # Standard output log
#SBATCH --partition=gbr
#SBATCH --nodelist=node748
#SBATCH --mem=80G

export PYTHONUNBUFFERED=1

pwd; hostname; date

echo "Running NsPOD"

# Space-separated list of lambda_TV values
srun --cpu-bind=cores python3 Crossing_StraightCubic_waves_TV.py --lambda_TV 0.1 1.0 10.0
