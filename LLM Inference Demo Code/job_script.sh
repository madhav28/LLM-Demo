#!/bin/bash --login

# Set required resources
#SBATCH --job-name=demo
#SBATCH --ntasks=1
#SBATCH --gpus=v100:1
#SBATCH --mem=32G
#SBATCH --time=0:05:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lellaom@msu.edu
#SBATCH --output=demo.out
#SBATCH --error=demo.err

# load virtual environment
source /mnt/home/lellaom/venv/demo_venv/bin/activate

# run python code
cd "/mnt/home/lellaom/LLM Inference Demo Code"
srun python few_shot_classification.py

# write job information to output file
scontrol show job $SLURM_JOB_ID
module load powertools
js -j $SLURM_JOB_ID