#!/bin/bash --login

# Set required resources
#SBATCH --job-name=demo
#SBATCH --ntasks=1
#SBATCH --gpus=v100:1
#SBATCH --mem=32G
#SBATCH --time=0:60:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lellaom@msu.edu
#SBATCH --output=demo.out
#SBATCH --error=demo.err

# load virtual environment
source /mnt/scratch/lellaom/venv/bin/activate

# run python code
cd "/mnt/home/lellaom/LLM Fine-tuning Demo Codes/Decoder Fine-tuning"
srun python decoder_fine_tuning.py

# write job information to output file
scontrol show job $SLURM_JOB_ID
module load powertools
js -j $SLURM_JOB_ID