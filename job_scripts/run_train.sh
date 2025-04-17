#!/bin/bash
#SBATCH --job-name=train
#SBATCH --account=csd886
#SBATCH --partition=gpu
#SBATCH --constraint="lustre"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=377308M
#SBATCH --gpus=4
#SBATCH --time=4:00:00
#SBATCH --output=script_logs/%x.o%j.txt
#SBATCH --error=script_logs/%x.e%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_90
#SBATCH --mail-user=etwatson@ucsd.edu

module purge
module load slurm
module load gpu
module load gcc/10.2.0
module load cuda/11.2.2
module load cudnn/8.1.1.33-11.2

echo "Activating virtual environment..."
source .venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }

srun --unbuffered python train.py || { echo "Python script failed"; exit 1; }

echo "Job completed."