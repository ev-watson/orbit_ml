#!/bin/bash
#SBATCH --job-name=SR
#SBATCH --account=csd886
#SBATCH --partition=compute
#SBATCH --constraint="lustre"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=249208M
#SBATCH --time=24:00:00
#SBATCH --output=script_logs/%x.o%j.txt
#SBATCH --error=script_logs/%x.e%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=etwatson@ucsd.edu

module purge
module load slurm
module load cpu
module load gcc/10.2.0

echo "Activating virtual environment..."
source .venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }

echo "Starting Python script..."
srun --unbuffered python pysr_main.py || { echo "Python script failed"; exit 1; }

echo "Job completed."
