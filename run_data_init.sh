#!/bin/bash
#SBATCH --job-name=data_init
#SBATCH --account=csd886
#SBATCH --partition=compute
#SBATCH --constraint="lustre"
#SBATCH --array=0,1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=script_logs/%x.o%j_%a.txt
#SBATCH --error=script_logs/%x.e%j_%a.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=etwatson@ucsd.edu

module purge
module load slurm
module load cpu
module load gcc/10.2.0

objects=("merc" "earth")

OBJECT=${objects[$SLURM_ARRAY_TASK_ID]}

echo "Activating virtual environment..."
source .venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }

echo "Starting Python script..."
srun --unbuffered python data_init.py --object="$OBJECT" || { echo "Python script failed"; exit 1; }

echo "Job completed."
