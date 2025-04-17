#!/bin/bash
#SBATCH --job-name=multihopt
#SBATCH --account=csd886
#SBATCH --partition=gpu
#SBATCH --constraint="lustre"
#SBATCH --array=0,1,2,3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=377308M
#SBATCH --gpus=4
#SBATCH --time=24:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --output=script_logs/%x.o%j_%a.txt
#SBATCH --error=script_logs/%x.e%j_%a.txt
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_90
#SBATCH --mail-user=etwatson@ucsd.edu
#SBATCH --exclude=exp-16-[57-59],exp-17-[57-59]

module purge
module load slurm
module load gpu
module load gcc/10.2.0
module load cuda/11.2.2
module load cudnn/8.1.1.33-11.2

optimizers=("sgd" "adamw" "nadam" "radam" "adabound" "swats" "lion")

# Shuffles optimizers deterministically based on the master job features, same between all 4 job array tasks
binary_seed=$(echo -n "${SLURM_JOB_START_TIME}" | sha256sum | xxd -r -p)
mapfile -t shuffled < <(shuf --random-source=<(echo -n "$binary_seed") -e "${optimizers[@]}")

OPTIMIZER=${shuffled[$SLURM_ARRAY_TASK_ID]}

echo "Activating virtual environment..."
source .venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }

echo "Running hopt.py with optimizer: $OPTIMIZER"
srun --unbuffered python hopt.py --opt="$OPTIMIZER" || { echo "Python script failed"; exit 1; }

echo "Job completed."