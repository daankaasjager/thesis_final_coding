#!/bin/bash
#SBATCH --time=0:59:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=7GB


# --- Load Environment ---
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1
source /scratch/s3905845/venvs/thesis/bin/activate
cd /scratch/s3905845/thesis_final_coding

# --- Set Temp Directory ---
export TMPDIR=/scratch/s3905845/tmp
export WANDB_TEMP=$TMPDIR
mkdir -p $TMPDIR

srun python main.py \
    mode="predict_properties" \
    experiment.name="ape_80"

srun python main.py \
    mode="predict_properties" \
    experiment.name="ape_70"