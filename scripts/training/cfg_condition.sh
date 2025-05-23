#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5GB

module purge

module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1

source /scratch/s3905845/venvs/thesis/bin/activate

# ROOT DIRECTORY OF THE PROGRAM, this makes sure output is saved in the right place
cd /scratch/s3905845/thesis_final_coding

export TMPDIR=/scratch/s3905845/tmp
export WANDB_TEMP=$TMPDIR
mkdir -p $TMPDIR

# (B) Print out which python just to check
echo "venv, python is: $(which python)"

srun python /scratch/s3905845/thesis_final_coding/main.py \
    mode=train \
    model=small \
    row_limit=null \
    checkpointing.fresh_data=false \
    checkpointing.retrain_tokenizer=false \
    checkpointing.resume_from_ckpt=false \
    checkpointing.retrain_ape_vocab=false\
    preprocessing.augment=true \
    preprocessing.discretize=true \
    conditioning.properties=['sa_score','vmin_r','vbur_vbur'] \
    conditioning.prepend=false \
    conditioning.embeddings=false \
    conditioning.cfg=true \
    conditioning.cfg_prob=0.2 \
    trainer.devices=1 \
    trainer.strategy=auto \
    trainer.accelerator=cuda \
    trainer.precision='bf16-mixed' \
    trainer.log_every_n_steps=300 \
    loader.global_batch_size=128 \
    loader.num_workers=8 \
    wandb.job_type=training \
    wandb.name=cfg_conditioning
    

