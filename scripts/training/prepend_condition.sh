#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10GB

module purge

module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1

source /scratch/s3905845/venvs/thesis/bin/activate

# (B) Print out which python just to check
echo "venv, python is: $(which python)"

srun python main.py \
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
    conditioning.prepend=true \
    conditioning.embeddings=false \
    conditioning.cfg=false \
    trainer.devices=4 \
    trainer.strategy=auto \
    trainer.accelerator=cuda \
    trainer.precision='bf16-mixed' \
    trainer.log_every_n_steps=300 \
    loader.global_batch_size=64 \
    loader.num_workers=4 \
    wandb.job_type=training \
    wandb.name=prepend
    

