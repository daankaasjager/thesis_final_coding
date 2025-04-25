#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:4
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
    row_limit=null \
    checkpointing.fresh_data=false \
    checkpointing.retrain_tokenizer=false \
    checkpointing.resume_from_ckpt=true \
    checkpointing.resume_ckpt_path=checkpoints/best-v6.ckpt \
    wandb.job_type=training \
    trainer.devices=4 \
    trainer.strategy=ddp \
    trainer.accelerator=cuda \
    trainer.precision='bf16-mixed' \
    loader.global_batch_size=400 \
    trainer.log_every_n_steps=300 \
    loader.num_workers=16