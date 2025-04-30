#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=1
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
    checkpointing.fresh_data=true \
    checkpointing.retrain_tokenizer=true \
    checkpointing.resume_from_ckpt=false \
    wandb.name=test \
    trainer.devices=1 \
    trainer.strategy=auto \
    trainer.accelerator=cuda \
    wandb.job_type=training \
    trainer.precision='bf16-mixed' \
    loader.global_batch_size=20 \
    trainer.log_every_n_steps=300 \
    loader.num_workers=4 \
    model=tiny