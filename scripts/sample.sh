#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10GB

module purge

module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1


source /scratch/s3905845/venvs/thesis/bin/activate

# ROOT DIRECTORY OF THE PROGRAM, this makes sure output is saved in the right place
cd /scratch/s3905845/thesis_final_coding

# (B) Print out which python just to check
echo "venv, python is: $(which python)"

srun python /scratch/s3905845/thesis_final_coding/main.py \
    mode=generate \
    local_paths.sampled_data="/scratch/s3905845/thesis_final_coding/data/kraken/sampled_data/generated_$(date +%Y%m%d_%H%M%S).json" \
    sampling.steps=64 \
    sampling.num_sample_batches=100 \
    trainer.devices=1 \
    trainer.strategy=auto \
    trainer.accelerator=cuda \
    trainer.log_every_n_steps=300 \
    trainer.precision='bf16-mixed' \
    loader.global_batch_size=16 \
    loader.num_workers=4 \
    eval.checkpoint_path="/scratch/s3905845/thesis_final_coding/checkpoints/prepend_conditioning.ckpt" \
    wandb.job_type=sampling \
    wandb.name=prepend_condition_sample