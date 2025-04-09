#!/bin/bash
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1


source /scratch/s3905845/venvs/thesis/bin/activate

# (B) Print out which python just to check
echo "venv, python is: $(which python)"


# (D) Optional: Export environment variables your code might need
pip install --force-reinstall --no-deps "pydantic<2.0"
pip install "wandb<0.15.9"


# to use moses, different python version is needed.
module load Python/3.10.8-GCCcore-12.2.0
source $HOME/venvs/evaluate_molecules/bin/activate