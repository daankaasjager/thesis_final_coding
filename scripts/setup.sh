#!/bin/bash
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1
source /scratch/s3905845/venvs/thesis/bin/activate

# (B) Print out which python just to check
echo "venv, python is: $(which python)"


# to use moses, different python version is needed.
module load Python/3.10.8-GCCcore-12.2.0
source $HOME/venvs/evaluate_molecules/bin/activate

# sort out imports and format code
black .
isort .
ruff check . --fix

#