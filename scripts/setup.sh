#!/bin/bash
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1
module load poetry/1.6.1-GCCcore-13.2.0 

# If you are in a non-GUI environment, set the keyring backend to null
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring

# Set the venv directory
poetry config virtualenvs.in-project false
poetry config virtualenvs.path /scratch/s3905845/venvs

poetry lock
poetry install

echo "venv, python is: $(which python)"

# Dev import and linting check
black .
isort .
ruff check . --fix