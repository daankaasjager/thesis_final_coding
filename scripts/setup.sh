#!/bin/bash
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1
source /scratch/s3905845/venvs/thesis/bin/activate

# (B) Print out which python just to check
echo "venv, python is: $(which python)"


# to use moses, different python version is needed.
module purge
module load Python/3.8.6-GCCcore-10.2.0
source /scratch/s3905845/venvs/evaluate_molecules/bin/activate
# 2. Upgrade pip, setuptools, wheel
pip install --upgrade pip setuptools wheel

# 3. Install numpy first (helps with some Cython dependencies)
pip install numpy cython

# 4. Install pomegranate
pip install pomegranate

pip install molsets

# sort out imports and format code
black .
isort .
ruff check . --fix

#