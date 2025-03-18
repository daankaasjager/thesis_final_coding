#!/bin/bash

# 1) Purge modules and load the ones you need
module purge
module load 2023.01
module load Python/3.11.5-GCCcore-13.2.0
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load CUDA/12.1

# 2) Create a new Python venv in $HOME/venvs/thesis_a100 (doesn't overwrite existing 'thesis')
python -m venv --without-pip $HOME/venvs/thesis_a100

# 3) Activate the new environment
source $HOME/venvs/thesis_a100/bin/activate

# 4) Print out which python to verify it's from 'thesis_a100' environment
echo "venv, python is: $(which python)"

curl -sS https://bootstrap.pypa.io/get-pip.py | python

pip install --no-cache-dir --upgrade pip setuptools woyheel
pip install --no-cache-dir -r requirements.txt

pip install --force-reinstall --no-deps "pydantic<2.0"
pip install "wandb<0.15.9"
