#!/bin/bash
#SBATCH --time=3:40:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4GB
sleep 180000