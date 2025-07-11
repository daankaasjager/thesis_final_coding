#!/bin/bash
#SBATCH --time=2:40:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5GB
sleep 180000