#!/bin/bash
#SBATCH --time=3:30:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10GB

sleep 180000