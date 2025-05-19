#!/bin/bash
#!/bin/bash
#SBATCH --time=1:30:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10GB
sleep 180000