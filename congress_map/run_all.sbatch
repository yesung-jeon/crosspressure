#!/bin/bash
# Example SLURM wrapper. Edit the partition / environment lines for your cluster, then: sbatch run_all.sbatch
#SBATCH --job-name=congress_map
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate <your-env>
# export HF_HOME=<cache dir with room for ~16 GB>
nvidia-smi --query-gpu=name,memory.total --format=csv
bash run_all.sh
