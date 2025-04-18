#!/bin/bash
#
#SBATCH --job-name="EM-trans"
#SBATCH --partition=gpu-v100
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --account=research-eemcs-ese

module load 2023r1 
module load cuda/11.6

srun mmd_vit_exp_30/train_gmm_vit_lh.py

