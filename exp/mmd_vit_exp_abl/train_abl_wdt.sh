#!/bin/bash
#
#SBATCH --job-name="EM-trans_abl"
#SBATCH --partition=gpu-v100
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --account=research-eemcs-ese

module load 2023r1 
module load cuda/11.6

srun mmd_vit_exp_abl/train_gmm_vit_lh.py

