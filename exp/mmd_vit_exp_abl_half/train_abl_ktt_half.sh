#!/bin/bash
#
#SBATCH --job-name="EM-trans_abl"
#SBATCH --partition=gpu-v100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --account=research-eemcs-ese

module load 2023r1 
module load cuda/11.6

srun mmd_vit_exp_abl_half/train_knowledge_transfer_t.py

