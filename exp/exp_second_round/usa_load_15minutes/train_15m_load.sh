#!/bin/bash

#SBATCH --job-name="15m-usa-load"
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=w.xia@tudelft.nl


module load 2023
module load Miniconda3/23.5.2-0

