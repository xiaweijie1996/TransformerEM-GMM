#!/bin/bash

#SBATCH --job-name="15m-usa-load_nomerge"
#SBATCH --partition=gpu_a100
#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=w.xia@tudelft.nl


module load 2023
module load Miniconda3/23.5.2-0

source $HOME/TransformerEM-GMM/.venv/bin/activate

cp $HOME/TransformerEM-GMM/exp/data_process_for_data_collection_all/new_data_15minute_grid_nomerge.pkl "$TMPDIR"

python $HOME/TransformerEM-GMM/exp/exp_second_round/usa_load_15minutes/train_load_nomerge.py "$TMPDIR/new_data_15minute_grid_nomerge.pkl"