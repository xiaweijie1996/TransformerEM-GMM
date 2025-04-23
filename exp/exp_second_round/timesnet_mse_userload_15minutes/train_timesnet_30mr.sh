#!/bin/bash

#SBATCH --job-name="15m_transformer_merge"
#SBATCH --partition=gpu_a100
#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --mail-type=END
#SBATCH --mail-user=w.xia@tudelft.nl


mmodule load 2023
module load Miniconda3/23.5.2-0

source $HOME/TransformerEM-GMM/.venv/bin/activate


cp $HOME/TransformerEM-GMM/exp/data_process_for_data_collection_all/transformer_data_15minutes.pkl "$TMPDIR"

python $HOME/TransformerEM-GMM/exp/exp_second_round/timesnet_mse_transformer_15minutes/timesnet.py "$TMPDIR/transformer_data_15minutes.pkl"