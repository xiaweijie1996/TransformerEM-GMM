#!/bin/bash

#SBATCH --job-name="15m-transformer_nomerge"
#SBATCH --partition=gpu_a100
#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=w.xia@tudelft.nl


module load 2023
module load Miniconda3/23.5.2-0

source $HOME/TransformerEM-GMM/.venv/bin/activate

cp $HOME/TransformerEM-GMM/exp/data_process_for_data_collection_all/transformer_data_15minutes.pkl "$TMPDIR"

python $HOME/TransformerEM-GMM/exp/exp_second_round/transformer_15minutes/train_transformer_nomerge.py "$TMPDIR/transformer_data_15minutes.pkl"