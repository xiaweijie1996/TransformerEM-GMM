
import os 
import sys
_parent_path = os.path.join(os.path.dirname(__file__), '..', '..')
print(_parent_path)
sys.path.append(_parent_path)

import wandb
import torch
import torch.optim as optim

import exp.model.gmm_transformer as gmm_model
import exp.asset.gmm_train_tool as gmm_train_tool
import exp.asset.plot_eva as pae
from exp.asset.dataloader import Dataloader_nolabel
torch.set_default_dtype(torch.float64)

# load data
batch_size = 64*2
split_ratio = (0.8,0.1,0.1) 
data_path = 'exp/data_process_for_data_collection_all/all_data_aug_30mr.pkl'
data_path = 'exp/data_process_for_data_collection_all/all_data_aug.pkl'
dataset = Dataloader_nolabel(data_path,  batch_size=batch_size
                    , split_ratio=split_ratio)
print('lenthg of train data: ', dataset.__len__()*split_ratio[0])
print('lenthg of test data: ', dataset.__len__()*split_ratio[1])