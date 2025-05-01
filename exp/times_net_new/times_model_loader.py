import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch 
import torch.nn as nn
from timesnet_config import TimesBlockConfig 
import timesnet_model as tm
import timesnet_train as tt
import asset.timesnet_loader as timesloader

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# check the model
configs = TimesBlockConfig()
model = tm.Model(configs).to(device)

# model path
model_path = 'exp/times_net_new/timesnet.pth'

# load model
model.load_state_dict(torch.load(model_path))

# give random input
data_path = '/home/weijiexia/paper3/data_process_for_data_collection_all/all_data_aug.pkl'
data_loader = timesloader.TimesNetLoader(data_path, 
                                             batch_size=30, 
                                             split_ration=(0.1, 0.1, 0.1),
                                             full_length=366)

model.eval()
full_series, index_mask = data_loader.load_train_data_times()
train_data, train_mask, random_mask, scaler = tt.normalize_and_mask(full_series, index_mask, device)
y_hat = model(train_data, None, random_mask)
print(y_hat.shape)
            