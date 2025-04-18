import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)),'..','..','..') 

import torch
import torch.nn as nn   
# Assuming these are your modules
import model.gmm_transformer as gmm_model

# Set default dtype to float64
torch.set_default_dtype(torch.float64)

# Determine the device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the encoder parameters
random_sample_num = 25
n_components = 6    
chw = (1, random_sample_num,  25)
para_dim = n_components*2
hidden_d = 96*1
out_d = 96
n_heads = 4
mlp_ratio = 3.0
n_blocks = 6

# Create the encoder model
encoder = gmm_model.ViT_encodernopara(chw, hidden_d, out_d, n_heads, mlp_ratio, n_blocks).to(device)

# load state dict of the model
model_path = '/home/weijiexia/paper3/mmd_vit_exp_30/model/_encoder_25_82814.pth'
model = torch.load(model_path, map_location=device)
encoder.load_state_dict(model)

# example input
x1 = torch.rand(1, n_components * 2, 25).to(device)
x2 = torch.rand(1, random_sample_num, 25).to(device)
x = torch.cat((x1, x2), dim=1)*0.1

# do not change the value but swap the second dim of x2
_indx = torch.randperm(random_sample_num)
_x2 = x2[:, _indx, :]
_x = torch.cat((x1, _x2), dim=1)*0.1
print(x.mean(), _x.mean(), x.var(), _x.var())
print('x - _x', ((x[:,:n_components * 2,:]-_x[:,:n_components * 2,:])**2).sum())

_y = encoder(_x)
y = encoder(x)
print(((y[:,:n_components * 2,:]-_y[:,:n_components * 2,:])**2).sum())


