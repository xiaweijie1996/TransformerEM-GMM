#!/usr/bin/env python3
import os 
import sys
_parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(_parent_path)

import wandb
import torch
import tqdm
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
# sklearn gmm
from sklearn.mixture import GaussianMixture

import model.gmm_transformer as gmm_model
import exp_second_round.eva.quant_shalpy.tool as tl
import asset.em_pytorch as ep
import asset.plot_eva as pae
from asset.dataloader import Dataloader_nolabel
import asset.random_sampler as rs
import asset.le_improved as le

torch.set_default_dtype(torch.float64)

# load data
batch_size =  100
split_ratio = (0.8,0.1,0.1)
# data_path =  'exp/data_process_for_data_collection_all/all_data.pkl'
data_path =  'exp/data_process_for_data_collection_all/transformer_data_15minutes.pkl' 
dataset = Dataloader_nolabel(data_path,  batch_size=batch_size
                    , split_ratio=split_ratio)
print('lenthg of train data: ', dataset.__len__()*split_ratio[0])
print('lenthg of test data: ', dataset.__len__()*split_ratio[1])

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# define the hyperparameters
random_sample_num = 40
num_epochs = int(500000)
sub_epoch = int(dataset.__len__()*split_ratio[0]/batch_size)
lr = 0.0001
n_components = 8
min_random_sample_num = 32

# define the encoder
# chw = (1, random_sample_num,  25)
# para_dim = n_components*2
# hidden_d = 24
# out_d = 24
# n_heads = 4
# mlp_ratio = 12
# n_blocks = 6
chw = (1, random_sample_num,  97)
para_dim = n_components*2
hidden_d = 96
out_d = 96
n_heads = 4
mlp_ratio = 12
n_blocks = 4
encoder = gmm_model.ViT_encodernopara(chw, hidden_d, out_d, n_heads, mlp_ratio, n_blocks).to(device)
_model_scale = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
print('number of parameters: ', _model_scale)

# Define a gmm embedding layer
embedding_para = torch.nn.Embedding(n_components*2, 1).to(device)
emb_empty_token = torch.nn.Embedding(1, chw[2]).to(device)

# encoder_model_path = 'exp/exp_second_round/loglikehoodest/model/solar_encoder_6_592086.pth'
# embedding_para_model_path = 'exp/exp_second_round/loglikehoodest/model/solar_embedding_6_592086.pth'
# emb_empty_token_model_path = 'exp/exp_second_round/loglikehoodest/model/solar_emb_empty_token_6_592086.pth'
encoder_model_path = 'exp/exp_second_round/gaussian_number_exame/8gaussian/transformer_encoder_40_6306166.pth'
embedding_para_model_path = 'exp/exp_second_round/gaussian_number_exame/8gaussian/transformer_embedding_40_6306166.pth'
emb_empty_token_model_path = 'exp/exp_second_round/gaussian_number_exame/8gaussian/transformer_emb_empty_token_40_6306166.pth'
encoder.load_state_dict(torch.load(encoder_model_path, map_location=device, weights_only=False))
embedding_para = torch.load(embedding_para_model_path, map_location=device, weights_only=False)
emb_empty_token = torch.load(emb_empty_token_model_path, map_location=device, weights_only=False)

samples = dataset.load_train_data()
samples = torch.tensor(samples, dtype=torch.float64).to(device)
orignal_samples = samples.clone()
a = 0

p1s = []
p2s = []
p12s = []
for sample in samples:
    encoder.eval()
    a += 1
    # sample[:, :-1] = (sample[:, :-1] - sample[:, :-1].min()) / (sample[:, :-1].max() - sample[:, :-1].min() + 1e-15)
    _min, _ = sample[:, :-1].min(axis=0, keepdim=True)
    _max, _ = sample[:, :-1].max(axis=0, keepdim=True)
    scaled_sample = sample.clone()
    scaled_sample[:, :-1] = (scaled_sample[:, :-1] - _min) / (_max - _min + 1e-15)
    scaled_sample = scaled_sample.unsqueeze(0)
    print(a)
    for _ in range(1):
        train_sample_part_scaled = rs.random_sample(scaled_sample, 'random', min_random_sample_num)
        # print('train_sample_part_scaled shape: ', train_sample_part_scaled.shape)
        # print('scaled_sample shape: ', scaled_sample.shape)
        
        # Compute loss of with two components v12
        _loss_nodrop, _random_num, _new_para, _param, r_samples, r_samples_part, _mm = tl.get_loss_le(scaled_sample, train_sample_part_scaled, encoder,
                                                                                random_sample_num, 1, n_components, embedding_para, emb_empty_token, 'True', device)
        likehood_nodrop = - _loss_nodrop.item()
        print('likehood_nodrop: ', likehood_nodrop)
        
        # Compute loss of with only knowledge transfer v1
        _loss_trransfer, _random_num, _new_para, _param, r_samples, r_samples_part, _mm = tl.get_loss_lenoem(scaled_sample, train_sample_part_scaled, encoder,
                                                                                random_sample_num, 1, n_components, embedding_para, emb_empty_token, 'True', device)
        likehood_transfer = - _loss_trransfer.item()
        print('likehood_transfer: ', likehood_transfer)
        
        # Compute loss of with only wt v1
        likehood_ems = []
        for i in range(1, 2):
            # fit the model 
            _ms, _covs = ep.GMM_PyTorch_Batch(n_components, train_sample_part_scaled[:, :, :-1].shape[-1]).fit(train_sample_part_scaled[:, :, :-1],i)
            new_para = torch.cat((_ms.reshape(1,-1), _covs.reshape(1,-1)), dim=1)
            _loss = le.le_loss(scaled_sample[:, :, :-1], n_components, new_para)
            # print('likehood_ems: ', -_loss.item())
            # check the loss if it is nan
            if torch.isnan(_loss).any():
                pass
            else:
                likehood_ems.append(-_loss.item())
        
        likehood_ems = np.max(likehood_ems)
        print('likehood_ems: ', likehood_ems, 'index', np.argmax(likehood_ems))
        
        baseline = 0
        # comput p12
        p12 = likehood_nodrop - baseline
        # print('p12: ', p12)
        # Computp pha1 
        p1 = 0.5*(likehood_ems-baseline)+0.5*(likehood_nodrop-likehood_transfer)
        # print('p1: ', p1)
        # Computp pha2
        p2 = 0.5*(likehood_transfer-baseline)+0.5*(likehood_nodrop-likehood_ems)
        # print('p2: ', p2)
        p1s.append(p1)
        p2s.append(p2)
        p12s.append(p12)
        
print('--------------------------')
print('p1s: ', np.mean(p1s))
print('p2s: ', np.mean(p2s))
print('p12s: ', np.mean(p12s))