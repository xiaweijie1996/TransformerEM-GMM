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
import exp_second_round.eva.eva_comp.tool as tl
import asset.em_pytorch as ep
import asset.plot_eva as pae
from asset.dataloader import Dataloader_nolabel
import asset.random_sampler as rs
import asset.le_improved as le
import asset.plot_eva as plot_eva

torch.set_default_dtype(torch.float64)

# load data
batch_size =  50
split_ratio = (0.8,0.1,0.1)
data_path =  'exp/data_process_for_data_collection_all/transformer_data_15minutes.pkl' 
dataset = Dataloader_nolabel(data_path,  batch_size=batch_size
                    , split_ratio=split_ratio)
print('lenthg of train data: ', dataset.__len__()*split_ratio[0])
print('lenthg of test data: ', dataset.__len__()*split_ratio[1])

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

random_sample_num = 40
hidden_d = 96
out_d = 96
n_heads = 4
mlp_ratio = 12
n_blocks = 4

# -----------------------1 component-----------------------
# define the hyperparameters
n_components1 = 1

chw = (1, random_sample_num,  97)
para_dim = n_components1*2
encoder1 = gmm_model.ViT_encodernopara(chw, hidden_d, out_d, n_heads, mlp_ratio, n_blocks).to(device)

# Define a gmm embedding layer
embedding_para = torch.nn.Embedding(n_components1*2, 1).to(device)
emb_empty_token = torch.nn.Embedding(1, chw[2]).to(device)

encoder_model_path1 = f'exp/exp_second_round/gaussian_number_exame/{n_components1}gaussian/transformer_encoder_40_6306166.pth'
embedding_para_model_path1 = f'exp/exp_second_round/gaussian_number_exame/{n_components1}gaussian/transformer_embedding_40_6306166.pth'
emb_empty_token_model_path1 = f'exp/exp_second_round/gaussian_number_exame/{n_components1}gaussian/transformer_emb_empty_token_40_6306166.pth'
encoder1.load_state_dict(torch.load(encoder_model_path1, map_location=device, weights_only=False))
embedding_para1 = torch.load(embedding_para_model_path1, map_location=device, weights_only=False)
emb_empty_token1 = torch.load(emb_empty_token_model_path1, map_location=device, weights_only=False)


# -----------------------2 component-----------------------
# define the hyperparameters
n_components2 = 2

chw = (1, random_sample_num,  97)
para_dim = n_components2*2
encoder2 = gmm_model.ViT_encodernopara(chw, hidden_d, out_d, n_heads, mlp_ratio, n_blocks).to(device)

# Define a gmm embedding layer
embedding_para = torch.nn.Embedding(n_components2*2, 1).to(device)
emb_empty_token = torch.nn.Embedding(1, chw[2]).to(device)

encoder_model_path2 = f'exp/exp_second_round/gaussian_number_exame/{n_components2}gaussian/transformer_encoder_40_6306166.pth'
embedding_para_model_path2 = f'exp/exp_second_round/gaussian_number_exame/{n_components2}gaussian/transformer_embedding_40_6306166.pth'
emb_empty_token_model_path2 = f'exp/exp_second_round/gaussian_number_exame/{n_components2}gaussian/transformer_emb_empty_token_40_6306166.pth'
encoder2.load_state_dict(torch.load(encoder_model_path2, map_location=device, weights_only=False))
embedding_para2 = torch.load(embedding_para_model_path2, map_location=device, weights_only=False)
emb_empty_token2 = torch.load(emb_empty_token_model_path2, map_location=device, weights_only=False)

# -----------------------4 component-----------------------
# define the hyperparameters
n_components4 = 4
chw = (1, random_sample_num,  97)
para_dim = n_components4*2
encoder4 = gmm_model.ViT_encodernopara(chw, hidden_d, out_d, n_heads, mlp_ratio, n_blocks).to(device)
# Define a gmm embedding layer
embedding_para = torch.nn.Embedding(n_components4*2, 1).to(device)
emb_empty_token = torch.nn.Embedding(1, chw[2]).to(device)
encoder_model_path4 = f'exp/exp_second_round/gaussian_number_exame/{n_components4}gaussian/transformer_encoder_40_6306166.pth'
embedding_para_model_path4 = f'exp/exp_second_round/gaussian_number_exame/{n_components4}gaussian/transformer_embedding_40_6306166.pth'
emb_empty_token_model_path4 = f'exp/exp_second_round/gaussian_number_exame/{n_components4}gaussian/transformer_emb_empty_token_40_6306166.pth'
encoder4.load_state_dict(torch.load(encoder_model_path4, map_location=device, weights_only=False))
embedding_para4 = torch.load(embedding_para_model_path4, map_location=device, weights_only=False)
emb_empty_token4 = torch.load(emb_empty_token_model_path4, map_location=device, weights_only=False)

# -----------------------8 component-----------------------
# define the hyperparameters
n_components8 = 8
chw = (1, random_sample_num,  97)
para_dim = n_components8*2
encoder8 = gmm_model.ViT_encodernopara(chw, hidden_d, out_d, n_heads, mlp_ratio, n_blocks).to(device)
# Define a gmm embedding layer
embedding_para = torch.nn.Embedding(n_components8*2, 1).to(device)
emb_empty_token = torch.nn.Embedding(1, chw[2]).to(device)
encoder_model_path8 = f'exp/exp_second_round/gaussian_number_exame/{n_components8}gaussian/transformer_encoder_40_6306166.pth'
embedding_para_model_path8 = f'exp/exp_second_round/gaussian_number_exame/{n_components8}gaussian/transformer_embedding_40_6306166.pth'
emb_empty_token_model_path8 = f'exp/exp_second_round/gaussian_number_exame/{n_components8}gaussian/transformer_emb_empty_token_40_6306166.pth'
encoder8.load_state_dict(torch.load(encoder_model_path8, map_location=device, weights_only=False))
embedding_para8 = torch.load(embedding_para_model_path8, map_location=device, weights_only=False)
emb_empty_token8 = torch.load(emb_empty_token_model_path8, map_location=device, weights_only=False)


samples = dataset.load_train_data()
samples = torch.tensor(samples, dtype=torch.float64).to(device)
orignal_samples = samples.clone()
a = 0

mmd1s = []
mmd2s = []
mmd4s = []
mmd8s = []
for min_random_sample_num in [4, 8, 16, 32]:
    mmd1_acc = 0
    mmd2_acc = 0
    mmd4_acc = 0
    mmd8_acc = 0
    for sample in samples:
        encoder1.eval()
        encoder2.eval()
        encoder4.eval()
        encoder8.eval()
        print(a)
        a += 1
        
        # sample[:, :-1] = (sample[:, :-1] - sample[:, :-1].min()) / (sample[:, :-1].max() - sample[:, :-1].min() + 1e-15)
        _min, _ = sample[:, :-1].min(axis=0, keepdim=True)
        _max, _ = sample[:, :-1].max(axis=0, keepdim=True)
        
        scaled_sample = sample.clone()
        scaled_sample[:, :-1] = (scaled_sample[:, :-1] - _min) / (_max - _min + 1e-15)
        scaled_sample = scaled_sample.unsqueeze(0)
        
        train_sample_part_scaled = rs.random_sample(scaled_sample, 'random', min_random_sample_num)
        
        _loss_1, _random_num, _new_para1, _param, r_samples, r_samples_part, _mm = tl.get_loss_le(scaled_sample, train_sample_part_scaled, encoder1,
                                                                                    random_sample_num, 1, n_components1, embedding_para1, emb_empty_token1, 'True', device)
    
        _loss_2, _random_num, _new_para2, _param, r_samples, r_samples_part, _mm = tl.get_loss_le(scaled_sample, train_sample_part_scaled, encoder2,
                                                                                    random_sample_num, 1, n_components2, embedding_para2, emb_empty_token2, 'True', device)
        
        _loss_4, _random_num, _new_para4, _param, r_samples, r_samples_part, _mm = tl.get_loss_le(scaled_sample, train_sample_part_scaled, encoder4,
                                                                                    random_sample_num, 1, n_components4, embedding_para4, emb_empty_token4, 'True', device)
        
        _loss_8, _random_num, _new_para8, _param, r_samples, r_samples_part, _mm = tl.get_loss_le(scaled_sample, train_sample_part_scaled, encoder8,
                                                                                    random_sample_num, 1, n_components8, embedding_para8, emb_empty_token8, 'True', device)
        
        samples1, gmm1 = plot_eva.sample_from_gmm(n_components1, _new_para1, 0)
        samples2, gmm2 = plot_eva.sample_from_gmm(n_components2, _new_para2, 0)
        samples4, gmm4 = plot_eva.sample_from_gmm(n_components4, _new_para4, 0)
        samples8, gmm8 = plot_eva.sample_from_gmm(n_components8, _new_para8, 0)
        
        # Compute mmd
        mmd1 = plot_eva.compute_mmd(samples1, scaled_sample[0, :, :-1])
        mmd2 = plot_eva.compute_mmd(samples2, scaled_sample[0, :, :-1])
        mmd4 = plot_eva.compute_mmd(samples4, scaled_sample[0, :, :-1])
        mmd8 = plot_eva.compute_mmd(samples8, scaled_sample[0, :, :-1])
        
        mmd1_acc += mmd1 * 0.6
        mmd2_acc += mmd2 * 0.6
        mmd4_acc += mmd4 * 0.6
        mmd8_acc += mmd8 * 0.6
    
    mmd1s.append(mmd1_acc / len(samples))
    mmd2s.append(mmd2_acc / len(samples))
    mmd4s.append(mmd4_acc / len(samples))
    mmd8s.append(mmd8_acc / len(samples))
    print('mmd1: ', mmd1_acc / len(samples))
    print('mmd2: ', mmd2_acc / len(samples))
    print('mmd4: ', mmd4_acc / len(samples))
    print('mmd8: ', mmd8_acc / len(samples))
    print('------------------')
    

front_size = 18
plt.figure(figsize=(6, 4))
plt.plot([4, 8, 16, 32], mmd1s, label='1 component', marker='o')
plt.plot([4, 8, 16, 32], mmd2s, label='2 component', marker='o')
plt.plot([4, 8, 16, 32], mmd4s, label='4 component', marker='o')
plt.plot([4, 8, 16, 32], mmd8s, label='8 component', marker='o')
plt.xlabel('Number of Shots [-]', fontsize=front_size)
plt.ylabel('Maximum Mean Discrepancy [-]', fontsize=front_size)
# plt.title('MMD Comparison of Different Components')
plt.legend()
plt.xticks([4, 8, 16, 32], fontsize=front_size-6)
plt.yticks(fontsize=front_size-6)
plt.legend(fontsize=12)
plt.grid()
plt.tight_layout()
# Save the figure
plt.savefig('exp/exp_second_round/eva/eva_comp/mmd_com.png')
