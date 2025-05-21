import os
import sys
_parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(_parent_path)

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import numpy as np

from scipy.stats import ks_2samp, wasserstein_distance

import model.gmm_transformer as gmm_model
from asset.dataloader import Dataloader_nolabel
import asset.random_sampler as rs
import asset.gmm_train_tool as gmm_train_tool
import asset.em_pytorch as ep
import asset.plot_eva as plot_eva

import asset.timesnet_loader as timesloader
import exp_second_round.timesnet_mse_userload_15minutes.timesnet_utils as ut
from exp_second_round.timesnet_mse_userload_15minutes.timesnet_config import TimesBlockConfig 
import exp_second_round.timesnet_mse_userload_15minutes.timesnet_train as tt
import exp_second_round.timesnet_mse_userload_15minutes.timesnet as timesnet

# Set default dtype to float64
torch.set_default_dtype(torch.float64)
# Determine the device to use
device = torch.device( 'cpu')

#%%
"Load Transformer model"
# Define the encoder parameters
random_sample_num = 40
n_components = 6    
chw = (1, random_sample_num,  97)
para_dim = n_components*2
hidden_d = 96
out_d = 96
n_heads = 4
mlp_ratio = 12
n_blocks = 4
# Create the encoder model
encoder = gmm_model.ViT_encodernopara(chw, hidden_d, out_d, n_heads, mlp_ratio, n_blocks).to(device)
# load state dict of the model
model_path = 'exp/exp_second_round/user_load_15minutes/model/_encoder_40_6306166.pth'
model = torch.load(model_path, map_location=device)
encoder.load_state_dict(model)
embedding_para = torch.nn.Embedding(n_components*2, 1).to(device)
emb_empty_token = torch.nn.Embedding(1, chw[2]).to(device)
path_embedding = 'exp/exp_second_round/user_load_15minutes/model/_embedding_40_6306166.pth'
emb_weight = torch.load(path_embedding, map_location=device, weights_only=False)
path_empty = 'exp/exp_second_round/user_load_15minutes/model/_emb_empty_token_40_6306166.pth'
empty_token_vec = torch.load(path_empty, map_location=device,weights_only=False)
# load data
batch_size =  1
split_ratio = (0.8,0.1,0.1)
data_path =  'exp/data_process_for_data_collection_all/new_data_15minute_grid_nomerge.pkl' ## 
dataset = Dataloader_nolabel(data_path,  batch_size=batch_size
                    , split_ratio=split_ratio)
test_data = dataset.load_test_data(batch_size)
test_data_org = test_data.copy()
print('test_data shape: ', test_data.shape)
# normalize the input data
min_test_data = test_data[:,:, :-1].min(axis=1).reshape(batch_size , 1, chw[2]-1)
max_test_data = test_data[:,:, :-1].max(axis=1).reshape(batch_size , 1, chw[2]-1)
test_data[:,:, :-1] = (test_data[:,:, :-1]  - min_test_data)/(max_test_data -min_test_data+1e-15)
test_data = torch.tensor(test_data, dtype=torch.float64).to(device)

# Compte the parameters
# _random_num = 32 # torch.randint(1, random_sample_num+1, (1,)).item() # Number of the shots

for _random_num in [8]:
    
    "Transform the data to GMM parameters"
    encoder.eval()
    start_time = time.time()
    _test_sample_part = rs.random_sample(test_data , 'random', _random_num)
    _test_sample_part[:, :, -1] = _test_sample_part[:, :, -1]/365 # simple data embedding
    _test_sample_part_emb = gmm_train_tool.pad_and_embed(_test_sample_part, random_sample_num, _random_num,
                                            emb_empty_token, device)
    _ms, _covs = ep.GMM_PyTorch_Batch(n_components, _test_sample_part[:,:, :-1].shape[-1]).fit(_test_sample_part[:,:, :-1], 1) # _ms: (b, n_components, 24), _covs: (b, n_components, 24)
    
    _ms = torch.zeros(_ms.shape[0], n_components, 96).to(device)
    _covs = torch.ones(_covs.shape[0], n_components, 96).to(device)
    
    # concatenate the mean and variance to have (b, n_components*2, 25)
    _param_emb, _param = gmm_train_tool.concatenate_and_embed_params(_ms, _covs, n_components, embedding_para, device)

    # feed into the encoder
    _test_sample_part_emb = torch.cat((_param_emb, _test_sample_part_emb), dim=1)
    encoder_out = encoder(_test_sample_part_emb)
    _new_para = encoder_out[:, :n_components*2, :]
    _new_para = encoder.output_adding_layer(_new_para, _param)
    end_time = time.time()
    print('Time taken to transform the data: ', end_time - start_time)

    mean = _new_para[:, :n_components* 96].view(-1, n_components, 96)
    cov = _new_para[:, n_components* 96:].view(-1, n_components, 96)
    # recover the scale
    recovered_test_data = test_data[:, :, :-1].clone() * (max_test_data -min_test_data+1e-15) + min_test_data
    recover_test_sample_part = _test_sample_part[:, :, :-1].clone() * (max_test_data -min_test_data+1e-15) + min_test_data
    recover_test_sample_part = _test_sample_part[:, :, :-1].clone()

    "evaluate the model"
    mmd = 0

    # sample from the GMM
    _num=0
    
    # Sample from the GMM
    samples, gmm = plot_eva.sample_from_gmm(n_components, _new_para, _num)
    # samples = samples * (max_test_data[_num] -min_test_data[_num]+1e-15) + min_test_data[_num]
    
    mmd += plot_eva.compute_mmd(samples, test_data[_num, :, :-1].cpu().detach().numpy())

    # Plot the samples
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)

    # plot the samples the colors indicate the sum of the samples
    samples = samples * (max_test_data[_num] -min_test_data[_num]+1e-15) + min_test_data[_num]
    for i in range(samples.shape[0]):
        _sum = samples[i, :-1].sum()
        color = plt.cm.viridis(_sum / test_data.max())
        plt.plot(samples[i, :-1], alpha=0.05, c=color)
    plt.title('Samples from GMM')
    plt.xlabel('Time')
    plt.ylabel('Value')

    plt.subplot(2, 1, 2)
    # plot the real samples
    for i in range(test_data.shape[1]):
        _sum = test_data[_num, i, :-1].sum()
        color = plt.cm.viridis(_sum / test_data.max())
        plt.plot(recovered_test_data[_num, i, :-1].cpu().detach().numpy(), alpha=0.05, c=color)
    plt.title('real')
    plt.xlabel('Time')
    plt.ylabel('Value')
    
    # save the plot
    plt.savefig(f'exp/exp_second_round/eva/eveload_for_finetune/samples_from_gmm_{_random_num}.png')
    plt.close()

    print(f'mmd of {_random_num}-shots: ', mmd/batch_size)

