import os
import sys
_parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(_parent_path)

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import ks_2samp, wasserstein_distance
import numpy as np  

import model.gmm_transformer as gmm_model
from asset.dataloader import Dataloader_nolabel
import asset.random_sampler as rs
import asset.gmm_train_tool as gmm_train_tool
import asset.em_pytorch as ep
import asset.plot_eva as plot_eva
import exp_second_round.eva.eva_function as eva_function

import asset.timesnet_loader as timesloader
from exp_second_round.timesnet_mse_userload_15minutes.timesnet_config import TimesBlockConfig 
from exp_second_round.timenet_transformer_mmd.timesnet_config import TimesBlockConfig as TimesBlockConfig_mmd
import exp_second_round.timesnet_solar_mmd_15minutes.timesnet_train_mmd as tt_mmd
import exp_second_round.timesnet_mse_userload_15minutes.timesnet_train as tt
import exp_second_round.timesnet_mse_userload_15minutes.timesnet as timesnet

# Set default dtype to float64
torch.set_default_dtype(torch.float64)
# Determine the device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
"Load TimesNet mmd model"
# check the model
configs =  TimesBlockConfig_mmd()
model_timesnetmmd = timesnet.Model(configs).to(device)

# print number of parameters
num_params = sum(p.numel() for p in model_timesnetmmd.parameters())
print(num_params)

data_path = 'exp/data_process_for_data_collection_all/transformer_data_15minutes.pkl'
data_loader = timesloader.TimesNetLoader(data_path, 
                                            batch_size=60, 
                                            split_ration=(0.8, 0.1, 0.1),
                                            full_length=366)

model_path = 'exp/exp_second_round/timenet_transformer_mmd/timesnet_MMD_3836096.pth'
model_timesnetmmd.load_state_dict(torch.load(model_path, map_location=device))

#%%
"Load TimesNet mse model"
# check the model
configs = TimesBlockConfig()
model_timesnetmse = timesnet.Model(configs).to(device)

# print number of parameters
num_params = sum(p.numel() for p in model_timesnetmse.parameters())
print(num_params)

model_path = 'exp/exp_second_round/timesnet_transformer_mse/30mr_timesnet_1281536.pth'
model_timesnetmse.load_state_dict(torch.load(model_path, map_location=device))

#%%
"Load Transformer model"
# Define the encoder parameters
random_sample_num = 40
n_components = 4    
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
model_path = 'exp/exp_second_round/transformer_15minutes/model/transformer_encoder_40_6306166.pth'
model = torch.load(model_path, map_location=device)
encoder.load_state_dict(model)
embedding_para = torch.nn.Embedding(n_components*2, 1).to(device)
emb_empty_token = torch.nn.Embedding(1, chw[2]).to(device)
path_embedding = 'exp/exp_second_round/transformer_15minutes/model/transformer_embedding_40_6306166.pth'
emb_weight = torch.load(path_embedding, map_location=device, weights_only=False)
path_empty = 'exp/exp_second_round/transformer_15minutes/model/transformer_emb_empty_token_40_6306166.pth'
empty_token_vec = torch.load(path_empty, map_location=device,weights_only=False)

#%%
"Load data"
# load data
batch_size = 40
split_ratio = (0.8,0.1,0.1)
data_path = 'exp/data_process_for_data_collection_all/transformer_data_15minutes.pkl'
dataset = Dataloader_nolabel(data_path,  batch_size=batch_size
                    , split_ratio=split_ratio)
# batch_size = dataset.__len__() * split_ratio[0]
print('length of train data: ', dataset.__len__())
# batch_size = int(batch_size)

test_data = dataset.load_test_data(batch_size)
# test_data = dataset.load_train_data()
copy_test_data = test_data.copy()
copy_test_data = torch.tensor(copy_test_data, dtype=torch.float64).to(device)

# normalize the input data
min_test_data = test_data[:,:, :-1].min(axis=1).reshape(batch_size , 1, chw[2]-1)
max_test_data = test_data[:,:, :-1].max(axis=1).reshape(batch_size , 1, chw[2]-1)
test_data[:,:, :-1] = (test_data[:,:, :-1]  - min_test_data)/(max_test_data -min_test_data+1e-15)
test_data = torch.tensor(test_data, dtype=torch.float64).to(device)
# replace nan with 0
test_data = torch.nan_to_num(test_data, nan=0.0, posinf=0.0, neginf=0.0)

for _random_num in [4, 8, 16, 32]:
    
    "Transform the data to GMM parameters"
    encoder.eval()
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

    mean = _new_para[:, :n_components* 96].view(-1, n_components, 96)
    cov = _new_para[:, n_components* 96:].view(-1, n_components, 96)
    
    "recover the scale"
    recovered_test_data = test_data[:, :, :-1].clone() * (max_test_data -min_test_data+1e-15) + min_test_data
    recover_test_sample_part = _test_sample_part[:, :, :-1].clone() * (max_test_data -min_test_data+1e-15) + min_test_data

    "TimesNetmse model"
    full_series, index_mask = data_loader.load_test_data_times(copy_test_data)
    test_data_timenetmse, random_mask, scaler = tt.normalize_and_mask_fix(full_series, index_mask, device, _random_num)
    model_timesnetmse.eval()
    y_hatmse = model_timesnetmse(test_data_timenetmse.double(), None, random_mask.double())
    # Scale back to original
    y_hatmse = y_hatmse * (scaler[1] - scaler[0]) + scaler[0]
    ymse_filtered_per_sample = y_hatmse
    
    mask_bt = 1- index_mask.sum(dim=2) 
    mask_bt = mask_bt > 0
    ymse_filtered_per_sample = [
        y_hatmse[i, mask_bt[i], :]           # Tensor of shape [Z_i, F]
        for i in range(y_hatmse.shape[0])
    ]
    
    "TimesNetmmd model"
    model_timesnetmmd.eval()
    test_data_timenetmmd, random_mask, scaler = tt_mmd.normalize_and_mask(full_series, index_mask, device, _random_num)
    y_hatmmd = model_timesnetmmd(test_data_timenetmmd.double(), None, random_mask.double())

    # Scale back to original
    y_hatmmd = y_hatmmd * (scaler[1] - scaler[0]) + scaler[0]
    ymmd_filtered_per_sample = y_hatmmd
    
    mask_bt = 1- index_mask.sum(dim=2) 
    mask_bt = mask_bt > 0
    ymmd_filtered_per_sample = [
        y_hatmmd[i, mask_bt[i], :]           # Tensor of shape [Z_i, F]
        for i in range(y_hatmmd.shape[0])
    ]
    
    "evaluate the model"
    mmd = 0
    kl = 0
    ks = 0
    ws = 0
    msem = 0

    mmd_partreal = 0
    kl_partreal = 0
    ks_partreal = 0
    ws_partreal = 0
    msem_partreal = 0
    
    mmd_timesnetmse = 0
    kl_timesnetmse = 0
    ks_timesnetmse = 0
    ws_timesnetmse = 0
    msem_timesnetmse = 0

    mmd_timesnetmmd = 0
    kl_timesnetmmd = 0
    ks_timesnetmmd = 0
    ws_timesnetmmd = 0
    msem_timesnetmmd = 0
    
    t_samples_list = []
    r_samples_list = []
    r_samples_part_list = []
    timesnet_sample_list = []
    timesnet_sample_mmd_list = []
    for i in tqdm(range(batch_size)):
        # sample from the GMM
        _num=i
        # Sample from the GMM
        samples_gmm, gmm = plot_eva.sample_from_gmm(n_components, _new_para, _num)
        samples_gmm = samples_gmm * (max_test_data[_num] -min_test_data[_num]) + min_test_data[_num]
        
        # samplee from the test data
        samples = test_data[_num, :, :-1] # + noise # prevent the nan
        samples = samples * (max_test_data[_num] -min_test_data[_num]+1e-15) + min_test_data[_num]
        
        mmd += plot_eva.compute_mmd(samples_gmm, test_data[_num, :, :-1].cpu().detach().numpy())
        kl += plot_eva.compute_kl_divergence(samples_gmm, test_data[_num, :, :-1].cpu().detach().numpy())
        ks += ks_2samp(samples_gmm.flatten(), test_data[_num, :, :-1].cpu().detach().numpy().flatten())[0]
        ws += wasserstein_distance(samples_gmm.flatten(), test_data[_num, :, :-1].cpu().detach().numpy().flatten())
        msem = plot_eva.calculate_autocorrelation_mse(samples,  samples_gmm)
        
        # # Partial real data
        _part_real = recover_test_sample_part[_num, :, :]
        _part_real = _part_real + torch.rand(_part_real.shape).to(device) * 0.01
        _part_real = _part_real.cpu().detach().numpy()
        mmd_partreal += plot_eva.compute_mmd(samples, _part_real)
        kl_partreal += plot_eva.compute_kl_divergence(samples, _part_real)
        ks_partreal += ks_2samp(samples.flatten(), _part_real.flatten())[0]
        ws_partreal += wasserstein_distance(samples.flatten(), _part_real.flatten())
        msem_partreal = plot_eva.calculate_autocorrelation_mse(samples, _part_real)
        
        # # TimesNet mse
        _part_real_timesnetmse = ymse_filtered_per_sample[_num].cpu().detach().numpy()
        mmd_timesnetmse += plot_eva.compute_mmd(samples, _part_real_timesnetmse)
        kl_timesnetmse += plot_eva.compute_kl_divergence(samples, _part_real_timesnetmse)
        ks_timesnetmse += ks_2samp(samples.flatten(), _part_real_timesnetmse.flatten())[0]
        ws_timesnetmse += wasserstein_distance(samples.flatten(), _part_real_timesnetmse.flatten())
        msem_timesnetmse = plot_eva.calculate_autocorrelation_mse(samples, _part_real_timesnetmse)
        
        # # timesnetmmd
        _part_real_timesnetmmd = ymmd_filtered_per_sample[_num].cpu().detach().numpy()
        mmd_timesnetmmd += plot_eva.compute_mmd(samples, _part_real_timesnetmmd)
        kl_timesnetmmd += plot_eva.compute_kl_divergence(samples, _part_real_timesnetmmd)
        ks_timesnetmmd += ks_2samp(samples.flatten(), _part_real_timesnetmmd.flatten())[0]
        ws_timesnetmmd += wasserstein_distance(samples.flatten(), _part_real_timesnetmmd.flatten())
        _part_real_timesnetmmd = _part_real_timesnetmmd + np.random.rand(_part_real_timesnetmmd.shape[0], _part_real_timesnetmmd.shape[1]) * 0.01 # prevent the nan
        msem_timesnetmmd = plot_eva.calculate_autocorrelation_mse(samples, _part_real_timesnetmmd)

        
        r_samples_list.append(recovered_test_data[_num, :, :-1].cpu().detach().numpy())
        t_samples_list.append(samples_gmm)
        r_samples_part_list.append(_part_real)
        timesnet_sample_list.append(_part_real_timesnetmse)
        timesnet_sample_mmd_list.append(_part_real_timesnetmmd)
        
    print(f'mmd of {_random_num}-shots: ', mmd/batch_size)
    print(f'kl of {_random_num}-shots: ', kl/batch_size)
    print(f'ks of {_random_num}-shots: ', ks/batch_size)
    print(f'ws of {_random_num}-shots: ', ws/batch_size)
    print(f'msem of {_random_num}-shots: ', msem/batch_size)

    print(f'mmd_partreal of {_random_num}-shots: ', mmd_partreal/batch_size)
    print(f'kl_partreal of {_random_num}-shots: ', kl_partreal/batch_size)
    print(f'ks_partreal of {_random_num}-shots: ', ks_partreal/batch_size)
    print(f'ws_partreal of {_random_num}-shots: ', ws_partreal/batch_size)
    print(f'msem_partreal of {_random_num}-shots: ', msem_partreal/batch_size)
    
    print(f'mmd_timesnetmse of {_random_num}-shots: ', mmd_timesnetmse/batch_size)
    print(f'kl_timesnetmse of {_random_num}-shots: ', kl_timesnetmse/batch_size)
    print(f'ks_timesnetmse of {_random_num}-shots: ', ks_timesnetmse/batch_size)
    print(f'ws_timesnetmse of {_random_num}-shots: ', ws_timesnetmse/batch_size)
    print(f'msem_timesnetmse of {_random_num}-shots: ', msem_timesnetmse/batch_size)
    
    print(f'mmd_timesnetmmd of {_random_num}-shots: ', mmd_timesnetmmd/batch_size)
    print(f'kl_timesnetmmd of {_random_num}-shots: ', kl_timesnetmmd/batch_size)
    print(f'ks_timesnetmmd of {_random_num}-shots: ', ks_timesnetmmd/batch_size)
    print(f'ws_timesnetmmd of {_random_num}-shots: ', ws_timesnetmmd/batch_size)
    print(f'msem_timesnetmmd of {_random_num}-shots: ', msem_timesnetmmd/batch_size)

    # save the results in a text file
    with open('exp/exp_second_round/eva/evatransformer/sample_from_gmm.txt', 'a') as f:
        f.write(f'mmd of {_random_num}-shots: {mmd/batch_size}\n')
        f.write(f'kl of {_random_num}-shots: {kl/batch_size}\n')
        f.write(f'ks of {_random_num}-shots: {ks/batch_size}\n')
        f.write(f'ws of {_random_num}-shots: {ws/batch_size}\n')
        f.write(f'msem of {_random_num}-shots: {msem/batch_size}\n')

        f.write(f'mmd_partreal of {_random_num}-shots: {mmd_partreal/batch_size}\n')
        f.write(f'kl_partreal of {_random_num}-shots: {kl_partreal/batch_size}\n')
        f.write(f'ks_partreal of {_random_num}-shots: {ks_partreal/batch_size}\n')
        f.write(f'ws_partreal of {_random_num}-shots: {ws_partreal/batch_size}\n')
        f.write(f'msem_partreal of {_random_num}-shots: {msem_partreal/batch_size}\n')
        
        f.write(f'mmd_timesnetmse of {_random_num}-shots: {mmd_timesnetmse/batch_size}\n')
        f.write(f'kl_timesnetmse of {_random_num}-shots: {kl_timesnetmse/batch_size}\n')
        f.write(f'ks_timesnetmse of {_random_num}-shots: {ks_timesnetmse/batch_size}\n')
        f.write(f'ws_timesnetmse of {_random_num}-shots: {ws_timesnetmse/batch_size}\n')
        f.write(f'msem_timesnetmse of {_random_num}-shots: {msem_timesnetmse/batch_size}\n')
        
        f.write(f'mmd_timesnetmmd of {_random_num}-shots: {mmd_timesnetmmd/batch_size}\n')
        f.write(f'kl_timesnetmmd of {_random_num}-shots: {kl_timesnetmmd/batch_size}\n')
        f.write(f'ks_timesnetmmd of {_random_num}-shots: {ks_timesnetmmd/batch_size}\n')
        f.write(f'ws_timesnetmmd of {_random_num}-shots: {ws_timesnetmmd/batch_size}\n')
        f.write(f'msem_timesnetmmd of {_random_num}-shots: {msem_timesnetmmd/batch_size}\n')
        
        f.write('\n')




