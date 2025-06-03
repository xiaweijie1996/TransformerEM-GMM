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
device =  'cpu'

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
# model_path = 'exp/exp_second_round/transformer_15minutes/model/transformer_encoder_40_6306166.pth'
model_path = 'exp/exp_second_round/transformer_15minutes/model2/transformer_encoder_40_6306166.pth'
model = torch.load(model_path, map_location=device, weights_only=False)
encoder.load_state_dict(model)
embedding_para = torch.nn.Embedding(n_components*2, 1).to(device)
emb_empty_token = torch.nn.Embedding(1, chw[2]).to(device)
# path_embedding = 'exp/exp_second_round/transformer_15minutes/model/transformer_embedding_40_6306166.pth'
path_embedding = 'exp/exp_second_round/transformer_15minutes/model2/transformer_embedding_40_6306166.pth'
emb_weight = torch.load(path_embedding, map_location=device, weights_only=False)
# path_empty = 'exp/exp_second_round/transformer_15minutes/model/transformer_emb_empty_token_40_6306166.pth'
path_empty = 'exp/exp_second_round/transformer_15minutes/model2/transformer_emb_empty_token_40_6306166.pth'
empty_token_vec = torch.load(path_empty, map_location=device, weights_only=False)

#%%
"Load data"
# load data
batch_size = 3
split_ratio = (0.8,0.1,0.1)
data_path = 'exp/data_process_for_data_collection_all/transformer_data_15minutes.pkl'
dataset = Dataloader_nolabel(data_path,  batch_size=batch_size
                    , split_ratio=split_ratio)
# batch_size = dataset.__len__() * split_ratio[0]
print('length of train data: ', dataset.__len__())
# batch_size = int(batch_size)

# test_data = dataset.load_test_data(batch_size)
test_data = dataset.load_test_data(batch_size)
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
        _num=i
        # Sample from the GMM
        samples_gmm, gmm = plot_eva.sample_from_gmm(n_components, _new_para, _num)
        samples_gmm = samples_gmm * (max_test_data[_num] -min_test_data[_num]) + min_test_data[_num]
        
        samples = test_data[_num, :, :-1] 
        samples = samples * (max_test_data[_num] -min_test_data[_num]+1e-15) + min_test_data[_num]

        # Partial real data
        _part_real = recover_test_sample_part[_num, :, :]
        
        # TimesNet mse
        _part_real_timesnetmse = ymse_filtered_per_sample[_num].cpu().detach().numpy()

        # # timesnetmmd
        _part_real_timesnetmmd = ymmd_filtered_per_sample[_num].cpu().detach().numpy()

        r_samples_list.append(recovered_test_data[_num, :, :-1].cpu().detach().numpy())
        t_samples_list.append(samples_gmm)
        r_samples_part_list.append(_part_real)
        timesnet_sample_list.append(_part_real_timesnetmse)
        timesnet_sample_mmd_list.append(_part_real_timesnetmmd)
    # - ------add plot
    save_path = f'exp/exp_second_round/eva/evatransformer/plot2/case_transformer_{_random_num}.png'
    eva_function.create_plots(t_samples_list, r_samples_list, r_samples_part_list, timesnet_sample_list, timesnet_sample_mmd_list, save_path)
        
    # - ------add plot
    # # Plot the samples
    # plt.figure(figsize=(10, 6))
    # plt.subplot(4, 1, 1)

    # # plot the samples the colors indicate the sum of the samples
    # samples_gmm = samples_gmm 
    # for i in range(samples.shape[0]):
    #     _sum = samples_gmm[i, :-1].sum()
    #     color = plt.cm.viridis(_sum / test_data.max())
    #     plt.plot( samples_gmm[i, :-1], alpha=0.05, c=color)
    # plt.title('Samples from GMM')
    # plt.xlabel('Time')
    # plt.ylabel('Value')

    # plt.subplot(4, 1, 2)
    # # plot the real samples
    # for i in range(test_data.shape[1]):
    #     _sum = test_data[_num, i, :-1].sum()
    #     color = plt.cm.viridis(_sum / test_data.max())
    #     plt.plot(recovered_test_data[_num, i, :-1].cpu().detach().numpy(), alpha=0.05, c=color)
    # plt.title('real')
    # plt.xlabel('Time')
    # plt.ylabel('Value')
    
    # plt.subplot(4, 1, 3)
    # # plot the timesnet mse samples
    # for i in range(_part_real_timesnetmse.shape[0]):
    #     _sum = _part_real_timesnetmse[i, :-1].sum()
    #     color = plt.cm.viridis(_sum / test_data.max())
    #     plt.plot(_part_real_timesnetmse[i, :-1], alpha=0.05, c=color)
    # plt.title('TimesNet mse')
    # plt.xlabel('Time')
    # plt.ylabel('Value')
    # plt.tight_layout()
    
    # plt.subplot(4, 1, 4)
    # # plot the timesnet mmd samples
    # for i in range(_part_real_timesnetmmd.shape[0]):
    #     _sum = _part_real_timesnetmmd[i, :-1].sum()
    #     color = plt.cm.viridis(_sum / test_data.max())
    #     plt.plot(_part_real_timesnetmmd[i, :-1], alpha=0.05, c=color)
    # plt.title('TimesNet mmd')
    # plt.xlabel('Time')
    # plt.ylabel('Value')
    # plt.tight_layout()
    
    # # save the plot
    # plt.savefig(f'exp/exp_second_round/eva/evasolar/plot/samples_from_gmm_{_random_num}.png')
    # plt.close()

    