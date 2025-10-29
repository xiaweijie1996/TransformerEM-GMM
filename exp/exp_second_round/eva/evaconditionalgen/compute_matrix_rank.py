import os
import sys
_parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(_parent_path)

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np  
from scipy.stats import ks_2samp, wasserstein_distance
from tqdm import tqdm

import exp_second_round.conditional_generativemodels.vae_model as vaemodel
from asset.dataloader import Dataloader_nolabel
import asset.random_sampler as rs

import model.gmm_transformer as gmm_model
import asset.gmm_train_tool_noem as gmm_train_tool
import asset.em_pytorch as ep
import asset.plot_eva as plot_eva
import exp_second_round.eva.evaconditionalgen.eva_function as eva_function

torch.set_default_dtype(torch.float64)

# -----------------------------------Load model and data-----------------------------------
# import the dataloader
batch_size = 32
split_ratio = (0.1,0.1,0.8)
data_path =  'exp/data_process_for_data_collection_all/new_data_15minute_grid_nomerge.pkl'
dataset = Dataloader_nolabel(data_path,  batch_size=batch_size
                    , split_ratio=split_ratio)
dataset.images = dataset.images + np.abs(np.random.normal(0, 0.01, dataset.images.shape)) 
print('lenthg of test data: ', dataset.__len__()*split_ratio[0])
print('lenthg of test data: ', dataset.__len__()*split_ratio[1])

# device
device = 'cpu'

# define the hyperparameters
# random_sample_num_vae = 32
for random_sample_num_vae in [4, 8, 16, 32]: # 8, 16, 32
    num_epochs = int(10000)
    input_shape=(250, 96)      # (C, L)
    latent_channels = 16
    hidden_dims= [32, 128, 256, 568]
    cond_dims = [32, 128, 256]

    # load gmm
    random_sample_num = 40
    n_components = 4 
    
    # define the encoder
    chw = (1, random_sample_num,  97)
    para_dim = n_components*2
    hidden_d = 48 * 2
    out_d = 96
    n_heads = 4
    mlp_ratio = 6
    n_blocks = 4
    
    # load r0
    encoder_r0 = gmm_model.ViT_encodernopara(chw, hidden_d, out_d, n_heads, mlp_ratio, n_blocks).to(device) # Create the encoder model
    embedding_para_r0 = torch.nn.Embedding(n_components*2 +1, 1).to(device) # +1 for gmm component withgts embedding of empty token
    emb_empty_token_r0 = torch.nn.Embedding(1, chw[2]).to(device)
    path_r0 = 'exp/exp_second_round/gaussian_variance_exame/r0'
    _model_scale = sum(p.numel() for p in encoder_r0.parameters() if p.requires_grad)
    encoder_r0.load_state_dict(torch.load(os.path.join(path_r0, f'transformer_encoder_{random_sample_num}_{_model_scale}.pth'), map_location=device))
    embedding_para_r0 = torch.load(os.path.join(path_r0, f'transformer_embedding_{random_sample_num}_{_model_scale}.pth'), map_location=device, weights_only=False)
    emb_empty_token_r0 = torch.load(os.path.join(path_r0, f'transformer_emb_empty_token_{random_sample_num}_{_model_scale}.pth'), map_location=device, weights_only=False)
    print(f'Load r0 model from {path_r0}, model scale: {_model_scale}')
    encoder_r0.eval()
    
    # load r1
    encoder_r1 = gmm_model.ViT_encodernopara(chw, hidden_d, out_d, n_heads, mlp_ratio, n_blocks).to(device) # Create the encoder model
    embedding_para_r1 = torch.nn.Embedding(n_components*2 , 1).to(device) # +1 for gmm component withgts embedding of empty token
    emb_empty_token_r1 = torch.nn.Embedding(1, chw[2]).to(device)
    path_r1 = 'exp/exp_second_round/gaussian_variance_exame/r1'
    _model_scale = sum(p.numel() for p in encoder_r1.parameters() if p.requires_grad)
    encoder_r1.load_state_dict(torch.load(os.path.join(path_r1, f'transformer_encoder_{random_sample_num}_{_model_scale}.pth'), map_location=device))
    embedding_para_r1 = torch.load(os.path.join(path_r1, f'transformer_embedding_{random_sample_num}_{_model_scale}.pth'), map_location=device, weights_only=False)
    emb_empty_token_r1 = torch.load(os.path.join(path_r1, f'transformer_emb_empty_token_{random_sample_num}_{_model_scale}.pth'), map_location=device, weights_only=False)
    print(f'Load r1 model from {path_r1}, model scale: {_model_scale}')
    encoder_r1.eval()
    
    # load r3
    encoder_r3 = gmm_model.ViT_encodernopara(chw, hidden_d, out_d, n_heads, mlp_ratio, n_blocks).to(device) # Create the encoder model
    embedding_para_r3 = torch.nn.Embedding(n_components*4 , 1).to(device) # +1 for gmm component withgts embedding of empty token
    emb_empty_token_r3 = torch.nn.Embedding(1, chw[2]).to(device)
    path_r3 = 'exp/exp_second_round/gaussian_variance_exame/r3'
    _model_scale = sum(p.numel() for p in encoder_r3.parameters() if p.requires_grad)
    encoder_r3.load_state_dict(torch.load(os.path.join(path_r3, f'transformer_encoder_{random_sample_num}_{_model_scale}.pth'), map_location=device))
    embedding_para_r3 = torch.load(os.path.join(path_r3, f'transformer_embedding_{random_sample_num}_{_model_scale}.pth'), map_location=device, weights_only=False) 
    emb_empty_token_r3 = torch.load(os.path.join(path_r3, f'transformer_emb_empty_token_{random_sample_num}_{_model_scale}.pth'), map_location=device, weights_only=False)
    print(f'Load r3 model from {path_r3}, model scale: {_model_scale}')
    encoder_r3.eval()
    
    # load r5
    encoder_r5 = gmm_model.ViT_encodernopara(chw, hidden_d, out_d, n_heads, mlp_ratio, n_blocks).to(device) # Create the encoder model
    embedding_para_r5 = torch.nn.Embedding(n_components*6 , 1).to(device) # +1 for gmm component withgts embedding of empty token
    emb_empty_token_r5 = torch.nn.Embedding(1, chw[2]).to(device)
    path_r5 = 'exp/exp_second_round/gaussian_variance_exame/r5'
    _model_scale = sum(p.numel() for p in encoder_r5.parameters() if p.requires_grad)
    encoder_r5.load_state_dict(torch.load(os.path.join(path_r5, f'transformer_encoder_{random_sample_num}_{_model_scale}.pth'), map_location=device))
    embedding_para_r5 = torch.load(os.path.join(path_r5, f'transformer_embedding_{random_sample_num}_{_model_scale}.pth'), map_location=device, weights_only=False)
    emb_empty_token_r5 = torch.load(os.path.join(path_r5, f'transformer_emb_empty_token_{random_sample_num}_{_model_scale}.pth'), map_location=device, weights_only=False)
    print(f'Load r5 model from {path_r5}, model scale: {_model_scale}')
    encoder_r5.eval()
    
    # Load r10
    encoder_r10 = gmm_model.ViT_encodernopara(chw, hidden_d, out_d, n_heads, mlp_ratio, n_blocks).to(device) # Create the encoder model
    embedding_para_r10 = torch.nn.Embedding(n_components*11, 1).to(device) # +1 for gmm component withgts embedding of empty token
    emb_empty_token_r10 = torch.nn.Embedding(1, chw[2]).to(device)
    path_r10 = 'exp/exp_second_round/gaussian_variance_exame/r10'
    _model_scale = sum(p.numel() for p in encoder_r10.parameters() if p.requires_grad)
    encoder_r10.load_state_dict(torch.load(os.path.join(path_r10, f'transformer_encoder_{random_sample_num}_{_model_scale}.pth'), map_location=device))
    embedding_para_r10 = torch.load(os.path.join(path_r10, f'transformer_embedding_{random_sample_num}_{_model_scale}.pth'), map_location=device, weights_only=False)
    emb_empty_token_r10 = torch.load(os.path.join(path_r10, f'transformer_emb_empty_token_{random_sample_num}_{_model_scale}.pth'), map_location=device, weights_only=False)
    print(f'Load r10 model from {path_r10}, model scale: {_model_scale}')
    encoder_r10.eval()
    
    # load data
    test_sample = dataset.load_test_data(batch_size)
    test_sample = torch.tensor(test_sample, dtype=torch.float64).to(device)

    # normalize the input data
    _test_min,_ = test_sample.min(axis=1, keepdim=True)
    _test_max,_ = test_sample.max(axis=1, keepdim=True)
    test_sample = (test_sample - _test_min)/(_test_max-_test_min+1e-15)

    _test_sample_part = rs.random_sample(test_sample , 'random', random_sample_num_vae)
    _test_sample_part = _test_sample_part.to(device)

    
    # -----------------------------------Plot the result-----------------------------------
   
    # ---------------gmm r0
    _test_sample_part_emb_r0 = gmm_train_tool.pad_and_embed(_test_sample_part, random_sample_num, random_sample_num_vae,
                                            emb_empty_token_r0, device)
    
    
    _ms, _covs = ep.GMM_PyTorch_Batch(n_components, _test_sample_part[:,:, :-1].shape[-1]).fit(_test_sample_part[:,:, :-1], 1) # _ms: (b, n_components, 24), _covs: (b, n_components, 24)
    _ms = torch.zeros(_ms.shape[0], n_components, 96).to(device)
    _covs = torch.ones(_ms.shape[0], n_components, 96).to(device)
    
    # concatenate the mean and variance to have (b, n_components*2, 25)
    _param_embr0, _param_r0 = gmm_train_tool.concatenate_and_embed_params(_ms, _covs, n_components, embedding_para_r0, device)
    # feed into the encoder
    _test_sample_part_emb_fix = torch.cat((_param_embr0, _test_sample_part_emb_r0), dim=1)
    encoder_out_r0 = encoder_r0(_test_sample_part_emb_fix)
    _new_para_r0 = encoder_out_r0[:, :n_components*2, :]
    _new_para_r0 = encoder_r0.output_adding_layer(_new_para_r0, _param_r0)
    # get the mean and covariance
    samples_r0_coll = [ ]
    for _i in tqdm(range(len(_new_para_r0))):
        _sample_r0 , _ = plot_eva.sample_from_gmm(n_components, _new_para_r0, _num=_i)
        # Drop nan sample_r0, is numpy array
        _sample_r0 = _sample_r0[~np.isnan(_sample_r0).any(axis=1)]
        samples_r0_coll.append(_sample_r0)
        
    print(samples_r0_coll[0].shape)
    
    # ---------------gmm r1
    _test_sample_part_emb_r1 = gmm_train_tool.pad_and_embed(_test_sample_part, random_sample_num, random_sample_num_vae,
                                            emb_empty_token_r1, device)
    
    _ms, _covs = ep.GMM_PyTorch_Batch(n_components, _test_sample_part[:,:, :-1].shape[-1]).fit(_test_sample_part[:,:, :-1], 1) # _ms: (b, n_components, 24), _covs: (b, n_components, 24)
    _ms = torch.zeros(_ms.shape[0], n_components, 96).to(device)
    _covs = torch.ones(_ms.shape[0], n_components, 96).to(device)
    
    _param_emb_r1, _param_r1 = gmm_train_tool.concatenate_and_embed_params(_ms, _covs, n_components, embedding_para_r1, device)
    # feed into the encoder
    _test_sample_part_emb_r1 = torch.cat((_param_emb_r1, _test_sample_part_emb_r1), dim=1)
    encoder_out_r1 = encoder_r1(_test_sample_part_emb_r1)
    _new_para_r1 = encoder_out_r1[:, :n_components*2, :]
    _new_para_r1 = encoder_r1.output_adding_layer(_new_para_r1, _param_r1)
    # get the mean and covariance
    samples_r1_coll = [ ]
    weights_1d = torch.full((n_components,), 1.0 / n_components, device=device)
    for _i in tqdm(range(len(_new_para_r1))):
        means_u, U_u, lam_u = plot_eva.unpack_rank_iso(_new_para_r1, n_components, 96, 1, 0.01, device=device)
        _samples_r1 = plot_eva.sample_rank_iso(means_u, U_u, lam_u, n_samples=250, weights=weights_1d)
        # drop nan, _sample_r1 is numpy array
        _samples_r1 = _samples_r1[~np.isnan(_samples_r1).any(axis=1)]
        
        samples_r1_coll.append(_samples_r1)

    print(samples_r1_coll[0].shape)
    
    # ---------------gmm r3
    _test_sample_part_emb_r3 = gmm_train_tool.pad_and_embed(_test_sample_part, random_sample_num, random_sample_num_vae,
                                            emb_empty_token_r3, device)
    _ms, _covs = ep.GMM_PyTorch_Batch(n_components, _test_sample_part[:,:, :-1].shape[-1]).fit(_test_sample_part[:,:, :-1], 1) # _ms: (b, n_components, 24), _covs: (b, n_components, 24)
    _ms = torch.zeros(_ms.shape[0], n_components, 96).to(device)
    _covs = torch.ones(_ms.shape[0], n_components*3, 96).to(device)
    
    print('ms covs shape: ', _ms.shape, _covs.shape)
    _param_emb_r3, _param_r3 = gmm_train_tool.concatenate_and_embed_params(_ms, _covs, n_components, embedding_para_r3, device)
    # feed into the encoder
    _test_sample_part_emb_r3 = torch.cat((_param_emb_r3, _test_sample_part_emb_r3), dim=1)
    encoder_out_r3 = encoder_r3(_test_sample_part_emb_r3)
    _new_para_r3 = encoder_out_r3[:, :n_components*4, :]
    print(_new_para_r3.shape, _param_r3.shape)
    _new_para_r3 = encoder_r3.output_adding_layer(_new_para_r3, _param_r3)
    # get the mean and covariance
    samples_r3_coll = [ ]
    weights_1d = torch.full((n_components,), 1.0 / n_components, device=device)
    for _i in tqdm(range(len(_new_para_r3))):
        means_u, U_u, lam_u = plot_eva.unpack_rank_iso(_new_para_r3, n_components, 96, 3, 0.01, device=device)
        _samples_r3 = plot_eva.sample_rank_iso(means_u, U_u, lam_u, n_samples=250, weights=weights_1d)
        # drop nan
        _samples_r3 = _samples_r3[~np.isnan(_samples_r3).any(axis=1)]
        samples_r3_coll.append(_samples_r3)  
    print(samples_r3_coll[0].shape)
    
    # ---------------gmm r5
    _test_sample_part_emb_r5 = gmm_train_tool.pad_and_embed(_test_sample_part, random_sample_num, random_sample_num_vae,
                                            emb_empty_token_r5, device)
    _ms, _covs = ep.GMM_PyTorch_Batch(n_components, _test_sample_part[:,:, :-1].shape[-1]).fit(_test_sample_part[:,:, :-1], 1) # _ms: (b, n_components, 24), _covs: (b, n_components, 24)
    _ms = torch.zeros(_ms.shape[0], n_components, 96).to(device)
    _covs = torch.ones(_ms.shape[0], n_components*5, 96).to(device)   
    _param_emb_r5, _param_r5 = gmm_train_tool.concatenate_and_embed_params(_ms, _covs, n_components, embedding_para_r5, device)
    # feed into the encoder
    _test_sample_part_emb_r5 = torch.cat((_param_emb_r5, _test_sample_part_emb_r5), dim=1)
    encoder_out_r5 = encoder_r5(_test_sample_part_emb_r5)
    _new_para_r5 = encoder_out_r5[:, :n_components*6, :]
    _new_para_r5 = encoder_r5.output_adding_layer(_new_para_r5, _param_r5)
    # get the mean and covariance
    samples_r5_coll = [ ]
    weights_1d = torch.full((n_components,), 1.0 / n_components, device=device)
    for _i in tqdm(range(len(_new_para_r5))):
        means_u, U_u, lam_u = plot_eva.unpack_rank_iso(_new_para_r5, n_components, 96, 5, 0.01, device=device)
        _samples_r5 = plot_eva.sample_rank_iso(means_u, U_u, lam_u, n_samples=250, weights=weights_1d)
        # drop nan
        _samples_r5 = _samples_r5[~np.isnan(_samples_r5).any(axis=1)]
        samples_r5_coll.append(_samples_r5)
    print(samples_r5_coll[1].shape)
    
    # ---------------gmm r10
    _test_sample_part_emb_r10 = gmm_train_tool.pad_and_embed(_test_sample_part, random_sample_num, random_sample_num_vae,
                                            emb_empty_token_r10, device)
    _ms, _covs = ep.GMM_PyTorch_Batch(n_components, _test_sample_part[:,:, :-1].shape[-1]).fit(_test_sample_part[:,:, :-1], 1) # _ms: (b, n_components, 24), _covs: (b, n_components, 24)
    _ms = torch.zeros(_ms.shape[0], n_components, 96).to(device)
    _covs = torch.ones(_ms.shape[0], n_components*10, 96).to(device)   
    _param_emb_r10, _param_r10 = gmm_train_tool.concatenate_and_embed_params(_ms, _covs, n_components, embedding_para_r10, device)
    # feed into the encoder
    _test_sample_part_emb_r10 = torch.cat((_param_emb_r10, _test_sample_part_emb_r10), dim=1)
    encoder_out_r10 = encoder_r10(_test_sample_part_emb_r10)
    _new_para_r10 = encoder_out_r10[:, :n_components*11, :]
    _new_para_r10 = encoder_r10.output_adding_layer(_new_para_r10, _param_r10)
    # get the mean and covariance
    samples_r10_coll = [ ]
    weights_1d = torch.full((n_components,), 1.0 / n_components, device=device)
    for _i in tqdm(range(len(_new_para_r10))):  
        means_u, U_u, lam_u = plot_eva.unpack_rank_iso(_new_para_r10, n_components, 96, 10, 0.01, device=device)
        _samples_r10 = plot_eva.sample_rank_iso(means_u, U_u, lam_u, n_samples=250, weights=weights_1d)
        # drop nan
        _samples_r10 = _samples_r10[~np.isnan(_samples_r10).any(axis=1)]
        # print('_samples_r10 shape: ', _samples_r10.shape)
        samples_r10_coll.append(_samples_r10)
    print(samples_r10_coll[1].shape)
    
    # -----------------------------------Evaluation-----------------------------------
    mmd_r0 = 0
    kl_r0 = 0
    ks_r0 = 0
    ws_r0 = 0
    msem_r0 = 0
    
    mmd_r1 = 0
    kl_r1 = 0
    ks_r1 = 0
    ws_r1 = 0
    msem_r1 = 0
    
    mmd_r3 = 0
    kl_r3 = 0
    ks_r3 = 0
    ws_r3 = 0
    msem_r3 = 0
    
    mmd_r5 = 0
    kl_r5 = 0
    ks_r5 = 0
    ws_r5 = 0
    msem_r5 = 0
    
    mmd_r10 = 0
    kl_r10 = 0
    ks_r10 = 0
    ws_r10 = 0
    msem_r10 = 0

    for i in tqdm(range(len(test_sample))): # range(len(test_sample))
        # samples scaled
        samples_real = test_sample[i]
        
        # gmm samples
        samples_r0 = torch.tensor(samples_r0_coll[i])
        samples_r1 = torch.tensor(samples_r1_coll[i])
        samples_r3 = torch.tensor(samples_r3_coll[i])
        samples_r5 = torch.tensor(samples_r5_coll[i])
        samples_r10 = torch.tensor(samples_r10_coll[i])
        
        # recover the samples
        _max = _test_max[i][:,:-1]
        _min = _test_min[i][:,:-1]
        samples_real = samples_real[:,:-1] *  (_max - _min) + _min
        samples_r0 = samples_r0 *  (_max - _min) + _min
        samples_r1 = samples_r1 *  (_max - _min) + _min
        samples_r3 = samples_r3 *  (_max - _min) + _min
        samples_r5 = samples_r5 *  (_max - _min) + _min
        samples_r10 = samples_r10 *  (_max - _min) + _min
        
        # check if any nan in samples
        assert not torch.isnan(samples_real).any(), 'nan in real samples'
        assert not torch.isnan(samples_r0).any(), 'nan in r0 samples'
        assert not torch.isnan(samples_r1).any(), 'nan in r1 samples'
        assert not torch.isnan(samples_r3).any(), 'nan in r3 samples'
        assert not torch.isnan(samples_r5).any(), 'nan in r5 samples'
        assert not torch.isnan(samples_r10).any(), 'nan in r10 samples'
        # check the shape of the sampel 
        print('shape of samples: ', samples_real.shape, samples_r0.shape, samples_r1.shape, samples_r3.shape, samples_r5.shape, samples_r10.shape)
        # gmm r0
        mmd_r0 += plot_eva.compute_mmd(samples_real.detach().numpy(), samples_r0.detach().numpy())
        kl_r0 += plot_eva.compute_kl_divergence(samples_real.detach().numpy(), samples_r0.detach().numpy())
        ks_r0 += ks_2samp(samples_real.flatten().detach().numpy(), samples_r0.flatten().detach().numpy())[0]
        ws_r0 += wasserstein_distance(samples_real.flatten().detach().numpy(), samples_r0.flatten().detach().numpy())
        msem_r0 += plot_eva.calculate_autocorrelation_mse(samples_real.detach().numpy(), samples_r0.detach().numpy())   
        
        # gmm r1
        mmd_r1 += plot_eva.compute_mmd(samples_real.detach().numpy(), samples_r1.detach().numpy())
        kl_r1 += plot_eva.compute_kl_divergence(samples_real.detach().numpy(), samples_r1.detach().numpy())
        ks_r1 += ks_2samp(samples_real.flatten().detach().numpy(), samples_r1.flatten().detach().numpy())[0]
        ws_r1 += wasserstein_distance(samples_real.flatten().detach().numpy(), samples_r1.flatten().detach().numpy())
        msem_r1 += plot_eva.calculate_autocorrelation_mse(samples_real.detach().numpy(), samples_r1.detach().numpy())
        
        # gmm r3
        mmd_r3 += plot_eva.compute_mmd(samples_real.detach().numpy(), samples_r3.detach().numpy())
        kl_r3 += plot_eva.compute_kl_divergence(samples_real.detach().numpy(), samples_r3.detach().numpy())
        ks_r3 += ks_2samp(samples_real.flatten().detach().numpy(), samples_r3.flatten().detach().numpy())[0]
        ws_r3 += wasserstein_distance(samples_real.flatten().detach().numpy(), samples_r3.flatten().detach().numpy())
        msem_r3 += plot_eva.calculate_autocorrelation_mse(samples_real.detach().numpy(), samples_r3.detach().numpy())
       
        # gmm r5
        mmd_r5 += plot_eva.compute_mmd(samples_real.detach().numpy(), samples_r5.detach().numpy())
        kl_r5 += plot_eva.compute_kl_divergence(samples_real.detach().numpy(), samples_r5.detach().numpy())
        ks_r5 += ks_2samp(samples_real.flatten().detach().numpy(), samples_r5.flatten().detach().numpy())[0]
        ws_r5 += wasserstein_distance(samples_real.flatten().detach().numpy(), samples_r5.flatten().detach().numpy())
        msem_r5 += plot_eva.calculate_autocorrelation_mse(samples_real.detach().numpy(), samples_r5.detach().numpy())
       
        # gmm r10
        mmd_r10 += plot_eva.compute_mmd(samples_real.detach().numpy(), samples_r10.detach().numpy())
        kl_r10 += plot_eva.compute_kl_divergence(samples_real.detach().numpy(), samples_r10.detach().numpy())
        ks_r10 += ks_2samp(samples_real.flatten().detach().numpy(), samples_r10.flatten().detach().numpy())[0]
        ws_r10 += wasserstein_distance(samples_real.flatten().detach().numpy(), samples_r10.flatten().detach().numpy())
        msem_r10 += plot_eva.calculate_autocorrelation_mse(samples_real.detach().numpy(), samples_r10.detach().numpy())
        
    with open('exp/exp_second_round/eva/evaconditionalgen/sample_from_gmm_rank_variance.txt', 'a') as f:
        f.write(f'For {random_sample_num_vae}-shots:\n')
        f.write(f'GMM Rank 0:\n')
        f.write(f'mmd: {mmd_r0/len(test_sample)}\n')
        f.write(f'kl: {kl_r0/len(test_sample)}\n')
        f.write(f'ks: {ks_r0/len(test_sample)}\n')
        f.write(f'ws: {ws_r0/len(test_sample)}\n')
        f.write(f'msem: {msem_r0/len(test_sample)}\n')
        
        f.write(f'GMM Rank 1:\n')
        f.write(f'mmd: {mmd_r1/len(test_sample)}\n')
        f.write(f'kl: {kl_r1/len(test_sample)}\n')
        f.write(f'ks: {ks_r1/len(test_sample)}\n')
        f.write(f'ws: {ws_r1/len(test_sample)}\n')
        f.write(f'msem: {msem_r1/len(test_sample)}\n')
        
        f.write(f'GMM Rank 3:\n')
        f.write(f'mmd: {mmd_r3/len(test_sample)}\n')
        f.write(f'kl: {kl_r3/len(test_sample)}\n')
        f.write(f'ks: {ks_r3/len(test_sample)}\n')
        f.write(f'ws: {ws_r3/len(test_sample)}\n')
        f.write(f'msem: {msem_r3/len(test_sample)}\n')
        
        f.write(f'GMM Rank 5:\n')
        f.write(f'mmd: {mmd_r5/len(test_sample)}\n')
        f.write(f'kl: {kl_r5/len(test_sample)}\n')
        f.write(f'ks: {ks_r5/len(test_sample)}\n')
        f.write(f'ws: {ws_r5/len(test_sample)}\n')
        f.write(f'msem: {msem_r5/len(test_sample)}\n')
        
        f.write(f'GMM Rank 10:\n')
        f.write(f'mmd: {mmd_r10/len(test_sample)}\n')
        f.write(f'kl: {kl_r10/len(test_sample)}\n')
        f.write(f'ks: {ks_r10/len(test_sample)}\n')
        f.write(f'ws: {ws_r10/len(test_sample)}\n')
        f.write(f'msem: {msem_r10/len(test_sample)}\n')
        f.write('\n')