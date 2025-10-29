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
import asset.gmm_train_tool as gmm_train_tool
import asset.em_pytorch as ep
import asset.plot_eva as plot_eva
import exp_second_round.eva.evaconditionalgen.eva_function as eva_function

torch.set_default_dtype(torch.float64)

# -----------------------------------Load model and data-----------------------------------
# import the dataloader
batch_size = 20
split_ratio = (0.8,0.1,0.5)
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
for random_sample_num_vae in [4, 8, 16, 32]:
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
    hidden_d = 48
    out_d = 96
    n_heads = 4
    mlp_ratio = 6
    n_blocks = 4
    
    # Load model of fixed weights
    encoder_fix = gmm_model.ViT_encodernopara(chw, hidden_d, out_d, n_heads, mlp_ratio, n_blocks).to(device) # Create the encoder model
    embedding_para_fix = torch.nn.Embedding(n_components*2 +1, 1).to(device) # +1 for gmm component withgts embedding of empty token
    emb_empty_token = torch.nn.Embedding(1, chw[2]).to(device)
    path_fix = 'exp/exp_second_round/gaussian_weight_exame/fixedweights_same'
    _model_scale = sum(p.numel() for p in encoder_fix.parameters() if p.requires_grad)
    # encoder_fix = torch.load(os.path.join(path_fix, f'transformer_encoder_{random_sample_num}_{_model_scale}.pth'), map_location=device)
    encoder_fix.load_state_dict(torch.load(os.path.join(path_fix, f'transformer_encoder_{random_sample_num}_{_model_scale}.pth'), map_location=device))
    embedding_para_fix = torch.load(os.path.join(path_fix, f'transformer_embedding_{random_sample_num}_{_model_scale}.pth'), map_location=device, weights_only=False)
    emb_empty_token = torch.load(os.path.join(path_fix, f'transformer_emb_empty_token_{random_sample_num}_{_model_scale}.pth'), map_location=device, weights_only=False)
    print(f'Load fixed weight model from {path_fix}, model scale: {_model_scale}')
    encoder_fix.eval()
    
    # load model of random weights1
    encoder_rand1 = gmm_model.ViT_encodernopara(chw, hidden_d, out_d, n_heads, mlp_ratio, n_blocks).to(device) # Create the encoder model
    embedding_para = torch.nn.Embedding(n_components*2 +1, 1).to(device) # +1 for gmm component withgts embedding of empty token
    emb_empty_token = torch.nn.Embedding(1, chw[2]).to(device)
    path_random = 'exp/exp_second_round/gaussian_weight_exame/fixedweights_randome1'
    _model_scale = sum(p.numel() for p in encoder_rand1.parameters() if p.requires_grad)
    # encoder_rand1 =torch.load(os.path.join(path_random, f'transformer_encoder_{random_sample_num}_{_model_scale}.pth'), map_location=device)
    encoder_rand1.load_state_dict(torch.load(os.path.join(path_random, f'transformer_encoder_{random_sample_num}_{_model_scale}.pth'), map_location=device))
    embedding_para = torch.load(os.path.join(path_random, f'transformer_embedding_{random_sample_num}_{_model_scale}.pth'), map_location=device, weights_only=False)
    emb_empty_token = torch.load(os.path.join(path_random, f'transformer_emb_empty_token_{random_sample_num}_{_model_scale}.pth'), map_location=device, weights_only=False)
    print(f'Load random weight model from {path_random}, model scale: {_model_scale}')
    encoder_rand1.eval()
    
    # load mode of random weights2
    encoder_random2 = gmm_model.ViT_encodernopara(chw, hidden_d, out_d, n_heads, mlp_ratio, n_blocks).to(device) # Create the encoder model
    embedding_para_random2 = torch.nn.Embedding(n_components*2 +1, 1).to(device) # +1 for gmm component withgts embedding of empty token
    emb_empty_token = torch.nn.Embedding(1, chw[2]).to(device)
    path_random2 = 'exp/exp_second_round/gaussian_weight_exame/fixedweights_randome2'
    _model_scale = sum(p.numel() for p in encoder_random2.parameters() if p.requires_grad)
    # encoder_random2 = torch.load(os.path.join(path_random2, f'transformer_encoder_{random_sample_num}_{_model_scale}.pth'), map_location=device)
    encoder_random2.load_state_dict(torch.load(os.path.join(path_random2, f'transformer_encoder_{random_sample_num}_{_model_scale}.pth'), map_location=device))
    embedding_para_random2 = torch.load(os.path.join(path_random2, f'transformer_embedding_{random_sample_num}_{_model_scale}.pth'), map_location=device, weights_only=False)
    emb_empty_token = torch.load(os.path.join(path_random2, f'transformer_emb_empty_token_{random_sample_num}_{_model_scale}.pth'), map_location=device, weights_only=False)
    print(f'Load random2 weight model from {path_random2}, model scale: {_model_scale}')
    encoder_random2.eval()
    
    # load model of flexible weights
    encoder_flex = gmm_model.ViT_encodernopara(chw, hidden_d, out_d, n_heads, mlp_ratio, n_blocks).to(device) # Create the encoder model
    embedding_para_flex = torch.nn.Embedding(n_components*2 +1, 1).to(device) # +1 for gmm component withgts embedding of empty token
    emb_empty_token = torch.nn.Embedding(1, chw[2]).to(device)
    path_flex = 'exp/exp_second_round/gaussian_weight_exame/flexibleweights'
    _model_scale = sum(p.numel() for p in encoder_flex.parameters() if p.requires_grad)
    # encoder_flex = torch.load(os.path.join(path_flex, f'transformer_encoder_{random_sample_num}_{_model_scale}.pth'), map_location=device)
    encoder_flex.load_state_dict(torch.load(os.path.join(path_flex, f'transformer_encoder_{random_sample_num}_{_model_scale}.pth'), map_location=device))
    embedding_para_flex = torch.load(os.path.join(path_flex, f'transformer_embedding_{random_sample_num}_{_model_scale}.pth'), map_location=device, weights_only=False)
    emb_empty_token = torch.load(os.path.join(path_flex, f'transformer_emb_empty_token_{random_sample_num}_{_model_scale}.pth'), map_location=device, weights_only=False)
    print(f'Load flexible weight model from {path_flex}, model scale: {_model_scale}')
    encoder_flex.eval() 

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
   
    # ---------------gmm fixed weights
    _test_sample_part_emb = gmm_train_tool.pad_and_embed(_test_sample_part, random_sample_num, random_sample_num_vae,
                                            emb_empty_token, device)
    
    
    _ms, _covs = ep.GMM_PyTorch_Batch(n_components, _test_sample_part[:,:, :-1].shape[-1]).fit(_test_sample_part[:,:, :-1], 1) # _ms: (b, n_components, 24), _covs: (b, n_components, 24)
    _ms = torch.zeros(_ms.shape[0], n_components, 96).to(device)
    _covs = torch.ones(_ms.shape[0], n_components, 96).to(device)
    
    # concatenate the mean and variance to have (b, n_components*2, 25)
    _param_emb_fix, _param_fix = gmm_train_tool.concatenate_and_embed_params(_ms, _covs, n_components, embedding_para_fix, device)
    # feed into the encoder
    _test_sample_part_emb_fix = torch.cat((_param_emb_fix, _test_sample_part_emb), dim=1)
    encoder_out_fix = encoder_fix(_test_sample_part_emb_fix)
    _new_para_fix = encoder_out_fix[:, :n_components*2, :]
    _new_para_fix = encoder_fix.output_adding_layer(_new_para_fix, _param_fix)
    # get the mean and covariance
    samples_fix = [ ]
    for _i in tqdm(range(len(_new_para_fix))):
        _sample , _ = plot_eva.sample_from_gmm(n_components, _new_para_fix, _num=_i)
        samples_fix.append(_sample)
        
    # ---------------gmm random weights1
    _test_sample_part_emb = gmm_train_tool.pad_and_embed(_test_sample_part, random_sample_num, random_sample_num_vae,
                                            emb_empty_token, device)
    _ms, _covs = ep.GMM_PyTorch_Batch(n_components, _test_sample_part[:,:, :-1].shape[-1]).fit(_test_sample_part[:,:, :-1], 1) # _ms: (b, n_components, 24), _covs: (b, n_components, 24)
    _ms = torch.zeros(_ms.shape[0], n_components, 96).to(device)
    _covs = torch.ones(_ms.shape[0], n_components, 96).to(device)
    # concatenate the mean and variance to have (b, n_components*2, 25)
    _param_emb, _param = gmm_train_tool.concatenate_and_embed_params(_ms, _covs, n_components, embedding_para, device)
    # feed into the encoder
    _test_sample_part_emb_rand1 = torch.cat((_param_emb, _test_sample_part_emb), dim=1)
    encoder_out_rand1 = encoder_rand1(_test_sample_part_emb_rand1)
    _new_para_rand1 = encoder_out_rand1[:, :n_components*2, :]
    _new_para_rand1 = encoder_rand1.output_adding_layer(_new_para_rand1, _param)
    # get the mean and covariance
    samples_rand1 = [ ]
    for _i in tqdm(range(len(_new_para_rand1))):
        _sample , _ = plot_eva.sample_from_gmm(n_components, _new_para_rand1, _num=_i)
        samples_rand1.append(_sample)
        
    # ---------------gmm random weights2
    _test_sample_part_emb = gmm_train_tool.pad_and_embed(_test_sample_part, random_sample_num, random_sample_num_vae,
                                            emb_empty_token, device)
    _ms, _covs = ep.GMM_PyTorch_Batch(n_components, _test_sample_part[:,:, :-1].shape[-1]).fit(_test_sample_part[:,:, :-1], 1) # _ms: (b, n_components, 24), _covs: (b, n_components, 24)
    _ms = torch.zeros(_ms.shape[0], n_components, 96).to(device)
    _covs = torch.ones(_ms.shape[0], n_components, 96).to(device)
    # concatenate the mean and variance to have (b, n_components*2, 25)
    _param_emb_random2, _param_random2 = gmm_train_tool.concatenate_and_embed_params(_ms, _covs, n_components, embedding_para_random2, device)
    # feed into the encoder
    _test_sample_part_emb_rand2 = torch.cat((_param_emb_random2, _test_sample_part_emb), dim=1)
    encoder_out_rand2 = encoder_random2(_test_sample_part_emb_rand2)
    _new_para_rand2 = encoder_out_rand2[:, :n_components*2, :]
    _new_para_rand2 = encoder_random2.output_adding_layer(_new_para_rand2, _param_random2)
    # get the mean and covariance
    samples_rand2 = [ ]
    for _i in tqdm(range(len(_new_para_rand2))):
        _sample , _ = plot_eva.sample_from_gmm(n_components, _new_para_rand2, _num=_i)
        samples_rand2.append(_sample)
    
    # ---------------gmm flexible weights
    _test_sample_part_emb = gmm_train_tool.pad_and_embed(_test_sample_part, random_sample_num, random_sample_num_vae,
                                            emb_empty_token, device)
    _ms, _covs = ep.GMM_PyTorch_Batch(n_components, _test_sample_part[:,:, :-1].shape[-1]).fit(_test_sample_part[:,:, :-1], 1) # _ms: (b, n_components, 24), _covs: (b, n_components, 24)
    _ms = torch.zeros(_ms.shape[0], n_components, 96).to(device)
    _covs = torch.ones(_ms.shape[0], n_components, 96).to(device)
    # concatenate the mean and variance to have (b, n_components*2, 25)
    _param_emb_flex, _param_flex = gmm_train_tool.concatenate_and_embed_params(_ms, _covs, n_components, embedding_para_flex, device)
    # feed into the encoder
    _test_sample_part_emb_flex = torch.cat((_param_emb_flex, _test_sample_part_emb), dim=1)
    encoder_out_flex = encoder_flex(_test_sample_part_emb_flex)
    _new_para_flex = encoder_out_flex[:, :n_components*2, :]
    _new_para_flex = encoder_flex.output_adding_layer(_new_para_flex, _param_flex)
    # get the mean and covariance
    samples_flex = [ ]
    for _i in tqdm(range(len(_new_para_flex))):
        _sample , _ = plot_eva.sample_from_gmm(n_components, _new_para_flex, _num=_i)
        samples_flex.append(_sample)
    
    print(len(samples_rand1))
    print(_test_max.shape)  
    # ---------------gmm random weights2
    
    
    mmd_fix = 0
    kl_fix = 0
    ks_fix = 0
    ws_fix = 0
    msem_fix = 0

    mmd_rand1 = 0
    kl_rand1 = 0
    ks_rand1 = 0
    ws_rand1 = 0
    msem_rand1 = 0
    
    mmd_rand2 = 0
    kl_rand2 = 0
    ks_rand2 = 0
    ws_rand2 = 0
    msem_rand2 = 0
    
    mmd_flex = 0
    kl_flex = 0
    ks_flex = 0
    ws_flex = 0
    msem_flex = 0 

    for i in tqdm(range(len(test_sample))): # range(len(test_sample))
        # samples scaled
        samples_real = test_sample[i]
        
        # gmm fixed weights
        samples_gmm_fix = torch.tensor(samples_fix[i])
        samples_gmm_rand1 = torch.tensor(samples_rand1[i])
        samples_gmm_rand2 = torch.tensor(samples_rand2[i])
        samples_gmm_flex = torch.tensor(samples_flex[i])
        
        # recover the samples
        _max = _test_max[i][:,:-1]
        _min = _test_min[i][:,:-1]
        samples_real = samples_real[:,:-1] *  (_max - _min) + _min
        samples_gmm_fix = samples_gmm_fix *  (_max - _min) + _min
        samples_gmm_rand1 = samples_gmm_rand1 *  (_max - _min) + _min
        samples_gmm_rand2 = samples_gmm_rand2 *  (_max - _min) + _min
        samples_gmm_flex = samples_gmm_flex *  (_max - _min) + _min
        
        # mmd += plot_eva.compute_mmd(samples_real.detach().numpy(), samples_vae.detach().numpy())
        # kl += plot_eva.compute_kl_divergence(samples_real.detach().numpy(), samples_vae.detach().numpy())
        # ks += ks_2samp(samples_real.flatten().detach().numpy(), samples_vae.flatten().detach().numpy())[0]
        # ws += wasserstein_distance(samples_real.flatten().detach().numpy(), samples_vae.flatten().detach().numpy())
        # msem += plot_eva.calculate_autocorrelation_mse(samples_real.detach().numpy(), samples_vae.detach().numpy())
        mmd_fix += plot_eva.compute_mmd(samples_real.detach().numpy(), samples_gmm_fix.detach().numpy())
        kl_fix += plot_eva.compute_kl_divergence(samples_real.detach().numpy(), samples_gmm_fix.detach().numpy())
        ks_fix += ks_2samp(samples_real.flatten().detach().numpy(), samples_gmm_fix.flatten().detach().numpy())[0]
        ws_fix += wasserstein_distance(samples_real.flatten().detach().numpy(), samples_gmm_fix.flatten().detach().numpy())
        msem_fix += plot_eva.calculate_autocorrelation_mse(samples_real.detach().numpy(), samples_gmm_fix.detach().numpy())
        
        mmd_rand1 += plot_eva.compute_mmd(samples_real.detach().numpy(), samples_gmm_rand1.detach().numpy())
        kl_rand1 += plot_eva.compute_kl_divergence(samples_real.detach().numpy(), samples_gmm_rand1.detach().numpy())
        ks_rand1 += ks_2samp(samples_real.flatten().detach().numpy(), samples_gmm_rand1.flatten().detach().numpy())[0]
        ws_rand1 += wasserstein_distance(samples_real.flatten().detach().numpy(), samples_gmm_rand1.flatten().detach().numpy())
        msem_rand1 += plot_eva.calculate_autocorrelation_mse(samples_real.detach().numpy(), samples_gmm_rand1.detach().numpy())
        
        mmd_rand2 += plot_eva.compute_mmd(samples_real.detach().numpy(), samples_gmm_rand2.detach().numpy())
        kl_rand2 += plot_eva.compute_kl_divergence(samples_real.detach().numpy(), samples_gmm_rand2.detach().numpy())
        ks_rand2 += ks_2samp(samples_real.flatten().detach().numpy(), samples_gmm_rand2.flatten().detach().numpy())[0]
        ws_rand2 += wasserstein_distance(samples_real.flatten().detach().numpy(), samples_gmm_rand2.flatten().detach().numpy())
        msem_rand2 += plot_eva.calculate_autocorrelation_mse(samples_real.detach().numpy(), samples_gmm_rand2.detach().numpy())
        
        mmd_flex += plot_eva.compute_mmd(samples_real.detach().numpy(), samples_gmm_flex.detach().numpy())
        kl_flex += plot_eva.compute_kl_divergence(samples_real.detach().numpy(), samples_gmm_flex.detach().numpy())
        ks_flex += ks_2samp(samples_real.flatten().detach().numpy(), samples_gmm_flex.flatten().detach().numpy())[0]
        ws_flex += wasserstein_distance(samples_real.flatten().detach().numpy(), samples_gmm_flex.flatten().detach().numpy())
        msem_flex += plot_eva.calculate_autocorrelation_mse(samples_real.detach().numpy(), samples_gmm_flex.detach().numpy())
        
    with open('exp/exp_second_round/eva/evaconditionalgen/sample_from_gmm_fixedweights.txt', 'a') as f:
        f.write(f'For {random_sample_num_vae}-shots:\n')
        f.write(f'GMM Fixed weights:\n')
        f.write(f'mmd: {mmd_fix/len(test_sample)}\n')
        f.write(f'kl: {kl_fix/len(test_sample)}\n')
        f.write(f'ks: {ks_fix/len(test_sample)}\n')
        f.write(f'ws: {ws_fix/len(test_sample)}\n')
        f.write(f'msem: {msem_fix/len(test_sample)}\n')
        
        f.write(f'GMM Random weights1:\n')
        f.write(f'mmd: {mmd_rand1/len(test_sample)}\n')
        f.write(f'kl: {kl_rand1/len(test_sample)}\n')
        f.write(f'ks: {ks_rand1/len(test_sample)}\n')
        f.write(f'ws: {ws_rand1/len(test_sample)}\n')
        f.write(f'msem: {msem_rand1/len(test_sample)}\n')
        
        f.write(f'GMM Random weights2:\n')
        f.write(f'mmd: {mmd_rand2/len(test_sample)}\n')
        f.write(f'kl: {kl_rand2/len(test_sample)}\n')
        f.write(f'ks: {ks_rand2/len(test_sample)}\n')
        f.write(f'ws: {ws_rand2/len(test_sample)}\n')
        f.write(f'msem: {msem_rand2/len(test_sample)}\n')
        
        f.write(f'GMM Flexible weights:\n')
        f.write(f'mmd: {mmd_flex/len(test_sample)}\n')
        f.write(f'kl: {kl_flex/len(test_sample)}\n')
        f.write(f'ks: {ks_flex/len(test_sample)}\n')
        f.write(f'ws: {ws_flex/len(test_sample)}\n')
        f.write(f'msem: {msem_flex/len(test_sample)}\n')
        f.write('\n')
    # with open('exp/exp_second_round/eva/evaconditionalgen/sample_from_vae.txt', 'a') as f:
    #     f.write(f'mmd of {random_sample_num_vae}-shots: {mmd/batch_size}\n')
    #     f.write(f'kl of {random_sample_num_vae}-shots: {kl/batch_size}\n')
    #     f.write(f'ks of {random_sample_num_vae}-shots: {ks/batch_size}\n')
    #     f.write(f'ws of {random_sample_num_vae}-shots: {ws/batch_size}\n')
    #     f.write(f'msem of {random_sample_num_vae}-shots: {msem/batch_size}\n')
    #     f.write('\n')