import os
import sys
_parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(_parent_path)

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np  

import exp_second_round.conditional_generativemodels.vae_model as vaemodel
from asset.dataloader import Dataloader_nolabel
import asset.random_sampler as rs

import model.gmm_transformer as gmm_model
import asset.gmm_train_tool as gmm_train_tool
import asset.em_pytorch as ep
import asset.plot_eva as plot_eva
import exp_second_round.eva.evaconditionalgen.eva_function as eva_function

# -----------------------------------Load model and data-----------------------------------
# import the dataloader
batch_size = 2
split_ratio = (0.8,0.1,0.1)
data_path =  'exp/data_process_for_data_collection_all/new_data_15minute_grid_nomerge.pkl'
dataset = Dataloader_nolabel(data_path,  batch_size=batch_size
                    , split_ratio=split_ratio)
dataset.images = dataset.images + np.abs(np.random.normal(0, 0.01, dataset.images.shape)) 
print('lenthg of test data: ', dataset.__len__()*split_ratio[0])
print('lenthg of test data: ', dataset.__len__()*split_ratio[1])

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define the hyperparameters
random_sample_num_vae = 32
num_epochs = int(10000)
input_shape=(250, 96)      # (C, L)
latent_channels = 16
hidden_dims= [32, 128, 256, 568]
cond_dims = [32, 128, 256]

# define the model
vae = vaemodel.ConvPoolVAE1D(input_shape=input_shape, condition_shape=(random_sample_num_vae, 96), 
                                    latent_channels=latent_channels,
                                    hidden_dims=hidden_dims, cond_dims=cond_dims).to(device)
    
# load vae 4 shot
if random_sample_num_vae == 4:
    path_vae = f'exp/exp_second_round/conditional_generativemodels/4shot/vae_1336858_{random_sample_num_vae}.pth'
if random_sample_num_vae == 8:
    path_vae = 'exp/exp_second_round/conditional_generativemodels/8shot/vae_1337242_8.pth'
if random_sample_num_vae == 16:
    path_vae = 'exp/exp_second_round/conditional_generativemodels/16shot/vae_1338010_16.pth'
if random_sample_num_vae == 32:
    path_vae = 'exp/exp_second_round/conditional_generativemodels/32shot/vae_1339546_32.pth'
    
vae.load_state_dict(torch.load(path_vae, map_location=device))

# load gmm
random_sample_num = 40
n_components = 6    
chw = (1, random_sample_num,  97)
para_dim = n_components*2
hidden_d = 96
out_d = 96
n_heads = 4
mlp_ratio = 12
n_blocks = 4

encoder = gmm_model.ViT_encodernopara(chw, hidden_d, out_d, n_heads, mlp_ratio, n_blocks).to(device) # Create the encoder model

model_path = 'exp/exp_second_round/user_load_15minutes/model/_encoder_40_6306166.pth' # load state dict of the model
model = torch.load(model_path, map_location=device)
encoder.load_state_dict(model)
embedding_para = torch.nn.Embedding(n_components*2, 1).to(device)
emb_empty_token = torch.nn.Embedding(1, chw[2]).to(device)
path_embedding = 'exp/exp_second_round/user_load_15minutes/model/_embedding_40_6306166.pth'
emb_weight = torch.load(path_embedding, map_location=device, weights_only=False)
path_empty = 'exp/exp_second_round/user_load_15minutes/model/_emb_empty_token_40_6306166.pth'
empty_token_vec = torch.load(path_empty, map_location=device,weights_only=False)

# load data
# test_sample = dataset.load_test_data()
test_sample = dataset.load_train_data()
test_sample = torch.tensor(test_sample, dtype=torch.float32).to(device)

# normalize the input data
_test_min,_ = test_sample.min(axis=1, keepdim=True)
_test_max,_ = test_sample.max(axis=1, keepdim=True)
test_sample = (test_sample - _test_min)/(_test_max-_test_min+1e-15)

# -----------------------------------Plot the result-----------------------------------
# vae
vae.eval()
# random_sample a number between min_random_sample_num and random_sample_num
_test_sample_part = rs.random_sample(test_sample , 'random', random_sample_num_vae)

# move the data to the device
_test_sample_part = _test_sample_part.to(device)

# feed into the model
_test_sample_part = _test_sample_part.double()
test_sample = test_sample.double()
recon, mu, logvar = vae(test_sample[:,:,:-1], _test_sample_part[:,:,:-1])

# gmm
# change _test_sample_part to float
encoder.eval()
_test_sample_part_emb = gmm_train_tool.pad_and_embed(_test_sample_part, random_sample_num, random_sample_num_vae,
                                        emb_empty_token, device)
_ms, _covs = ep.GMM_PyTorch_Batch(n_components, _test_sample_part[:,:, :-1].shape[-1]).fit(_test_sample_part[:,:, :-1], 1) # _ms: (b, n_components, 24), _covs: (b, n_components, 24)
_ms = torch.zeros(_ms.shape[0], n_components, 96).to(device)
_covs = torch.ones(_ms.shape[0], n_components, 96).to(device)

    
# concatenate the mean and variance to have (b, n_components*2, 25)
_param_emb, _param = gmm_train_tool.concatenate_and_embed_params(_ms, _covs, n_components, embedding_para, device)

# feed into the encoder
_test_sample_part_emb = torch.cat((_param_emb, _test_sample_part_emb), dim=1)
encoder_out = encoder(_test_sample_part_emb)
_new_para = encoder_out[:, :n_components*2, :]
_new_para = encoder.output_adding_layer(_new_para, _param)

# get the mean and covariance
mean = _new_para[:, :n_components* 96].view(-1, n_components, 96)
cov = _new_para[:, n_components* 96:].view(-1, n_components, 96)

t_samples_list = []
r_samples_list = []
r_samples_part_list = []
vae_sample_list = []
for i in range(mean.shape[0]):
    # samples scaled
    samples_gmm, gmm = plot_eva.sample_from_gmm(n_components, _new_para, i) # num
    samples_vae = recon[i]
    samples_partial = _test_sample_part[i]
    samples_real = test_sample[i]
    
    # recover the samples
    _max = _test_max[i][:,:-1]
    _min = _test_min[i][:,:-1]
    samples_gmm = torch.tensor(samples_gmm, dtype=torch.float32).to(device)
    samples_gmm = samples_gmm * (_max - _min) + _min
    samples_vae = samples_vae *  (_max - _min) + _min
    samples_partial = samples_partial[:,:-1] *  (_max - _min) + _min
    samples_real = samples_real[:,:-1] *  (_max - _min) + _min
    
    # add the samples to the list
    t_samples_list.append(samples_gmm.cpu().detach().numpy())
    r_samples_list.append(samples_real.cpu().detach().numpy())
    r_samples_part_list.append(samples_partial.cpu().detach().numpy())
    vae_sample_list.append(samples_vae.cpu().detach().numpy())
    
save_path = f'exp/exp_second_round/conditional_generativemodels/{random_sample_num_vae}shot/plot/{random_sample_num_vae}_plot.png'
eva_function.create_plots(t_samples_list, r_samples_list, r_samples_part_list, vae_sample_list, save_path)
        