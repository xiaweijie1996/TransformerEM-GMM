import os
import sys
_parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(_parent_path)

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np  

import exp_second_round.conditional_generativemodels.ddpm_model as ddpmmodel
import exp_second_round.conditional_generativemodels.nf_model as nfmodel
from asset.dataloader import Dataloader_nolabel
import asset.random_sampler as rs

import model.gmm_transformer as gmm_model
import asset.gmm_train_tool as gmm_train_tool
import asset.em_pytorch as ep
import asset.plot_eva as plot_eva
import exp_second_round.eva.evaconditionalgen.eva_function as eva_function

# -----------------------------------Load model and data-----------------------------------
# import the dataloader
batch_size = 80
split_ratio = (0 ,1, 0)
data_path =  'exp/data_process_for_data_collection_all/new_data_15minute_grid_nomerge.pkl'
dataset = Dataloader_nolabel(data_path,  batch_size=batch_size
                    , split_ratio=split_ratio)
dataset.images = dataset.images + np.abs(np.random.normal(0, 0.01, dataset.images.shape)) 
print('lenthg of test data: ', dataset.__len__()*split_ratio[0])
print('lenthg of test data: ', dataset.__len__()*split_ratio[1])

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define the hyperparameters
random_sample_num_vae = 4
num_epochs = int(10000)
input_shape=(250, 96)      # (C, L)
hidden_dims= [32, 128, 256, 568]
cond_dims = [32, 128, 256]

# define the model ddpm
ddpm = ddpmmodel.FFD_NL(in_channels=250, 
                        hidden_channels=240,
                        condition_channels=random_sample_num_vae,
                        t_max=300,
                        betas=ddpmmodel.linear_beta_schedule(300)).to(device)
# load vae 4 shot
if random_sample_num_vae == 4:
    path = 'exp/exp_second_round/conditional_generativemodels/4shot/ddpm_710170_4shot.pt'
elif random_sample_num_vae == 8:
    path = 'exp/exp_second_round/conditional_generativemodels/8shot/ddpm_720250_16shot.pt'
elif random_sample_num_vae == 16:
    path = 'exp/exp_second_round/conditional_generativemodels/16shot/ddpm_720250_16shot.pt'
elif random_sample_num_vae ==32:
    path = 'exp/exp_second_round/conditional_generativemodels/32shot/ddpm_720250_16shot.pt'
    
ddpm.load_state_dict(torch.load(path, map_location=device))

# define the model flow
flow = nfmodel.CNicemModel(input_c=250, 
                            hidden_c=128, 
                            condition_c=random_sample_num_vae, 
                            n_layers=3).to(device)

# load flow 4 shot
if random_sample_num_vae == 4:
    path = 'exp/exp_second_round/conditional_generativemodels/4shot/flow_885774_4shot.pt'
elif random_sample_num_vae == 8:
    path = 'exp/exp_second_round/conditional_generativemodels/8shot/flow_882414_8shot.pt'
elif random_sample_num_vae == 16:
    path = 'exp/exp_second_round/conditional_generativemodels/16shot/flow_882414_16shot.pt'
elif random_sample_num_vae ==32:
    path = 'exp/exp_second_round/conditional_generativemodels/32shot/flow_882414_32shot.pt'
flow.load_state_dict(torch.load(path, map_location=device))

# load data
_sample_indx=[17, 23]
test_sample = dataset.load_test_data(batch_size, _sample_indx)  # (B,N,L)
print('test sample shape: ', test_sample.shape)
# Use 5 and 6 th sample for testing

# test_sample = test_sample[_index, :, :]   # (B,N,L)
# test_sample = test_sample[:, :, :-1]   # (B,N,L)
test_sample = torch.tensor(test_sample, dtype=torch.float32).to(device)

# normalize the input data
_test_min,_ = test_sample.min(axis=1, keepdim=True)
_test_max,_ = test_sample.max(axis=1, keepdim=True)
test_sample = (test_sample - _test_min)/(_test_max-_test_min+1e-15)

# random_sample a number between min_random_sample_num and random_sample_num
_test_sample_part = rs.random_sample(test_sample , 'random', random_sample_num_vae)

# move the data to the device
_test_sample_part = _test_sample_part.to(device)

# feed into the model
_test_sample_part = _test_sample_part.double()
test_sample = test_sample.double()

# -----------------------------------Plot the result-----------------------------------
# ddpm eval
ddpm.eval()

recon = ddpmmodel.sampling(ddpm, _test_sample_part[:, : ,:-1])
recon = recon.float()

print(recon.shape, test_sample.shape, _test_sample_part.shape)

# flow eval
flow.eval()
z = torch.randn(recon.shape, device=device)
# print(z.shape, _test_sample_part[:, : ,:-1].shape)
recon_flow, _ = flow.inverse(z, _test_sample_part[:, : ,:-1])


f_samples_list = []
r_samples_list = []
r_samples_part_list = []
ddpm_sample_list = []
for i in range(len(_sample_indx)):
    print('Processing sample: ', i/len(_sample_indx))
    # samples scaled
    samples_ddpm = recon[i]
    samples_partial = _test_sample_part[i]
    samples_real = test_sample[i]
    samples_flow = recon_flow[i]
    # recover the samples
    _max = _test_max[i][:,:-1]
    _min = _test_min[i][:,:-1]
    
    samples_ddpm = samples_ddpm *  (_max - _min) + _min
    samples_flow = samples_flow *  (_max - _min) + _min
    samples_partial = samples_partial[:,:-1] *  (_max - _min) + _min
    samples_real = samples_real[:,:-1] *  (_max - _min) + _min
    
    # add the samples to the list
    r_samples_list.append(samples_real.cpu().detach().numpy())
    r_samples_part_list.append(samples_partial.cpu().detach().numpy())
    ddpm_sample_list.append(samples_ddpm.cpu().detach().numpy())
    # t_samples_list.append(samples_gmm.cpu().detach().numpy())
    f_samples_list.append(samples_flow.cpu().detach().numpy() * 0.5)
    
t_samples_list = r_samples_part_list
save_path = f'exp/exp_second_round/conditional_generativemodels/{random_sample_num_vae}shot/plot/{random_sample_num_vae}_plot_ddpmflow.png'
eva_function.create_plots2(f_samples_list, r_samples_list, r_samples_part_list, ddpm_sample_list, save_path)
        