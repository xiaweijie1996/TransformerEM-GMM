import os
import sys
_parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(_parent_path)

import torch
import torch.nn as nn   

import model.gmm_transformer as gmm_model
from asset.dataloader import Dataloader_nolabel
import asset.random_sampler as rs
import asset.gmm_train_tool as gmm_train_tool
import asset.em_pytorch as ep

# Set default dtype to float64
torch.set_default_dtype(torch.float64)

# Determine the device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define the encoder parameters
random_sample_num = 96
n_components = 6    
chw = (1, random_sample_num,  97)
para_dim = n_components*2
hidden_d = 96*2
out_d = 96
n_heads = 1
mlp_ratio = 1
n_blocks = 2

# Create the encoder model
encoder = gmm_model.ViT_encodernopara(chw, hidden_d, out_d, n_heads, mlp_ratio, n_blocks).to(device)

# load state dict of the model
model_path = 'exp/exp_second_round/user_load_15minutes/model/loadmerge_encoder_96_704366.pth'
model = torch.load(model_path, map_location=device)
encoder.load_state_dict(model)

embedding_para = torch.nn.Embedding(n_components*2, 1).to(device)
emb_empty_token = torch.nn.Embedding(1, chw[2]).to(device)

path_embedding = 'exp/exp_second_round/user_load_15minutes/model/loadmerge_embedding_96_704366.pth'
emb_weight = torch.load(path_embedding, map_location=device, weights_only=False)
path_empty = 'exp/exp_second_round/user_load_15minutes/model/loadmerge_emb_empty_token_96_704366.pth'
empty_token_vec = torch.load(path_empty, map_location=device,weights_only=False)

# load data
batch_size =  64
split_ratio = (0.8,0.1,0.1)
data_path =  'exp/data_process_for_data_collection_all/new_data_15minute_grid_merge.pkl' ## 
dataset = Dataloader_nolabel(data_path,  batch_size=batch_size
                    , split_ratio=split_ratio)
print('lenthg of test data: ', dataset.__len__()*split_ratio[1])
test_data = dataset.load_test_data(64)

# normalize the input data
min_test_data = test_data[:,:, :-1].min(axis=1, keepdim=True)
max_test_data = test_data[:,:, :-1].max(axis=1, keepdim=True)
test_data = (test_data - min_test_data)/(max_test_data-min_test_data+1e-15)

_random_num = torch.randint(1, random_sample_num+1, (1,)).item()
_test_sample_part = rs.random_sample(test_data , 'random', _random_num)
_test_sample_part[:, :, -1] = _test_sample_part[:, :, -1]/365 # simple data embedding

_test_sample_part_emb = gmm_train_tool.pad_and_embed(_test_sample_part, random_sample_num, _random_num,
                                           emb_empty_token, device)
_ms, _covs = ep.GMM_PyTorch_Batch(n_components, _test_sample_part[:,:, :-1].shape[-1]).fit(_test_sample_part[:,:, :-1], 1) # _ms: (b, n_components, 24), _covs: (b, n_components, 24)

# concatenate the mean and variance to have (b, n_components*2, 25)
_param_emb, _param = gmm_train_tool.concatenate_and_embed_params(_ms, _covs, n_components, embedding_para, device)

# feed into the encoder
_test_sample_part_emb = torch.cat((_param_emb, _test_sample_part_emb), dim=1)
encoder_out = encoder(_test_sample_part_emb)
_new_para = encoder_out[:, :n_components*2, :]
_new_para = encoder.output_adding_layer(_new_para, _param)

mean = _new_para[:, :n_components* 96].view(-1, n_components, 96)
cov = _new_para[:, n_components* 96:].view(-1, n_components, 96)

print('mean shape: ', mean.shape)
print('cov shape: ', cov.shape)