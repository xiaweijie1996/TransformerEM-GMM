from typing import Union

import torch
import torch.nn as nn   
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

import model.gmm_transformer as gmm_model
import asset.dataloader as dl
import asset.em_pytorch as ep
import asset.random_sampler as rs
import asset.gmm_train_tool as gtt
import asset.plot_eva as pe

class GMMsTransPipline:
    def from_pretrained(
        n_components: int = 6,
        resolution: int = 24):
        
        # check the model type
        if resolution == 24:
            n_components = 6,
            hidden_d = 24 * 4,
            out_d = 24,
            n_heads = 4,
            mlp_ratio = 8,
            n_blocks = 6,
            encoder_path = r'mmd_vit_exp_30\model\_encoder_25_4537398.pth',
            path_para = r'mmd_vit_exp_30\model\_embedding_25_4537398.pth',
            path_token = r'mmd_vit_exp_30\model\_emb_empty_token_25_4537398.pth',
            random_sample_num = None
        pass





def load_model(
    n_components = 6,
    hidden_d = 24 * 4,
    out_d = 24,
    n_heads = 4,
    mlp_ratio = 8,
    n_blocks = 6,
    encoder_path = r'mmd_vit_exp_30\model\_encoder_25_4537398.pth',
    path_para = r'mmd_vit_exp_30\model\_embedding_25_4537398.pth',
    path_token = r'mmd_vit_exp_30\model\_emb_empty_token_25_4537398.pth',
    random_sample_num = None
    ):

    chw = (1, random_sample_num,  25)
    
    # Set device to GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the transformer model
    encoder = gmm_model.ViT_encodernopara(chw, hidden_d, out_d, n_heads, mlp_ratio, n_blocks).to(device)
    _model_scale = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print('Number of parameters of encoder:', _model_scale)

    # Load the pre-trained model state
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))

    state_dict_para = torch.load(path_para, map_location=device)
    state_dict_token = torch.load(path_token, map_location=device)
    
    return encoder, state_dict_para, state_dict_token

def load_valdata(
    batch_size = 10,
    split_ratio = (0.8,0.1,0.1),
    data_path = r'data_process_for_data_collection_all\all_data_aug.pkl'):
    dataloader = dl.Dataloader_nolabel(data_path, batch_size, split_ratio)
    return dataloader


def inference(encoder, para_emb, token_emb, _val_data, sample_num, n_components=6, device='cpu'):
    encoder.eval()
    # _val_data = dataloader.load_vali_data(size=1)
    _val_data = torch.tensor(_val_data, dtype=torch.float64).to(device)
    
    # normalize the input data
    _val_min,_ = _val_data[:,:, :-1].min(axis=1, keepdim=True)
    _val_max,_ = _val_data[:,:, :-1].max(axis=1, keepdim=True)
    _val_data[:,:, :-1] = (_val_data[:,:, :-1] - _val_min)/(_val_max-_val_min+1e-15)
    _val_sample_part = rs.random_sample(_val_data, 'random', sample_num)
    _val_sample_part[:, :, -1] = _val_sample_part[:, :, -1]/365 # simple date embedding
    
    # padding the empty token to _train_sample_part to have shape (b, random_sample_num, 25)
    _val_sample_part_emb = gtt.pad_and_embed(_val_sample_part, 25, sample_num, token_emb, device)
    
    # use ep to do one iteration of the EM algorithm
    _ms, _covs = ep.GMM_PyTorch_Batch(n_components, 24).fit(_val_sample_part[:,:, :-1], 1)
    
    # concatenate the mean and variance to have (b, n_components*2, 25)
    _param_emb, _param = gtt.concatenate_and_embed_params(_ms, _covs, n_components, para_emb, device)
    
    # feed into the encoder
    _val_sample_part_emb = torch.cat((_param_emb, _val_sample_part_emb), dim=1)
    encoder_out = encoder(_val_sample_part_emb)
    
    _new_para = encoder_out[:, :n_components*2, :]
    _new_para = encoder.output_adding_layer(_new_para, _param)
        
    return _new_para, _param, _val_data[:, :, :-1], _val_sample_part[:, :, :-1], (_val_min, _val_max)


def sample_and_scale(
    _new_para = None,
    _mm = None,
    _val_data = None,
    _val_sample_part = None,
    sample_amount = 250,
    n_components = 6,
    ):
    _min = _mm[0].cpu().detach().numpy()
    _max = _mm[1].cpu().detach().numpy()
    _samples, _gmm = pe.sample_from_gmm(n_components=n_components , _new_para=_new_para, _num=0, _num_samples=sample_amount)
    t_samples = _samples * (_max - _min) + _min
    t_samples[t_samples < 0] = 0
    r_samples_scale_back =_val_data * (_max - _min) + _min
    r_samples_part_scale_back = _val_sample_part * (_max - _min) + _min
    return t_samples[0], r_samples_scale_back[0], r_samples_part_scale_back[0]
