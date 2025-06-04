#!/usr/bin/env python3
import os 
import sys
_parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(_parent_path)

import wandb
import torch
import numpy as np
import torch.optim as optim

import model.gmm_transformer as gmm_model
import asset.gmm_train_tool_noem as gmm_train_tool
import asset.em_pytorch as ep
import asset.plot_eva as pa 
from asset.dataloader import Dataloader_nolabel

torch.set_default_dtype(torch.float64)

# load data
batch_size =  128
split_ratio = (0.8,0.1,0.1)
data_path =  'exp/data_process_for_data_collection_all/transformer_data_15minutes.pkl' # sys.argv[1]  #'exp/data_process_for_data_collection_all/new_data_15minute_grid_nomerge.pkl' ## 
dataset = Dataloader_nolabel(data_path,  batch_size=batch_size
                    , split_ratio=split_ratio)
dataset.images = dataset.images # + np.abs(np.random.normal(0, 0.01, dataset.images.shape)) 
print('lenthg of train data: ', dataset.__len__()*split_ratio[0])
print('lenthg of test data: ', dataset.__len__()*split_ratio[1])

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# define the hyperparameters
random_sample_num = 40
num_epochs = int(5000000)
sub_epoch = int(dataset.__len__()*split_ratio[0]/batch_size)
save_model = 'exp/exp_second_round/transformer_15minutes/model2/'
save_image = 'exp/exp_second_round/transformer_15minutes/gen_img2/'
lr = 0.001
n_components = 6
min_random_sample_num = 8

# define the encoder
chw = (1, random_sample_num,  97)
para_dim = n_components*2
hidden_d = 96
out_d = 96
n_heads = 4
mlp_ratio = 12
n_blocks = 4
encoder = gmm_model.ViT_encodernopara(chw, hidden_d, out_d, n_heads, mlp_ratio, n_blocks).to(device)
_model_scale = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
print('number of parameters: ', _model_scale)

# Define a gmm embedding layer
embedding_para = torch.nn.Embedding(n_components*2, 1).to(device)
emb_empty_token = torch.nn.Embedding(1, chw[2]).to(device)

# define the optimizer and loss function and cyclic learning rate scheduler
optimizer = optim.AdamW(list(encoder.parameters()), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False)
scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=5e-5, max_lr=lr, step_size_up=1000, mode='triangular', cycle_momentum=False)

# log number of parameters of encoder and decoder
wandb.init(project=f'transformer_{random_sample_num}_train_nomerge')
wandb.log({'num_parameters_encoder': _model_scale})

# train the model2
mid_loss = 100000

for epoch in range(num_epochs):
   
    # train.model
    encoder.train()
    embedding_para.train()
    emb_empty_token.train()
    
    # _loss, _new_para, _param, train_sample[:, :, :-1], _train_sample_part[:, n_components*2:, :-1], var, (_train_min, _train_max) 
    _loss, _random_num, _new_para, _param, r_samples, r_samples_part, _mm = gmm_train_tool.get_loss_le(dataset, encoder,
                                                                            random_sample_num, min_random_sample_num, n_components, 
                                                                            embedding_para, emb_empty_token, 'True', device)
    optimizer.zero_grad()
    _loss.backward()
    optimizer.step()
    scheduler.step()
        
    wandb.log({'loss_train': _loss.item(), 'random_num': _random_num, 'epoch': epoch, 'lr': optimizer.param_groups[0]['lr']})
    print('epoch: ', epoch, 'loss_test: ', _loss.item(), 'random_num: ', _random_num)
    
    if epoch % 100 == 0:
        encoder.eval()
        embedding_para.eval()
        emb_empty_token.eval()
        _loss, _random_num, _new_para, _param, r_samples, r_samples_part, _mm = gmm_train_tool.get_loss_le(dataset, encoder,
                                                                            random_sample_num, min_random_sample_num, n_components, 
                                                                            embedding_para, emb_empty_token, 'False', device)
        
        
        wandb.log({'loss_test': _loss.item(), 'random_num': _random_num, 'epoch':epoch})
        
        save_path = save_image+f'_{_model_scale}.png'
        llk_e = pa.plot_samples(save_path, batch_size, n_components, _mm, _new_para, r_samples, r_samples_part, _param, figsize=(10, 15))
    
    # save the model and embeding
    if _loss.item() < mid_loss:
        mid_loss = _loss.item()
        print('save model and embedding')
        torch.save(encoder.state_dict(), save_model + f'transformer_encoder_{random_sample_num}_{_model_scale}.pth')
        torch.save(embedding_para, save_model + f'transformer_embedding_{random_sample_num}_{_model_scale}.pth')
        torch.save(emb_empty_token, save_model + f'transformer_emb_empty_token_{random_sample_num}_{_model_scale}.pth')
