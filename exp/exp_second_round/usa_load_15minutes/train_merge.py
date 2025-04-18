#!/usr/bin/env python3
import os 
import sys
_parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(_parent_path)

import wandb
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

import model.gmm_transformer as gmm_model
import asset.gmm_train_tool as gmm_train_tool
import asset.em_pytorch as ep
import asset.plot_eva as pae
from asset.dataloader import Dataloader_nolabel
torch.set_default_dtype(torch.float64)

# load data
batch_size =  12
split_ratio = (0.8,0.1,0.1)
data_path =  'exp/data_process_for_data_collection_all/new_data_15minute_grid_nomerge.pkl' ## 
dataset = Dataloader_nolabel(data_path,  batch_size=batch_size
                    , split_ratio=split_ratio)
print('lenthg of train data: ', dataset.__len__()*split_ratio[0])
print('lenthg of test data: ', dataset.__len__()*split_ratio[1])

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# define the hyperparameters
random_sample_num = 96
num_epochs = int(50000)
sub_epoch = int(dataset.__len__()*split_ratio[0]/batch_size)
save_model = 'exp/exp_second_round/usa_load_15minutes/model/'
save_image = 'exp/exp_second_round/usa_load_15minutes/gen_img/'
lr = 0.0001
n_components = 6
min_random_sample_num = 8

# define the encoder
chw = (1, random_sample_num,  97)
para_dim = n_components*2
hidden_d = 96*4
out_d = 96
n_heads = 2
mlp_ratio = 1
n_blocks = 2
encoder = gmm_model.ViT_encodernopara(chw, hidden_d, out_d, n_heads, mlp_ratio, n_blocks).to(device)
_model_scale = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
print('number of parameters: ', _model_scale)

# Define a gmm embedding layer
embedding_para = torch.nn.Embedding(n_components*2, 1).to(device)
emb_empty_token = torch.nn.Embedding(1, chw[2]).to(device)

# define the optimizer and loss function and cyclic learning rate scheduler
optimizer = optim.Adam(list(encoder.parameters()), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False)
scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=5e-5, max_lr=1e-3, step_size_up=sub_epoch*2, mode='triangular', cycle_momentum=False)

# log number of parameters of encoder and decoder
wandb.init(project=f'load_usa_{random_sample_num}_train_merge', entity='xiaweijie1996')
wandb.log({'num_parameters_encoder': _model_scale})

# train the model2
mid_loss = 100000

mixer = GradScaler()
for epoch in range(num_epochs):
    torch.cuda.empty_cache()

    for j in range(sub_epoch):
        # train.model
        encoder.train()
        optimizer.zero_grad()
        with autocast():
        # _loss, _new_para, _param, train_sample[:, :, :-1], _train_sample_part[:, n_components*2:, :-1], var, (_train_min, _train_max) 
            _loss, _random_num, _new_para, _param, r_samples, r_samples_part, _mm = gmm_train_tool.get_loss_le(dataset, encoder,
                                                                                random_sample_num, min_random_sample_num, n_components, 
                                                                                embedding_para, emb_empty_token, 'True', device)
        mixer.scale(_loss).backward()
        mixer.step(optimizer)
        mixer.update()
        scheduler.step()
        
    if epoch % 1 == 0:
        encoder.eval()
        wandb.log({'loss_train': _loss.item()})
        _loss_collection = []
        for _ in range(20):
            _loss, _random_num, _new_para, _param, r_samples, r_samples_part, _mm = gmm_train_tool.get_loss_le(dataset, encoder,
                                                                                random_sample_num, min_random_sample_num, n_components, 
                                                                                embedding_para, emb_empty_token, 'False', device)
            _loss_collection.append(_loss)
        _loss = torch.stack(_loss_collection).mean()
        # print('epoch: ', epoch, 'loss_test: ', _loss.item(), 'random_num: ', _random_num)
        wandb.log({'loss_test': _loss.item(), 'random_num': _random_num, 'epoch':epoch})
        
        # save the model and embeding
        if _loss.item() < mid_loss:
            mid_loss = _loss.item()
            torch.save(encoder.state_dict(), save_model + f'_encoder_{random_sample_num}_{_model_scale}.pth')
            torch.save(embedding_para, save_model + f'_embedding_{random_sample_num}_{_model_scale}.pth')
            torch.save(emb_empty_token, save_model + f'_emb_empty_token_{random_sample_num}_{_model_scale}.pth')

        # e, c = pae.evaluation(n_components, _new_para, r_samples, r_samples_part)

        # wandb.log({'mmd_t':e[0],'mmd_r':e[1], 'mmd_gmm':e[2]})
        # wandb.log({'c_t':c[0],'c_r':c[1], 'c_gmm':c[2]})

    if epoch % 50 == 0:
        save_path = save_image+str(epoch)+f'_{random_sample_num}_{_model_scale}.png'
        llk_e = pae.plot_samples(save_path, batch_size, n_components, _mm, _new_para, r_samples, r_samples_part, _param, figsize=(10, 15))
        
        