#!/usr/bin/env python3

import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import wandb
import torch
import torch.optim as optim

import model.gmm_transformer as gmm_model
import asset.gmm_train_abl as gmm_train_abl
import asset.plot_eva as pae
from asset.dataloader import Dataloader_nolabel
torch.set_default_dtype(torch.float64)

# load data
batch_size = 64*20
split_ratio = (0.8,0.1,0.1)
data_path = '/home/weijiexia/paper3/data_process_for_data_collection_all/all_data_aug_30mr.pkl'
dataset = Dataloader_nolabel(data_path,  batch_size=batch_size
                    , split_ratio=split_ratio)
print('lenthg of train data: ', dataset.__len__()*split_ratio[0])
print('lenthg of test data: ', dataset.__len__()*split_ratio[1])

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# define the hyperparameters
random_sample_num = 49
num_epochs = int(50)
sub_epoch = int(dataset.__len__()*split_ratio[0]/batch_size)
save_model = '/home/weijiexia/paper3/mmd_vit_exp_abl_half/model/'
save_image = '/home/weijiexia/paper3/mmd_vit_exp_abl_half/gen_img/'
lr = 0.0001
n_components = 6
min_random_sample_num = 8

# define the encoder
chw = (1, random_sample_num,  49)
para_dim = n_components*2
hidden_d = 30
out_d = 48
n_heads = 2
mlp_ratio = 2
n_blocks = 3
encoder = gmm_model.ViT_encodernopara(chw, hidden_d, out_d, n_heads, mlp_ratio, n_blocks).to(device)
_model_scale = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
print('number of parameters: ', _model_scale)

# Define a gmm embedding layer
embedding_para = torch.nn.Embedding(n_components*2, chw[2]).to(device)
emb_empty_token = torch.nn.Embedding(1, chw[2]).to(device)

# define the optimizer and loss function and cyclic learning rate scheduler
optimizer = optim.AdamW(list(encoder.parameters()), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False)
scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=5e-5, max_lr=1e-3, step_size_up=sub_epoch*2, mode='triangular', cycle_momentum=False)

# log number of parameters of encoder and decoder
wandb.init(project=f'small_{random_sample_num}_30m_wdt', entity='xiaweijie1996')
wandb.log({'num_parameters_encoder': _model_scale})

# train the model
mid_loss = 100000

for epoch in range(num_epochs):
    for j in range(sub_epoch):
        # train.model
        encoder.train()
        
        # _loss, _new_para, _param, train_sample[:, :, :-1], _train_sample_part[:, n_components*2:, :-1], var, (_train_min, _train_max) 
        _loss, _random_num, _new_para, _param, r_samples, r_samples_part, _mm = gmm_train_abl.get_loss_le(dataset, encoder,
                                                                              random_sample_num, min_random_sample_num, n_components, 
                                                                              embedding_para, emb_empty_token, 'True', device)
        optimizer.zero_grad()
        _loss.backward()
        optimizer.step()
        scheduler.step()
        print('epoch: ', epoch, 'loss_train: ', _loss.item(), 'random_num: ', _random_num)
        
    if epoch % 1 == 0:
        encoder.eval()
        wandb.log({'loss_train': _loss.item()})
        _loss_collection = []
        for _ in range(1):
            _loss, _random_num, _new_para, _param, r_samples, r_samples_part, _mm = gmm_train_abl.get_loss_le(dataset, encoder,
                                                                                random_sample_num, min_random_sample_num, n_components, 
                                                                                embedding_para, emb_empty_token, 'False', device)
            _loss_collection.append(_loss)
        _loss = torch.stack(_loss_collection).mean()
        # print('epoch: ', epoch, 'loss_test: ', _loss.item(), 'random_num: ', _random_num)
        wandb.log({'loss_test': _loss.item(), 'random_num': _random_num, 'epoch':epoch})
        
        # save the model and embeding
        if _loss.item() < mid_loss:
            mid_loss = _loss.item()
            torch.save(encoder.state_dict(), save_model + f'small_30m_encoder_wdt_{random_sample_num}_{_model_scale}.pth')
            torch.save(embedding_para, save_model + f'small_30m_embedding_wdt_{random_sample_num}_{_model_scale}.pth')
            torch.save(emb_empty_token, save_model + f'small_30m_emb_empty_token_wdt_{random_sample_num}_{_model_scale}.pth')

    if epoch % 10 == 0:
        save_path = save_image+str(epoch)+f'small_wdt_{random_sample_num}_{_model_scale}.png'
        llk_e = pae.plot_samples(save_path, batch_size, n_components, _mm, _new_para, r_samples, r_samples_part, _param, figsize=(10, 15))
        
        