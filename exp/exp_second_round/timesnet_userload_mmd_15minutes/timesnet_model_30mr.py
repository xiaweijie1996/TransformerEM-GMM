#!/usr/bin/env python3
import os 
import sys
_parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(_parent_path)

import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import asset.timesnet_loader as timesloader
import exp_second_round.timesnet_userload_mmd_15minutes.timesnet_utils as ut
from exp_second_round.timesnet_userload_mmd_15minutes.timesnet_config import TimesBlockConfig 
import exp_second_round.timesnet_userload_mmd_15minutes.timesnet_train_mmd as tt
import wandb

class TimesBlock(nn.Module):
    
    def __init__(self, configs):    ##configs is the configuration defined for TimesBlock
        super(TimesBlock, self).__init__() 
        self.seq_len = configs.seq_len   ##sequence length 
        self.pred_len = configs.pred_len ##prediction length
        self.k = configs.top_k    ##k denotes how many top frequencies are 
                                                                #taken into consideration
        # parameter-efficient design
        self.conv = nn.Sequential(
            ut.Inception_Block_V1(configs.d_model, configs.d_ff,
                            num_kernels=configs.num_kernels),
            nn.GELU(),
            ut.Inception_Block_V1(configs.d_ff, configs.d_model,
                            num_kernels=configs.num_kernels)
        )
    
    def forward(self, x):
            B, T, N = x.size()
                #B: batch size  T: length of time series  N:number of features
            period_list, period_weight = ut.FFT_for_Period(x, self.k)
            
                #FFT_for_Period() will be shown later. Here, period_list([top_k]) denotes 
                #the top_k-significant period and period_weight([B, top_k]) denotes its weight(amplitude)

            res = []
            for i in range(self.k):
                period = period_list[i]

                # padding : to form a 2D map, we need total length of the sequence, plus the part 
                # to be predicted, to be divisible by the period, so padding is needed
                if (self.seq_len + self.pred_len) % period != 0:
                    length = (
                                    ((self.seq_len + self.pred_len) // period) + 1) * period
                    padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                    out = torch.cat([x, padding], dim=1)
                else:
                    length = (self.seq_len + self.pred_len)
                    out = x

                # reshape: we need each channel of a single piece of data to be a 2D variable,
                # Also, in order to implement the 2D conv later on, we need to adjust the 2 dimensions 
                # to be convolutioned to the last 2 dimensions, by calling the permute() func.
                # Whereafter, to make the tensor contiguous in memory, call contiguous()
                out = out.reshape(B, length // period, period,
                                N).permute(0, 3, 1, 2).contiguous()
                
                #2D convolution to grap the intra- and inter- period information
                
                out = self.conv(out)

                # reshape back, similar to reshape
                out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
                
                #truncating down the padded part of the output and put it to result
                res.append(out[:, :(self.seq_len + self.pred_len), :])
                
            res = torch.stack(res, dim=-1) #res: 4D [B, length , N, top_k]

            # adaptive aggregation
            #First, use softmax to get the normalized weight from amplitudes --> 2D [B,top_k]
            period_weight = F.softmax(period_weight, dim=1) 

            #after two unsqueeze(1),shape -> [B,1,1,top_k],so repeat the weight to fit the shape of res
            period_weight = period_weight.unsqueeze(
                1).unsqueeze(1).repeat(1, T, N, 1)
            
            #add by weight the top_k periods' result, getting the result of this TimesBlock
            res = torch.sum(res * period_weight, -1)

            # residual connection
            res = res + x
            return res


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        #params init
        configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        #stack TimesBlock for e_layers times to form the main part of TimesNet, named model
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        
        #embedding & normalization
        # enc_in is the encoder input size, the number of features for a piece of data
        # d_model is the dimension of embedding
        self.enc_embedding = ut.DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                        configs.dropout)
        self.layer = configs.e_layers # num of encoder layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        #define the some layers for different tasks

        self.projection = nn.Linear(
            configs.d_model, configs.c_out, bias=True)
        
    def imputation(self, x_enc, x_mark_enc, mask):

        x_enc = x_enc.masked_fill(mask == 0, 0)
        
        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        
        
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        return dec_out

    
    def forward(self, x_enc, x_mark_enc,  mask=None):
        dec_out = self.imputation(
            x_enc, x_mark_enc, mask)
        return dec_out  # [B, L, D] return the whole sequence with missing value estimated
    

if __name__ == '__main__':
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # check the model
    configs = TimesBlockConfig()
    model = Model(configs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # print number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(num_params)

    data_path = sys.argv[1] # 'exp/data_process_for_data_collection_all/new_data_15minute_grid_nomerge.pkl'
    data_loader = timesloader.TimesNetLoader(data_path, 
                                             batch_size=60, 
                                             split_ration=(0.8, 0.1, 0.1),
                                             full_length=366)
    
    sub_epoch = int(50000/60)
    epoch = 1000000
    max_loss = 10000
    
    wandb.init(project='timesnet_mmd_userload_15minutes',)
    tt.train_and_evaluate(model, data_loader, optimizer, device, epoch, sub_epoch, 1, num_params)
    