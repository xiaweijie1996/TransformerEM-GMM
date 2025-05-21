import os, sys
_parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(_parent_path)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import wandb
import torch

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


import torch.optim as optim
import torch.nn.functional as F

from asset.dataloader import Dataloader_nolabel
# import exp_second_round.finetune.vae_model as vaemodel
import exp_second_round.conditional_generativemodels.vae_model as vaemodel
import asset.plot_eva as eva
import asset.random_sampler as rs

def loss_function(recon_x, x, mu, logvar, reduction='sum'):
    """
    VAE loss = reconstruction loss + KL divergence
      recon_x: reconstructed output, shape (B, C, L)
      x      : target input,       shape (B, C, L)
      mu, logvar: latent stats,     shape (B, z, L_enc)

    Returns:
      total_loss = 10 * recon_BCE + kld
      recon_BCE, kld
    """
    # reconstruction loss (BCE over all elements)
    recon_loss = F.mse_loss(recon_x, x, reduction=reduction)
    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # kl = 0
    return recon_loss * 10 + kld, recon_loss, kld


if __name__ == "__main__":
    batch_size = 7
    split_ratio = (0.8,0.1,0.1)
    data_path =  'exp/data_process_for_data_collection_all/new_data_15minute_grid_nomerge.pkl'
    dataset = Dataloader_nolabel(data_path,  batch_size=batch_size
                        , split_ratio=split_ratio)
  
    print('lenthg of train data: ', dataset.__len__()*split_ratio[0])
    print('lenthg of test data: ', dataset.__len__()*split_ratio[1])
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # define the hyperparameters
    # num_epochs = int(1000000)
    # input_shape=(250, 96)      # (C, L)
    # latent_channels = 16
    # hidden_dims= [128, 256, 512]
    random_sample_num = 8
    num_epochs = int(10000)
    input_shape=(250, 96)      # (C, L)
    latent_channels = 16
    hidden_dims= [32, 128, 256, 568]
    cond_dims = [32, 128, 256]
        
    # define the model
    # vaemodel = vaemodel.ConvPoolVAE1D(input_shape, latent_channels=latent_channels, hidden_dims=hidden_dims).to(device)
    vaemodel = vaemodel.ConvPoolVAE1D(input_shape=input_shape, condition_shape=(random_sample_num, 96), 
                                      latent_channels=latent_channels,
                                      hidden_dims=hidden_dims, cond_dims=cond_dims).to(device)
    
    # load the pretrained model
    pretrained_model_path = 'exp/exp_second_round/finetune/vae_1337242_8.pth'
    
    # define the optimizer
    optimizer = optim.Adam(vaemodel.parameters(), lr=1e-4)

    # initialize wandb
    wandb.init(project="generative_model_finetune_vae")
    
    # log amount of parameters
    parameters = sum(p.numel() for p in vaemodel.parameters())
    # wandb.log({"amount of parameters": sum(p.numel() for p in vaemodel.parameters())})
    
    # train the model
    mid_loss = 100000000000
    train_sample = dataset.load_test_data(size=1)
    train_sample = train_sample[:, : , :-1]
    train_sample = torch.tensor(train_sample, dtype=torch.float32).to(device)
    
    start_time = time.time()
    for epoch in range(num_epochs):
        # normalize the input data
        _train_min,_ = train_sample.min(axis=1, keepdim=True)
        _train_max,_ = train_sample.max(axis=1, keepdim=True)
        train_sample_scaled = (train_sample - _train_min)/(_train_max-_train_min+1e-15)
        
        _train_sample_part = rs.random_sample(train_sample, 'random', random_sample_num)
         
        # feed into the model
        train_sample_scaled = train_sample_scaled.double()
        _train_sample_part = _train_sample_part.double()
        recon, mu, logvar = vaemodel(train_sample_scaled, _train_sample_part)
        
        # calculate the loss
        loss, recon_loss, kld = loss_function(recon, train_sample_scaled, mu, logvar)
        
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # recover the input data
        recon = recon * (_train_max - _train_min) + _train_min
        mmd = eva.compute_mmd(recon.cpu().detach().numpy()[0], train_sample.cpu().detach().numpy()[0])
        
        delt_time = time.time() - start_time
        print(f"Epoch [{epoch+1}], Loss: {loss.item():.4f}, Recon Loss: {recon_loss.item():.4f}, ,mmd: {mmd:.4f} , Time: {delt_time:.4f}")
        wandb.log({"epoch": epoch, "loss": loss.item(), "recon_loss": recon_loss.item(), "mmd": mmd, "time": delt_time})

        if epoch % 50 == 0:
            # save the model 
            # mid_loss = loss.item()
            # torch.save(vaemodel.state_dict(), f'exp/exp_second_round/finetune/checkpoint/vae_{epoch}.pth')
        
            # plot the generated results
            recon = recon.cpu().detach().numpy()
            recon = recon.reshape(-1, 96)
            
            # plot the original data
            label_fontsize = 15
            tick_fontsize = 15
            fig, ax = plt.subplots(figsize=(6, 6))
            _sum = train_sample[0].sum(axis=1)
            # Setup colormap and normalization
            color_map = plt.cm.coolwarm
            norm = mcolors.Normalize(vmin=_sum.min(), vmax=_sum.max())
            for _i in range(train_sample.shape[1]):
                color = color_map(norm(_sum[_i].cpu().detach().numpy()))
                plt.plot(train_sample[0, _i].cpu().detach().numpy(), c=color, alpha=0.5)
            sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
            sm.set_array([])  # dummy—required by colorbar
            
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label('Daily Consumption [kWh]', fontsize=label_fontsize)
            cbar.ax.tick_params(labelsize=tick_fontsize)
            # plt.title('Original data')
            hours = [f'{h//4:02d}:{(h%4)*15:02d}' for h in range(0, 96, 16)]
            
            # x and y axis start from 0
            ax.set_xlim(0, 96)
            ax.set_ylim(train_sample[0].min().cpu().detach().numpy(), 1.2 * train_sample[0].max().cpu().detach().numpy())
            
            ax.set_xticks(np.arange(0, 96, 16))
            ax.set_xticklabels(hours, fontsize=tick_fontsize)
            # Rotate the x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            plt.xlabel('Hour of Day [-]', fontsize=label_fontsize)
            plt.ylabel('Electricity Consumption [kWh]', fontsize=label_fontsize)
            fig.tight_layout()
            plt.savefig(f'exp/exp_second_round/finetune/plot/original.png')
            plt.close()
            
            # plot the generated data
            fig, ax = plt.subplots(figsize=(6, 6))
            _sum = recon.sum(axis=1)
            # Setup colormap and normalization
            color_map = plt.cm.coolwarm
            norm = mcolors.Normalize(vmin=_sum.min(), vmax=_sum.max())
            for _i in range(recon.shape[0]):
                color = color_map(norm(_sum[_i]))
                plt.plot(recon[ _i], c=color, alpha=0.5)
            sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
            sm.set_array([])
            # dummy—required by colorbar
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label('Daily Consumption [kWh]', fontsize=label_fontsize)
            cbar.ax.tick_params(labelsize=tick_fontsize)
            # plt.title('Generated data')
            hours = [f'{h//4:02d}:{(h%4)*15:02d}' for h in range(0, 96, 16)]
            ax.set_xticks(np.arange(0, 96, 16))
            ax.set_xticklabels(hours, fontsize=tick_fontsize)
            
            # x and y axis start from 0
            ax.set_xlim(0, 96)
            ax.set_ylim(train_sample[0].min().cpu().detach().numpy(), 1.2 * train_sample[0].max().cpu().detach().numpy())
            
            # Rotate the x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            plt.xlabel('Hour of Day [-]', fontsize=label_fontsize)
            plt.ylabel('Electricity Consumption [kWh]', fontsize=label_fontsize)
            fig.tight_layout()
            plt.savefig(f'exp/exp_second_round/finetune/plot/generated_{epoch}.png')
            plt.close()
        
            
        # break