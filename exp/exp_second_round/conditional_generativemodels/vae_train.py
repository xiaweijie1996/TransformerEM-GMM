import os, sys
_parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(_parent_path)

import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.optim as optim
import torch.nn.functional as F

from asset.dataloader import Dataloader_nolabel
import asset.random_sampler as rs
import exp_second_round.conditional_generativemodels.vae_model as vaemodel

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
    return recon_loss * 5 + kld, recon_loss, kld


if __name__ == "__main__":
    batch_size =  32
    split_ratio = (0.8,0.1,0.1)
    data_path =  'exp/data_process_for_data_collection_all/new_data_15minute_grid_nomerge.pkl'
    dataset = Dataloader_nolabel(data_path,  batch_size=batch_size
                        , split_ratio=split_ratio)
    dataset.images = dataset.images + np.abs(np.random.normal(0, 0.01, dataset.images.shape)) 
    print('lenthg of train data: ', dataset.__len__()*split_ratio[0])
    print('lenthg of test data: ', dataset.__len__()*split_ratio[1])
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # define the hyperparameters
    random_sample_num = 32
    num_epochs = int(10000)
    input_shape=(250, 96)      # (C, L)
    latent_channels = 16
    hidden_dims= [32, 128, 256, 568]
    cond_dims = [32, 128, 256]
    
    # define the model
    vaemodel = vaemodel.ConvPoolVAE1D(input_shape=input_shape, condition_shape=(random_sample_num, 96), 
                                      latent_channels=latent_channels,
                                      hidden_dims=hidden_dims, cond_dims=cond_dims).to(device)
    
    # define the optimizer
    optimizer = optim.Adam(vaemodel.parameters(), lr=0.0001)

    # initialize wandb
    wandb.init(project="conditional_generative_models_{}shot".format(random_sample_num))
    
    # log amount of parameters
    parameters = sum(p.numel() for p in vaemodel.parameters())
    print('amount of parameters: ', parameters)
    wandb.log({"amount of parameters": sum(p.numel() for p in vaemodel.parameters())})
    
    # train the model
    mid_loss = 100000000000
    for epoch in range(num_epochs):
        # load the training data
        train_sample = dataset.load_train_data()
        train_sample = torch.tensor(train_sample, dtype=torch.float32).to(device)
        train_sample = train_sample[:, :, :-1]
        
        # normalize the input data
        _train_min,_ = train_sample.min(axis=1, keepdim=True)
        _train_max,_ = train_sample.max(axis=1, keepdim=True)
        train_sample = (train_sample - _train_min)/(_train_max-_train_min+1e-15)
        
        # random_sample a number between min_random_sample_num and random_sample_num
        _train_sample_part = rs.random_sample(train_sample, 'random', random_sample_num)
        
        # move the data to the device
        _train_sample_part = _train_sample_part.to(device)
        
        # feed into the model
        recon, mu, logvar = vaemodel(train_sample, _train_sample_part)
        
        # calculate the loss
        loss, recon_loss, kld = loss_function(recon, train_sample, mu, logvar)
        
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Recon Loss: {recon_loss.item():.4f}, KLD: {kld.item():.4f}")
           
        wandb.log({"loss": loss.item(), "recon_loss": recon_loss.item(), "kld": kld.item()})
    
        # save the model
        if loss.item() < mid_loss:
            mid_loss = loss.item()
            torch.save(vaemodel.state_dict(), f'exp/exp_second_round/conditional_generativemodels/{random_sample_num}shot/vae_{parameters}_{random_sample_num}.pth')
        

        if epoch % 1000 == 0:
            # plot condition, recon, and original
            fig, ax = plt.subplots(3, 1, figsize=(15, 30))
            
            plt.subplot(3, 1, 1)    
            plt.title('Condition')
            _sample = _train_sample_part[0].cpu().detach().numpy()
            plt.plot(_sample .T, alpha=0.5)
            plt.xlabel('time')
            plt.ylabel('value')
            
            
            plt.subplot(3, 1, 2)
            plt.title('Reconstructed')
            _sample = recon[0].cpu().detach().numpy()
            plt.plot(_sample.T, alpha=0.5)
            plt.xlabel('time')
            plt.ylabel('value')
            
            plt.subplot(3, 1, 3)
            plt.title('Original')
            _sample = train_sample[0].cpu().detach().numpy()
            plt.plot(_sample.T, alpha=0.5)
            plt.xlabel('time')
            plt.ylabel('value')
    
            plt.tight_layout()
    
            plt.savefig(f'exp/exp_second_round/conditional_generativemodels/{random_sample_num}shot/vae_epoch_{random_sample_num}.png')
            plt.close(fig)
            wandb.log({"epoch": epoch, "condition": wandb.Image(fig)})
                