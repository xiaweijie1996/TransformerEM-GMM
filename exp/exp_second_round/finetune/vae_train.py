import os, sys
_parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(_parent_path)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.optim as optim
import torch.nn.functional as F

from asset.dataloader import Dataloader_nolabel
import exp_second_round.finetune.vae_model as vaemodel

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
    batch_size = 30
    split_ratio = (0.8,0.1,0.1)
    data_path =  'exp/data_process_for_data_collection_all/transformer_data_15minutes.pkl'
    dataset = Dataloader_nolabel(data_path,  batch_size=batch_size
                        , split_ratio=split_ratio)
  
    print('lenthg of train data: ', dataset.__len__()*split_ratio[0])
    print('lenthg of test data: ', dataset.__len__()*split_ratio[1])
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # define the hyperparameters
    num_epochs = int(1000000)
    input_shape=(250, 96)      # (C, L)
    latent_channels = 16
    hidden_dims= [128, 256, 512]
        
    # define the model
    vaemodel = vaemodel.ConvPoolVAE1D(input_shape, latent_channels=latent_channels, hidden_dims=hidden_dims).to(device)
    
    # define the optimizer
    optimizer = optim.Adam(vaemodel.parameters(), lr=0.0005)

    # initialize wandb
    wandb.init(project="generative_model_finetune")
    
    # log amount of parameters
    parameters = sum(p.numel() for p in vaemodel.parameters())
    wandb.log({"amount of parameters": sum(p.numel() for p in vaemodel.parameters())})
    
    # train the model
    mid_loss = 100000000000
    
    for epoch in range(num_epochs):
        # load the training data
        train_sample = dataset.load_train_data()
        rain_sample = train_sample[:, : , :-1]
        train_sample = torch.tensor(train_sample, dtype=torch.float32).to(device)
        
        # normalize the input data
        _train_min,_ = train_sample.min(axis=1, keepdim=True)
        _train_max,_ = train_sample.max(axis=1, keepdim=True)
        train_sample = (train_sample - _train_min)/(_train_max-_train_min+1e-15)
        
        # feed into the model
        recon, mu, logvar = vaemodel(train_sample)
        
        # calculate the loss
        loss, recon_loss, kld = loss_function(recon, train_sample, mu, logvar)
        
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Recon Loss: {recon_loss.item():.4f}, KLD: {kld.item():.4f}")
        # print(f"mu: {mu[0].cpu().detach().numpy()}", f"logvar: {logvar[0].cpu().detach().numpy()}")
        
        wandb.log({"loss": loss.item(), "recon_loss": recon_loss.item(), "kld": kld.item()})
        wandb.log({"epoch": epoch})
        # wandb.log({"mu": mu[0].cpu().detach().numpy(), "logvar": logvar[0].cpu().detach().numpy()})
        # save the model
        if loss.item() < mid_loss:
            mid_loss = loss.item()
            torch.save(vaemodel.state_dict(), f'exp/exp_second_round/finetune/vae_{parameters}.pth')
        

        if epoch % 5000 == 0:
            # plot condition, recon, and original
            fig, ax = plt.subplots(2, 1, figsize=(10, 10))
            
            plt.subplot(2, 1, 2)
            plt.title('Reconstructed')
            _sample = recon[0].cpu().detach().numpy()
            plt.plot(_sample.T, alpha=0.5)
            plt.xlabel('time')
            plt.ylabel('value')
            
            plt.subplot(2, 1, 1)
            plt.title('Original')
            _sample = train_sample[0].cpu().detach().numpy()
            plt.plot(_sample.T, alpha=0.5)
            plt.xlabel('time')
            plt.ylabel('value')
    
            plt.tight_layout()
    
            plt.savefig(f'exp/exp_second_round/finetune/vae_gen.png')
            plt.close(fig)
            wandb.log({"epoch": epoch, "condition": wandb.Image(fig)})
                