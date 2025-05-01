import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import os
import datetime 


def loss_function_vae(recon_x, x, mu, logvar):
    """Define the loss function of vae"""
    mse = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kld, mse, kld


def train_models(
    input_dim = 48,
    hidden_dim = 10,
    path_data =  r'/home/wxia/researchdrive/paper1/cleaned_data/uk_data_agg.csv',
    saving_path =  r'/home/wxia/researchdrive/paper1/Models/models_agg/vae/uk_agg',
    lr = 0.0002,
    batch_size = 64,
    epochs = 400,
    save_control = False,
    model_typle = '30m',
    ):
        
    """
    This the the training function of the model
    z_dim: The dimension of noise
    model_typle: model typles related to resoultion of the data we have 15 minutes (15m), 30m, and 60m
    lr: learning rate
    epochs: epochs
    k: training ratio (train discriminator once, then train generator 5 times)
    path_data: path of input data
    saving_path: path to save the data and model
    save_control: save the model or not
    """
        
    # load the data & data loader
    data = pd.read_csv(path_data)
    data_loader, scaler = dataloader(data, batch_size=batch_size, shuffle=True)

    # define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define the model
    if model_typle == '30m':
        vae = mds.VAE_30m(input_shape=input_dim, latent_dim=hidden_dim).to(device)
    elif model_typle == '60m':
        vae = mds.VAE_60m(input_shape=input_dim, latent_dim=hidden_dim).to(device)
    vae = nn.DataParallel(vae)

    # define the optimizer
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    # store the loss
    losses = []
    msees = []
    klds = []

    # train the model
    print('Start training the model!')
    for epoch in range(epochs):
        print(f'Epoch [{epoch+1}/{epochs}]')
        for i, data in enumerate(data_loader):
            # renconstruct the data
            recon_data, mu, logvar = vae(data.to(device))
            
            # compute the loss
            loss, mse_l, kld_l = loss_function_vae(recon_data, data.to(device), mu, logvar) 
            
            #optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            msees.append(mse_l.item())
            klds.append(kld_l.item())
            
            # compute the js divergence
            p = data.cpu().detach().numpy()
            q = recon_data.cpu().detach().numpy()
            
            if save_control:
                print('saving the models')
                time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                path = saving_path+f'/models'
                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(vae.module.state_dict(), path+f'/{model_typle}__{epoch}_{time}.pt')
                  
    return scaler
    