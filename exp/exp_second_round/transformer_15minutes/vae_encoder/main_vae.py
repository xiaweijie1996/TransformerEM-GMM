import os, sys
_parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(_parent_path)

import wandb
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

import exp_second_round.transformer_15minutes.vae_encoder.tools_vae as tools_vae
import exp_second_round.transformer_15minutes.vae_encoder.vae_model as mds
from asset.dataloader import Dataloader_nolabel

torch.set_default_dtype(torch.float64)

# load data
batch_size =  32
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

# Define the hyperparameters of VAE
input_dim = 96
hidden_dim = 24
vae_model = mds.VAE(input_shape=input_dim, latent_dim=hidden_dim).to(device)

save_model = 'exp/exp_second_round/transformer_15minutes/vae_encoder/'
save_image = 'exp/exp_second_round/transformer_15minutes/vae_encoder/'
lr = 0.001
epochs = 1000
sub_epoch = int(dataset.__len__()*split_ratio[0]/batch_size)

# define the optimizer and loss function and cyclic learning rate scheduler
optimizer = torch.optim.Adam(vae_model.parameters(), lr=lr)
_model_cycle = sum(p.numel() for p in vae_model.parameters() if p.requires_grad)
print('number of parameters: ', _model_cycle)

loss_function = tools_vae.loss_function_vae
wandb.init(project="vae_transformer_15minutes")
wandb.config = {
    "learning_rate": lr,
    "epochs": 1000,
    "batch_size": batch_size,
    "input_dim": input_dim,
    "hidden_dim": hidden_dim,
    "model_type": '15m',
}

loss_mid = 1000000000000
for epoch in range(epochs):
    for j in range(sub_epoch):
        vae_model.train()
        train_sample = dataset.load_train_data()
        train_sample = torch.tensor(train_sample, dtype=torch.float64).to(device)
    
        # normalize the input data
        _train_min,_ = train_sample[:,:, :-1].min(axis=1, keepdim=True)
        _train_max,_ = train_sample[:,:, :-1].max(axis=1, keepdim=True)
        train_sample = (train_sample[:,:, :-1] - _train_min)/(_train_max-_train_min+1e-15)
        reshape_train_sample = train_sample.view(-1, 96)
        
        recon_data, mu, logvar = vae_model(reshape_train_sample)
        loss, mse_l, kld_l = loss_function(recon_data, reshape_train_sample, mu, logvar)
        wandb.log({"loss": loss.item(), "mse_loss": mse_l.item(), "kld_loss": kld_l.item()})
        
        #optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # check the loss in test set
    test_sample = dataset.load_test_data()
    test_sample = torch.tensor(test_sample, dtype=torch.float64).to(device)
    _test_min,_ = test_sample[:,:, :-1].min(axis=1, keepdim=True)
    _test_max,_ = test_sample[:,:, :-1].max(axis=1, keepdim=True)
    test_sample = (test_sample[:,:, :-1] - _test_min)/(_test_max-_test_min+1e-15)
    reshape_test_sample = test_sample.view(-1, 96)
    recon_data, mu, logvar = vae_model(reshape_test_sample)
    loss_test, mse_l, kld_l = loss_function(recon_data, reshape_test_sample, mu, logvar)
    wandb.log({"loss_test": loss_test.item(), "mse_loss_test": mse_l.item(), "kld_loss_test": kld_l.item()})

    if loss_test.item() < loss_mid:
        loss_mid = loss_test.item()
        torch.save(vae_model.state_dict(), os.path.join(save_model, 'vae_model.pth'))

        # Plot the recon data
        _plot_sample = test_sample[0]* (_test_max[0]-_test_min[0]+1e-15) + _test_min[0]
        _plot_recon = recon_data[: 250] * (_test_max[0]-_test_min[0]+1e-15) + _test_min[0]
        _plot_sample = _plot_sample.cpu().detach().numpy()
        _plot_recon = _plot_recon.cpu().detach().numpy()
        _plot_sample = _plot_sample.reshape(-1, 96)
        _plot_recon = _plot_recon.reshape(-1, 96)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(_plot_sample.T, label='Original', alpha=0.5, c='blue')
        plt.xlabel('Time')
        plt.ylabel('Value')
        
        plt.subplot(1, 2, 2)
        plt.plot(_plot_recon.T, label='Reconstructed', alpha=0.5, c='blue')
        plt.xlabel('Time')
        plt.ylabel('Value')
        
        plt.savefig(os.path.join(save_image, f'recon_epoch.png'))
        plt.close()
        
        wandb.log({"recon_epoch": wandb.Image(os.path.join(save_image, f'recon_epoch.png'))})