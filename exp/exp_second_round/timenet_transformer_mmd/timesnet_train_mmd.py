import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import wandb
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def normalize_and_mask(data, index_mask, device, ratio=None):
    # data = data.view(data.size(0), -1, 1)
    
    # normalize the data between 0 and 1 for each sample in the batch
    _min = data.min(dim=1).values
    _max = data.max(dim=1).values
    _min = _min.unsqueeze(1)
    _max = _max.unsqueeze(1)
    data = (data - _min) / (_max - _min + 1e-10)
    
    data = data.to(device)

    # find rows with at least one '1'
    tensor = (index_mask.sum(dim=2) >= 24).float()
    
    # convert to numpy array
    np_array = tensor.numpy()

    # number of ones to keep in each row
    num_ones_to_keep = np.random.randint(8, 98)
    wandb.log({'num_ones_to_keep': num_ones_to_keep})

    # get the indices of ones in the array
    indices = np.argwhere(np_array == 1)

    # create an array to hold the output
    output_array = np.zeros_like(np_array)

    # sort indices by row, then shuffle within rows to randomize which ones are kept
    np.random.shuffle(indices)

    # keep track of how many ones have been placed in each row
    placed_ones = np.zeros(np_array.shape[0], dtype=int)

    # place ones in the output array
    for i in range(indices.shape[0]):
        row, col = indices[i]
        if placed_ones[row] < num_ones_to_keep:
            output_array[row, col] = 1
            placed_ones[row] += 1

    # convert back to tensor
    modified_tensor = torch.tensor(output_array)
    random_mask = modified_tensor.unsqueeze(-1).repeat(1, 1, 96)

    return data, random_mask.to(device), (_min, _max)

# class MMDLoss(nn.Module):
#     def __init__(self, kernel_type='rbf', bandwidth=1.0):
#         super(MMDLoss, self).__init__()
#         self.kernel_type = kernel_type
#         self.bandwidth = bandwidth

#     def forward(self, y, y_hat):
#         return self.compute_mmd(y, y_hat)

#     def compute_mmd(self, x, y):
#         """Computes the MMD loss between two sets of samples."""
#         if self.kernel_type == 'rbf':
#             return self.mmd_rbf(x, y, self.bandwidth)
#         else:
#             raise ValueError("Unsupported kernel type: {}".format(self.kernel_type))

#     def mmd_rbf(self, x, y, bandwidth):
#         """Compute MMD using the RBF kernel."""
#         xx, yy, xy = self._pairwise_distances(x, y)
        
#         k_xx = torch.exp(-xx / (2 * bandwidth**2))
#         k_yy = torch.exp(-yy / (2 * bandwidth**2))
#         k_xy = torch.exp(-xy / (2 * bandwidth**2))
        
#         mmd = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
#         return mmd

#     def _pairwise_distances(self, x, y):
#         """Compute pairwise squared Euclidean distances."""
#         x_flat = x.view(-1, x.size(-1))
#         y_flat = y.view(-1, y.size(-1))
        
#         # Compute the squared distances
#         xx = torch.cdist(x_flat, x_flat, p=2).pow(2)
#         yy = torch.cdist(y_flat, y_flat, p=2).pow(2)
#         xy = torch.cdist(x_flat, y_flat, p=2).pow(2)
        
#         return xx, yy, xy
    
def gaussian_kernel(x: torch.Tensor,
                    y: torch.Tensor,
                    sigma: float = 1.0) -> torch.Tensor:
    """
    Compute the Gaussian (RBF) kernel between all pairs in x and y.
    Args:
      x: Tensor of shape (B, L, F)
      y: Tensor of shape (B, M, F)
      sigma: bandwidth of the kernel
    Returns:
      kernel matrix of shape (B, L, M)
    """
    # x_i shape: (B, L, 1, F), y_j shape: (B, 1, M, F)
    x_i = x.unsqueeze(2)
    y_j = y.unsqueeze(1)
    # pairwise squared Euclidean distances: (B, L, M)
    dist_sq = ((x_i - y_j) ** 2).sum(-1)
    # RBF kernel
    return torch.exp(-dist_sq / (2 * sigma ** 2))


def mmd_loss(x: torch.Tensor,
             y: torch.Tensor,
             sigma: float = 1.0) -> torch.Tensor:
    """
    Compute the unbiased MMD^2 loss between x and y, averaged over the batch.
    Args:
      x: Tensor of shape (B, L, F)
      y: Tensor of shape (B, L, F)
      sigma: bandwidth for the RBF kernel
    Returns:
      scalar MMD loss
    """
    B, L, _ = x.shape
    # Kxx, Kyy: (B, L, L), Kxy: (B, L, L)
    Kxx = gaussian_kernel(x, x, sigma)
    Kyy = gaussian_kernel(y, y, sigma)
    Kxy = gaussian_kernel(x, y, sigma)

    # For unbiased estimate, subtract diagonal terms from Kxx and Kyy
    # (optional; for large L you can also use the simple avg version below)
    mask = ~torch.eye(L, dtype=torch.bool, device=x.device)
    # sum over i != j and normalize by L*(L-1)
    xx = Kxx.masked_select(mask.unsqueeze(0)).view(B, -1).mean(dim=1)
    yy = Kyy.masked_select(mask.unsqueeze(0)).view(B, -1).mean(dim=1)
    # all pairs in Kxy, normalize by L^2
    xy = Kxy.view(B, -1).mean(dim=1)

    # MMD^2 per batch element
    mmd2 = xx + yy - 2 * xy
    # return average over batch
    return mmd2.mean()


def train_and_evaluate(model, data_loader, optimizer, device, epochs, sub_epoch, test_steps, num_params):
    current_loss =10000
    for epoch in range(epochs):
        # for sub_epoch in range(sub_epochs):
            # Training
        model.train()
        full_series, index_mask = data_loader.load_train_data_times()

        train_data, random_mask, scaler = normalize_and_mask(full_series, index_mask, device)
        
        y_hat = model(train_data, None, random_mask.to(device)) # train data will be masked
        # loss = ((y_hat -train_data)**2) * index_mask.to(device)
        # loss =  mmd_loss(train_data, y_hat)
        # loss = loss.mean()
        
        # Compute the mse of the mean\
        _mean_train = train_data.mean(dim=1)
        _mean_y_hat = y_hat.mean(dim=1)
        mse_mean = ((train_data - y_hat)**2).mean(dim=1)
        # Compute the mse of the std
        _std_train = train_data.std(dim=1)
        _std_y_hat = y_hat.std(dim=1)
        mse_std = ((train_data - y_hat)**2).std(dim=1)
        loss = mse_std*50+mse_mean # mmd_loss(train_data, y_hat)
        loss = loss.mean()
        
        wandb.log({'loss_train': loss.item()})
        print(f"Epoch {epoch + 1}, Sub-Epoch {sub_epoch + 1}, Loss: {loss.item()}")
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
        # Testing
        model.eval()
        losses = []
        with torch.no_grad():
            for _test in range(test_steps):
                full_series, index_mask = data_loader.load_test_data_times()
                test_data, random_mask, scaler = normalize_and_mask(full_series, index_mask, device)
                
                y_hat = model(test_data, None, random_mask)
     
                # Comptue the mse of the mean
                _mean_test = test_data.mean(dim=1)
                _mean_y_hat = y_hat.mean(dim=1)
                mse_mean = ((test_data - y_hat)**2).mean(dim=1)
                
                # Compute the mse of the std
                _std_test = test_data.std(dim=1)
                _std_y_hat = y_hat.std(dim=1)
                mse_std = ((test_data - y_hat)**2).std(dim=1)
                
                loss = mse_std  *50 +mse_mean  # mmd_loss(test_data, y_hat)
                loss = loss.mean()
                losses.append(loss.item())
        
        # print(f"Epoch {epoch + 1}, Test Loss: {sum(losses) / len(losses)}")
        
        _loss = sum(losses) / len(losses)
        wandb.log({'loss_test': _loss})
        if _loss < current_loss:
            current_loss = _loss
            torch.save(model.state_dict(), f'exp/exp_second_round/timenet_transformer_mmd/timesnet_MMD_{num_params}.pth')
        
        if epoch % 10 == 0:
            
            _plot = (train_data[0, :, :].cpu()*index_mask[0, :, :]).detach().numpy()
            # scale back to original
            _plot = _plot* (scaler[1][0, 0, :].cpu().detach().numpy() - scaler[0][0, 0, :].cpu().detach().numpy()) + scaler[0][0, 0, :].detach().numpy()
            
            plt.figure(figsize=(20, 5))
            plt.subplot(2, 1, 1)
            plt.plot(_plot[:365,:].T, alpha = 0.2)
            
            _pre  = (y_hat[0, :, :].cpu()*index_mask[0, :, :]).detach().numpy()
            _pre = _pre* (scaler[1][0, 0, :].cpu().detach().numpy() - scaler[0][0, 0, :].cpu().detach().numpy()) + scaler[0][0, 0, :].detach().numpy()
            plt.subplot(2, 1, 2)
            plt.plot(_pre[:365,:].T, alpha = 0.2)
            plt.legend(['x', 'y_hat'])
            plt.savefig(f'exp/exp_second_round/timenet_transformer_mmd/timesnet_MMD_{num_params}.png')
            plt.close()
            wandb.log({'plot': wandb.Image(f'exp/exp_second_round/timenet_transformer_mmd/timesnet_MMD_{num_params}.png')})
            # print('-------------------')