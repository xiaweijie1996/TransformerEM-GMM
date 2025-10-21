import os
import sys
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
print(_parent_dir)
sys.path.append(_parent_dir)

import torch
import torch.nn as nn

class Conv1DBlock(torch.nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv1DBlock, self).__init__()
        self.model = nn.Sequential(
            torch.nn.Conv1d(in_channels, hid_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(hid_channels),
            torch.nn.ReLU(),
            
            torch.nn.Conv1d(hid_channels, hid_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(hid_channels),
            torch.nn.ReLU(),
            
            torch.nn.Conv1d(hid_channels, out_channels, kernel_size, stride, padding)
        )

        
    def forward(self, x):
        return self.model(x)
    
    
class CNiceModelBasic(torch.nn.Module):
    def __init__(self, 
                 input_c: int = 2,
                 hidden_c: int = 64,
                 condition_c: int = 128,
                 split_ratio: float = 0.5,
                 scaler_dim: int = 96
                 ):
        
        super(CNiceModelBasic, self).__init__()
        self.input_c = input_c
        self.hidden_c = hidden_c
        self.condition_c = condition_c
        self.split_ratio = split_ratio
        
        # Define the layers using BasicFFN
        self.fc1 = Conv1DBlock(
            in_channels=self.input_c + self.condition_c,
            hid_channels=self.hidden_c,
            out_channels=self.input_c
        )
        
        self.fc2 = Conv1DBlock(
            in_channels=self.input_c + self.condition_c,
            hid_channels=self.hidden_c,
            out_channels=self.input_c
        )
        
        # add scaler parameters if needed here with shape (1, 1, dim)
        scaler = nn.Parameter(torch.ones(1, 1, scaler_dim))
        self.register_parameter('scaler', scaler)

    def forward_direction(self, x, c):
       
        # Split the input tensor
        half_dim = x.size(-1) // 2
        x11, x12 = x[:, :, :half_dim], x[:, :, half_dim:]
        
        # x2
        x21 = x11
        # print('x21 shape: ', x21.shape, ' c shape: ', c.shape, 'x', x.shape)
        x22 = x12 + self.fc1(torch.cat([x11, c], dim=1))
        # print('x21 shape: ', x21.shape, ' c shape: ', c.shape, 'x', x.shape)
        
        # x3
        x32 = x22
        x31 = x21 + self.fc2(torch.cat([x22, c], dim=1))
        
        # Combine the outputs
        x3 = torch.cat([x31, x32], dim=-1)
        
        # scaler operation
        x3 = x3 * torch.exp(self.scaler)
        
        log_det_jacobian = torch.sum(self.scaler) * x.size(0)
        return x3, log_det_jacobian
    
    def inverse_direction(self, x3, c):
        half_dim = x3.size(-1) // 2
        
        
        # inverse scaler operation
        x3 = x3 * torch.exp(-self.scaler)
        log_det_jacobian = -torch.sum(self.scaler) * x3.size(0)
        
        # Split the input tensor
        x31, x32 = x3[:, :, :half_dim ], x3[:, :, half_dim :]
        
        # x2
        x22 = x32
        x21 = x31 - self.fc2(torch.cat([x32, c], dim=1))
        
        # x1
        x11 = x21
        x12 = x22 - self.fc1(torch.cat([x21, c], dim=1))
        
        # Combine the outputs
        x1 = torch.cat([x11, x12], dim=-1)

        return x1, log_det_jacobian
    
class CNicemModel(torch.nn.Module):
    def __init__(self, 
                    input_c: int = 2,
                    hidden_c: int = 64,
                    condition_c: int = 128,
                    n_layers: int = 1,
                    split_ratio: float = 0.5,
                    scaler_dim: int = 96
                    ):
            
            super(CNicemModel, self).__init__()
            self.input_c = input_c
            self.hidden_c = hidden_c
            self.condition_c = condition_c
            self.n_layers = n_layers
            self.split_ratio = split_ratio
            
            self.model = nn.ModuleList()
            for _ in range(self.n_layers):
                self.model.append(
                    CNiceModelBasic(
                        input_c=self.input_c,
                        hidden_c=self.hidden_c,
                        condition_c=self.condition_c,
                        split_ratio=self.split_ratio,
                        scaler_dim=scaler_dim
                    )
                )
            
    def forward(self, x, c):
        log_det_jacobian = 0
        for layer in self.model:
            x, det = layer.forward_direction(x, c)
            log_det_jacobian += det
        return x, log_det_jacobian

    def inverse(self, x, c):
        log_det_jacobian = 0
        for layer in reversed(self.model):
            x, det = layer.inverse_direction(x, c)
            log_det_jacobian += det
        return x, log_det_jacobian
    

if __name__ == "__main__":
    B, N, L = 4, 1, 20
    C_N = 5
    x = torch.randn(B, N, L)
    c = torch.randn(B, C_N, L//2)
    model = CNicemModel(input_c=N, condition_c=C_N, n_layers=4, scaler_dim=L)
    y, log_det = model.forward(x, c)
    x_recon, log_det_inv = model.inverse(y, c)
    print("Input shape:", x.shape)
    print("Condition shape:", c.shape)
    print("Output shape:", y.shape)
    
    error = torch.abs(x - x_recon).mean()
    print("Reconstruction error (should be close to 0):", error.item())
    
    print(torch.allclose(x, x_recon, atol=1e-5))