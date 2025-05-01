import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self, input_shape=48, latent_dim=10):
        super(VAE, self).__init__()

        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.out_scale1 = 96 # control the model complexity
        
        self.encoder = nn.Sequential(
                # First layer
                nn.Linear(self.input_shape,self.out_scale1*10),
                nn.BatchNorm1d(self.out_scale1*10),
                nn.LeakyReLU(),
                
                # second layer
                nn.Linear(self.out_scale1*10,self.out_scale1*5),
                nn.BatchNorm1d(self.out_scale1*5),
                nn.LeakyReLU(),
                )
        
        self.mu = nn.Linear(self.out_scale1*5, self.latent_dim) 
        self.logvar = nn.Linear(self.out_scale1*5, self.latent_dim)

        self.decoder = nn.Sequential(
                # First layer
                nn.Linear(self.latent_dim,self.out_scale1*5),
                nn.BatchNorm1d(self.out_scale1*5),
                nn.LeakyReLU(),
                
                # second layer
                nn.Linear(self.out_scale1*5,self.out_scale1*10),
                nn.BatchNorm1d(self.out_scale1*10),
                nn.LeakyReLU(),
                
                # third layer
                nn.Linear(self.out_scale1*10,self.input_shape),
                nn.BatchNorm1d(self.input_shape),
                nn.Tanh()
            )
            
            
    def encode(self, x):
        out = self.encoder(x)
        return self.mu(out), self.logvar(out)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
