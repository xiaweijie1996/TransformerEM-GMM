import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvPoolVAE1D(nn.Module):
    def __init__(
        self,
        input_shape=(125, 96),       # (C, L)
        condition_shape=(8, 96),     # (C_cond, L_cond)
        latent_channels=16,
        hidden_dims= [16, 128, 324],
        cond_dims = [32, 64, 12]
    ):
        super().__init__()
        C, L = input_shape
        C_cond, L_cond = condition_shape

        # --- Encoder (1D convs) ---
        enc_layers = []
        in_ch = C
        for h in hidden_dims:
            enc_layers += [
                nn.Conv1d(in_ch, h, kernel_size=3, padding=1),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
            ]
            in_ch = h
        self.encoder = nn.Sequential(*enc_layers)

        # project to mu and logvar
        self.conv_mu     = nn.Conv1d(hidden_dims[-1], latent_channels, kernel_size=1)
        self.conv_logvar = nn.Conv1d(hidden_dims[-1], latent_channels, kernel_size=1)

        # --- Map conditioning (1D) ---
        mapcond_layers = []
        in_ch = C_cond
        for h in cond_dims:
            mapcond_layers += [
                nn.Conv1d(in_ch, h, kernel_size=3, padding=1),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
            ]
            in_ch = h
        # final 1x1 to latent size
        mapcond_layers += [nn.Conv1d(cond_dims[-1], latent_channels, kernel_size=1) ]
        self.mapcond = nn.Sequential(*mapcond_layers)

        # --- Decoder (1D upsampling) ---
        dec_layers = []
        hidden_dims_rev = hidden_dims[::-1]
        in_ch = latent_channels * 2  # z + cond
        for h in hidden_dims_rev:
            dec_layers += [
                nn.ConvTranspose1d(in_ch, h, kernel_size=3, padding=1),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
            ]
            in_ch = h
        # final reconstruction to 1 channel
        dec_layers += [
            nn.Conv1d(in_ch, input_shape[0], kernel_size=3, padding=1),
        ]
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x, cond):
        # _input = torch.cat([x, cond], dim=1) if cond is not None else x
        f = self.encoder(x)
        mu     = self.conv_mu(f)
        logvar = self.conv_logvar(f)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, cond):
        z_cond = self.mapcond(cond) 
        _z = torch.cat([z, z_cond], dim=1) 
        recon = self.decoder(_z)
        return recon

    def forward(self, x, cond):
        mu, logvar = self.encode(x, cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond)
        return recon, mu, logvar


if __name__ == "__main__":
    # Example usage
    model = ConvPoolVAE1D(input_shape=(250, 96), condition_shape=(8, 96), latent_channels=16)
    x = torch.randn(4, 250, 96)
    recon, mu, logvar = model(x, x)
    print("recon:", recon.shape)   # should be (4,1,96)
    print("mu   :", mu.shape)      # -> (4,16,12)
    print("logvar:", logvar.shape) # -> (4,16,12)
    print('params:', sum(p.numel() for p in model.parameters()))