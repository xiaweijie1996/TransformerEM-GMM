import torch
import torch.nn as nn
import torch.nn.functional as F

class FFD_NL(nn.Module):
    def __init__(self, 
                 in_channels, 
                 hidden_channels=128, 
                 condition_channels=4, 
                 out_channels=None,
                 t_max=1000, 
                 betas=None):
        super().__init__()
        assert betas is not None, "Provide a length-t_max betas schedule."
        assert betas.ndim == 1, "betas must be 1-D."
        assert len(betas) == t_max, "len(betas) must equal t_max."

        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.condition_channels = condition_channels
        self.t_max = t_max

        self.net = nn.Sequential(
            nn.Conv1d(in_channels + 1 + self.condition_channels, hidden_channels, kernel_size=3, padding=1),
            # nn.GroupNorm(4, hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            # nn.GroupNorm(4, hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            # nn.GroupNorm(4, hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_channels, out_channels, kernel_size=3, padding=1),
        )

        # Register buffers so they move with .to(device)
        self.register_buffer("betas", betas.clone().float())
        self.register_buffer("alphas", (1.0 - self.betas))
        self.register_buffer("cumprod_alphas", torch.cumprod(self.alphas, dim=0))

    def forward(self, x, t, cond=None):
        # t expected shape: (B,)
        if t.ndim != 1:
            t = t.view(-1)
        B, _, L = x.shape
        # normalize t to [0,1] then broadcast
        t_norm = (t.float() / (self.t_max - 1)).view(B, 1, 1).expand(B, 1, L)
        x_emb = torch.cat([x, t_norm], dim=1)
        if cond is not None:
            # (optional) sanity checks
            # assert cond.size(1) == self.condition_channels and cond.size(2) == L
            x_emb = torch.cat([x_emb, cond], dim=1)
        return self.net(x_emb)

def linear_beta_schedule(T):
    # Return exactly T values; simple monotonic schedule (typical)
    return torch.linspace(1e-4, 0.02, T)

def training(model, x0, t, cum_alpha_sqrt, cum_one_minus_alpha_sqrt, cond=None):
    # t must be (B,)
    if t.ndim != 1:
        t = t.view(-1)
    B = x0.size(0)
    noise = torch.randn_like(x0)

    a_sqrt = cum_alpha_sqrt[t].view(B, 1, 1)
    one_minus_a_sqrt = cum_one_minus_alpha_sqrt[t].view(B, 1, 1)
    x_t = a_sqrt * x0 + one_minus_a_sqrt * noise

    noise_pred = model(x_t, t, cond=cond)
    return F.mse_loss(noise_pred, noise)

@torch.no_grad()
def sampling(model, cond):
    model.eval()
    device = cond.device
    B, _, L = cond.shape
    N = model.in_channels
    T = model.t_max

    x_t = torch.randn(B, N, L, device=device)
    for t in reversed(range(T)):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        noise_pred = model(x_t, t_batch, cond=cond)

        a_t = model.alphas[t]                  # scalar tensor on device (buffer)
        beta_t = model.betas[t]
        cum_a_t = model.cumprod_alphas[t]
        # DDPM update (epsilon prediction)
        coeff = (1.0 / torch.sqrt(a_t))
        mean = coeff * (x_t - (beta_t / torch.sqrt(1.0 - cum_a_t)) * noise_pred)

        if t > 0:
            # simple choice: sigma = sqrt(beta_t)
            noise = torch.randn_like(x_t)
            x_t = mean + torch.sqrt(beta_t) * noise
        else:
            x_t = mean
    return x_t

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    torch.manual_seed(0)

    batch_size = 1
    x0 = torch.randn(batch_size, 10, 24)
    cond = torch.randn(batch_size, 4, 24)
    t_max = 30

    betas = linear_beta_schedule(t_max)
    model = FFD_NL(in_channels=10, condition_channels=4, t_max=t_max, betas=betas, hidden_channels=32)

    cum_alpha_sqrt = model.cumprod_alphas.sqrt()
    cum_one_minus_alpha_sqrt = (1 - model.cumprod_alphas).sqrt()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    save_dir = "exp/exp_second_round/conditional_generativemodels"
    os.makedirs(save_dir, exist_ok=True)

    for step in range(10000):
        model.train()
        t = torch.randint(0, t_max, (batch_size,), dtype=torch.long)  # (B,)
        loss = training(model, x0, t, cum_alpha_sqrt, cum_one_minus_alpha_sqrt, cond=cond)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            model.eval()
            sampled_data = sampling(model, cond)
            x0_ave = x0.mean(dim=0, keepdim=True)
            sampled_data_ave = sampled_data.mean(dim=0, keepdim=True)

            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.title("Original Data (Average over Batch)")
            plt.plot(x0_ave[0].cpu().numpy().T)
            plt.subplot(2, 1, 2)
            plt.title("Sampled Data (Average over Batch)")
            plt.plot(sampled_data_ave[0].cpu().numpy().T)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/ddpm_step_test.png")
            plt.close()

            print(f"[{step}] loss={loss.item():.6f} | L1 avg diff={torch.mean(torch.abs(sampled_data - x0)).item():.6f}")
