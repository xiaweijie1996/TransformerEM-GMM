import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Simple Conv1d denoiser for (B, N, L)
# -------------------------------
class FFD_NL(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, condition_channels=4, t_max=1000, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.condition_channels = condition_channels
        self.t_max = t_max  # expect t ∈ [0, t_max-1]

        self.net = nn.Sequential(
            nn.Conv1d(in_channels + 1 + self.condition_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, hidden_channels), 
            # nn.LayerNorm(hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, hidden_channels), 
            # nn.LayerNorm(hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, hidden_channels), 
            # nn.LayerNorm(hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, t, cond=None):
        B, _, L = x.shape
        assert cond is not None, "Provide cond of shape (B, C_cond, L)"
        if cond.shape[-1] != L:
            # align condition length if needed
            cond = F.interpolate(cond, size=L, mode="linear", align_corners=False)

        # scale t to [0,1]
        t_norm = (t.float().view(B, 1, 1) / max(1, (self.t_max - 1))).clamp(0, 1)
        t_chan = t_norm.expand(B, 1, L)
        x_in = torch.cat([x, t_chan, cond], dim=1)
        return self.net(x_in)


# -------------------------------
# Diffusion schedule & q(x_t|x0)
# -------------------------------

def linear_beta_schedule(T):
    return torch.linspace(1e-4, 2e-2, T)

def prepare_buffers(T, device):
    betas  = linear_beta_schedule(T).to(device)
    alphas = 1.0 - betas
    a_bar  = torch.cumprod(alphas, dim=0)
    a_bar_prev = torch.cat([torch.ones(1, device=device), a_bar[:-1]], dim=0)

    buf = {
        "betas": betas,
        "alphas": alphas,
        "a_bar": a_bar,
        "a_bar_prev": a_bar_prev,
        "sqrt_recip_alpha": torch.sqrt(1.0 / alphas),
        "sqrt_one_minus_a_bar": torch.sqrt(1.0 - a_bar),
        "posterior_variance": betas * (1.0 - a_bar_prev) / (1.0 - a_bar)  # Ho et al.
    }
    return buf

def forward_diffusion_sample(x0, t, sqrt_a_bar, sqrt_one_minus_a_bar):
    noise = torch.randn_like(x0)
    x_t = sqrt_a_bar[t].view(-1,1,1) * x0 + sqrt_one_minus_a_bar[t].view(-1,1,1) * noise
    return x_t, noise

@torch.no_grad()
def p_sample_step(model, x_t, t, cond, buf):
    betas_t  = buf["betas"][t].view(-1,1,1)
    sra_t    = buf["sqrt_recip_alpha"][t].view(-1,1,1)
    sqrt_om  = buf["sqrt_one_minus_a_bar"][t].view(-1,1,1)
    post_var = buf["posterior_variance"][t].clamp_min(1e-20).view(-1,1,1)

    eps = model(x_t, t, cond)  # predict noise
    mean = sra_t * (x_t - betas_t / sqrt_om * eps)

    noise = torch.randn_like(x_t)
    nonzero = (t > 0).float().view(-1,1,1)
    return mean + nonzero * torch.sqrt(post_var) * noise

@torch.no_grad()
def sample_from_noise(model, cond, N, L, T=1000, steps=None, device=None, clip=None):
    device = device or next(model.parameters()).device
    B = cond.size(0)
    x = torch.randn(B, N, L, device=device)
    buf = prepare_buffers(T, device)

    if steps is None or steps >= T:
        t_schedule = torch.arange(T-1, -1, -1, device=device)
    else:
        t_schedule = torch.linspace(T-1, 0, steps, device=device).long()

    model.eval()
    for tt in t_schedule:
        t_vec = torch.full((B,), int(tt), device=device, dtype=torch.long)
        x = p_sample_step(model, x, t_vec, cond, buf)
        if clip is not None:
            x = x.clamp(*clip)
    return x


# -------------------------------
# Minimal training & usage example
# -------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Shapes
    B, N, L = 2, 5, 5
    Cc, Lc  = 1, 5
    T = 30

    # Schedules
    buf = prepare_buffers(T, device)

    # Model
    model = FFD_NL(in_channels=N, hidden_channels=64, condition_channels=Cc, t_max=T).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)

    print(" amount of model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    # Fake data
    x0   = torch.randn(B, N, L, device=device)
    cond = torch.randn(B, Cc, Lc, device=device)

    for it in range(20000):
        # sample a fresh timestep per sample
        t = torch.randint(0, T, (B,), device=device)

        # forward diffusion for those t
        x_t, noise = forward_diffusion_sample(x0, t, buf["a_bar"].sqrt(), (1 - buf["a_bar"]).sqrt())

        # predict eps
        eps_pred = model(x_t, t, cond=cond)

        loss = F.mse_loss(eps_pred, noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if it % 100 == 0:
            print(f"step {it} | loss {loss.item():.4f}")
            
             # sampling demo (few steps for speed)
            samples = sample_from_noise(model, cond, N, L, T=T, steps=50, device=device, clip=(-3,3))
            # print("samples:", samples.shape)

            # plot reconstruction
            plt.figure(figsize=(12,6))
            i = 0
            plt.subplot(1,2,i*2+1)
            plt.imshow(x0[i].cpu(), aspect="auto", origin="lower")
            plt.title("x0 (orig)")
            plt.subplot(1,2,i*2+2)
            plt.imshow(samples[i].cpu().detach(), aspect="auto", origin="lower")
            plt.title("x0 (synth)")
            plt.tight_layout()
            plt.savefig("exp/exp_second_round/conditional_generativemodels/ddpm_step_test.png")
                    
        