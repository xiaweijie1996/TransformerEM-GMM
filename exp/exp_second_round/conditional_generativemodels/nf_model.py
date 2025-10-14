import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -------------------------
# Small helpers
# -------------------------
def _groupnorm(num_channels, max_groups=8):
    # pick a safe num_groups that divides num_channels
    for g in reversed(range(1, max_groups + 1)):
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    # fallback to LayerNorm-like via GroupNorm with 1 group
    return nn.GroupNorm(1, num_channels)

# -------------------------
# 1x1 Invertible Conv (channel mixing)
# -------------------------
class Invertible1x1Conv1D(nn.Module):
    """
    Glow-style invertible 1x1 convolution for (B, C, L).
    Determinant is det(W)^L.
    """
    def __init__(self, num_channels):
        super().__init__()
        w_init = torch.qr(torch.randn(num_channels, num_channels))[0]  # orthogonal
        self.weight = nn.Parameter(w_init)

    def forward(self, x, logdet=None):
        # x: (B, C, L)
        B, C, L = x.shape
        W = self.weight
        z = F.conv1d(x, W.unsqueeze(-1))  # (C_out=C)
        if logdet is not None:
            # log|det(W)| * L * B (sum over batch later)
            log_abs_det = torch.logdet(W.float()).to(x.dtype)
            logdet = logdet + L * log_abs_det
        return z, logdet

    def inverse(self, z, logdet=None):
        B, C, L = z.shape
        W_inv = torch.inverse(self.weight.float()).to(z.dtype)
        x = F.conv1d(z, W_inv.unsqueeze(-1))
        if logdet is not None:
            log_abs_det = torch.logdet(self.weight.float())
            logdet = logdet - L * log_abs_det
        return x, logdet

# -------------------------
# Affine coupling (channel split)
# -------------------------
class AffineCoupling1D(nn.Module):
    """
    RealNVP-style affine coupling for (B, C, L) with condition (B, C_cond, L).
    Split along channels: x = [xA, xB]
    Predict s,t from [xA, cond] to transform xB.
    """
    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        hidden_channels: int = 128,
        clamp: float = 2.0,   # for stable exp(scale)
    ):
        super().__init__()
        assert in_channels % 2 == 0, "in_channels should be even (for channel split)."
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.half = in_channels // 2
        self.clamp = clamp

        # A small conv net that outputs 2*half channels: [s, t]
        self.net = nn.Sequential(
            nn.Conv1d(self.half + cond_channels, hidden_channels, kernel_size=3, padding=1),
            _groupnorm(hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            _groupnorm(hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(hidden_channels, 2 * self.half, kernel_size=3, padding=1),
        )

    def _st(self, xA, cond):
        # concat on channel dim
        h = torch.cat([xA, cond], dim=1)  # (B, half + C_cond, L)
        st = self.net(h)                  # (B, 2*half, L)
        s, t = st.chunk(2, dim=1)
        # clamp the scale -> stable exp, like Glow/RealNVP
        s = self.clamp * torch.tanh(s / self.clamp)
        return s, t

    def forward(self, x, cond):
        """
        x:    (B, C, L)
        cond: (B, C_cond, L)
        returns: z, logdet (scalars per batch item summed across space/channels)
        """
        xA, xB = torch.split(x, [self.half, self.half], dim=1)   # channel split
        s, t = self._st(xA, cond)

        zB = xB * torch.exp(s) + t
        z = torch.cat([xA, zB], dim=1)

        # logdet = sum over s (per sample, over channels and length)
        logdet = torch.sum(s, dim=[1, 2])  # (B,)
        return z, logdet

    def inverse(self, z, cond):
        zA, zB = torch.split(z, [self.half, self.half], dim=1)
        s, t = self._st(zA, cond)

        xB = (zB - t) * torch.exp(-s)
        x = torch.cat([zA, xB], dim=1)

        logdet = -torch.sum(s, dim=[1, 2])  # inverse changes sign
        return x, logdet

# -------------------------
# Full Flow: stack K blocks (+ optional 1x1 convs)
# -------------------------
class Flow1D(nn.Module):
    """
    A stack of K affine coupling layers with optional invertible 1x1 convs.
    Data:    x in (B, C, L)
    Cond:    c in (B, C_cond, L)
    Base:    standard normal N(0, I) over (C, L)
    """
    def __init__(
        self,
        channels: int,
        cond_channels: int,
        hidden_channels: int = 128,
        K: int = 6,
        use_1x1: bool = True,
        clamp: float = 2.0,
    ):
        super().__init__()
        self.channels = channels
        self.cond_channels = cond_channels
        self.blocks = nn.ModuleList()
        self.mixers = nn.ModuleList() if use_1x1 else None

        for _ in range(K):
            self.blocks.append(
                AffineCoupling1D(
                    in_channels=channels,
                    cond_channels=cond_channels,
                    hidden_channels=hidden_channels,
                    clamp=clamp,
                )
            )
            if use_1x1:
                self.mixers.append(Invertible1x1Conv1D(channels))

    def forward(self, x, cond):
        """
        returns z, logdet (B,)
        """
        logdet = x.new_zeros(x.size(0))
        h = x
        for i, block in enumerate(self.blocks):
            if self.mixers is not None:
                h, logdet = self.mixers[i](h, logdet)
            h, ld = block(h, cond)
            logdet = logdet + ld
        z = h
        return z, logdet

    def inverse(self, z, cond):
        """
        maps z -> x, returns x, logdet (B,)
        """
        logdet = z.new_zeros(z.size(0))
        h = z
        for i in reversed(range(len(self.blocks))):
            h, ld = self.blocks[i].inverse(h, cond)
            logdet = logdet + ld
            if self.mixers is not None:
                h, logdet = self.mixers[i].inverse(h, logdet)
        x = h
        return x, logdet

    # Log-likelihood under standard normal base
    def nll(self, x, cond, reduce=True):
        """
        Negative log-likelihood (in nats).
        """
        z, logdet = self(x, cond)  # logdet: (B,)
        # base log prob: standard normal over every element
        # log p(z) = -0.5 * (z^2 + log(2pi))
        B = x.size(0)
        D = z[0].numel()
        log_pz = -0.5 * (z ** 2 + math.log(2 * math.pi))
        log_pz = log_pz.view(B, -1).sum(dim=1)  # (B,)

        log_px = log_pz + logdet  # (B,)
        nll = -log_px
        return nll.mean() if reduce else nll

# -------------------------
# Minimal test
# -------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, L = 2, 8, 16       # data (B, N, L)
    C_cond = 4               # condition channels (B, N', L)

    x = torch.randn(B, C, L)
    cond = torch.randn(B, C_cond, L)

    flow = Flow1D(channels=C, cond_channels=C_cond, hidden_channels=128, K=4, use_1x1=True)

    # Forward: x -> z
    z, logdet = flow(x, cond)
    # Inverse: z -> x
    x_rec, inv_logdet = flow.inverse(z, cond)

    rec_err = torch.mean((x - x_rec) ** 2).item()
    print("x  :", x.shape, "cond:", cond.shape)
    print("z  :", z.shape, "logdet:", logdet.shape)
    print("x̂  :", x_rec.shape, "rec_err:", rec_err)
