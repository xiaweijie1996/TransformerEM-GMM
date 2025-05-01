import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import math

def le_loss(X: torch.Tensor,
            n_components: int,
            _para: torch.Tensor,
            eps: float = 1e-6) -> torch.Tensor:
    """
    Compute the average negative log-likelihood of a diagonal-covariance GMM.

    Args:
      X           (b, N, d)        data
      n_components                number of mixture components K
      _para       (b, K*d*2)       concatenated [means, raw_vars]
      eps                          numerical stability

    Returns:
      scalar loss = -mean_{batch, samples} log p(x)
    """
    b, N, d = X.shape
    K = n_components
    device = X.device

    # Unpack parameters
    # means:       (b, K, d)
    # raw_vars:    (b, K, d)  (we'll turn this into positive variances)
    means, raw_vars = _para.split(K * d, dim=1)
    means   = means.view(b, K, d)
    raw_vars = raw_vars.view(b, K, d)

    # Ensure variances > 0
    # Option A: clamp (you can also do `vars = torch.exp(raw_vars)`)
    vars = torch.clamp(raw_vars, min=eps)

    # Mixture weights (you can also learn these)
    # here we just use uniform / linear weights as in your original
    w = torch.linspace(1/K, 1.0, K, device=device)
    log_w = torch.log(w / w.sum()).unsqueeze(0).unsqueeze(1)  # (1,1,K) → broadcast

    # Compute Gaussian log-pdf for all b, N, K in one go
    # diff: (b, N, K, d)
    diff = X.unsqueeze(2) - means.unsqueeze(1)
    inv_vars = 1.0 / vars.unsqueeze(1)  # (b,1,K,d)
    mahal = torch.sum(diff * diff * inv_vars, dim=-1)  # (b, N, K)

    # log normalizer: -½[d log(2π) + ∑_i log σ_i]
    log_det = 0.5 * torch.sum(torch.log(vars + eps), dim=-1)  # (b, K)
    const  = -0.5 * d * math.log(2 * math.pi)
    log_norm = const - log_det                                # (b, K)
    log_norm = log_norm.unsqueeze(1)                          # → (b,1,K)

    # component log-likelihoods: (b, N, K)
    log_comp = log_norm - 0.5 * mahal + log_w

    # log-sum-exp over components → (b, N)
    ll = torch.logsumexp(log_comp, dim=2)

    # return mean negative log-likelihood
    return -ll.mean()