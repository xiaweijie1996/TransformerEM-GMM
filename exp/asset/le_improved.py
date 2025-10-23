import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import math
import torch.nn.functional as F

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
  
 
def le_loss_flexibleweights(
    X: torch.Tensor,               # (b, N, d)
    n_components: int,             # K
    _para: torch.Tensor,           # (b, K*d*2) -> [means, raw_vars]
    weights: torch.Tensor,         # (b, K)     (unnormalized ok)
    eps: float = 1e-6
) -> torch.Tensor:
    b, N, d = X.shape
    K = n_components
    device = X.device

    # Unpack parameters
    means, raw_vars = _para.split(K * d, dim=1)  # both (b, K*d)
    means    = means.view(b, K, d)               # (b, K, d)
    raw_vars = raw_vars.view(b, K, d)            # (b, K, d)

    # Ensure positive, not-too-small variances (σ^2)
    # vars = torch.exp(raw_vars).clamp_min(eps)    # (b, K, d)
    vars = torch.clamp(raw_vars, min=eps)        # (b, K, d)
    # Normalize weights to a simplex and take log
    w = torch.linspace(1/K, 1.0, K, device=device)
    log_w = torch.log(w / w.sum()).unsqueeze(0).unsqueeze(1)  # (1,1,K) → broadcast


    # Compute Gaussian log-pdf terms for all b, N, K
    # diff: (b, N, K, d)
    diff = X.unsqueeze(2) - means.unsqueeze(1)
    inv_vars = 1.0 / vars.unsqueeze(1)                 # (b, 1, K, d)
    mahal = torch.sum(diff * diff * inv_vars, dim=-1)  # (b, N, K)

    # log normalizer: -½ [ d log(2π) + log |Σ| ] with diag Σ
    # log|Σ| = sum_i log(σ_i^2)
    const = -0.5 * d * math.log(2.0 * math.pi)
    log_det_half = 0.5 * torch.sum(torch.log(vars), dim=-1)  # (b, K)
    log_norm = (const - log_det_half).unsqueeze(1)            # (b, 1, K)

    # Component log-likelihoods and mixture
    log_comp = log_norm - 0.5 * mahal + log_w                 # (b, N, K)
    ll = torch.logsumexp(log_comp, dim=2)                     # (b, N)

    # Mean negative log-likelihood over batch & samples
    return -ll.mean()
  
def vec_to_martrix(vec: torch.Tensor) -> torch.Tensor:
  """_summary_

  Args:
      vec (B, N, d)
      
  Returns:
      mat (B, N, d, d) where
  """
  return vec.unsqueeze(-1) * vec.unsqueeze(-2)


def le_loss_rank_iso(
    X: torch.Tensor,               # (b, N, d)
    n_components: int,             # K
    _para: torch.Tensor,           # (b, K*d + K*d*r) -> [means, U_flat, lambda_raw]
    r: int,                        # rank
    scaler: float = 0.1,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Negative log-likelihood for a GMM with rank-r + isotropic covariances:
        Σ_k = U_k U_k^T + λ_k I

    Packing of _para (per batch item):
      - means     : (K*d)
      - U_flat    : (K*d*r)   -> reshape to (K, d, r)
      - lambda_raw: (K)       -> softplus -> λ>0

    Shapes:
      X:        (b, N, d)
      means:    (b, K, d)
      U:        (b, K, d, r)
      lambda:   (b, K)
    """
    b, N, d = X.shape
    K = n_components
    device, dtype = X.device, X.dtype

    # ---- Unpack parameters ----------------------------------------------------
    sz_means = K * d
    sz_U     = K * d * r
    expected = sz_means + sz_U 
    assert _para.shape[1] == expected, f"_para has { _para.shape[1] }, expected { expected }"

    means     = _para[:, :sz_means].view(b, K, d)                  # (b, K, d)
    U_flat    = _para[:, sz_means:sz_means+sz_U].view(b, K, d, r)  # (b, K, d, r)
    lambda_rw = torch.ones(b, K).to(X.device)   #_para[:, sz_means+sz_U:]                           # (b, K)
    lam       = lambda_rw *scaler                  # (b, K), strictly > 0

    # ---- Small r×r system: M = I + (1/λ) U^T U -------------------------------
    # Compute per (b,K): S = U^T U  (d×r -> r×r)
    # S: (b, K, r, r)
    S = torch.matmul(U_flat.transpose(-2, -1), U_flat)

    # Build I_r and broadcast lam
    Ir = torch.eye(r, device=device, dtype=dtype).view(1, 1, r, r)
    lam_inv = (1.0 / lam).unsqueeze(-1).unsqueeze(-1)              # (b, K, 1, 1)
    M = Ir + lam_inv * S                                           # (b, K, r, r)

    # Cholesky for stability (PD)
    Lm = torch.linalg.cholesky(M)                                  # (b, K, r, r)

    # ---- log|Σ| = d*log λ + log|I + (1/λ) U^T U| -----------------------------
    logdet_Sigma = d * torch.log(lam.clamp_min(eps))               # (b, K)
    logdet_Sigma = logdet_Sigma + 2.0 * torch.sum(
        torch.log(torch.diagonal(Lm, dim1=-2, dim2=-1).clamp_min(eps)), dim=-1
    )                                                              # (b, K)

    # ---- Mahalanobis via Woodbury --------------------------------------------
    # diff = x - μ
    diff = X.unsqueeze(2) - means.unsqueeze(1)                     # (b, N, K, d)

    # a = (1/λ) * diff
    a = diff / lam.unsqueeze(1).unsqueeze(-1)                      # (b, N, K, d)

    # bvec = U^T a  -> (b, N, K, r)
    bvec = torch.einsum('bnkd,bkdr->bnkr', a, U_flat)              # (b, N, K, r)

    # Solve y = M^{-1} bvec using Cholesky of M: (b, K, r, r)
    # reshape to (b, K, r, N) for triangular solves
    bT = bvec.permute(0, 2, 3, 1)                                  # (b, K, r, N)
    # Solve Lm * z = bT  (Lm is lower-triangular)
    y = torch.linalg.solve_triangular(Lm, bT, upper=False)

    # Solve Lm^T * y = z  (Lm^T is upper-triangular)
    y = torch.linalg.solve_triangular(Lm.transpose(-2, -1), y, upper=True)

    y = y.permute(0, 3, 1, 2)                                      # (b, N, K, r)

    # Quadratic form:
    # r^T Σ^{-1} r = r^T (1/λ I) r - bvec^T y
    rDinvr    = torch.sum(diff * a, dim=-1)                        # (b, N, K)
    correction = torch.sum(bvec * y, dim=-1)                       # (b, N, K)
    mahal = rDinvr - correction                                    # (b, N, K)

    # ---- Mixture weights ------------------------------------------------------
    w = torch.linspace(1/K, 1.0, K, device=device)
    log_w = torch.log(w / w.sum()).unsqueeze(0).unsqueeze(1)  # (1,1,K) → broadcast

    # ---- Log-likelihood and reduction ----------------------------------------
    const = -0.5 * d * math.log(2.0 * math.pi)
    log_norm = const - 0.5 * logdet_Sigma.unsqueeze(1)             # (b, 1, K)
    log_comp = log_norm - 0.5 * mahal + log_w                      # (b, N, K)
    ll = torch.logsumexp(log_comp, dim=2)                          # (b, N)

    return -ll.mean()
  
  
if __name__ == "__main__":
    # simple test
   b, N, d = 2, 4, 3
   K = 5
   r = 2
   X = torch.randn(b, N, d)
   means = torch.randn(b, K, d)
   U = torch.randn(b, K, d, r)
   _para = torch.cat([means.view(b, K*d), U.view(b, K*d*r)], dim=1)
   loss = le_loss_rank_iso(X, K, _para, r)
   print('loss: ', loss)