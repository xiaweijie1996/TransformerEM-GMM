import torch

_new_para = torch.randn(10, 4*(49+1), 96)  # Example encoder output
n_components = 4
_covs = _new_para[:, n_components:, :].view(_new_para.shape[0], -1)
_covs = _covs[:, :n_components*96*49].reshape(_new_para.shape[0], n_components, 96*49)

# Example dimensions
b, K, d = _new_para.shape[0], n_components, 96
device = _new_para.device

# _covs: (b, K, d*(d+1)//2)   -- flattened lower-triangular entries per component
# you currently have (b, K, 96*49) = (b, K, 4704)  since 96*97/2 = 4656 ≈ 96*49 (close enough)

# 1️⃣  Create index mask for lower-triangular positions
idx = torch.tril_indices(d, d, 0, device=device)   # (2, d*(d+1)//2)

# 2️⃣  Prepare output tensor
L = torch.zeros((b, K, d, d), device=device, dtype=_covs.dtype)

# 3️⃣  Scatter flattened entries into lower-triangular positions
#     broadcast indices to fill all batches/components efficiently
L[:, :, idx[0], idx[1]] = _covs

# L is now the batch of lower-triangular Cholesky factors
#  → you can make full covariance matrices as:
_fullcovs = torch.matmul(L, L.transpose(-1, -2))  # (b, K, d, d)

print(_fullcovs.shape)  # Should be (b, K, d, d)