
# %%
import torch
import math
import matplotlib.pyplot as plt

# all data float64
torch.set_default_dtype(torch.float64)

def gaussian_logpdf(x, means, covs, eps=1e-6):
    """
    Compute log N(x | means, diagonal(covs))
    x:    (b, N, 1, d)
    means:(b, 1, K, d)
    covs: (b, 1, K, d)
    returns: (b, N, K)
    """
    d = x.shape[-1]
    diff = x - means                         # (b, N, K, d)
    inv_cov = 1.0 / (covs + eps)             # (b, 1, K, d)
    exponent = -0.5 * torch.sum(diff**2 * inv_cov, dim=-1)  # (b, N, K)

    # log normalization: -½ d log(2π) - ½ Σ_i log σ_i
    log_det = 0.5 * torch.sum(torch.log(covs.squeeze(1) + eps), dim=-1)  # (b, K)
    log_norm = -0.5 * d * math.log(2 * math.pi) - log_det    # (b, K)

    return exponent + log_norm.unsqueeze(1)  # (b, N, K)

class GMM_PyTorch_Batch:
    def __init__(self, n_components, n_features, init_var=0.5):
        self.n_components = n_components
        self.n_features = n_features
        self._init_parameters(init_var)

    def _init_parameters(self, init_var):
        K, d = self.n_components, self.n_features
        self.weights0 = torch.full((K,), 1.0 / K, dtype=torch.float64)
        self.means0 = torch.randn((K, d), dtype=torch.float64)
        # initialize variances
        self.covariances0 = torch.full((K, d), init_var, dtype=torch.float64)

    def fit(self, X, n_iter=10, eps=1e-6):
        """
        X: tensor of shape (b, N, d)
        Returns: (means, covariances, weights) all shapes with batch dim b
        """
        device = X.device
        X = torch.as_tensor(X, dtype=torch.float64).to(device)
        b, N, d = X.shape
        K = self.n_components

        weights = self.weights0.unsqueeze(0).repeat(b, 1).to(device)      # (b, K)
        means   = self.means0.unsqueeze(0).repeat(b, 1, 1).to(device)      # (b, K, d)
        covs    = self.covariances0.unsqueeze(0).repeat(b, 1, 1).to(device) # (b, K, d)

        for _ in range(n_iter):
            # E-step (log-space)
            x_exp = X.unsqueeze(2)    # (b, N, 1, d)
            m_exp = means.unsqueeze(1)   # (b, 1, K, d)
            c_exp = covs.unsqueeze(1)    # (b, 1, K, d)

            log_pdf   = gaussian_logpdf(x_exp, m_exp, c_exp, eps)          # (b, N, K)
            log_w     = torch.log(weights.unsqueeze(1) + eps)             # (b, 1, K)
            log_joint = log_pdf + log_w                                    # (b, N, K)

            log_norm = torch.logsumexp(log_joint, dim=2, keepdim=True)    # (b, N, 1)
            resp     = torch.exp(log_joint - log_norm)                    # (b, N, K)

            # M-step
            Nk       = resp.sum(dim=1)                                    # (b, K)
            
            means = torch.einsum('bnk,bnd->bkd', resp, X) / (Nk.unsqueeze(-1) + eps)  # (b, K, d)

            diff = X.unsqueeze(2) - means.unsqueeze(1)                    # (b, N, K, d)
            covs = torch.einsum('bnk,bnkd->bkd', resp, diff**2) / (Nk.unsqueeze(-1) + eps)  # (b, K, d)

        self.weights     = weights.clone()
        self.means       = means.clone()
        self.covariances = covs.clone()
        return means, covs

    def sample(self, n_samples, order = 2, eps=1e-6):
        """
        Sample `n_samples` points from the fitted GMM. Only supports batch size b=1.
        Returns a numpy array of shape (n_samples, d).
        """
        assert hasattr(self, 'weights'), 'Model must be fitted before sampling.'
        # assert self.weights.shape[0] == 1, 'Sampling only supports batch size b=1.'

        weights = self.weights[order ]                                      # (K,)
        comp_idx = torch.multinomial(weights, n_samples, replacement=True)  # (n_samples,)
        eps_noise = torch.randn((n_samples, self.n_features), dtype=torch.float64)
        chosen_means = self.means[order , comp_idx]                         # (n_samples, d)
        chosen_stds  = torch.sqrt(self.covariances[order , comp_idx] + eps) # (n_samples, d)
        samples = chosen_means + eps_noise * chosen_stds               # (n_samples, d)
        return samples.numpy()


def demo_gmm_plot():
    """
    Demonstrate fitting and sampling for 2D data (b=1)."""
    b, N, d = 10, 500, 2
    X = torch.randn((b, N, d))
    # Concate another gaussian cluster
    X = torch.cat([X, torch.randn((b, N, d)) + 5], dim=1)  # (b, 2N, d)
    X = torch.cat([X, torch.randn((b, N, d)) - 5], dim=1)  # (b, 3N, d)

    gmm = GMM_PyTorch_Batch(n_components=3, n_features=d)
    gmm.fit(X, n_iter=20)
    samples = gmm.sample(1000)

    # Plot only if data is 2D
    if d == 2:
        plt.figure(figsize=(6, 6))
        plt.scatter(X[0, :, 0].numpy(), X[0, :, 1].numpy(), alpha=0.5, label='Original data')
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, label='GMM samples')
        plt.legend()
        plt.title('GMM Fit & Sampling (b=1, d=2)')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()
    else:
        raise ValueError('demo_gmm_plot only supports 2D (d=2) data.')

if __name__ == "__main__":
    demo_gmm_plot()

