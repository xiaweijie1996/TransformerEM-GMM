import torch
import torch.nn as nn
import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def le_loss(X, n_components, _para):
    device = X.device

    # Ensure X and _para are on the correct device
    X = X.to(device)

    b, N, d = X.shape
    K = n_components

    # Mixture weights
    weights = torch.linspace(1/n_components, 1, n_components, device=device)
    # weights = torch.exp(weights) / torch.exp(weights).sum()
    weights = weights.unsqueeze(0).repeat(b, 1).to(device)  # Use log weights for numerical stability

    # Reshape _para to get means and variances for each batch
    _means = _para[:, :n_components * d].view(b, n_components, d)
    _covariances = _para[:, n_components * d:].view(b, n_components, d) # Ensure variances are positive

    log_likelihoods = []

    for component in range(n_components):
        diff = X - _means[:, component, :].unsqueeze(1)
        _inverse = (1 / _covariances[:, component, :]).unsqueeze(1)
        diff_div = diff * _inverse
        exp_vector = torch.sum(diff_div * diff, dim=2)
        
        _sigma = torch.prod(_covariances[:, component, :], dim=1).unsqueeze(1)
        _sigma = 1 / torch.sqrt(_sigma)
        
        log_likelihood_component = -0.5 * exp_vector + torch.log(_sigma) + torch.log(weights[:, component].unsqueeze(1))
        log_likelihoods.append(log_likelihood_component)

    # Stack all the log-likelihoods 
    log_likelihoods = torch.stack(log_likelihoods, dim=2)

    # Use log-sum-exp trick
    max_log_likelihoods, _ = torch.max(log_likelihoods, dim=2, keepdim=True)
    stable_log_likelihoods = log_likelihoods - max_log_likelihoods
    log_likelihoods = max_log_likelihoods.squeeze(2) + torch.log(torch.sum(torch.exp(stable_log_likelihoods), dim=2))

    return -log_likelihoods.mean()

