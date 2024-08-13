import torch
from torch.distributions.multivariate_normal import MultivariateNormal

# all data float64
torch.set_default_dtype(torch.float64)

class GMM_Simplified_PyTorch:
    """
    A simplified version of GMM (fixed W) implemented in PyTorch.
    This GMM function can not solve GMM with batch data.
    
    args:
    n_components: int, number of components
    n_features: int, number of features
    
    """
    def __init__(self, n_components, n_features):
        self.n_components = n_components
        self.n_features = n_features
        self.load_parameters()
        
    def load_parameters(self): 
        self.weights = torch.linspace(1/self.n_components, 1, self.n_components)
        self.weights = self.weights / self.weights.sum()
        self.means = torch.ones((self.n_components, self.n_features), requires_grad=False)
        self.covariances = torch.ones((self.n_components, self.n_features), requires_grad=False) * 0.5  # Initialize variances to a small positive value
        
    def fit(self, X, n_iter=100):
        N, d = X.shape 
        K = self.n_components
        threshold = 1e-20
        
        for _ in range(n_iter):
            # E-step: Compute responsibilities
            responsibilities = torch.zeros((N, K))
            for k in range(K):
                diff = X - self.means[k]
                cov_matrix = torch.diag(self.covariances[k])
                
                # check if exists value less than threshold
                nan_indices = torch.isnan(cov_matrix)
                cov_matrix[nan_indices] = 0
                small_values_indices = cov_matrix < threshold
                random_block = torch.rand((d, d)) * threshold + threshold
                random_additions = random_block[small_values_indices] 
                cov_matrix[small_values_indices] = cov_matrix[small_values_indices] + random_additions
   
                inv_cov_matrix = torch.inverse(cov_matrix)
                log_det_cov_matrix = torch.logdet(cov_matrix)
                exponent = -0.5 * torch.einsum('ni,ij,nj->n', diff, inv_cov_matrix, diff)
                log_prob = -0.5 * (d * torch.log(2 * torch.tensor(torch.pi)) + log_det_cov_matrix) + exponent
                
                responsibilities[:, k] = self.weights[k] * torch.exp(log_prob)
            
            # check if exists value less than threshold
            nan_indices = torch.isnan(responsibilities)
            responsibilities[nan_indices] = 0
            small_values_indices = responsibilities < threshold
            random_block = torch.rand((N, K)) * threshold + threshold
            random_additions = random_block[small_values_indices] 
            responsibilities[small_values_indices] = responsibilities[small_values_indices] + torch.tensor(random_additions)
            
            responsibilities = responsibilities / (responsibilities.sum(dim=1, keepdim=True))
  
            # M-step: Update parameters
            N_k = responsibilities.sum(dim=0)
            new_means = torch.mm(responsibilities.t(), X) / N_k[:, None]
            new_covariances = torch.zeros_like(self.covariances)

            for k in range(K):
                diff = X - new_means[k]
                new_covariances[k] = (responsibilities[:, k][:, None] * diff**2).sum(dim=0) / N_k[k]

            self.means.data = new_means
            self.covariances.data = new_covariances
        
        current_state = torch.get_rng_state()
        torch.set_rng_state(current_state)
        
    def sample(self, n_samples=1):
        samples = []
        for _ in range(n_samples):
            component = torch.multinomial(self.weights, 1).item()
            cov_matrix = torch.diag(self.covariances[component])
            mvn = MultivariateNormal(self.means[component], cov_matrix)
            samples.append(mvn.sample())

        return torch.stack(samples).detach().numpy()
    
    def log_likelihood_estimate(self, data):
        N, d = data.shape
        K = self.n_components
        log_likelihoods = torch.zeros((N, K))
        
        for k in range(K):
            diff = data - self.means[k]
            cov_matrix = torch.diag(self.covariances[k])
            inv_cov_matrix = torch.inverse(cov_matrix)
            log_det_cov_matrix = torch.logdet(cov_matrix)
            exponent = -0.5 * torch.einsum('ni,ij,nj->n', diff, inv_cov_matrix, diff)
            log_prob = -0.5 * (d * torch.log(2 * torch.tensor(torch.pi)) + log_det_cov_matrix) + exponent
            log_likelihoods[:, k] = torch.log(self.weights[k]) + log_prob
        
        total_log_likelihood = torch.logsumexp(log_likelihoods, dim=1)
        return total_log_likelihood.sum().item()
    

class GMM_PyTorch_Batch:
    """
    A simplified version of GMM (fixed W) implemented in PyTorch.
    This GMM function can solve GMM with batch data.
    
    args:
    n_components: int, number of components
    n_features: int, number of features
    
    """
    def __init__(self, n_components, n_features):
        self.n_components = n_components
        self.n_features = n_features
        self.load_parameters()
        
    def load_parameters(self): 
        self.weights = torch.linspace(1/self.n_components, 1, self.n_components)
        self.weights = self.weights / self.weights.sum()
        self.means = torch.ones((self.n_components, self.n_features), requires_grad=False)
        self.covariances = torch.ones((self.n_components, self.n_features), requires_grad=False) * 0.5  # Initialize variances to a small positive value
        
    def fit(self, X, n_iter):   
        # X: (b, N, d)
        X = torch.tensor(X)
        device = X.device
        b, N, d = X.shape # batch, number of samples, number of features
        K = self.n_components
        
        # broadcast the parameters
        _means = self.means.unsqueeze(0).repeat(b, 1, 1).to(device)
        _covariances = self.covariances.unsqueeze(0).repeat(b, 1, 1).to(device)
        _weights = self.weights.unsqueeze(0).repeat(b, 1).to(device)
        threshold = 1e-20
        
        for _ in range(n_iter):
            # torch.manual_seed(0)
            # E-step: Compute responsibilities
            responsibilities = torch.zeros((b, N, K)).to(device)
            for k in range(K):
                diff = X - _means[:, k, :].unsqueeze(1)
                cov_matrix = torch.diag_embed(_covariances[:, k, :]).to(device)
                
                # check if exists value less than threshold
                nan_indices = torch.isnan(cov_matrix)
                cov_matrix[nan_indices] = 0
                small_values_indices = cov_matrix < threshold
                random_block = (torch.rand((d, d)) * threshold + threshold).to(device)
                random_block = random_block.unsqueeze(0).repeat(b, 1, 1)
                random_additions = random_block[small_values_indices]
                cov_matrix[small_values_indices] = cov_matrix[small_values_indices] + random_additions
                
                # print(_, 'cov: ',torch.diag(cov_matrix[176,:,:]))
                inv_cov_matrix = torch.inverse(cov_matrix)
                log_det_cov_matrix = torch.logdet(cov_matrix)
                
                exponent = -0.5 * torch.einsum('bni,bij,bnj->bn', diff, inv_cov_matrix, diff)
                log_prob = -0.5 * (d * torch.log(2 * torch.tensor(torch.pi)) + log_det_cov_matrix).unsqueeze(1).repeat(1, N) + exponent
                
                responsibilities[:, :, k] = _weights[:, k].unsqueeze(1) * torch.exp(log_prob)
               
            # check if exists value less than threshold
            nan_indices = torch.isnan(responsibilities)
            responsibilities[nan_indices] = 0
            small_values_indices = responsibilities < threshold
            random_block = (torch.rand((N, K)) * threshold + threshold).to(device)
            random_block = random_block.unsqueeze(0).repeat(b, 1, 1)
            random_additions = random_block[small_values_indices]
            responsibilities[small_values_indices] = responsibilities[small_values_indices]+ random_additions
            
            responsibilities = responsibilities / (responsibilities.sum(dim=2, keepdim=True))
            
            # M-step: Update parameters
            N_k = responsibilities.sum(dim=1)
            new_means = torch.einsum('bni,bnk->bki', X, responsibilities) / N_k.unsqueeze(2)
            
            new_covariances = torch.zeros_like(_covariances)
            for k in range(K):
                diff = X - new_means[:, k, :].unsqueeze(1)
                weighted_diff = diff * diff * responsibilities[:, :, k][:, :, None]
                new_covariances[:, k] = weighted_diff.sum(dim=1) / N_k[:, k][:, None]
                
            _means = new_means
            _covariances = new_covariances   
             
        current_state = torch.get_rng_state()
        torch.set_rng_state(current_state)
        
        return new_means, new_covariances
