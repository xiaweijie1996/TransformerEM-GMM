import torch
import matplotlib.pyplot as plt
import gmm_trans_package.asset.em_pytorch as ep_module
from scipy.stats import kendalltau, energy_distance
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.mixture import GaussianMixture 
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity

def plot_results(t_samples,_val_data, sampled_samples):
    # plot the generated samples
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    # plot the generated samples
    axs[0].plot(t_samples.T, c='b', alpha=0.1)
    axs[0].set_title('Generated Samples by our method')

    # plot the real samples
    axs[1].plot(_val_data.T, c='r', alpha=0.1)
    axs[1].set_title('Real Samples (complete data)')

    # plot the real samples
    axs[2].plot(sampled_samples.T, c='r', alpha=0.1)
    axs[2].set_title('Sampled Real Samples')

    plt.show()
def sample_from_gmm(n_components, _new_para, _num=0, _num_samples=300):
    _dim=  int(_new_para.shape[-1]/n_components/2)
     
    gmm = ep_module.GMM_Simplified_PyTorch(n_components, _dim)
    gmm.means = _new_para[_num, :n_components*_dim].view(n_components, _dim).cpu().detach()
    gmm.covariances = _new_para[_num, n_components*_dim:].view(n_components, _dim).cpu().detach()
    _samples = gmm.sample(_num_samples)
    _samples[_samples<0]=0
    return _samples, gmm

# save_path, batch_size, n_components, _mm, _new_para, r_samples, r_samples_part, _param
def plot_samples(save_path, batch_size, n_components, _mm, _new_para, r_samples, r_samples_part, _param, figsize=(10, 15)):
    fig, axs = plt.subplots(6, 1, figsize=figsize)
    
    # Dimension 
    _dim=  int(_new_para.shape[-1]/n_components/2)
    
    # Random integer between 0 and batch_size
    _num = 0
    _min = _mm[0][_num].cpu().detach().numpy()
    _max = _mm[1][_num].cpu().detach().numpy()
    r_samples_est = r_samples[_num].clone()
    r_samples_est = r_samples_est.cpu().detach()
    
    # Second subplot: Generated Samples
    _samples, _gmm = sample_from_gmm(n_components, _new_para)
    # Scale the samples back to the original scale
    t_samples = _samples * (_max - _min) + _min
    t_samples[t_samples < 0] = 0
    axs[1].plot(t_samples.T, c='b', alpha=0.1)
    axs[1].set_title('Generated Samples by transformer')
    # llk_gmm_t = _gmm.log_likelihood_estimate(r_samples_est)

    # Third subplot: Real Samples
    r_samples_scale_back = r_samples_est * (_max - _min) + _min
    axs[2].plot(r_samples_scale_back.numpy().T, c='r', alpha=0.1)
    axs[2].set_title('Real Samples (data)')
    
    # Fourth subplot: Generated Samples
    gmm = ep_module.GMM_Simplified_PyTorch(n_components, _dim)
    gmm.fit(r_samples_est, 300)
    _samples = gmm.sample(300)
    _samples = _samples * (_max - _min) + _min
    _samples[_samples < 0] = 0
    axs[3].plot(_samples.T, c='b', alpha=0.1)
    axs[3].set_title('Fit GMM to all real sample')
    # llk_gmm_f = gmm.log_likelihood_estimate(r_samples_est)

    gmm_param = torch.concat((gmm.means.view(-1), gmm.covariances.view(-1)))
    # First subplot: Generated vs Original
    axs[0].plot(_new_para[_num].cpu().detach().numpy(), c='b', alpha=0.5)
    axs[0].plot(gmm_param.cpu().detach().numpy(), c='r', alpha=0.5)
    axs[0].plot(_param[_num].view(-1).cpu().detach().numpy(), c='g', alpha=0.5)
    axs[0].legend(['Generated', 'gmm', 'gmm_torch'])
    axs[0].set_title('Predicted, GMM_fit, EM_embedding')
    
    r_samples_part_gmm = r_samples_part[_num].cpu().detach().numpy()
    r_samples_part_gmm_scaled = r_samples_part_gmm * (_max - _min) + _min
    axs[4].plot(r_samples_part_gmm_scaled.T, c='r', alpha=0.1)
    axs[4].set_title('Random sample data')
    
    # fit and sample from a gaussian mixture
    gmm_sklearn = GaussianMixture(n_components=n_components, random_state=0).fit(r_samples_part_gmm)
    gmm_sample, _ = gmm_sklearn.sample(300)
    gmm_sample[gmm_sample<0]=0
    gmm_sample = gmm_sample * (_max - _min) + _min
    axs[5].plot(gmm_sample.T, c='r', alpha=0.1)
    axs[5].set_title('GMM generated random sample data')
    
    # Display the plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# n_components, _new_para, r_samples, r_samples_part
def evaluation(n_components, _new_para, r_samples, r_samples_part, _num=0, _num_samples=300):
    t_sample, _ = sample_from_gmm(n_components, _new_para, _num, _num_samples)
    _r_sample = r_samples[_num].cpu().detach().numpy()
    _r_sampled_part =  r_samples_part[_num].cpu().detach().numpy()
    
    # fit and sample from a gaussian mixture
    gmm_sklearn = GaussianMixture(n_components=n_components, max_iter=300, random_state=0).fit(_r_sampled_part)
    gmm_sample, _ = gmm_sklearn.sample(300)
    gmm_sample[gmm_sample<0]=0
    # print(gmm_sample.shape)
    # compute energy distance
    mmd_rt = compute_mmd(t_sample, _r_sample)
    mmd_r_r = compute_mmd(_r_sampled_part, _r_sample)
    mmd_rg = compute_mmd(gmm_sample, _r_sample)
    
    # compute corr mse
    c_rt = calculate_autocorrelation_mse(t_sample, _r_sample)
    c_r_r = calculate_autocorrelation_mse(_r_sampled_part, _r_sample)
    c_rg = calculate_autocorrelation_mse(gmm_sample, _r_sample)
    
    return (mmd_rt, mmd_r_r, mmd_rg), (c_rt, c_r_r, c_rg)

def compute_kl_divergence(X, Y, bandwidth=0.1):
    kde_X = KernelDensity(bandwidth=bandwidth).fit(X)
    kde_Y = KernelDensity(bandwidth=bandwidth).fit(Y)

    # Create an evaluation grid
    eval_points = np.vstack([X, Y])
    
    log_density_X = kde_X.score_samples(eval_points)
    log_density_Y = kde_Y.score_samples(eval_points)

    # Convert log densities to probabilities
    density_X = np.exp(log_density_X)
    density_Y = np.exp(log_density_Y)

    return entropy(density_X, density_Y)

def compute_mmd(X, Y, kernel='rbf', gamma=1.0):
    if kernel == 'rbf':
        XX = rbf_kernel(X, X, gamma=gamma)
        YY = rbf_kernel(Y, Y, gamma=gamma)
        XY = rbf_kernel(X, Y, gamma=gamma)
    else:
        raise ValueError("Unsupported kernel type")

    return np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)

    
def calculate_energy_distances(dataset1, dataset2):
    """
    Calculate the energy distances between two dataset.

    """
    dataset1 = np.array(dataset1).flatten()
    dataset2 = np.array(dataset2).flatten()
    
    return energy_distance(dataset1, dataset2)

def kendalltau_corr(data):
    """compute the Kendall Tau correlation between one time series data"""
    data = np.array(data)
    correlation_matrix = np.zeros((data.shape[1], data.shape[1]))
    
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            correlation_matrix[i, j] = kendalltau(data[:, i], data[:, j])[0]
    return correlation_matrix


def calculate_autocorrelation_mse(dataset1, dataset2):
    """
    Calculate the correlation and mean square error between two dataset.

    """
    dataset1 = np.array(dataset1)
    dataset2 = np.array(dataset2)
    # compute the correlation matrix of data1
    correlation_matrix1 = kendalltau_corr(dataset1)
    # compute the correlation matrix of data2
    correlation_matrix2 = kendalltau_corr(dataset2)
    # compute the mean square error
    mse = mean_squared_error(correlation_matrix1, correlation_matrix2)
    return mse
 