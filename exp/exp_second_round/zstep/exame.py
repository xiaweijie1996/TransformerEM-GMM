import os 
import sys
_parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(_parent_path)

from sklearn.mixture import GaussianMixture
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle

import asset.random_sampler as rs
from asset.dataloader import Dataloader_nolabel
from asset.em_pytorch import GMM_Simplified_PyTorch

# load data
batch_size =  78
split_ratio = (1,0,0)
data_path =  'exp/data_process_for_data_collection_all/new_data_15minute_grid_nomerge.pkl' 
dataset = Dataloader_nolabel(data_path,  batch_size=batch_size
                    , split_ratio=split_ratio)
dataset.images = dataset.images # + np.abs(np.random.normal(0, 0.01, dataset.images.shape)) 
print('lenthg of train data: ', dataset.__len__()*split_ratio[0])
print('lenthg of test data: ', dataset.__len__()*split_ratio[1])
from asset.dataloader import Dataloader_nolabel


min_random_sample_num = 8
random_sample_num = 9
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_sample = dataset.load_train_data()
train_sample = torch.tensor(train_sample, dtype=torch.float64).to(device)


_train_min,_ = train_sample[:,:, :-1].min(dim=1, keepdim=True)
_train_max,_ = train_sample[:,:, :-1].max(dim=1, keepdim=True)
train_sample[:,:, :-1] = (train_sample[:,:, :-1] - _train_min)/(_train_max-_train_min+1e-15)

# random_sample a number between min_random_sample_num and random_sample_num
_random_num = torch.randint(min_random_sample_num, random_sample_num+1, (1,)).item()
_train_sample_part = rs.random_sample(train_sample, 'random', _random_num)
_train_sample_part[:, :, -1] = _train_sample_part[:, :, -1]/365 # simple data embedding

print('train sample shape: ', train_sample.shape, _train_sample_part.shape)
print(train_sample.mean(dim=1))
total_diff_mean_list = []
total_diff_cov_list = []
for n_iter_partial  in range(1, 10):
    total_diff_mean = 0
    total_diff_cov = 0
    for i in range(train_sample.shape[0]):
        
        # fit gmm model
        # gmm_model_partial = GMM_Simplified_PyTorch(n_components=4, n_features=_train_sample_part.shape[2])
        # gmm_model_partial.fit(_train_sample_part[i].cpu(), n_iter=n_iter_partial)
        gmm_model_partial = GaussianMixture(n_components=4, covariance_type='diag', max_iter=n_iter_partial, random_state=0)
        gmm_model_partial.fit(_train_sample_part[i][:,:-1].cpu().numpy())
        
        # fit gmm full model
        # gmm_model_full = GMM_Simplified_PyTorch(n_components=4, n_features=train_sample.shape[2])
        # gmm_model_full.fit(train_sample[i].cpu(), n_iter=200)
        gmm_model_full = GaussianMixture(n_components=4, covariance_type='diag', max_iter=200, random_state=0)
        gmm_model_full.fit(train_sample[i][:,:-1].cpu().numpy())
        
        print(train_sample[i].shape, _train_sample_part[i].shape)
        # mean_partial = gmm_model_partial.means.detach().cpu().numpy()
        # cov_partial = gmm_model_partial.covariances.detach().cpu().numpy()
        # weights_partial = gmm_model_partial.weights.detach().cpu().numpy()
        
        # mean_full = gmm_model_full.means.detach().cpu().numpy()
        # cov_full = gmm_model_full.covariances.detach().cpu().numpy()
        # weights_full = gmm_model_full.weights.detach().cpu().numpy()
        
        mean_partial = gmm_model_partial.means_
        cov_partial = gmm_model_partial.covariances_
        weights_partial = gmm_model_partial.weights_
        print(mean_partial.mean(), cov_partial.mean())
        
        mean_full = gmm_model_full.means_
        cov_full = gmm_model_full.covariances_
        weights_full = gmm_model_full.weights_
        print(mean_full.mean(), cov_full.mean())
        
        _mean_diff = np.abs(mean_partial - mean_full).mean()
        _cov_diff = np.abs(cov_partial - cov_full).mean()
        _weights_diff = np.abs(weights_partial - weights_full).mean()
        
        total_diff_cov += _cov_diff
        total_diff_mean += _mean_diff
        print(f'Iter {n_iter_partial}, Sample {i}')
        
    print(f'Average Mean Diff: {total_diff_mean/train_sample.shape[0]}, Average Cov Diff: {total_diff_cov/train_sample.shape[0]}')
    total_diff_mean_list.append(total_diff_mean/train_sample.shape[0])
    total_diff_cov_list.append(total_diff_cov/train_sample.shape[0])

# Save the list as pickle file

with open('exp/exp_second_round/zstep/gmm_mean_cov_diff_list.pkl', 'wb') as f:
    pickle.dump({'mean_diff': total_diff_mean_list, 'cov_diff': total_diff_cov_list}, f)
    
# pLOT
plt.figure(figsize=(10,5))
# plot 2 figures in one plot
plt.subplot(1,2,1)
plt.plot(range(len(total_diff_mean_list)), total_diff_mean_list, label='Mean Diff', marker='o')
plt.xlabel('Number of EM Iterations (Partial GMM)')
plt.ylabel('Average Mean Difference')
plt.title('Mean Difference vs EM Iterations')
plt.grid()
plt.subplot(1,2,2)
plt.plot(range(len(total_diff_cov_list)), total_diff_cov_list, label='Cov Diff', marker='o', color='orange')
plt.xlabel('Number of EM Iterations (Partial GMM)')
plt.ylabel('Average Covariance Difference')
plt.title('Covariance Difference vs EM Iterations')
plt.grid()
plt.tight_layout()
plt.savefig('exp/exp_second_round/zstep/gmm_mean_cov_diff_plot.png')
plt.close()