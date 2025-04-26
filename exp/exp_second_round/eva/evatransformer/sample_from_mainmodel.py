import os
import sys
_parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(_parent_path)

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import ks_2samp, wasserstein_distance

import model.gmm_transformer as gmm_model
from asset.dataloader import Dataloader_nolabel
import asset.random_sampler as rs
import asset.gmm_train_tool as gmm_train_tool
import asset.em_pytorch as ep
import asset.plot_eva as plot_eva


# Set default dtype to float64
torch.set_default_dtype(torch.float64)

# Determine the device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define the encoder parameters
random_sample_num = 96
n_components = 6    
chw = (1, random_sample_num,  97)
para_dim = n_components*2
hidden_d = 96*1
out_d = 96
n_heads = 2
mlp_ratio = 2
n_blocks = 2

# random_sample_num = 96
# n_components = 3   
# chw = (1, random_sample_num,  97)
# para_dim = n_components*2
# hidden_d = 96*1
# out_d = 96
# n_heads = 2
# mlp_ratio = 4
# n_blocks = 3

# Create the encoder model
encoder = gmm_model.ViT_encodernopara(chw, hidden_d, out_d, n_heads, mlp_ratio, n_blocks).to(device)

# load state dict of the model
model_path = 'exp/exp_second_round/transformer_15minutes/model/transformer_encoder_96_251246.pth'
# model_path = 'exp/exp_second_round/transformer_15minutes/model/transformer_encoder_96_783090.pth'
model = torch.load(model_path, map_location=device)
encoder.load_state_dict(model)

embedding_para = torch.nn.Embedding(n_components*2, 1).to(device)
emb_empty_token = torch.nn.Embedding(1, chw[2]).to(device)

path_embedding = 'exp/exp_second_round/transformer_15minutes/model/transformer_embedding_96_251246.pth'
# path_embedding = 'exp/exp_second_round/transformer_15minutes/model/transformer_embedding_96_783090.pth'
emb_weight = torch.load(path_embedding, map_location=device, weights_only=False)
path_empty = 'exp/exp_second_round/transformer_15minutes/model/transformer_emb_empty_token_96_251246.pth'
# path_empty = 'exp/exp_second_round/transformer_15minutes/model/transformer_emb_empty_token_96_783090.pth'
empty_token_vec = torch.load(path_empty, map_location=device,weights_only=False)

# load data
batch_size =  1
split_ratio = (0.8,0.1,0.1)
data_path =  'exp/data_process_for_data_collection_all/transformer_data_15minutes.pkl' ## 
dataset = Dataloader_nolabel(data_path,  batch_size=batch_size
                    , split_ratio=split_ratio)
print('lenthg of test data: ', dataset.__len__()*split_ratio[1])
# test_data = dataset.load_test_data(batch_size )
test_data = dataset.load_train_data() # batch_size 

# normalize the input data
min_test_data = test_data[:,:, :-1].min(axis=1).reshape(batch_size , 1, chw[2]-1)
max_test_data = test_data[:,:, :-1].max(axis=1).reshape(batch_size , 1, chw[2]-1)
test_data[:,:, :-1] = (test_data[:,:, :-1]  - min_test_data)/(max_test_data -min_test_data+1e-15)
test_data = torch.tensor(test_data, dtype=torch.float64).to(device)

# Compte the parameters
# _random_num = 32 # torch.randint(1, random_sample_num+1, (1,)).item() # Number of the shots

for _random_num in [4, 8, 16, 32]:
    _test_sample_part = rs.random_sample(test_data , 'random', _random_num)
    _test_sample_part[:, :, -1] = _test_sample_part[:, :, -1]/365 # simple data embedding
    _test_sample_part_emb = gmm_train_tool.pad_and_embed(_test_sample_part, random_sample_num, _random_num,
                                            emb_empty_token, device)
    _ms, _covs = ep.GMM_PyTorch_Batch(n_components, _test_sample_part[:,:, :-1].shape[-1]).fit(_test_sample_part[:,:, :-1], 1) # _ms: (b, n_components, 24), _covs: (b, n_components, 24)

    # concatenate the mean and variance to have (b, n_components*2, 25)
    _param_emb, _param = gmm_train_tool.concatenate_and_embed_params(_ms, _covs, n_components, embedding_para, device)

    # feed into the encoder
    _test_sample_part_emb = torch.cat((_param_emb, _test_sample_part_emb), dim=1)
    encoder_out = encoder(_test_sample_part_emb)
    _new_para = encoder_out[:, :n_components*2, :]
    _new_para = encoder.output_adding_layer(_new_para, _param)

    mean = _new_para[:, :n_components* 96].view(-1, n_components, 96) 
    # mean = mean + torch.randn(mean.shape).to(device) * 0.01
    cov = _new_para[:, n_components* 96:].view(-1, n_components, 96)
    # cov = cov + torch.randn(cov.shape).to(device) * 0.01

    # recover the scale
    recovered_test_data = test_data[:, :, :-1].clone() * (max_test_data -min_test_data+1e-15) + min_test_data
    recover_test_sample_part = _test_sample_part[:, :, :-1].clone() * (max_test_data -min_test_data+1e-15) + min_test_data
    recover_test_sample_part = _test_sample_part[:, :, :-1].clone()

    mmd = 0
    kl = 0
    ks = 0
    ws = 0
    msem = 0

    mmd_partreal = 0
    kl_partreal = 0
    ks_partreal = 0
    ws_partreal = 0
    msem_partreal = 0
    

    for i in tqdm(range(batch_size)):
        # sample from the GMM
        _num=i
        # Sample from the GMM
        samples, gmm = plot_eva.sample_from_gmm(n_components, _new_para, _num)
        # samples = samples * (max_test_data[_num] -min_test_data[_num]+1e-15) + min_test_data[_num]
        mmd += plot_eva.compute_mmd(samples, test_data[_num, :, :-1].cpu().detach().numpy())
        kl += plot_eva.compute_kl_divergence(samples, test_data[_num, :, :-1].cpu().detach().numpy())
        ks += ks_2samp(samples.flatten(), test_data[_num, :, :-1].cpu().detach().numpy().flatten())[0]
        ws += wasserstein_distance(samples.flatten(), test_data[_num, :, :-1].cpu().detach().numpy().flatten())
        msem = plot_eva.calculate_autocorrelation_mse(samples, test_data[_num, :, :-1].cpu().detach().numpy())
        
        # Partial real data
        _part_real = recover_test_sample_part[_num, :, :].cpu().detach().numpy()
        mmd_partreal += plot_eva.compute_mmd(samples, _part_real)
        kl_partreal += plot_eva.compute_kl_divergence(samples, _part_real)
        ks_partreal += ks_2samp(samples.flatten(), _part_real.flatten())[0]
        ws_partreal += wasserstein_distance(samples.flatten(), _part_real.flatten())
        msem_partreal = plot_eva.calculate_autocorrelation_mse(samples, _part_real)
        
        
        # Plot the samples
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)

        # plot the samples the colors indicate the sum of the samples
        samples = samples * (max_test_data[_num] -min_test_data[_num]+1e-15) + min_test_data[_num]
        for i in range(samples.shape[0]):
            # print('samples shape check: ', samples.shape)
            _sum = test_data[_num, i, :-1].sum()
            color = plt.cm.viridis(_sum / test_data.max())
            plt.plot(samples[i, :-1], alpha=0.05, c=color)
        plt.title('Samples from GMM')
        plt.xlabel('Time')
        plt.ylabel('Value')

        plt.subplot(2, 1, 2)
        # plot the real samples
        for i in range(test_data.shape[1]):
            _sum = test_data[_num, i, :-1].sum()
            color = plt.cm.viridis(_sum / test_data.max())
            plt.plot(recovered_test_data[_num, i, :-1].cpu().detach().numpy(), alpha=0.05, c=color)
        plt.title('real')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.savefig(f'exp/exp_second_round/eva/evatransformer/plot/samples_from_gmm_{_random_num}.png')
        plt.show()



        print(f'mmd of {_random_num}-shots: ', mmd/batch_size)
        print(f'kl of {_random_num}-shots: ', kl/batch_size)
        print(f'ks of {_random_num}-shots: ', ks/batch_size)
        print(f'ws of {_random_num}-shots: ', ws/batch_size)
        print(f'msem of {_random_num}-shots: ', msem/batch_size)

        print(f'mmd_partreal of {_random_num}-shots: ', mmd_partreal/batch_size)
        print(f'kl_partreal of {_random_num}-shots: ', kl_partreal/batch_size)
        print(f'ks_partreal of {_random_num}-shots: ', ks_partreal/batch_size)
        print(f'ws_partreal of {_random_num}-shots: ', ws_partreal/batch_size)
        print(f'msem_partreal of {_random_num}-shots: ', msem_partreal/batch_size)
        
        # save the results in a text file
        with open('exp/exp_second_round/eva/evatransformer/sample_from_gmm.txt', 'a') as f:
            f.write(f'mmd of {_random_num}-shots: {mmd/batch_size}\n')
            f.write(f'kl of {_random_num}-shots: {kl/batch_size}\n')
            f.write(f'ks of {_random_num}-shots: {ks/batch_size}\n')
            f.write(f'ws of {_random_num}-shots: {ws/batch_size}\n')
            f.write(f'msem of {_random_num}-shots: {msem/batch_size}\n')

            f.write(f'mmd_partreal of {_random_num}-shots: {mmd_partreal/batch_size}\n')
            f.write(f'kl_partreal of {_random_num}-shots: {kl_partreal/batch_size}\n')
            f.write(f'ks_partreal of {_random_num}-shots: {ks_partreal/batch_size}\n')
            f.write(f'ws_partreal of {_random_num}-shots: {ws_partreal/batch_size}\n')
            f.write(f'msem_partreal of {_random_num}-shots: {msem_partreal/batch_size}\n')
            f.write('\n')




