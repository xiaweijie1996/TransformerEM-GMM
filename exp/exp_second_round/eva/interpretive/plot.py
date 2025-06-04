#!/usr/bin/env python3
import os 
import sys
_parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(_parent_path)

import wandb
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

import model.gmm_transformer as gmm_model
import exp_second_round.eva.interpretive.tool as tl
import asset.em_pytorch as ep
import asset.plot_eva as pae
from asset.dataloader import Dataloader_nolabel
import asset.random_sampler as rs

torch.set_default_dtype(torch.float64)

# load data
batch_size =  30
split_ratio = (0.8,0.1,0.1)
data_path =  'exp/data_process_for_data_collection_all/all_data.pkl'
dataset = Dataloader_nolabel(data_path,  batch_size=batch_size
                    , split_ratio=split_ratio)
print('lenthg of train data: ', dataset.__len__()*split_ratio[0])
print('lenthg of test data: ', dataset.__len__()*split_ratio[1])

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# define the hyperparameters
random_sample_num = 6
num_epochs = int(500000)
sub_epoch = int(dataset.__len__()*split_ratio[0]/batch_size)
lr = 0.0001
n_components = 4
min_random_sample_num = 4

# define the encoder
chw = (1, random_sample_num,  25)
para_dim = n_components*2
hidden_d = 24
out_d = 24
n_heads = 4
mlp_ratio = 12
n_blocks = 6
encoder = gmm_model.ViT_encodernopara(chw, hidden_d, out_d, n_heads, mlp_ratio, n_blocks).to(device)
_model_scale = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
print('number of parameters: ', _model_scale)

# Define a gmm embedding layer
embedding_para = torch.nn.Embedding(n_components*2, 1).to(device)
emb_empty_token = torch.nn.Embedding(1, chw[2]).to(device)

encoder_model_path = 'exp/exp_second_round/loglikehoodest/model/solar_encoder_6_592086.pth'
embedding_para_model_path = 'exp/exp_second_round/loglikehoodest/model/solar_embedding_6_592086.pth'
emb_empty_token_model_path = 'exp/exp_second_round/loglikehoodest/model/solar_emb_empty_token_6_592086.pth'
encoder.load_state_dict(torch.load(encoder_model_path, map_location=device, weights_only=False))
embedding_para = torch.load(embedding_para_model_path, map_location=device, weights_only=False)
emb_empty_token = torch.load(emb_empty_token_model_path, map_location=device, weights_only=False)

samples = dataset.load_test_data(batch_size)
samples = torch.tensor(samples, dtype=torch.float64).to(device)
orignal_samples = samples.clone()
a = 0

for sample in samples:
    encoder.eval()
    a += 1
    # sample[:, :-1] = (sample[:, :-1] - sample[:, :-1].min()) / (sample[:, :-1].max() - sample[:, :-1].min() + 1e-15)
    _min, _ = sample[:, :-1].min(axis=0, keepdim=True)
    _max, _ = sample[:, :-1].max(axis=0, keepdim=True)
    scaled_sample = sample.clone()
    scaled_sample[:, :-1] = (scaled_sample[:, :-1] - _min) / (_max - _min + 1e-15)
    scaled_sample = scaled_sample.unsqueeze(0)
    
    train_sample_part_scaled = rs.random_sample(scaled_sample, 'random', 6)
    print('train_sample_part_scaled shape: ', train_sample_part_scaled.shape)
    print('scaled_sample shape: ', scaled_sample.shape)
    
    _loss_nodrop, _random_num, _new_para, _param, r_samples, r_samples_part, _mm = tl.get_loss_le(scaled_sample, train_sample_part_scaled, encoder,
                                                                              6, 1, n_components, embedding_para, emb_empty_token, 'True', device)
    print(f'loss of no drop: ', _loss_nodrop)
    
    # iteratively cancel the first, second, and third, up to laset element of train_sample_part_scaled
    delta_loss = []
    dropped_rows = []
    for row in range(train_sample_part_scaled.shape[1]):
        # Drop one row from second dimension
        before = train_sample_part_scaled[:, :row, :]
        after = train_sample_part_scaled[:, row+1:, :]
        dropped_row = train_sample_part_scaled[:, row:row+1, :]
        dropped = torch.cat((before, after), dim=1)
        print('dropped shape: ', dropped.shape)

        # Compute loss with one row dropped
        _loss, *_ = tl.get_loss_le(
            scaled_sample,
            dropped,
            encoder,
            6, 1, n_components,
            embedding_para,
            emb_empty_token,
            'True',
            device
        )

        diff = torch.abs( _loss - _loss_nodrop)
        print(f'Loss delta (drop row {row}):', diff.item())
        delta_loss.append(diff.item())
        dropped_rows.append(dropped_row.squeeze(0).detach().numpy())

    # Convert delta_loss to numpy
    delta_loss = np.array(delta_loss)  # shape [N,]
    N = delta_loss.shape[0]

    # Find min/max indices as before
    idx_min = np.argmin(delta_loss)
    idx_max = np.argmax(delta_loss)

    # Prepare normalization and colormap
    norm = Normalize(vmin=delta_loss.min(), vmax=delta_loss.max())
    cmap = plt.get_cmap('coolwarm')

    # Prepare the impact image
    impact_img = delta_loss.reshape(-1, 1)

    # Unscale sample parts for plotting
    train_part = train_sample_part_scaled[0, :, :-1].cpu().numpy()  # shape [N, T]
    train_part = train_part * ((_max - _min).cpu().numpy() + 1e-15) + _min.cpu().numpy()

    # --- Plotting ---
    fig, axes = plt.subplots(1, 4, figsize=(10, 3),
                            gridspec_kw={'width_ratios': [0.3, 0.5, 0.5, 0.5]})

    # 1) Heatmap
    ax0 = axes[0]
    im = ax0.imshow(impact_img, aspect='auto', cmap=cmap, norm=norm)
    ax0.set_title("|Δ| of sampled Data", x=2.2, ha="right", va="bottom", pad=0)
    ax0.set_xticks([])
    ax0.set_yticks(np.arange(N))
    ax0.set_yticklabels(np.arange(1, N+1))
    ax0.invert_yaxis()
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="20%", pad=0.2)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("|Δ| log-likelihood")

    # 2) Dropped row min
    plot_row_min = dropped_rows[idx_min][:, :-1]
    plot_row_min = plot_row_min * ((_max - _min).cpu().numpy() + 1e-15) + _min.cpu().numpy()
    axes[1].plot(plot_row_min.T, color='blue')
    axes[1].set_title(f"ECP of minmum |Δ|")
    axes[1].set_xlabel("Hour of Day [-]")
    axes[1].set_ylabel("Electricity Consumption [kWh]")

    # 3) Dropped row max
    plot_row_max = dropped_rows[idx_max][:, :-1]
    plot_row_max = plot_row_max * ((_max - _min).cpu().numpy() + 1e-15) + _min.cpu().numpy()
    axes[2].plot(plot_row_max.T, color='red')
    axes[2].set_title(f"ECP of maximum |Δ|")
    axes[2].set_xlabel("Hour of Day [-]")
    axes[2].set_ylabel("Electricity Consumption [kWh]")

    # 4) Train sample part, colored by delta_loss
    ax3 = axes[3]
    for i in range(N):
        color = cmap(norm(delta_loss[i]))
        ax3.plot(train_part[i], color=color, alpha=0.8)
    ax3.set_title("Sampled Data")
    axes[3].set_xlabel("Hour of Day [-]")
    axes[3].set_ylabel("Electricity Consumption [kWh]")

    plt.tight_layout()
    plt.savefig(f"exp/exp_second_round/eva/interpretive/plot/plot_{a}.png")
    plt.close(fig)