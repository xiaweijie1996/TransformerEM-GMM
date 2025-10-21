# train_nf.py
# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

# repo paths & data
_parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(_parent_path)
from asset.dataloader import Dataloader_nolabel
import asset.random_sampler as rs

# your NF model
import nf_model_lin as nf

def main():
    # config
    batch_size = 16
    split_ratio = (0.8, 0.1, 0.1)
    data_path = 'exp/data_process_for_data_collection_all/new_data_15minute_grid_nomerge.pkl'
    save_dir_root = 'exp/exp_second_round/conditional_generativemodels'
    N, L = 250, 96
    random_sample_num = 8
    hidden_channels = 198
    K_blocks = 3
    lr = 2e-4
    weight_decay = 1e-4
    max_iters = 10001
    log_every = 100

    subdir = os.path.join(save_dir_root, f'{random_sample_num}shot')
    os.makedirs(subdir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data
    dataset = Dataloader_nolabel(data_path, batch_size=batch_size, split_ratio=split_ratio)

    # model & opt
    model = nf.CNicemModel(input_c=1, hidden_c=hidden_channels, condition_c=random_sample_num*2, n_layers=K_blocks, scaler_dim=L).to(device)
    num_para = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model #params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # load model
    path = os.path.join(save_dir_root, f'{random_sample_num}shot/flow_{num_para}_{random_sample_num}shot.pt')
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
    
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best = float('inf')
    ckpt_path = os.path.join(subdir, f'flow_{sum(p.numel() for p in model.parameters() if p.requires_grad)}_{random_sample_num}shot.pt')

    it = 0
    while it < max_iters:
        it = it + 1
        x = dataset.load_train_data()
        x = torch.tensor(x, dtype=torch.float32, device=device)[:, :, :-1]   # (B,N,L)

        # per-sample, per-channel min-max
        x_min = x.min(dim=1, keepdim=True).values
        x_max = x.max(dim=1, keepdim=True).values
        x0 = (x - x_min) / (x_max - x_min + 1e-15)   # (B,N,L)
        # condition: random N' channels
        cond = rs.random_sample(x0, 'random', random_sample_num).to(device)

        # forward
        # reshape x0 (B, N, L) → (B*N,1 , L)
        x0 = x0.reshape(-1, 1, L)
        # expland cond (B, N', L) → (B*N, N', L)
        cond = cond.unsqueeze(1).expand(-1, 250, -1, -1).reshape(-1, random_sample_num, L)
        cond = cond.reshape(cond.shape[0], -1, L//2)
        z, log_det = model.forward(x0, cond)
        loss = 0.5 * (z**2).sum(dim=(1,2)) - log_det
        
        loss = loss.mean()
        
        # backward
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        print(f"iter {it}: loss = {loss.item():.4f}, best = {best:.4f}", 'log_det mean: ', log_det.mean().item())
            
            
        if it % log_every == 0:
            
            if loss.item() < best:
                best = loss.item()
                torch.save(model.state_dict(), ckpt_path)
                print(f"  saved best model to {ckpt_path}")
                
            # plot generation
            model.eval()
            with torch.no_grad():
                # sample from N(0,I)
                L = x0.shape[2]
                z_sample = torch.randn(N, 1, L, device=device) 
                print('z_sample mean: ', z_sample.mean().item(), ' std: ', z_sample.std().item())
                print('x0 mean: ', x0.mean().item(), ' std: ', x0.std().item())
                print('z', z.mean().item(), ' std: ', z.std().item())   
        
                x_sample, _ = model.inverse(z_sample, cond[:N, :, :])
                x_sample = x_sample.cpu().numpy()
                # plot first sample's all channels and  real data
                plt.figure(figsize=(12,6))
                plt.subplot(2,1,1)
                plt.plot(x_sample[:,0,:].T, alpha=0.5, color='C0')
                plt.title(f"Generated Sample (iter {it})")
                plt.subplot(2,1,2)
                plt.plot(x0[:N, 0, : ].cpu().numpy().T, alpha=0.5, color='C1')
                plt.title(f"Real Data")
                plt.tight_layout()
                plt.savefig(os.path.join(subdir, f'sample_iter.png'))
                plt.close()
         
                
                
            model.train()
            
if __name__ == "__main__":
    main()
