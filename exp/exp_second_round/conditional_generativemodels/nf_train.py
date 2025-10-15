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
import nf_model as nf

def main():
    # config
    batch_size = 32
    split_ratio = (0.8, 0.1, 0.1)
    data_path = 'exp/data_process_for_data_collection_all/new_data_15minute_grid_nomerge.pkl'
    save_dir_root = 'exp/exp_second_round/conditional_generativemodels'
    N, L = 250, 96
    random_sample_num = 4
    hidden_channels = 2
    K_blocks = 2
    lr = 2e-4
    weight_decay = 1e-4
    max_iters = 100000
    log_every = 2

    subdir = os.path.join(save_dir_root, f'{random_sample_num}shot_flow')
    os.makedirs(subdir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data
    dataset = Dataloader_nolabel(data_path, batch_size=batch_size, split_ratio=split_ratio)

    # model & opt
    model = nf.CNicemModel(input_c=N, hidden_c=hidden_channels, condition_c=random_sample_num, n_layers=K_blocks).to(device)
    print(f"Model #params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best = float('inf')
    ckpt_path = os.path.join(subdir, f'flow_{sum(p.numel() for p in model.parameters() if p.requires_grad)}_{random_sample_num}shot.pt')

    it = 0
    while it < max_iters:
        x = dataset.load_train_data()
        x = torch.tensor(x, dtype=torch.float32, device=device)[:, :, :-1]   # (B,N,L)

        # per-sample, per-channel min-max
        x_min = x.min(dim=1, keepdim=True).values
        x_max = x.max(dim=1, keepdim=True).values
        x0 = (x - x_min) / (x_max - x_min + 1e-15)

        # condition: random N' channels
        cond = rs.random_sample(x0, 'random', random_sample_num).to(device)

        # forward
        z, log_det = model.forward(x0, cond)
        loss = 0.5 * (z**2).sum(dim=(1,2)) - log_det
        
        loss = loss.mean()
        
        # backward
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if it % log_every == 0:
            print(f"iter {it}: loss = {loss.item():.4f}, best = {best:.4f}")
            
            if loss.item() < best:
                best = loss.item()
                torch.save(model.state_dict(), ckpt_path)
                print(f"  saved best model to {ckpt_path}")
                
            # plot generation
            model.eval()
            with torch.no_grad():
                # sample from N(0,I)
                z_sample = torch.randn(batch_size, N, L, device=device) 
                
                # random cond
                cond = rs.random_sample(x0, 'random', random_sample_num).to(device)
                
                x_sample, _ = model.inverse(z_sample, cond)   # (B,N,L)
                
                # denormalize
                x_sample = x_sample * (x_max - x_min + 1e-15) + x_min
                
                x_sample = x_sample.cpu().numpy()
                
                # plot first sample's all channels
                plt.figure(figsize=(12,6))
                for i in range(N):
                    plt.plot(x_sample[0,i], alpha=0.1)
                plt.title(f"iter {it}, loss {loss.item():.4f}")
                plt.tight_layout()
                plt.savefig(os.path.join(subdir, f'sample_iter.png'))
                plt.close()
            model.train()
            
if __name__ == "__main__":
    main()
