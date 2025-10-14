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

@torch.no_grad()
def sample_from_flow(model, cond, N, L, temperature=1.0):
    device = cond.device
    B = cond.size(0)
    z = torch.randn(B, N, L, device=device) * temperature
    x, _ = model.inverse(z, cond)
    return x

def main():
    # config
    batch_size = 32
    split_ratio = (0.8, 0.1, 0.1)
    data_path = 'exp/data_process_for_data_collection_all/new_data_15minute_grid_nomerge.pkl'
    save_dir_root = 'exp/exp_second_round/conditional_generativemodels'
    N, L = 250, 96
    random_sample_num = 4
    hidden_channels = 256
    K_blocks = 8
    use_1x1 = True
    lr = 2e-4
    weight_decay = 1e-4
    max_iters = 100000
    log_every = 200
    temperature = 1.0

    subdir = os.path.join(save_dir_root, f'{random_sample_num}shot_flow')
    os.makedirs(subdir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)

    # data
    dataset = Dataloader_nolabel(data_path, batch_size=batch_size, split_ratio=split_ratio)

    # model & opt
    model = nf.Flow1D(
        channels=N,
        cond_channels=random_sample_num,
        hidden_channels=hidden_channels,
        K=K_blocks,
        use_1x1=use_1x1,
        clamp=2.0,
    ).to(device)
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

        nll = model.nll(x0, cond, reduce=True)
        opt.zero_grad(set_to_none=True)
        nll.backward()
        opt.step()

        if it % log_every == 0:
            print(f"iter {it} | nll {nll.item():.6f}")

            model.eval()
            with torch.no_grad():
                samples = sample_from_flow(model, cond, N=N, L=L, temperature=temperature)

                # quick preview
                i = 0
                plt.figure(figsize=(10,4))
                plt.subplot(1,2,1); plt.plot(x0[i].detach().cpu().T, alpha=0.05); plt.title("x0 (orig)")
                plt.subplot(1,2,2); plt.plot(samples[i].detach().cpu().T, alpha=0.05); plt.title("x0 (synth)")
                plt.tight_layout()
                plt.savefig(os.path.join(subdir, 'preview_iter.png'))
                plt.close()
            model.train()

            if nll.item() < best:
                best = nll.item()
                torch.save(model.state_dict(), ckpt_path)
                print("saved best")

        it += 1

    print("done | best nll:", best)
    print("ckpt:", ckpt_path)

if __name__ == "__main__":
    main()
