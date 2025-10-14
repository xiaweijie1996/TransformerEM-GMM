# train_ddpm.py
# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim

# ----------------------------
# Repo paths & data imports
# ----------------------------
_parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(_parent_path)

from asset.dataloader import Dataloader_nolabel
import asset.random_sampler as rs

# ----------------------------
# Import your DDPM module
# (must provide: FFD_NL, linear_beta_schedule, training, sampling)
# ----------------------------
import ddpm_model as ddpm


def main():
    # ===== Config =====
    batch_size        = 32
    split_ratio       = (0.8, 0.1, 0.1)
    data_path         = 'exp/data_process_for_data_collection_all/new_data_15minute_grid_nomerge.pkl'
    save_dir_root     = 'exp/exp_second_round/conditional_generativemodels'

    # model/data shapes
    N, L              = 250, 96        # (channels, length)
    random_sample_num = 8             # condition channels N'
    T                 = 300            # diffusion steps
    hidden_channels   = 240             # model hidden channels
    lr                = 2e-4
    weight_decay      = 1e-4
    max_iters         = 100000
    log_every         = 200

    # I/O
    subdir = os.path.join(save_dir_root, f'{random_sample_num}shot')
    os.makedirs(subdir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)

    # ===== Data =====
    dataset = Dataloader_nolabel(
        data_path,
        batch_size=batch_size,
        split_ratio=split_ratio
    )
    # tiny stabilizer (same as in your VAE script)
    dataset.images = dataset.images #+ np.abs(np.random.normal(0, 0.01, dataset.images.shape))

    print('length of train data: ', int(len(dataset) * split_ratio[0]))
    print('length of test  data: ', int(len(dataset) * split_ratio[1]))

    # ===== Model & Opt =====
    betas = ddpm.linear_beta_schedule(T)  # length-T schedule
    model = ddpm.FFD_NL(
        in_channels=N,
        hidden_channels=hidden_channels,
        condition_channels=random_sample_num,
        t_max=T,
        betas=betas
    ).to(device)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('amount of model parameters:', n_params)

    # ===== Precompute (on device) =====
    # (buffers are already on device; just cache square-roots for training)
    cum_alpha_sqrt = model.cumprod_alphas.sqrt()
    cum_one_minus_alpha_sqrt = (1.0 - model.cumprod_alphas).sqrt()

    best_loss = float('inf')
    ckpt_path = os.path.join(subdir, f'ddpm_{n_params}_{random_sample_num}shot.pt')

    # ===== Training loop =====
    it = 0
    while it < max_iters:
        # 1) Load a training batch -> (B, N, L+1)? (your VAE trims last step)
        train_sample = dataset.load_train_data()                       # numpy
        train_sample = torch.tensor(train_sample, dtype=torch.float32, device=device)
        train_sample = train_sample[:, :, :-1]                         # drop last step -> (B, N, L)
        assert train_sample.shape[1] == N and train_sample.shape[2] == L, \
            f"Got {train_sample.shape}, expected (B,{N},{L}) after trimming."

        # 2) Min-max normalize channel-wise per sample
        _min, _ = train_sample.min(dim=1, keepdim=True)
        _max, _ = train_sample.max(dim=1, keepdim=True)
        x0 = (train_sample - _min) / (_max - _min + 1e-15)             # (B, N, L)

        # 3) Build condition: randomly sample N' channels (and same L)
        cond = rs.random_sample(x0, 'random', random_sample_num).to(device)  # (B, N', L)
        assert cond.shape[1] == random_sample_num and cond.shape[2] == L

        B = x0.size(0)

        # 4) One optimization step on randomly sampled timesteps
        t = torch.randint(0, T, (B,), device=device, dtype=torch.long)       # (B,)
        # use training() from the ddpm module (computes eps, forward-noise, MSE)
        loss = ddpm.training(model, x0, t, cum_alpha_sqrt, cum_one_minus_alpha_sqrt, cond=cond)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        # (optional) gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        # 5) Logs, sampling preview, checkpointing
        if it % log_every == 0:
            print(f"iter {it} | loss {loss.item():.6f}")

            with torch.no_grad():
                # sampling() from ddpm module (starts from noise, uses model+cond)
                samples = ddpm.sampling(model, cond)  # (B, N, L)

            # plot the first sample (many lines; faint alpha)
            fig = plt.figure(figsize=(10, 4))
            i = 0
            plt.subplot(1, 2, 1)
            plt.plot(x0[i].detach().cpu().T, alpha=0.05)
            plt.title("x0 (orig)")
            plt.subplot(1, 2, 2)
            plt.plot(samples[i].detach().cpu().T, alpha=0.05)
            plt.title("x0 (synth)")
            plt.tight_layout()
            fig_path = os.path.join(subdir, 'preview_iter.png')
            plt.savefig(fig_path)
            plt.close(fig)

            # checkpoint best loss
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), ckpt_path)
                print("New best model saved.")

        it += 1

    print("Training finished. Best loss:", best_loss)
    print("Checkpoint saved to:", ckpt_path)


if __name__ == "__main__":
    main()
