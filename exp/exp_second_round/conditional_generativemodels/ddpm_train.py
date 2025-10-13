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
# must provide: FFD_NL, prepare_buffers, forward_diffusion_sample, sample_from_noise
# ----------------------------
import ddpm_model as ddpm  # <- you said to use: import ddpm

def main():
    # ===== Config =====
    batch_size        = 32
    split_ratio       = (0.8, 0.1, 0.1)
    data_path         = 'exp/data_process_for_data_collection_all/new_data_15minute_grid_nomerge.pkl'
    save_dir          = 'exp/exp_second_round/conditional_generativemodels'
    os.makedirs(save_dir, exist_ok=True)

    # model/data shapes
    N, L              = 250, 96        # (channels, length)
    random_sample_num = 32             # condition channels N'
    T                 = 300         # diffusion steps
    hidden_channels   = 24             # model hidden channels
    lr                = 2e-4
    weight_decay      = 1e-4
    max_iters         = 20000
    log_every         = 100
    sample_steps      = 50             # fast sampler preview; for best quality use None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)

    # ===== Data =====
    dataset = Dataloader_nolabel(
        data_path,
        batch_size=batch_size,
        split_ratio=split_ratio
    )
    # tiny stabilizer (same as in your VAE script)
    dataset.images = dataset.images + np.abs(np.random.normal(0, 0.01, dataset.images.shape))

    print('length of train data: ', int(len(dataset) * split_ratio[0]))
    print('length of test  data: ', int(len(dataset) * split_ratio[1]))

    # ===== Model & Opt =====
    model = ddpm.FFD_NL(
        in_channels=N,
        hidden_channels=hidden_channels,
        condition_channels=random_sample_num,
        t_max=T
    ).to(device)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('amount of model parameters:', n_params)

    # ===== Schedules =====
    buf = ddpm.prepare_buffers(T, device)
    sqrt_a_bar = buf["a_bar"].sqrt()
    sqrt_one_minus_a_bar = (1.0 - buf["a_bar"]).sqrt()

    best_loss = float('inf')
    ckpt_path = os.path.join(save_dir, f'ddpm_{n_params}_{random_sample_num}shot.pt')

    # ===== Training loop =====
    it = 0
    while it < max_iters:
        # 1) Load a training batch (shape: (B, N, L+1)? your VAE trims last step)
        train_sample = dataset.load_train_data()            # numpy
        train_sample = torch.tensor(train_sample, dtype=torch.float32, device=device)
        train_sample = train_sample[:, :, :-1]              # match your VAE: drop last time step -> (B, N, L)

        # 2) Min-max normalize channel-wise per sample
        _min, _ = train_sample.min(dim=1, keepdim=True)
        _max, _ = train_sample.max(dim=1, keepdim=True)
        x0 = (train_sample - _min) / (_max - _min + 1e-15)  # (B, N, L)

        # 3) Build condition: randomly sample N' channels (and same L)
        cond = rs.random_sample(x0, 'random', random_sample_num).to(device)  # (B, N', L)

        B = x0.size(0)

        # 4) One optimization step on randomly sampled timesteps
        t = torch.randint(0, T, (B,), device=device)
        x_t, noise = ddpm.forward_diffusion_sample(x0, t, sqrt_a_bar, sqrt_one_minus_a_bar)
        eps_pred = model(x_t, t, cond=cond)
        loss = F.mse_loss(eps_pred, noise)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # 5) Logs, sampling preview, checkpointing
        if it % log_every == 0:
            print(f"iter {it} | loss {loss.item():.6f}")

            with torch.no_grad():
                # quick preview sampling conditioned on same cond
                samples = ddpm.sample_from_noise(
                    model, cond, N=N, L=L, T=T,
                    steps=sample_steps, device=device, clip=(-3, 3)
                )  # (B, N, L)

            # plot first sample
            fig = plt.figure(figsize=(10, 4))
            i = 0
            # print("x0[i].shape:", x0[i].shape)
            plt.subplot(1, 2, 1)
            plt.plot(x0[i].detach().cpu(), alpha=0.05)
            plt.title("x0 (orig)")
            plt.subplot(1, 2, 2)
            plt.plot(samples[i].detach().cpu(), alpha=0.05)
            plt.title("x0 (synth)")
            plt.tight_layout()
            fig_path = os.path.join(save_dir, f'{random_sample_num}shot/preview_iter.png')
            plt.savefig(fig_path)
            plt.close(fig)

        #     # checkpoint best loss
        #     if loss.item() < best_loss:
        #         best_loss = loss.item()
        #         torch.save({
        #             "model": model.state_dict(),
        #             "opt": opt.state_dict(),
        #             "iter": it,
        #             "best_loss": best_loss,
        #             "config": {
        #                 "N": N, "L": L, "T": T, "random_sample_num": random_sample_num,
        #                 "lr": lr, "weight_decay": weight_decay
        #             }
        #         }, ckpt_path)

        it += 1

    print("Training finished. Best loss:", best_loss)
    print("Checkpoint saved to:", ckpt_path)

if __name__ == "__main__":
    main()
