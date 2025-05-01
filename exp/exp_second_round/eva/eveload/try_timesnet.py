import os
import sys
_parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(_parent_path)

import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import asset.timesnet_loader as timesloader
import exp_second_round.timesnet_mse_userload_15minutes.timesnet_utils as ut
from exp_second_round.timesnet_mse_userload_15minutes.timesnet_config import TimesBlockConfig 
import exp_second_round.timesnet_mse_userload_15minutes.timesnet_train as tt
import exp_second_round.timesnet_mse_userload_15minutes.timesnet as timesnet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# check the model
configs = TimesBlockConfig()
model = timesnet.Model(configs).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

# print number of parameters
num_params = sum(p.numel() for p in model.parameters())
print(num_params)

data_path = 'exp/data_process_for_data_collection_all/new_data_15minute_grid_merge.pkl'
data_loader = timesloader.TimesNetLoader(data_path, 
                                            batch_size=60, 
                                            split_ration=(0.8, 0.1, 0.1),
                                            full_length=366)

model_path = 'exp/exp_second_round/timesnet_mse_userload_15minutes/30mr_timesnet_1281536.pth'
model.load_state_dict(torch.load(model_path, map_location=device))

sub_epoch = int(50000/60)
epoch = 1000000
max_loss = 10000

model.eval()
with torch.no_grad():
    for _test in range(10):
        full_series, index_mask = data_loader.load_test_data_times()
        print('full_series shape: ', full_series.shape)
        test_data, random_mask, scaler = tt.normalize_and_mask(full_series, index_mask, device)
        print('test_data type: ', type(test_data))  
        print('test_data shape: ', test_data.shape)
        print('random_mask shape: ', random_mask.shape)
        y_hat = model(test_data, None, random_mask)

        _min = scaler[0]
        _max = scaler[1]
        
        # Scale back to original
        y_hat = y_hat * (scaler[1] - scaler[0]) + scaler[0]
        
        mask_bt = 1- index_mask.sum(dim=2) 
        mask_bt = mask_bt > 0
        y_filtered_per_sample = [
            y_hat[i, mask_bt[i], :]           # Tensor of shape [Z_i, F]
            for i in range(y_hat.shape[0])
        ]
        print('y_filtered_per_sample shape: ', y_filtered_per_sample[0].shape)
        
        # Plot the sample
        # for i in range(1):
        #     plt.figure(figsize=(10, 5))
        #     plt.plot(y_filtered_per_sample[i].cpu().numpy().T, alpha=0.2)
        #     plt.title(f"Sample {i} - Filtered Output")
        #     plt.xlabel("Time Step")
        #     plt.ylabel("Value")
        #     plt.legend(['y_hat'])
        #     plt.savefig(f'exp/exp_second_round/eva/eveload/plot/30mr_timesnet_{num_params}_sample_{i}.png')
        #     plt.close()
        
        # # Plot orginal data
        # for i in range(1):
        #     plt.figure(figsize=(10, 5))
        #     plt.plot(full_series[i, :, :].cpu().numpy().T, alpha=0.2)
        #     plt.title(f"Sample {i} - Original Data")
        #     plt.xlabel("Time Step")
        #     plt.ylabel("Value")
        #     plt.legend(['x'])
        #     plt.savefig(f'exp/exp_second_round/eva/eveload/plot/30mr_timesnet_{num_params}_sample_{i}_original.png')
        #     plt.close()
        break
