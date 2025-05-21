import os, sys
__parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(__parent_path)

import torch

import exp_second_round.eva.eva_function as eva_f

from exp_second_round.timesnet_mse_userload_15minutes.timesnet_config import TimesBlockConfig as TimesmseConfig 
from exp_second_round.timesnet_userload_mmd_15minutes.timesnet_config import TimesBlockConfig as TimesmmdConfig 

# ----------------------Load The Models----------------------
encoder_path = 'exp/exp_second_round/user_load_15minutes/model/_encoder_40_6306166.pth'
path_para = 'exp/exp_second_round/user_load_15minutes/model/_embedding_40_6306166.pth'
path_token = 'exp/exp_second_round/user_load_15minutes/model/_emb_empty_token_40_6306166.pth'
n_components = 6
hidden_d = 96
out_d = 96
n_heads = 4
mlp_ratio = 12
n_blocks = 4
random_sample_num = 40
encoder, _para, _token = eva_f.load_model(
            n_components=n_components,
            hidden_d=hidden_d,
            out_d=out_d,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            n_blocks=n_blocks,
            encoder_path=encoder_path,
            path_para=path_para,
            path_token=path_token,
            random_sample_num=random_sample_num,
            )
bathch_size = 10
data_path = 'exp/data_process_for_data_collection_all/new_data_15minute_grid_nomerge.pkl'
dataloader = eva_f.load_valdata(data_path=data_path, batch_size=bathch_size, split_ratio=(0.8, 0.1, 0.1))

timesnetmse_path = 'exp/exp_second_round/timesnet_userload_mmd_15minutes/timesnet_MMD_1281536.pth'
timesnet_model_mse = eva_f.load_timesnet_model(
    timesnet_path=timesnetmse_path,
    configs=TimesmseConfig,
    )

timesnetmmd_path = 'exp/exp_second_round/timesnet_userload_mmd_15minutes/timesnet_MMD_1281536.pth'
timesnet_model_mmd = eva_f.load_timesnet_model(
    timesnet_path=timesnetmmd_path,
    configs=TimesmseConfig,
    )

# -----------------------Plot the figure of ECP--------------------------------
for nums in [(4, 5), (8, 9), (16, 17), (24, 25)]:
    t_samples_list, r_samples_list, r_samples_part_list, timesnet_sample_list, timesnet_sample_mmd_list = [], [], [], [], []
    for _element in range(7): 
        val_data = dataloader.load_vali_data(size=1)
        mmds,  _data = eva_f.few_shot_eva(encoder, timesnet_model_mse, timesnet_model_mmd, _para, 
                                    _token, val_data, min_shot=nums[0], max_shot=nums[1], _iter=1)
        t_samples, r_samples, r_samples_part, timesnet_sample, timesnet_sample_mmd = _data
        t_samples_list.append(t_samples)
        r_samples_list.append(r_samples)
        r_samples_part_list.append(r_samples_part)
        timesnet_sample_list.append(timesnet_sample)
        timesnet_sample_mmd_list.append(timesnet_sample_mmd)
        
    eva_f.create_plots(t_samples_list, r_samples_list, r_samples_part_list, timesnet_sample_list, timesnet_sample_mmd_list)
# -----------------------Plot the figure of ECP--------------------------------



# ----------------------Plot MMD figure---------------------------------------    
mmds,  _data = few_shot_eva(encoder, timesnet_model_mse, timesnet_model_mmd, _para, _token, val_data)    
plot_mmd_comparison(mmds[0], mmds[1], mmds[2], mmds[3])
# ----------------------Plot MMD figure---------------------------------------    
