import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch

import asset.dataloader as dl

class TimesNetLoader():
    def __init__(self, 
                 data_path, 
                 batch_size=30, 
                 split_ration=(0.8, 0.1, 0.1),
                 full_length=366):
        
        # Define the dataloader
        self.dl = dl.Dataloader_nolabel(data_path, batch_size, split_ration)
        self.full_length = full_length  # Length of the full time series
        
    def load_train_data_times(self):
        train_data = self.dl.load_train_data()  # np.array (batch_size, 250, 25)
        
        b, l, f = train_data.shape
        
        # Full data is from 0 to full_length
        full_series = torch.zeros(b, self.full_length, f - 1, dtype=torch.float32)  # f-1 to exclude the index column
        
        # Convert train_data to a PyTorch tensor
        train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
        
        # Index of the original data with the full length
        indices = train_data_tensor[:, :, -1].long() - 1  # Indices range from 0 to full_length-1
        
        # Batch indices to match the shape of indices
        batch_indices = torch.arange(b).view(-1, 1).expand(b, l)
        
        # Use advanced indexing to fill the full_series
        full_series[batch_indices, indices] = train_data_tensor[:, :, :-1]
        
        # Create the mask for indices
        index_mask = torch.zeros(b, self.full_length, f - 1, dtype=torch.float32)
        index_mask[batch_indices, indices] = 1
        
        return full_series, index_mask # train_data
    
    def load_test_data_times(self, train_data_input = None):
        if train_data_input is None:
            train_data = self.dl.load_test_data()
        else:
            train_data = train_data_input # self.dl.load_test_data()  # np.array (batch_size, 250, 25)
        
        b, l, f = train_data.shape
        
        # Full data is from 0 to full_length
        full_series = torch.zeros(b, self.full_length, f - 1, dtype=torch.float32)  # f-1 to exclude the index column
        
        # Convert train_data to a PyTorch tensor
        train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
        
        # Index of the original data with the full length
        indices = train_data_tensor[:, :, -1].long() - 1  # Indices range from 0 to full_length-1
        
        # Batch indices to match the shape of indices
        batch_indices = torch.arange(b).view(-1, 1).expand(b, l)
        
        # Use advanced indexing to fill the full_series
        full_series[batch_indices, indices] = train_data_tensor[:, :, :-1]
        
        # Create the mask for indices
        index_mask = torch.zeros(b, self.full_length, f - 1, dtype=torch.float32)
        index_mask[batch_indices, indices] = 1
        
        return full_series, index_mask # train_data
    
if __name__ == '__main__':
    data_path = '/home/weijiexia/paper3/data_process_for_data_collection_all/all_data_aug.pkl'
    loader = TimesNetLoader(data_path)
    full_series,index_mask  = loader.load_train_data_times()
    print(full_series.shape)
