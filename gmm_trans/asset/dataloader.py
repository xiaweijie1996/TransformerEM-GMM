import numpy as np
import pickle

class Dataloader_nolabel:
    """
    A simple dataloader for loading data without label.
    
    Args:
    - data_path: str, path to the data file
    - batch_size: int, batch size
    - split_ratio: tuple, (train_ratio, test_ratio, vali_ratio)
    
    """
    def __init__(self, 
                 data_path, 
                 batch_size=32, 
                 split_ratio=(0.8, 0.1, 0.1)):
        
        with open(data_path, 'rb') as f:
            images = pickle.load(f)
        
        self.images = images # (N, 300, 24)
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self._train_idx, self._test_idx, self._vali_idx = self._train_test_vali_split()
    
    def __len__(self):
        return len(self.images)
            
    def _train_test_vali_split(self):
        _len = self.__len__()
        _split_train = int(_len * self.split_ratio[0])
        _split_test = int(_len * self.split_ratio[1])
        train_indices = np.arange(0, _split_train)
        test_indices = np.arange(_split_train, _split_train + _split_test)
        vali_indices = np.arange(_split_train + _split_test, _len)
        return train_indices, test_indices, vali_indices
    
    def load_train_data(self):
        _sample_indx = np.random.choice(self._train_idx, self.batch_size)
        _train_data = self.images[_sample_indx]
        return _train_data.astype(np.float64)
    
    def load_test_data(self, size=64):
        _sample_indx = np.random.choice(self._test_idx, size)
        _test_data = self.images[_sample_indx]
        return _test_data.astype(np.float64)
    
    def load_vali_data(self, size=64):
        _sample_indx = np.random.choice(self._vali_idx, size)
        _vali_data = self.images[_sample_indx]
        return _vali_data.astype(np.float64)
