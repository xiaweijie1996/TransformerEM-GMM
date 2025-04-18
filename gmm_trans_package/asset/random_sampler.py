import numpy as np

def random_sample(data, strategy, random_sampe=20, num_samples_all=250):
    if strategy == 'random':
        _sample_indx = np.random.choice(range(num_samples_all), random_sampe, replace=False)
        _sample_indx = np.sort(_sample_indx)
        _sample_indx = _sample_indx.astype(np.int32)
        
        _data = data[:, _sample_indx, :]
        
    if strategy == 'top':
        _data = data[:, :random_sampe, :]
        
    if strategy == 'bottom':
        _data = data[:, -random_sampe:, :]
        
    if strategy == 'middle':
        _data = data[:, 150-random_sampe//2:150+random_sampe//2, :]

    return _data