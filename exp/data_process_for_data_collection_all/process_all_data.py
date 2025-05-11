import pickle
import pandas as pd

path = 'exp/data_process_for_data_collection_all/all_data.pkl'

with open(path, 'rb') as f:
    data = pickle.load(f)
print(data[0])

