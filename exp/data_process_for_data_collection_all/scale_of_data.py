import pickle
import pandas as pd

# path = 'exp/data_process_for_data_collection_all/all_data.pkl'

# with open(path, 'rb') as f:
#     data = pickle.load(f)
# print(data.shape)


path = 'exp/data_process_for_data_collection_all/transformer_data_15minutes.pkl'
with open(path, 'rb') as f:
    station_data = pickle.load(f)
print(station_data.shape)

path = 'exp/data_process_for_data_collection_all/new_data_15minute_solar_nomerge.pkl'
with open(path, 'rb') as f:
    new_data = pickle.load(f)
print(new_data.shape)