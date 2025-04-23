import pandas as pd
import numpy as np
import pickle

path = "exp/data_process_for_data_collection_all/transformer_data.csv"

# 1. Load and parse timestamps
df = pd.read_csv(path)
df['date'] = pd.to_datetime(df['date'], utc=True)

# 2. Day features
df['day_of_year'] = df['date'].dt.dayofyear
df['day']        = df['date'].dt.normalize()   # midnight of that date
# The time of the data
df['time']       = df['date'].dt.time

# 3. Sort so each day’s 96 points are in order
df.sort_values(['day','date', 'time'], inplace=True)

# 4. Create a list of columns for each substation
skip = {'date','day','day_of_year', 'time'}
station_cols = [c for c in df.columns if c not in skip]


# 5. For each substation, group by day_of_year → stack into a (d,96) matrix
station_list = []
for col in station_cols:
    _df = df[['day_of_year','time', col]]
    
    # 5.1. Pivot the data
    _df = _df.pivot(index='day_of_year', columns='time', values=col)
    _df = _df.reset_index()
    _df = _df.rename(columns={'day_of_year': 'day'})
    
    # Put day at the end column
    cols = _df.columns.tolist()
    cols = cols[1:] + [cols[0]]
    _df = _df[cols]
    
    # Cancel rows with nan
    _df = _df.dropna()
    
    # Check the number of rows and randomly sampe 250 rows
    if len(_df) > 300:
        _df1 = _df.sample(250, random_state=1)
        station_list.append(_df1.values.reshape(1, -1, 97))
        _df1 = _df.sample(250, random_state=2)
        station_list.append(_df1.values.reshape(1, -1, 97))
        
    # Check the number of rows and randomly sample 250 rows
    elif len(_df) < 300 and len(_df) > 150:
        _df1 = _df.sample(250, random_state=1, replace=True)
        station_list.append(_df1.values.reshape(1, -1, 97))
        print(f"Station {col} has {len(_df)} rows, sampled {len(_df1)} rows")
    
# 6. Concatenate all the data to have shape (n, 250, 96)
station_data = np.concatenate(station_list, axis=0)


print(station_data.shape)

# 7. Save the data
with open('exp/data_process_for_data_collection_all/transformer_data_15minutes.pkl', 'wb') as f:
    pickle.dump(station_data, f)
