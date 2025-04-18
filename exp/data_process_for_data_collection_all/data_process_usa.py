import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm

# Read data
df_1 = pd.read_csv('exp/data_process_for_data_collection_all/15minute_data_austin.csv')
df_2 = pd.read_csv('exp/data_process_for_data_collection_all/15minute_data_california.csv')
df_3 = pd.read_csv('exp/data_process_for_data_collection_all/15minute_data_newyork.csv')

data_type = 'grid'

# process df_1
df_1 = df_1[['dataid', 'local_15min', 'grid', 'solar', 'solar2']]
# give 0 to nan
df_1 = df_1.fillna(0)
df_1['pure_load'] = df_1[data_type] # 

# count the number of dataid
dataid_list_df1 = df_1['dataid'].unique()
print('The number of dataid in df_1 is: ', len(dataid_list_df1))
print('the range of time is: ', df_1['local_15min'].min(), df_1['local_15min'].max())

# counte the numbder of datapoints for each dataid
count_list_df1 = []
for i in dataid_list_df1:
    count_list_df1.append(len(df_1[df_1['dataid'] == i]))

# process df_2
df_2 = df_2[['dataid', 'local_15min', 'grid', 'solar', 'solar2']]
# give 0 to nan
df_2 = df_2.fillna(0)
df_2['pure_load'] = df_2[data_type]

# count the number of dataid
dataid_list_df2 = df_2['dataid'].unique()
print('The number of dataid in df_2 is: ', len(dataid_list_df2))
print('the range of time is: ', df_2['local_15min'].min(), df_2['local_15min'].max())
      
# counte the numbder of datapoints for each dataid
count_list_df2 = []
for i in dataid_list_df2:
    count_list_df2.append(len(df_2[df_2['dataid'] == i]))

# process df_3
df_3 = df_3[['dataid', 'local_15min', 'grid', 'solar', 'solar2']]
# give 0 to nan
df_3 = df_3.fillna(0)
df_3['pure_load'] = df_3[data_type] # (df_3['grid'] + df_3['solar'] + df_3['solar2'])/4

# count the number of dataid
dataid_list_df3 = df_3['dataid'].unique()
print('The number of dataid in df_3 is: ', len(dataid_list_df3))
print('the range of time is: ', df_3['local_15min'].min(), df_3['local_15min'].max())

# counte the numbder of datapoints for each dataid
count_list_df3 = []
for i in dataid_list_df3:
    count_list_df3.append(len(df_3[df_3['dataid'] == i]))

# combine df_1, df_2, df_3
df =  pd.concat([df_1, df_2, df_3], axis=0) 
df['local_15min'] = pd.to_datetime(df['local_15min'].str[:-6])
df['date'] = df['local_15min'].dt.date
df['time'] = df['local_15min'].dt.strftime('%H:%M')

# # Pivot the DataFrame to get pure load values in columns based on time
pivot_df = df.pivot_table(
    index=['dataid', 'date'], columns='time', values=data_type, aggfunc='first')
pivot_df.reset_index(inplace=True)
pivot_df.index.name = None
# Day of year
pivot_df['day'] = pd.to_datetime(pivot_df['date']).dt.dayofyear
print(pivot_df.head())

# Count the amount of dataid inn pivot_df
dataid_list_pivot = pivot_df['dataid'].unique()
print('The number of dataid in pivot_df is: ', len(dataid_list_pivot))

# Enrich the data by random sampling 10 data points for each dataid
dataid_list = pivot_df['dataid'].unique()
num_samples = 5
new_data = []
for _ in tqdm(range(num_samples)):
    
    # Randomly select a dataid
    random_dataid = np.random.choice(dataid_list, size=25, replace=False)
    
    # Randomly select a row from the pivot_df for the selected dataid
    random_row = pivot_df[pivot_df['dataid'].isin(random_dataid)]
    
    # Reorganize the data as (10, days, 96)
    create_data = 0
    add_count = 0
    for dataid in random_dataid:
        # Get the data for the selected dataid
        dataid_data = random_row[random_row['dataid'] == dataid].drop(columns=['dataid', 'date'])
        
        # Convert to numpy array and reshape
        reshaped_data = dataid_data.values.reshape(1, -1, 97)
        
        # Check if shape 1 large than 250
        if reshaped_data.shape[1] > 250:
            # Randomly select 250 data points
            random_indices = np.random.choice(reshaped_data.shape[1], size=250, replace=False)
            reshaped_data = reshaped_data[:, random_indices, :]
            create_data = create_data + reshaped_data
            add_count = add_count + 1
        else:
            pass
        
    create_data = create_data / add_count
    print(create_data.shape)
    new_data.append(create_data)

new_data = np.concatenate(new_data, axis=0)
print('new_data shape: ', new_data[0, :, :])
print('new_data shape: ', new_data.shape)

# Plot the one dataid
plt.figure(figsize=(20, 10))
plt.plot(new_data[0, :, :-1], color='blue', alpha=0.5, label='pure_load')
plt.savefig('exp/data_process_for_data_collection_all/15minute_data.png')
