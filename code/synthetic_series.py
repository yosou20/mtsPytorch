#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random 
import numpy as np
import pandas as pd
from numpy import random
import os, sys
import matplotlib.pyplot as plt


# ## Creating synthetic data

# In[ ]:


# Predefined paramters for 3 order of AR(3)
random.seed(101) # utils to  generate the same random numbers
ar_n = 3                     # Order of the AR(n) data
ar_coeff = [0.7, -0.3, -0.1] # Coefficients b_3, b_2, b_1
noise_level = 0.1            # Noise added to the AR(n) data
length = 20000               # Number of data points to generate
 
# Random initial values
#ar_data1 = list(np.random.randn(ar_n))
data_list = []
# generate 30 time-series of data
for j in range(30):
    ar_data = list(np.random.randn(ar_n))   
    for i in range(length - ar_n):            
        next_val = (np.array(ar_coeff) @ np.array(ar_data[-3:])) + np.random.randn() * noise_level
        ar_data.append(next_val)
    data_list.append(ar_data)

# create dataframe from a list of the generated series
raw_data = pd.DataFrame(data_list)
# set the indices
idx_list = pd.date_range(end='2022-05-01', periods = 20000, freq= 'D')
raw_data  = raw_data.T
raw_data['Date'] = idx_list
raw_data.set_index(['Date'], inplace= True)
raw_data = raw_data.T
# Save the generated data csv file.
raw_data.to_csv('syntetic_data.csv', index=False, header=None)

#Data loading
df = pd.read_csv('syntetic_data.csv',header = None , index_col=False)
#print(df.shape)
#df.head()


# plt.show()
#df.shape
#df = np.transpose(df)
print('Number of features or sensors {}'.format(df.shape[0]))
print('Number of observations {}'.format(df.shape[1]))


# In[ ]:


df = np.transpose(df)
df


# ## the first few  series

# In[ ]:


plt.figure(figsize=(16,4))
plt.plot(df.iloc[:100,:5])
plt.title('Synthetic time-series')
plt.xlabel('Time attributes')
plt.ylabel('Serie values')
plt.legend
plt.show()


# ## Anomaly injection function

# In[ ]:


def create_anomaly(data, start_index=100,  duration=60, series_num=0):
    base_value = data.iloc[start_index, series_num]
    data.iloc[start_index:start_index + duration, series_num] = base_value     +  np.random.normal(loc=0, scale=0.8, size=duration)


# In[ ]:


start_idx_list = [300, 1200, 2000]
periods_list = [20, 60, 60]
sensor_list = [0, 16, 25]
for start_idx, period, sensor in zip(start_idx_list, periods_list, sensor_list):
    create_anomaly(df, start_idx, period, sensor)
    


# In[ ]:


fig, ax = plt.subplots(3,1, figsize=(16,8))
ax[0].set_title('Synthetic time-series with anomalies')
ax[0].plot(df.iloc[0:1000,0], color='blue')
ax[0].fill_between(np.arange(300,300+20), y1=min(df.iloc[:,0]), y2=max(df.iloc[:,0]), color='red', alpha=0.3)
ax[0].plot(df.iloc[0:1000,0], color='blue')
ax[0].set_ylabel('Series 0')
# add anomalies to senor root 16
ax[1].plot(df.iloc[0:2500,16], color='blue')
ax[1].fill_between(np.arange(1200,1200+60), y1=min(df.iloc[:,16]), y2=max(df.iloc[:,16]), color='red', alpha=0.3)
ax[1].plot(df.iloc[0:2500,16], color='blue')
ax[1].set_ylabel('Series 16')

# add anomalies to senor root 25
ax[2].plot(df.iloc[0:2500,25], color='blue')
ax[2].fill_between(np.arange(2000,2000+60), y1=min(df.iloc[:,25]), y2=max(df.iloc[:,25]), color='red', alpha=0.3)
ax[2].plot(df.iloc[0:2500,25], color='blue')
ax[2].set_ylabel('Series 25')
plt.show()


# In[ ]:


df


# In[ ]:


idx_list = pd.date_range(end='2022-05-01', periods = 20000, freq= 'D')
idx_list
#df = df.T
df['Date'] = idx_list
df.index


# In[ ]:


df.set_index(['Date'], inplace= True)
df.T


# In[ ]:


df.values


# ## Saving data to csv file format

# In[ ]:


df = pd.DataFrame(data_list)
df.to_csv('synthetic_data.csv', index=None, header=None)


# ## Creating feature matrices with a single sliding window

# In[ ]:


def feature_matrices_gen(win):
    data = pd.read_csv('syntetic_data.csv', header = None)
    gap_time = 10
    series_number = data.shape[0]
    series_length = data.shape[1]
    signature_matrices_number = int(series_length /gap_time)
    if win == 0:
        print("the size of win cannot be 0")
    raw_data = data
    raw_data= np.asarray(raw_data)
    signature_matrices = np.zeros((signature_matrices_number, series_number, series_number))

    for t in range(win, signature_matrices_number):
        raw_data_t = raw_data[:, t - win:t]
        signature_matrices[t] = np.dot(raw_data_t, raw_data_t.T) / win
    print("series_number is", series_number)
    print("series_length is", series_length)
    print("signature_matrices_number is", signature_matrices_number)

    return signature_matrices, 
    


# In[ ]:


raw_df = feature_matrices_gen(10)
#raw_df


# ## Creating feature matrices with a list of sliding windows

# In[ ]:


win_size = [10, 20, 30]
signature_matrices = []

# Generation signature matrices according the win size w
for w in win_size:
    signature_matrices.append(feature_matrices_gen(w))

signature_matrices = np.asarray(signature_matrices)
print("the shape of signature_matrices is", signature_matrices.shape)

