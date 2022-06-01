#!/usr/bin/env python
# coding: utf-8

# In[70]:


import random 
import numpy as np
import pandas as pd
from numpy import random
import os, sys
import matplotlib.pyplot as plt
import glob # utils for reading multiple files from folder
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#  ## Gas  data loading

# In[44]:


#!pip3 install glob2


# In[82]:


path='C:/Users/souare/OneDrive - Syddansk Universitet/Master_thesis_docs/MasterProjectCode/gas_data/'
#data_list = os.listdir('./gas_data')
#path = 'C/data/gas_data/'
all_files = glob.glob(path + "*.csv")
df_list = []
for file in all_files:
    #print(file)
    df = pd.read_csv(file, header= 0,  index_col=False)
    df.reset_index(inplace=True)
    df.drop(['index','Time (s)'], inplace=True, axis=1)  # remove index for now
    #df.drop(['index'], inplace=True, axis = 1)
    df.columns = [''] * len(df.columns)
    df_list.append(df)
df = pd.concat(df_list)
df.columns = [x for x in range(0, len(df.columns))]

print('Number of features or sensors {}'.format(df.shape[1]))
print('Number of observations {}'.format(df.shape[0]))


# ## Normalizing the data

# In[89]:


max_value = np.max(df, axis=1)
min_value = np.min(df, axis=1)
df = (np.transpose(df) - min_value)/(max_value - min_value + 1e-6)
df = np.transpose(df) 
df.shape


# In[90]:


df = pd.DataFrame(df)


# In[91]:


df


# ## Adding random noise to data

# In[92]:


def create_anomaly(data, start_index=100,  duration=60, series_num=0):
    base_value = data.iloc[start_index, series_num]
    data.iloc[start_index:start_index + duration, series_num] = base_value     +  np.random.normal(loc=0, scale=0.8, size=duration)


# In[93]:


start_idx_list = [300, 1200, 2000]
periods_list = [20, 60, 60]
sensor_list = [1, 16, 18]
for start_idx, period, sensor in zip(start_idx_list, periods_list, sensor_list):
    create_anomaly(df, start_idx, period, sensor)


# In[94]:


df


# In[95]:


fig, ax = plt.subplots(3,1, figsize=(16,8))
ax[0].set_title('Synthetic time-series with anomalies')
ax[0].plot(df.iloc[0:1000,1], color='blue')
ax[0].fill_between(np.arange(300,300+20), y1=min(df.iloc[:,1]), y2=max(df.iloc[:,1]), color='red', alpha=0.3)
ax[0].plot(df.iloc[0:1000,1], color='blue')
ax[0].set_ylabel('Series 1')
# add anomalies to senor root 16
ax[1].plot(df.iloc[0:2500,16], color='blue')
ax[1].fill_between(np.arange(1200,1200+60), y1=min(df.iloc[:,16]), y2=max(df.iloc[:,16]), color='red', alpha=0.3)
ax[1].plot(df.iloc[0:2500,16], color='blue')
ax[1].set_ylabel('Series 16')

# add anomalies to senor root 18
ax[2].plot(df.iloc[0:2500,18], color='blue')
ax[2].fill_between(np.arange(2000,2000+60), y1=min(df.iloc[:,18]), y2=max(df.iloc[:,18]), color='red', alpha=0.3)
ax[2].plot(df.iloc[0:2500,18], color='blue')
ax[2].set_ylabel('Series 18')
plt.show()


# In[103]:


#df.iloc[299:350,1]


# In[100]:


df


# ## saving the data to  csv file

# In[104]:


df.to_csv('gas_data_abnr.csv', index=None, header=None)


# In[ ]:




