# import package utils for the project
import  utils as util
import numpy as np
import pandas as pd
import os, sys
import math
import scipy
import matplotlib.pyplot as plt
import itertools as it
import string
import re
import random

# import data normalization
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#import data visualization
import seaborn as sns
import plotly.express as px
from sklearn.metrics import mean_squared_error


#========================================================================

# initialized the variables
window_size = util.window_size
step_size = util.step_max
min_index = util.min_index
max_index = util.max_index
gap_time = util.gap_time


#load raw data paths
raw_synth_ts_path = util.raw_synth_ts_path
raw_synth_ts_path1 = util.raw_synth_ts_path1
raw_real_ts_path =  util.raw_real_ts_path
#train_ts_path = util.train_ts_path
#test_ts_path = util.test_ts_path
reconstructed_ts_path = util.reconstructed_ts_path
save_data_path = util.save_data_path

# train test split
train_start =util.train_start_id
train_end =  util.train_end_id

test_start = util.test_start_id
test_end = util.test_end_id

# Synthetic data generation and preparation
# Predefined paramters
random.seed(101) # set random seed
ar_n = 3                     # Create an AR(3) order of data
ar_coeff = [0.7, -0.3, -0.1] # set the coefficients b_3, b_2, b_1
noise_level = 0.1            # Noise added to the AR(n) data
length = 20000               # Number of data points to generate

data_list = [] # time-series list
#create a multivariate time-series with 30 features.
for j in range(30):
    ar_data = list(np.random.randn(ar_n))
    #mat_col = np.zeros((3, length-ar_n))    
    for i in range(length - ar_n):            
        next_val = (np.array(ar_coeff) @ np.array(ar_data[-3:])) + np.random.randn() * noise_level
        ar_data.append(next_val)
    data_list.append(ar_data)

# Reindex and save time-series to csv file format
raw_data = pd.DataFrame(data_list)
idx_list = pd.date_range(end='2022-05-01', periods = 20000, freq= 'D')
raw_data  = raw_data.T
raw_data['Date'] = idx_list
raw_data.set_index(['Date'], inplace= True)
raw_data = raw_data.T
raw_data.to_csv('syntetic_data.csv', index=False, header=None)
#print(max(window_size))
#Data loading
#df = pd.read_csv('syntetic_data.csv',header = None, index_col=False)

f_matrix_ts_path = save_data_path + "matrix_data/"
if not os.path.exists(f_matrix_ts_path):
    os.makedirs(f_matrix_ts_path)

# Creating feature and self-matrices
def feature_and_self_matrices_generator():
    #Data loading
    df = pd.read_csv(raw_synth_ts_path1, header = None, index_col=False)
    df = np.array(df, dtype=np.float64)
    #print(df.shape[0])
    # perform MinMaxScaler normalization 
    
    # scaler = MinMaxScaler()     
    # scaler = scaler.fit(df)
    # df = scaler.transform(df)

    max_value = np.max(df, axis=1)
    min_value = np.min(df, axis=1)
    df = (np.transpose(df) - min_value)/(max_value - min_value + 1e-6)
    df = np.transpose(df) 
    n_feature = df.shape[0]   # number of features in the dataset
    n_obs = df.shape[1]       # number of observations
    print("Number of features or time-series is: {}".format(n_feature))
    print("Number of observation: {}".format(n_obs))
    print("Number of feature matrices is {}".format(int(n_obs/gap_time)))
    
    print("================================================")
    print("Generating feature matrices....")
    print("================================================")
    #print(df)    
    # standard normalization
    
    # scaler = StandardScaler()
    # scaler = scaler.fit(df)
    # df = scaler.transform(df)
    # print(df)
    # Generate features matrices for range of window size
    
    for win in window_size:
        matrix_list = []
        print("Feature matrices for window size " + str(win)+ " are created!")
        for t in range(min_index, max_index, gap_time):
            # initialize feature matrix at t
            f_matrix = np.zeros((n_feature, n_feature))
            if t >= max(window_size):
                for  i in range(n_feature):
                    for j in range(n_feature):
                        f_matrix[i, j] = np.inner(df[i, t-win:t], df[j,t - win:t])/win
                        f_matrix[j, i] = f_matrix[i, j]
            matrix_list.append(f_matrix)
        f_matrix_path = f_matrix_ts_path + "matrix_win_" + str(win)
        np.save(f_matrix_path, matrix_list)
        del matrix_list[:]
    print("Feature matrix generation complete")

# generate feature matrices using single window-size
def signature_matrices_generation(df, win):
   
    df = raw_data
    if win == 0:
        print("The size of win cannot be 0")

    raw_data = np.asarray(raw_data)
    n_feature_matrices = raw_data.shape[1]/gap_time
    signature_matrices = np.zeros(raw_data.shape[1]/gap_time, raw_data[0], raw_data[0])

    for t in range(win, n_feature_matrices):
        raw_data_t = raw_data[:, t - win:t]
        signature_matrices[t] = np.dot(raw_data_t, raw_data_t.T) / win

    return signature_matrices

def train_test_split_generator():
    
    print("================================================")
    print("Generating train test split series....")
    print("================================================")
    
    # create train data folder
    f_matrix_ts_path = save_data_path + "matrix_data/"
    train_ts_path = f_matrix_ts_path + "train_data/"
    if not os.path.exists(train_ts_path):
        os.makedirs(train_ts_path)
    # create test data folders
    test_ts_path = f_matrix_ts_path +"test_data/"
    if not os.path.exists(test_ts_path):
        os.makedirs(test_ts_path)
    
    #data from the windows list 
    
    matrix_list = []    
    for w in range(len(window_size)):        
        win_data_path = f_matrix_ts_path + "matrix_win_" + str(window_size[w]) + ".npy"
        matrix_list.append(np.load(win_data_path))
    
    train_test_list = [[train_start, train_end], [test_start, test_end]]
    for d in range(len(train_test_list)):
        for idx in range(int(train_test_list[d][0]/gap_time), int(train_test_list[d][1]/gap_time)):
            # set sequence step
            step_matrix_list = []
            for step_idx in range(step_size, 0, -1):
                win_matrix = []
                for i in range(len(window_size)):
                    win_matrix.append(matrix_list[i][idx - step_idx])
                step_matrix_list.append(win_matrix)
            if idx >= (train_start/(gap_time) + window_size[-1]/(gap_time) + step_size) and idx < (train_end/gap_time):
                dest_dir = os.path.join(train_ts_path, 'train_data_' + str(idx))
                np.save(dest_dir, step_matrix_list)
            elif idx >= (test_start/(gap_time)) and idx < (test_end/(gap_time)):
                dist_dir = os.path.join(test_ts_path, 'test_data_'+ str(idx))
                np.save(dist_dir, step_matrix_list)
            
            del step_matrix_list[:]
    print("Train test split  generation is finished!.")


if __name__ == '__main__':
    feature_and_self_matrices_generator()
    train_test_split_generator()
    
    