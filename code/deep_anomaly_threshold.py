import utils as util
import numpy as np
import argparse
import matplotlib.pyplot as plt
import string
import re
import math
import os
import random
import torch
import pandas as pd

# utils for visualization
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_squared_error

random.seed(101)

# get variables
tfactor = util.tfactor
gap_time = util.gap_time
valid_start = int(util.valid_start_id/gap_time)
valid_end = int(util.valid_end_id/gap_time)
test_start = int(util.test_start_id/gap_time)
test_end = int(util.test_end_id/gap_time)
thred_b = util.threhold

v_id = valid_id - valid_start  #length of validation set
t_id = test_id - test_start # length of test set

# compute anomaly score on validation and test dataset
v_anomaly_score_list = np.zeros((v_id, 1)) #stores validation scores
t_anomaly_score_list = np.zeros((t_id, 1)) # stores test scores

matrix_data_path = util.matrix_data_path
test_data_path = matrix_data_path + "test_data/"
reconst_matrix_path = matrix_data_path + "reconst_matrix/"

criterion = torch.nn.MSELoss()

def deep_anomaly_detector( valid_start, test_start, tfactor = 1.52):
    
    error_list = []
    abnormal_points = []
    threshold_temp = []
    
    for i in range(valid_start, test_end): 
        get_test_matrix = os.path.join(test_data_path, "test_data_" + str(i) + '.npy')
        load_test_matrix = np.load(get_test_matrix)
        #print(load_test_matrix.shape)
        get_reconst_dir = os.path.join(reconst_matrix_path, "reconst_matrix_" + str(i) + '.npy')
        load_reconst_matrix = np.load(get_reconst_dir)
        # get self matrix
        get_test_self_matrix = np.array(load_test_matrix)[-1][0] 
        
        # get the reconstruction self matrix
        get_reconst_self_matrix = np.array(load_reconst_matrix)[0][0]           

        #compute number of broken element in residual matrix
        get_loss_matrix = np.square(np.subtract(get_test_self_matrix, get_reconst_self_matrix))
        # Compute the maximum of reconstruction losses
        loss_max = get_loss_matrix.max(axis=1)
        # Compute the minimum of reconstruction losses		
        loss_min = get_loss_matrix.min(axis=0)
        std_loss = np.std(get_loss_matrix) # standard deviation of the losses	
        
        upper_bound = loss_max + std_loss*t_factor
        low_bound = loss_min - std_loss*t_factor
        
        # get anomaly points
        temp_anomaly =  get_loss_matrix[ get_loss_matrix > upper_bound31.max()                                     
                                        | get_loss_matrix < low_bound] #0.6
        abnormal_points.append(temp_anomaly)
        
        # get anomaly score with local threshold    
        df_temp = get_loss_matrix[get_loss_matrix > util.threhold_temp]  # 0.04 # 
        threshold_temp = len(df_temp)	
                        
        if i < valid_end:
            v_anomaly_score_list[i - valid_start] = threshold_temp
        else:
            t_anomaly_score_list[i - test_start] = threshold_temp
    v_anomaly_score_max = np.max(v_anomaly_score_list.ravel())
    t_anomaly_score_list = t_anomaly_score_list.ravel()

    print(t_anomaly_score_list)
    print(v_anomaly_score_list)
    
    anomaly_t = [list(x) for x in abnormal_points]
    anomaly_t = list(filter(lambda x: x, anomaly_t))
    return anomay_t, t_anomaly_score_list

