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

tfactor = util.tfactor
gap_time = util.gap_time
valid_start = int(util.valid_start_id/gap_time)
valid_end = int(util.valid_end_id/gap_time)
test_start = int(util.test_start_id/gap_time)
test_end = int(util.test_end_id/gap_time)
thred_b = util.threhold

# compute anomaly score on validation and test dataset
valid_anomaly_score = np.zeros((valid_end - valid_start, 1)) #stores validation score
test_anomaly_score = np.zeros((test_end - test_start, 1))

matrix_data_path = util.matrix_data_path
test_data_path = matrix_data_path + "test_data/"
reconstructed_data_path = matrix_data_path + "reconstructed_data/"
criterion = torch.nn.MSELoss()
error_list = []
abnormal_points = []
threshold_temp = []
for i in range(valid_start, test_end): # range from test start to the end
	path_temp_1 = os.path.join(test_data_path, "test_data_" + str(i) + '.npy')
	gt_matrix_temp = np.load(path_temp_1)
	#print(gt_matrix_temp.shape)

	path_temp_2 = os.path.join(reconstructed_data_path,
	                           "reconstructed_data_" + str(i) + '.npy')
	#path_temp_2 = os.path.join(reconstructed_data_path, "pcc_matrix_full_test_" + str(i) + '_pred_output.npy')
	reconstructed_matrix_temp = np.load(path_temp_2)	
	# reconstructed_matrix_temp = np.transpose(reconstructed_matrix_temp, [0, 3, 1, 2])
	#print(reconstructed_matrix_temp.shape)
	#first (short) duration scale for evaluation
	select_gt_matrix = np.array(gt_matrix_temp)[-1][0]  # get last step matrix

	select_reconstructed_matrix = np.array(reconstructed_matrix_temp)[0][0]   
    

	#compute number of broken element in residual matrix
	select_matrix_error = np.square(np.subtract(select_gt_matrix, select_reconstructed_matrix))
	# Compute the maximum of reconstruction errors
	res = select_matrix_error.max(axis=1)
	#print(res.mean().max().max())
	#print(res.CooksDistance()) # it doen't work yet
	res_min = select_matrix_error.min(axis=0)
	std_res = np.std(res)
	#print(res_min)
	
	mean_test_data =np.mean(select_gt_matrix, axis=1)
	#print(res)
	upper_bound = res + select_gt_matrix 
 
	upper_bound3 = mean_test_data + std_res
	#upper_bound31 = upper_bound3.max()*1.5
	upper_bound31 = upper_bound3.max()
	#print(upper_bound31)
	#upper_bound2= upper_bound.max(axis=1)*0.9998
	#print(upper_bound2)
	#low_bound = res_min - std_res*0.5	
	#print('low bound',low_bound)
	#print('max std',std_res)
	#print(np.std(select_matrix_error))
	#error_list.append(select_matrix_error)
	#select_matrix_error
	df_temp = select_matrix_error[select_matrix_error > 0.030]  # 0.04 # Thred_b
	#error_list.append(df_temp)
	temp_anomaly = select_gt_matrix[select_gt_matrix > upper_bound31.max()] #0.6
	abnormal_points.append(temp_anomaly)
	#print(temp_anomaly) 
	#print(len(temp_anomaly))
	#print(len(select_gt_matrix))
	#print(select_gt_matrix)
	threshold_temp.append(np.abs(np.std(res).max(axis=0)))
	
 
	#df_temp = select_matrix_error[select_matrix_error > upper_bound]
 	
	#num_broken = len(select_matrix_error[select_matrix_error >  thred_b])
	#num_broken2 = len(select_matrix_error[select_matrix_error > upper_bound31.max()*0.02 ]) #0.0055
	num_broken =len(df_temp)	
	#print(np.abs(np.std(res).max(axis=0)))
	#print('threshold1',num_broken)
	#print(num_broken2)
	
 
	

	#print num_broken
	if i < valid_end:
		valid_anomaly_score[i - valid_start] = num_broken
	else:
		test_anomaly_score[i - test_start] = num_broken
valid_anomaly_max = np.max(valid_anomaly_score.ravel())
test_anomaly_score = test_anomaly_score.ravel()

print(test_anomaly_score)
#print(valid_anomaly_score.ravel())



#Check the number of files in the test folder
list2 = os.listdir(test_data_path)
number_files = len(list2)
#print(number_files)

# Check the length of the abnormality list
#print('number of abnormal points',len(abnormal_points))

#check the indices of all the points
#for abn in abnormal_points:
#   print('index {}, element {}'.format(abnormal_points.index(abn), abn))

#print(len(temp_anomaly))
#filtering the number of points
anomaly_t = [list(x) for x in abnormal_points]
anomaly_t = list(filter(lambda x: x, anomaly_t))
#print(anomaly_t)
#print(len(anomaly_t))
    
#print(error)
#print(len(error))
#print(x[0:400])
#print(len(select_gt_matrix))   
#print(len(select_reconstructed_matrix)) 
#print(len(select_matrix_error)) 

#print(len(error_list[error_list> thred_b]))

#plot anomaly score curve and identification result

anomaly_pos = np.zeros(5)
root_cause_gt = np.zeros((5, 3))
anomaly_span = [10, 30, 90]
root_cause_f = open("../data/test_anomaly.csv", "r")
row_index = 0
for line in root_cause_f:
    line=line.strip()
    anomaly_axis = int(re.split(',',line)[0])
    anomaly_pos[row_index] = anomaly_axis/gap_time - test_start - anomaly_span[row_index%3]/gap_time
    print(anomaly_pos[row_index])
    root_list = re.split(',',line)[1:]
    for k in range(len(root_list)-1):
        root_cause_gt[row_index][k] = int(root_list[k])
    row_index += 1
root_cause_f.close()

fig, axes = plt.subplots(figsize=(16,7))
#plt.plot(test_anomaly_score, 'black', linewidth = 2)
test_num = test_end - test_start
print(test_num)
#plt.xticks(fontsize = 13)
plt.ylim((0, 100))
#plt.yticks(np.arange(0, 101, 20), fontsize = 13)
plt.plot(test_anomaly_score, color = 'black', linewidth = 1)
threshold = np.full((test_num), valid_anomaly_max*0.50)
axes.plot(threshold, color = 'green', linestyle = '--',linewidth = 1)

for k in range(len(anomaly_pos)):
   axes.axvspan(anomaly_pos[k], anomaly_pos[k] + anomaly_span[k%3]/gap_time, color='red', linewidth=2)
   # labels = [' ', '0e3', '2e3', '4e3', '6e3', '8e3', '10e3']
   #axes.set_xticklabels(labels, rotation = 25, fontsize = 15)

plt.xlabel('Time-series indices', fontsize = 13)
plt.ylabel('Test Anomaly score', fontsize = 13)
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
axes.yaxis.set_ticks_position('left')
axes.xaxis.set_ticks_position('bottom')
fig.subplots_adjust(bottom=0.25)
fig.subplots_adjust(left=0.25)
plt.title("SACLAD-MTS", size = 15)
plt.savefig('../outputs/anomaly_score.jpg')
plt.show()

xp = [t for t in range(test_num)]

#px.line(test_anomaly_score, xp)
    

def deep_anomaly_detector(valid_start, test_end, tfactor =1.52):
    abnormal_points1 = []
    anomaly_dict = {}
    for i in range(valid_start, test_end):
        
        path_temp_1 = os.path.join(test_data_path, "test_data_" + str(i) + '.npy')
        gt_matrix_temp = np.load(path_temp_1)
        path_temp_2 = os.path.join(reconstructed_data_path,
                            "reconstructed_data_" + str(i) + '.npy')
        reconstructed_matrix_temp = np.load(path_temp_2)
        #get last ste matrix from array of test patrices
        select_gt_matrix = np.array(gt_matrix_temp)[-1][0] 
		#get a list of re-constructed matrices for first sliding window win = 10
        select_reconstructed_matrix = np.array(reconstructed_matrix_temp)[0][0]
        #reconstructed_matrix_temp = np.load(path_temp_2)
        #select_gt_matrix = np.array(gt_matrix_temp)[-1][0]  # get last step matrix

        #select_reconstructed_matrix = np.array(reconstructed_matrix_temp)[0][0]   
    

		#compute number of broken element in residual matrix
        select_matrix_error = np.square(np.subtract(select_gt_matrix, select_reconstructed_matrix))
        res = select_matrix_error.max(axis=1)
       	#res_min = select_matrix_error.min(axis=1)
        #std_res = np.std(res)

       	mean_test_data =np.mean(select_gt_matrix, axis=1)
		#print(res)
        #upper_bound = res + select_gt_matrix 

        upper_bound3 = mean_test_data + std_res
       	upper_bound31 = upper_bound3.max()*tfactor
		#print(upper_bound31)
     	#upper_bound2= upper_bound.max(axis=1)*0.9998
		#print(upper_bound2)
  		#low_bound = res_min - std_res*0.5	
		#0.6
       	temp_anomaly = select_gt_matrix[select_gt_matrix > upper_bound31.max()]
        #print (i, end = " ")
        #print(temp_anomaly)
        anomaly_dict[i] = temp_anomaly
        #print(anomaly_dict)
        
        #print (temp_anomaly[i + valid_start])
        abnormal_points1.append(temp_anomaly)
    anomaly_t = [list(x) for x in abnormal_points1]
    anomaly_t = list(filter(lambda x: x, anomaly_t))
    df = pd.DataFrame(anomaly_t, columns=['abnormal'])
    df['position'] = [x for x in range(len(anomaly_t))]
    
    dct = {k: None if not v else v for k, v in anomaly_dict.items() }
    #print(dct)
    
    for key, value in dct.items():
        if value is None:
            value = 0
        dct[key] = value
    #print(dct.values())
    dict_df = pd.DataFrame(dct.items())
    dict_df.columns = ['Index', 'Value']
    dict_df['Value'] = dict_df['Value'].str[0] # remove the array bracket.
    #print("------print dict------")
    #print(dict_df)
    #dict_df.plot()
    
    #print(anomaly_dict.values())    
    
    
    

    
    
    # df.set_index('index', inplace=True)
    return df, dict_df

if __name__ == '__main__':
    df, dict_df =deep_anomaly_detector(valid_start, test_end, 1.95) #1.98
    # Good the print this here !!!!
    print(df.values)
    print(df.index)
    print(len(df))
    print(df)
    
    #fig = px.scatter(df, 'position', 'abnormal')
    print("----anomaly dictionary----")
    print(dict_df[dict_df['Value']>0]) 
    #fig = px.line(test_anomaly_score)   
    #fig = px.line(threshold)
    #fig = px.scatter(dict_df,  dict_df['Index'], dict_df['Value'])
    #fig.show()

	#df.plot.scatter('position','abnormal', color='red')

    #plt.show()
    
   # print(test_anomaly_score['Index'])
    
    xa = [x for x in range(0, 1201)]
    print(np.max(xa))

    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter( x= xa, y = test_anomaly_score,mode='lines', name='lines'))
    fig.add_trace(go.Scatter(x = xa, y = threshold,
                             mode='lines', name='lines'))
#fig.add_trace(px.line(threshold))


    fig.show()