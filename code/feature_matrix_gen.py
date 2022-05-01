import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

#utils for data normalization
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

#utils for importing data
import yfinance as yf
from yahoofinancials import YahooFinancials



# declare variables utils

parser = argparse.ArgumentParser(
    description = "Feature and self-matrces generation functions")
#parser.add_argument('--time_series', type= str,
                                # default="node", help="time-series type")
parser.add_argument('--step_max', type= int, default=5,
                    help="maximum step in ConvLSTM")
parser.add_argument('--gap_time', type= int, default=10,
                    help="gap time between each segment")
parser.add_argument('--window_size', type= int, default= [10, 20, 30], #[10, 20, 30],
                    help="window size of each segment")
parser.add_argument('--min_time', type= int, default=0,
                    help="minimum time point")
parser.add_argument('--max_time', type= int, default= 2781, # 400, #20000,
                    help="maximum time point")
parser.add_argument('--train_start_point', type= int, default=0,
                    help="train starting point")
parser.add_argument('--train_end_point', type= int, default= 2000, #300,  #8000,
                    help="train end point")
parser.add_argument('--test_start_point', type= int, default= 2000,# 300, #8000,
                    help="test starting point")
parser.add_argument('--test_end_point', type= int, default= 2781, # 400,#20000,
                    help="test end point")
#parser.add_argument('--raw_data_path', type= str, default = "../data/synthetic_data_with_anomaly-s-1.csv",
#                    help="path to raw data")
parser.add_argument('--save_data_path', type= str, default ="../data/",
                    help="path to sva data")

#get variables
args = parser.parse_args()

#print(args)
#time_series = args.time_series
step_max = args.step_max
min_time = args.min_time
max_time = args.max_time
gap_time = args.gap_time
window_size = args.window_size

# get train and test parameters

train_start = args.train_start_point
train_end = args.train_end_point

test_start = args.test_start_point
test_end = args.test_end_point

#raw_data_path = args.raw_data_path
save_data_path = args.save_data_path

#ts_colname = "agg_time_interval"
#agg_freg = "5min"

matrix_data_path = save_data_path +"matrix_data/"

if not os.path.exists(matrix_data_path):
    os.makedirs(matrix_data_path)
    
def feature_and_self_matrices_generator():
    df = yf.download('BTC-USD', 
                      #start='2022-02-02', 
                      #end='2022-03-01', 
                      progress=False)
    #print(df.index)
    df.reset_index(inplace=True)
   # print(df.head())
   # df.columns =df.shape[1]
    #data = pd.read_csv(raw_data_path, delimiter=",")
    df.drop(['Date'], inplace=True, axis=1)
    df.columns = [''] * len(df.columns)
    df.columns = [0, 1, 2, 3, 4, 5]  
   
   
   
    df = np.array(df, dtype=np.float64)
    #print(df)
    #data = df # np.transpose(df)
    #sensor_n = df.shape[0]
    #print(df.shape)
    #print(sensor_n)
    #tdata = data
   
    #min-max normalization
    
    # max_value = np.max(data, axis=1)
    # min_value = np.min(data, axis=1)
    # data = (np.transpose(data) - min_value)/(max_value-min_value + 1e-6)
    # tdata = np.transpose(data)
    
    # Standard normalization
    
    scaler = StandardScaler()
    scaler = scaler.fit(df)
    df = scaler.transform(df)
    
    # Perform EDA here!!!
    
    plt.plot(df[:100, :5])
    df = np.transpose(df)
    print(df)
    sensor_n = df.shape[0]
    print(sensor_n)
    
    plt.show()
    
    
    data = df
   # matrix generation for a list of window_size
    for w in range(len(window_size)):
        matrix_all =[]
        win = window_size[w]
        print("feature matrix with windows "+ str(win)+ "...")
        for t in range(min_time, max_time, gap_time):
            #print(t)
            matrix_t =np.zeros((sensor_n, sensor_n))
            
            if t >= 60:
                for i in range(sensor_n):
                    for j in range(i, sensor_n):
                        matrix_t[i][j] = np.inner(data[i, t-win:t], data[j, t-win:t])/(win)
                        matrix_t[j][i] = matrix_t[i][j] 
            matrix_all.append(matrix_t)
        path_temp = matrix_data_path + "matrix_win_"+ str(win)
        np.save(path_temp, matrix_all)
        del matrix_all[:]
    print("feature matrix generation completed")
    
    

def train_test_data_generator():
    print("Generating train and test data....")
    matrix_data_path = save_data_path + "matrix_data/"
    train_data_path = matrix_data_path + "train_data/"
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)
    test_data_path = matrix_data_path + "test_data/"
    if not os.path.exists(test_data_path):
        os.makedirs(test_data_path)
    
    data_all =[]
    for w in range(len(window_size)):
        path_temp = matrix_data_path +"matrix_win_" +str(window_size[w])+".npy"
        data_all.append(np.load(path_temp))
    
    train_test_time = [[train_start, train_end],[test_start, test_end]]
    for i in range(len(train_test_time)):
        for data_id in range(int(train_test_time[i][0]/gap_time), int(train_test_time[i][1]/gap_time)):
            
            step_multi_matrix = []
            for step_id in range(step_max, 0, -1):
                multi_matrix = []
                for i in range(len(window_size)):
                    multi_matrix.append(data_all[i][data_id - step_id])
                step_multi_matrix.append(multi_matrix)     
         
                    
            if data_id >=  (train_start/gap_time + window_size[-1]/gap_time + step_max) and data_id < (train_end/gap_time):
                path_temp = os.path.join(train_data_path, 'train_data_' +str(data_id) )
                np.save(path_temp, step_multi_matrix)
        
            elif data_id >= (test_start/gap_time) and data_id < (test_end/gap_time):
                path_temp = os.path.join(test_data_path, 'test_data_' + str(data_id))
                np.save(path_temp, step_multi_matrix)
            
            del step_multi_matrix[:]
    print("Train and test data are generated.")
            
          
        
            
        


if __name__ == '__main__':
    
   feature_and_self_matrices_generator()
   train_test_data_generator()
    
    #if time_series == "node":
    #    feature_and_self_matrices_generator()
   #train_test_data_generator()
 

