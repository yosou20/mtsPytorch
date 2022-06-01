# Constants utils for the model
window_size =[10, 30, 60] # range og window size
step_max = 5 # maximum step of convolutional LSTM
gap_time = 10  # gap time between each  subsequences

#min and max index
min_index = 0 # first time index of ts
max_index = 20000 # the index of the last observation

#raw_data_path = '../data/synthetic_data_with_anomaly-s-1.csv'  # path to load raw data
raw_synth_ts_path = '../data/synthetic_data.csv'  # path to  synthetic dataset 1
raw_synth_ts_path1 = '../data/synthetic_data_abn.csv'  # path to synthetic raw dataset 2
raw_real_ts_path  = '../data/gas_data/' # path to real-world raw data

reconstructed_ts_path = "../data/reconstructed/" # path to reconstructed ts.

#save data directory
save_data_path = '../data/' 
matrix_data_path = "../data/feature_matrix/"


#path to train test files

splits = ["train", "test"]
train_data_path = "./data/feature_matrix/train_data/"
test_data_path = "./data/feature_matrix/test_data/"
shuffle = {'train': True, 'test': False}



train_start_id = 0  #training start index
train_end_id = 8000 # training end index

test_start_id = 8000 #test start index
test_end_id = 20000  #test end index

valid_start_id = 8000 #validation start indec
valid_end_id = 10000

training_iters = 5 #initial itwration
save_model_step = 1 # number of saved train model model

learning_rate = 0.0002 # learning rate default

threhold = 0.005 # Threshold default

tfactor = 1.52 #default value