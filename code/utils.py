# Constants utils for the model
gap_time = 10  # gap time between each  subsequences
window_size =[10, 30, 60] # range og window size
step_max = 5 # maximum step of convolutional LSTM

#min and max index
min_index = 0 # first time index of ts
max_index = 20000 # the index of the last observation

#raw_data_path = '../data/synthetic_data_with_anomaly-s-1.csv'  # path to load raw data
raw_synth_ts_path = '../data/synthetic_data.csv'  # path to load raw data
raw_real_ts_path  = '../data/gas_data/' # path to real-world raw data
model_path = '../MSCRED/'
#train_ts_path = "../data/train/" # path to train_ts 
#test_ts_path = "../data/test/" # path to test_ts
reconstructed_ts_path = "../data/reconstructed/" # path to reconstructed ts.

# save data directory
save_data_path = '../data/' 
matrix_data_path = "../data/matrix_data/"


# path to train test files

splits = ["train", "test"]
train_data_path = "./data/matrix_data/train_data/"
test_data_path = "./data/matrix_data/test_data/"
shuffle = {'train': True, 'test': False}



train_start_id = 0
train_end_id = 8000

test_start_id = 8000
test_end_id = 20000

valid_start_id = 8000
valid_end_id = 10000

training_iters = 5
save_model_step = 1

learning_rate = 0.0002

threhold = 0.005
alpha = 1.5
tfactor = 1.52 #default value