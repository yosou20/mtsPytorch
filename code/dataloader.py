import torch
import sys, os
import numpy as np
from  torch.utils.data import DataLoader
import code.utils as util

def load_data():
    """_Loading train and test data_

    Returns:
        _datase_: _train and test data_
    """
    

test_data_path = util.test_data_path
train_data_path = util.train_data_path
def load_data():
    dataset = {}
    train_file_list = os.listdir(train_data_path)
    test_file_list = os.listdir(test_data_path)
    train_file_list.sort(key = lambda x:int(x[11:-4])) # file names
    test_file_list.sort(key = lambda x:int(x[10:-4]))
    train_data, test_data = [],[]   
    for file in train_file_list:   
        train_file_path = train_data_path + file
        train_matrix = np.load(train_file_path)        
        train_data.append(train_matrix)

    for file in test_file_list:
        test_file_path = test_data_path + file
        test_matrix = np.load(test_file_path)       
        test_data.append(test_matrix)

    dataset["train"] = torch.from_numpy(np.array(train_data)).float()
    dataset["test"] = torch.from_numpy(np.array(test_data)).float()

    dataloader = {x: torch.utils.data.DataLoader(
                                dataset=dataset[x], batch_size=1, shuffle=util.shuffle[x]) 
                                for x in util.splits}
    return dataloader

