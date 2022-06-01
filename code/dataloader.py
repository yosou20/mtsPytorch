import torch
import sys, os
import numpy as np
from  torch.utils.data import DataLoader
import code.utils as util

def load_data():
    """_summary_

    Returns:
        _type_: _description_
    """
    # test_data_path = util.test_data_path
    # train_data_path = util.train_data_path
    # train_test_set = {}
    # train_file_list = os.listdir(train_data_path) # get train set files
    # test_file_list = os.listdir(test_data_path)
    # train_file_list.sort(key=lambda x:int(x[11:-4]))
    # test_file_list.sort(key=lambda x:int(x[11:-4]))
    
    # train_data, test_data = [],[]
    # train_test_file_list =[train_file_list, test_file_list]
    # for file_list in train_test_file_list:
    #     if file_list == train_test_file_list[0]:
    #         for file in file_list:
    #             dest_dir = train_data_path + file
    #             train_matrix = np.load(dest_dir)
    #             train_data.append(train_matrix)
    #     if file_list == train_test_file_list[1]:
    #         for file in file_list:
    #             dest_dir = test_data_path + file
    #             test_matrix = np.load(dest_dir)
    #             test_data.append(test_matrix)
        
    # train_test_set['train'] = torch.from_numpy(np.array('train_data')).float()
    # train_test_set['test'] = torch.from_numpy(np.array('test_data')).float()
    
    # dataloader = { x: DataLoader(
    #     dataset = train_test_set[x],
    #     batch_size = 1, shuffle = util.shuffle[x]) for x in util.splits} # batch_size can be changed
    # return dataloader

test_data_path = util.test_data_path
train_data_path = util.train_data_path
def load_data():
    dataset = {}
    train_file_list = os.listdir(train_data_path)
    test_file_list = os.listdir(test_data_path)
    train_file_list.sort(key = lambda x:int(x[11:-4]))
    test_file_list.sort(key = lambda x:int(x[10:-4]))
    train_data, test_data = [],[]
    #train_test_file_list = [train_test_file_list, test_file_list]
    #for file_list in train_test_file_list:
    for file in train_file_list:   
        train_file_path = train_data_path + file
        train_matrix = np.load(train_file_path)
        #train_matrix = np.transpose(train_matrix, (0, 2, 3, 1))
        train_data.append(train_matrix)

    for file in test_file_list:
        test_file_path = test_data_path + file
        test_matrix = np.load(test_file_path)
        #test_matrix = np.transpose(test_matrix, (0, 2, 3, 1))
        test_data.append(test_matrix)

    dataset["train"] = torch.from_numpy(np.array(train_data)).float()
    dataset["test"] = torch.from_numpy(np.array(test_data)).float()

    dataloader = {x: torch.utils.data.DataLoader(
                                dataset=dataset[x], batch_size=1, shuffle=util.shuffle[x]) 
                                for x in util.splits}
    return dataloader

