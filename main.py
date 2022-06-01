import torch
import torch.nn as nn
import torch.functional as F 
from tqdm import tqdm # for smarter progress bar
from code.sacladmts import SACLADMTS
from code.dataloader import load_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
# utils  to check model parameters
from torchsummary import summary
from torchvision import models
# utils to compute model metrics
from torchmetrics import Accuracy, MetricCollection, Precision, Recall
from torchmetrics import F1Score
from torchmetrics.classification import Accuracy

def train(dataLoader, model, optimizer, epochs, device):
    model = model.to(device)
    print("----Model training on {} -------".format(device))
    train_loss_list = []
    train_acc_list = []
    f1_score_list = []
    for epoch in range(1, epochs+1):
        train_l_sum, n = 0.0, 0
        train_accuracy = 0.0
        runing_f1_score = 0.0
        f1 = F1Score()
        
        
        for x in tqdm(dataLoader):
            x = x.to(device)
            x = x.squeeze()
            #print(type(x))
            train_loss = torch.mean((model(x)-x[-1].unsqueeze(0))**2) # mean square resisdual of loss
            train_loss_list.append(train_loss)
            train_l_sum += train_loss
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            if (train_loss < 0.005).float():
                train_accuracy += 1.0
                train_acc_list.append(train_accuracy)
            train_accuracy 
            #f1_temp = f1(model(x).unsqueeze(0), x[-1].unsqueeze(0))
            #f1_score_list.append(f1_temp)
            
            n += 1
            #print("[Epoch %d/%d][Batch %d/%d] [loss: %f]" % (epoch+1, epochs, n, len(dataLoader), l.item()))
            
        print("[Epoch %d/%d] [ %f] [ %f]" % (epoch, epochs, train_l_sum/(n), train_accuracy/n))
    #anomaly_t = [list(x) for x in train_loss_list]
    #print(anomaly_t)
    return train_loss_list,train_acc_list   
    # plt.plot(np.arange(epoch-1), train_loss_list, label='train loss')
    # plt.plot(np.arange(epoch-1), train_loss_list, label='train accuracy')
    # plt.xlabel('number of epochs')
    # plt.legend()
    # plt.show()

def test(dataLoader, model):
    print("------Testing the model on {}-------".format(device))
    index = 800
    loss_list = []
    test_index_list = []
    test_acc_list = []
    test_accuracy = 0.0
    reconstructed_data_path = "./data/matrix_data/reconstructed_data/"
    if not os.path.exists(reconstructed_data_path):
        os.makedirs(reconstructed_data_path)
    with torch.no_grad():
        for x in dataLoader:
            x = x.to(device)
            x = x.squeeze()
            reconstructed_matrix = model(x) 
            path_temp = os.path.join(reconstructed_data_path, 'reconstructed_data_' + str(index) + ".npy")
            np.save(path_temp, reconstructed_matrix.cpu().detach().numpy())
            test_loss = torch.mean((reconstructed_matrix -x[-1])**2) # criterion(reconstructed_matrix, x[-1].unsqueeze(0)).mean()
            loss_list.append(test_loss)
            if (test_loss < 0.005).float():
                test_accuracy += 1.0
                test_acc_list.append(test_accuracy)
            #test_accuracy           
            
            index += 1
            #epoch_loss = test_loss.item()/index
    
        #print("[test_index %d] [loss: %f] [test accuracy: %f]" %  (index, test_loss.item(), 100*test_accuracy/index))
        print(" %d,  %f,  %f" %  (index, test_loss.item(), 100*test_accuracy/index))
        #print(loss_list)
        #loss_list2 = [list(x) for x in loss_list]
        #print(loss_list2)
        #plt.plot(np.array(loss_list), 'r')
    #test_data = pd.DataFrame(index, loss_list, test_acc_list)
    #print(test_data)
    #return loss_list, test_accuracy, test_acc_list    
    # plt.plot(np.arange(epoch-1), train_loss_list, label='train loss')
    # plt.plot(np.arange(epoch-1), train_loss_list, label='train accuracy')
    # plt.xlabel('number of epochs')
    # plt.legend()
    # plt.show()
 
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device is", device)
    dataLoader = load_data()     
    sacladmts = SACLADMTS(3, 256)
        

    # model training
    # mscred.load_state_dict(torch.load("./checkpoints/model1.pth"))
    optimizer = torch.optim.Adam(sacladmts.parameters(), lr = 0.0002)
    train(dataLoader["train"], sacladmts, optimizer,50, device)
    #summary(mscred, (5, 512, 64), 32 )
    #summary(mscred, input_size, batch_size=-1, device='cpu')
    print("Model saving ....")
    torch.save(sacladmts.state_dict(), "./checkpoints/model2.pth")
   
    # # model testing
    sacladmts.load_state_dict(torch.load("./checkpoints/model2.pth"))
    sacladmts.to(device)
    
    test(dataLoader["test"], sacladmts)