import torch
import torch.nn as nn
import torch.functional as F 
from tqdm import tqdm # for smarter progress bar
from code.sacladmts import SACLADMTS
from code.dataloader import load_data
import matplotlib.pyplot as plt
import numpy as np
import os
# utils  to check model parameters
from torchsummary import summary
from torchvision import models
# utils to compute model metrics
from torchmetrics import Accuracy, MetricCollection, Precision, Recall
from torchmetrics.classification import Accuracy

def train(dataLoader, model, optimizer, epochs, device):
    model = model.to(device)
    print("------training on {} -------".format(device))
    train_loss_list = []
    for epoch in range(1, epochs+1):
        train_l_sum, n = 0.0, 0
        train_accuracy = 0.0
        
        
        for x in tqdm(dataLoader):
            x = x.to(device)
            x = x.squeeze()
            #print(type(x))
            train_loss = torch.mean((model(x)-x[-1])**2) # mean square resisdual of loss
            train_loss_list.append(object)
            train_l_sum += train_loss
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            if (train_loss < 0.005).float():
                train_accuracy += 1.0
            train_accuracy 
            
            n += 1
            #print("[Epoch %d/%d][Batch %d/%d] [loss: %f]" % (epoch+1, epochs, n, len(dataLoader), l.item()))
            
        print("[Epoch %d/%d] [loss: %f] [train_acc: %f]" % (epoch, epochs, train_l_sum/(n), 100*train_accuracy/n))


def test(dataLoader, model):
    print("------Testing the model on {}-------".format(model.to(device)))
    index = 800
    loss_list = []
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
            test_accuracy           
            
            index += 1
            print("[test_index %d] [loss: %f] [test accuracy: %f]" %  (index, test_loss.item(), 100*test_accuracy/index))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device is", device)
    dataLoader = load_data()
    #mscred = MSCRED(3, 256)   
    sacladmts = SACLADMTS(3, 256)
        

    # model training
    # mscred.load_state_dict(torch.load("./checkpoints/model1.pth"))
    optimizer = torch.optim.Adam(sacladmts.parameters(), lr = 0.0002)
    train(dataLoader["train"], sacladmts, optimizer, 1, device)
    #summary(mscred, (5, 512, 64), 32 )
    #summary(mscred, input_size, batch_size=-1, device='cpu')
    print("Model saving ....")
    torch.save(sacladmts.state_dict(), "./checkpoints/model2.pth")
   
    # # model testing
    sacladmts.load_state_dict(torch.load("./checkpoints/model2.pth"))
    sacladmts.to(device)
    
    test(dataLoader["test"], sacladmts)