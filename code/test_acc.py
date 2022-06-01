import torch
from  sacladmts import SACLADMTS
from conv2d_lstm import ConvLSTM
# import torch.nn as nn
# import torch.functional as F 
# from tqdm import tqdm # for smarter progress bar
# from sacladmts import SACLADMTS
# from dataloader import load_data
# import matplotlib.pyplot as plt
# import numpy as np
# import os



#def test(): 
#     # Load the model that we saved at the end of the training loop 
#     model = SACLADMTS(3, 256) 
#     path = "Model2.pth" 
#     model.load_state_dict(torch.load(path)) 
     
#     running_accuracy = 0 
#     total = 0 
 
#     with torch.no_grad(): 
#         for data in test_loader: 
#             inputs, outputs = data 
#             outputs = outputs.to(torch.float32) 
#             predicted_outputs = model(inputs) 
#             _, predicted = torch.max(predicted_outputs, 1) 
#             total += outputs.size(0) 
#             running_accuracy += (predicted == outputs).sum().item() 
 
#         print('Accuracy of the model based on the test set of', test_split ,'inputs is: %d %%' % (100 * running_accuracy / total))    
 
 
# # Optional: Function to test which species were easier to predict  
# Specify a path
PATH = "../checkpoints/model2.pth"

# Save
#torch.save(net, PATH)

# Load
model = SACLADMTS(3,256)

checkpoint = torch.load(PATH)
#model.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epochs']
loss = checkpoint['loss']