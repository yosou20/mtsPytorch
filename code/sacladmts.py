import torch
import torch.nn as nn
import numpy as np
from code.conv2d_lstm import ConvLSTM
import code.utils as util


from torchsummary import summary
from torchvision import models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CnnEncoder(nn.Module):
    def __init__(self, in_channels_encoder):
        super(CnnEncoder, self).__init__()
        ## Add block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels_encoder, 32, 3, (1, 1), 1),
            nn.SELU() # utils for vanishing gradient and normalization   
            #nn.ReLU() #   
        
        )
        ## Add block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, (2, 2), 1),
            nn.SELU() # utils for vanishing gradient and normalization 
            #nn.ReLU() # 
        ) 
        ## Add block 3 
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 2, (2, 2), 1),
            nn.SELU()
            #nn.ReLU() #
            
        )
        ## Add block 4   
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, 2, (2, 2), 0),
            nn.SELU() # utils for vanishing gradient and normalization
            #nn.ReLU() #
        )
    def forward(self, X):
        block1_out = self.block1(X)
        block2_out = self.block2(block1_out)
        block3_out = self.block3(block2_out)
        block4_out = self.block4(block3_out)
        return block1_out, block2_out, block3_out, block4_out    

class Conv_LSTM(nn.Module):
    def __init__(self):
        super(Conv_LSTM, self).__init__()
        ## Add convLSTM bock 1
        self.lstm_bock1 = ConvLSTM(
            input_channels=32, hidden_channels=[32], 
            kernel_size=3, step=5, effective_step=[4])
        
        self.lstm_bock2 = ConvLSTM(
            input_channels=64, hidden_channels=[64],
            kernel_size=3, step=5, effective_step=[4])
        
        self.lstm_bock3 = ConvLSTM(
            input_channels=128, hidden_channels=[128],
            kernel_size=3, step=5, effective_step=[4])
        
        self.lstm_bock4 = ConvLSTM(
            input_channels=256, hidden_channels=[256], 
            kernel_size=3, step=5, effective_step=[4])

    def forward(self, block1_out, block2_out, block3_out, block4_out):
        lstm_bock1_output = self.lstm_bock1(block1_out)       
        lstm_bock2_output = self.lstm_bock2(block2_out)        
        lstm_bock3_output = self.lstm_bock3(block3_out)       
        lstm_bock4_output = self.lstm_bock4(block4_out)       
        return lstm_bock1_output[0][0], lstm_bock2_output[0][0], lstm_bock3_output[0][0], lstm_bock4_output[0][0]
        

class CnnDecoder(nn.Module):
    def __init__(self, in_channels):
        super(CnnDecoder, self).__init__()
        self.t_block4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, 2, 2, 0, 0), # perform operation
            nn.SELU(),
           # nn.ReLU()
        )
        self.t_block3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 2, 2, 1, 1),
            nn.SELU(),
           # nn.ReLU()
        )
        self.t_block2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 3, 2, 1, 1),
            nn.SELU(),
            #nn.ReLU()
        )
        self.t_block1 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 3, 1, 1, 0),
            nn.SELU(),
           # nn.ReLU()
        )
    
    def forward(self, lstm_bock1_output, lstm_bock2_output, lstm_bock3_output, lstm_bock4_output):
        t_block4 = self.t_block4(lstm_bock4_output)
        t_block4_concat = torch.cat((t_block4, lstm_bock3_output), dim = 1)
        t_block3 = self.t_block3(t_block4_concat)
        t_block3_concat = torch.cat((t_block3, lstm_bock2_output), dim = 1)
        t_block2 = self.t_block2(t_block3_concat)
        t_block2_concat = torch.cat((t_block2, lstm_bock1_output), dim = 1)
        t_block1 = self.t_block1(t_block2_concat)
        return t_block1


class SACLADMTS(nn.Module):
    def __init__(self, in_channels_encoder, in_channels_decoder):
        super(SACLADMTS, self).__init__()
        # Add an instance of a  Convolutional  Encoder
        self.conv_encoder = CnnEncoder(in_channels_encoder)
        #Add Convolutional LSTM
        self.conv_lstm = Conv_LSTM()
        # Add Convolutional Decoder
        self.conv_decoder = CnnDecoder(in_channels_decoder)
    
    def forward(self, x):
        block1_out, block2_out, block3_out, block4_out = self.conv_encoder(x)
        lstm_bock1_output, lstm_bock2_output, lstm_bock3_output, lstm_bock4_output = self.conv_lstm(
                                block1_out, block2_out, block3_out, block4_out)

        model_output = self.conv_decoder(lstm_bock1_output,
                                         lstm_bock2_output, lstm_bock3_output, lstm_bock4_output)
        return model_output
