#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 12:12:27 2023

@author: jianxig
"""
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch

# 2D CNN encoder
class ResCNNEncoder(nn.Module):
    def __init__(self, cnn_pretrained_weight_dir, training=True):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResCNNEncoder, self).__init__()
        
        # The pre-trained model
        model_ft = models.resnet50(pretrained=False)
        # Replace the output layer with other FC layers
        # For CRNN, batchNorm1d uses momentum=0.01
        model_ft.fc=nn.Sequential(
                   nn.Linear(2048, 512),
                   nn.BatchNorm1d(512),
                   nn.ReLU(inplace=True),
                   nn.Linear(512, 128),
                   nn.BatchNorm1d(128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, 2))
        
        # Load the weights when we train the network
        if training:
            model_ft.load_state_dict(torch.load(cnn_pretrained_weight_dir))
        
        # Remove the output layer in the pre-trained model 
        # Note: The network output is B x 128
        model_ft.fc=nn.Sequential(*list(model_ft.fc.children())[:4])
        
        # Freeze the network when we train the network
        if training:
            for param in model_ft.parameters():
                param.requires_grad = False
            
        #
        self.resnet=model_ft

        
    def forward(self, x_3d, prev_cnn_embed_seq, bool_first_batch):
        cnn_embed_seq = []
        
        # For the first sequence, all images should be processed by CNN
        if bool_first_batch:
        
            for t in range(x_3d.size(1)):
                # ResNet CNN
                with torch.no_grad():
                    x = self.resnet(x_3d[:, t, :, :, :])  # ResNet (Process a specific batch)
    
                cnn_embed_seq.append(x)
    
            # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
            # (Me) '_' in 'transpose_' denotes 'in-place operation'
            # cnn_embed_seq: shape=(batch, time_step, input_size)
            cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        
        # For other sequence, only the last image is processed by CNN
        else:
            with torch.no_grad():
                # ResNet (Process the last batch)
                x = self.resnet(x_3d[:, -1, :, :, :]) 
            # Seperate the tensor in each time step
            list_prev_cnn_embed_seq = list(torch.unbind(prev_cnn_embed_seq, dim=1)) 
            # Remove the info in the first time step
            _ = list_prev_cnn_embed_seq.pop(0)
            # Insert the info at the last time step
            list_prev_cnn_embed_seq.append(x)
            # 
            cnn_embed_seq = torch.stack(list_prev_cnn_embed_seq, dim=0).transpose_(0, 1) 
        
        return cnn_embed_seq


class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, output_dim=50):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers   # RNN hidden layers
        self.h_RNN = h_RNN                 # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.output_dim = output_dim

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,        
            num_layers=h_RNN_layers,       
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.output_layer = nn.Linear(self.h_FC_dim, self.output_dim)

    def forward(self, x_RNN):
        
        # (Me) when we use nn.DataParallel,  it replicates original module to every GPU it uses, then weight tensors are 
        # fragmented again since there’s no gurantee that replicated tensors are still contiguous on memory space.
        # Therefore we should flatten_parameters again everytime the module is replicated to another GPU, and the best place 
        # to put function call would be the head of forward function (of nn.Module), because forward function of nn.Module 
        # on each GPU is called only one time when forward of nn.DataParallel is called
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)  
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """ 
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x = self.fc1(RNN_out[:, -1, :])   # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.output_layer(x)

        return x