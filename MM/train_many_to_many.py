#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 16:28:52 2023

@author: jianxig

Note: Think about MSE ignore_index: https://discuss.pytorch.org/t/mse-ignore-index/43934
"""
import pandas as pd
import DIY_data_loader_many_to_many 
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
import torch
import copy
from torchvision import transforms
import numpy as np
import network_many_to_many

# Wrap the results, then calculate the error for a specific batch
def batch_offset(np_outputs, np_labels):
    np_diff = np_outputs - np_labels
    bool_top_val=np_diff>=180
    bool_bottom_val=np_diff<=-180
    offset_larger_180=bool_top_val*(-360.0)
    offset_smaller_180=bool_bottom_val*(360.0)
    np_offset=offset_larger_180+offset_smaller_180
    
    return np_offset


# For training
def train_model(device, data_loader, model, loss_function, optimizer, dataset_sizes_dict):
    total_loss, total_loss_deg = 0, 0 
    # Set model as training mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()
    
    # To show the progress bar
    with tqdm(data_loader, unit='batch') as tepoch:
        # Iterate the dataset
        for batch_idx, sample_batch in enumerate(tepoch):
            X = sample_batch[0]#X.size: [B,T,3,224,224]
            y = sample_batch[1]#y.size: [B,T,2]
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            cnn_output = cnn_encoder(X)#cnn_output.size: [B,T,64]
            rnn_output = rnn_decoder(cnn_output)#rnn_output.size: [B,T,2]
            loss = loss_function(rnn_output, y.float())

            #
            loss.backward()
            optimizer.step()
    
            # Loss calculation (Note: The loss considers errors at all timesteps)
            total_loss += (loss.item())*y.size(0)
            # Calculate the loss in degrees (Note: It only considers errors at the last timesteps)
            np_labels=y.cpu().detach().numpy()
            np_labels_rad=np.arctan2(np_labels[:,-1,0], np_labels[:,-1,1])
            np_labels_deg=np_labels_rad*180/np.pi
            np_outputs=rnn_output.cpu().detach().numpy()
            np_outputs_rad=np.arctan2(np_outputs[:,-1,0], np_outputs[:,-1,1])
            np_outputs_deg=np_outputs_rad*180/np.pi
            np_offset_deg=batch_offset(np_outputs_deg, np_labels_deg)
            # MAE
            total_loss_deg += np.sum(np.abs(np_outputs_deg - np_labels_deg + np_offset_deg))

    avg_loss = total_loss / dataset_sizes_dict['train']
    print(f"Train loss: {avg_loss}")
    avg_loss_deg = total_loss_deg / dataset_sizes_dict['train']
    print(f'Train loss (deg): {avg_loss_deg}')
    
    return avg_loss, avg_loss_deg


# For validation
def validate_model(device, data_loader, model, loss_function, dataset_sizes_dict):
    total_loss, total_loss_deg = 0, 0
    # Set the model as testing mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()
    
    # Avoid back-propagation
    with torch.no_grad():
        
        # To show the progress bar
        with tqdm(data_loader, unit='batch') as tepoch:
            for batch_idx, sample_batch in enumerate(tepoch):
                X = sample_batch[0] # X.shape: (B, T, 3, 224, 224)
                y = sample_batch[1] # y.shape: (B, T, 2)
                X, y = X.to(device), y.to(device)
                cnn_output=cnn_encoder(X) # Output size: (B, T, 64)
                rnn_output = rnn_decoder(cnn_output) # Output size: (B, T, 2)
                
                # Loss calculation (Note: The loss considers errors at all timesteps)
                loss = loss_function(rnn_output, y.float())
                total_loss += (loss.item())*y.size(0)
                # Calculate the loss in degrees (Note: It only considers errors at the last timesteps)
                np_labels=y.cpu().detach().numpy()#np_labels.shape=(B,T,2)
                np_labels_rad=np.arctan2(np_labels[:,-1,0], np_labels[:,-1,1])#np_labels_rad.shape=(B,)
                np_labels_deg=np_labels_rad*180/np.pi
                np_outputs=rnn_output.cpu().detach().numpy()#np_outputs.shape(B,T,2)
                np_outputs_rad=np.arctan2(np_outputs[:,-1,0], np_outputs[:,-1,1])#np_outputs_rad.shape=(B,)
                np_outputs_deg=np_outputs_rad*180/np.pi
                np_offset_deg=batch_offset(np_outputs_deg, np_labels_deg)#np_offset_deg.shape=(B,)
                # MAE
                total_loss_deg += np.sum(np.abs(np_outputs_deg - np_labels_deg + np_offset_deg))

    avg_loss = total_loss / dataset_sizes_dict['val']
    print(f"Validation loss: {avg_loss}")    
    avg_loss_deg = total_loss_deg / dataset_sizes_dict['val']
    print(f'Validation loss (deg): {avg_loss_deg}')
    
    return avg_loss, avg_loss_deg


if __name__=='__main__':
    TIMESTEP=5 # 30 5
    BATCH_SIZE=32# 5 1
    EPOCH=20
    best_loss=1e5
    learning_rate = 1e-4
    NUM_WORKERS=4
    # Parameters for CNN
    res_size = 224        # ResNet image size
    CNN_pretrained_weight_dir='/home/jianxig/CNN_roll/dataset_v11/ResNet_FineT/2023-3-1/l1_RC/proposal/best_weight_1.pt'
    input_img_dir='/zfs/cucat/jianxig/img_dataset/'
    input_dir= '/zfs/cucat/jianxig/dataset_v11/CRNN/'
    output_dir='/home/jianxig/CRNN/v2_1/l1/many-to-many/proposal/'
    train_csv_dir= input_dir+'d1_train.csv' 
    valid_csv_dir=input_dir+'d1_develop.csv'
    #output_csv_dir=output_dir+'output.csv'
    csv_dir_history=output_dir+'history.csv'
    
    # Read the data
    df_train=pd.read_csv(train_csv_dir, sep='\s*,\s*', header=[0],engine='python')
    sin_train=df_train['sin'].to_numpy()
    cos_train=df_train['cos'].to_numpy()
    area_num_train=df_train['Area_Num'].to_numpy()
    image_name_train=df_train['Image_name'].tolist()
    df_valid=pd.read_csv(valid_csv_dir, sep='\s*,\s*', header=[0],engine='python')
    sin_validation=df_valid['sin'].to_numpy()
    cos_validation=df_valid['cos'].to_numpy()
    area_num_validation=df_valid['Area_Num'].to_numpy()
    image_name_validation=df_valid['Image_name'].tolist()

    # For saving history
    csv_history=open(csv_dir_history, 'w')
    csv_history.write('Epoch,Loss(train),Loss(deg)(train),Loss(validation),Loss(deg)(validation)\n')
    
    # Check if gpu is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print('Using GPU\n')
    else:
        print('Using CPU\n')
    
    # DIY data loader
    # (Note: The experiment results show that data augmentation improves the performance of CRNN, 
    # which is contradictory to the conclusion in my deep-learning paper)
    train_dataset = DIY_data_loader_many_to_many.SequenceDataset(sin_train, cos_train, area_num_train, \
                                                                image_name_train, input_img_dir, TIMESTEP, \
            transform=transforms.Compose([
            transforms.ColorJitter(brightness=(0.7,1.2),contrast=(0.7,1.2),saturation=(0.8,1.3)),\
            transforms.RandomResizedCrop((224,224), scale=(0.7,1.0), ratio=(1.0,1.0), interpolation=transforms.InterpolationMode.BICUBIC),\
            transforms.ToTensor(),\
            # The mean and std are obtained from ImageNet data set.
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
        
    validation_dataset = DIY_data_loader_many_to_many.SequenceDataset(sin_validation, cos_validation, area_num_validation, \
                                         image_name_validation, input_img_dir, TIMESTEP, \
                                         transform=transforms.Compose([
                                         transforms.Resize((res_size,res_size),interpolation=transforms.InterpolationMode.BICUBIC),\
                                         transforms.ToTensor(),\
                                         # The mean and std are obtained from ImageNet data set.
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
        
    # 
    dataset_sizes_dict={'train':len(train_dataset), 'val':len(validation_dataset)}

    # Note: shuffle==True only shuffles the data between each timestep. It never shuffle the data within
    # a timestep
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, \
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)#
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True,\
                                   num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)# 
    
    # Construct the network
    # EncoderCNN architecture
    cnn_encoder = network_many_to_many.ResCNNEncoder(CNN_pretrained_weight_dir, training=True).to(device)
    # DecoderRNN architecture
    CNN_embed_dim=128
    RNN_hidden_layers = 1
    RNN_hidden_nodes = 32
    RNN_FC_dim = 32
    RNN_dropout_p = 0.5       # dropout probability
    final_output_length=2
    rnn_decoder = network_many_to_many.DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, \
                                                 h_RNN=RNN_hidden_nodes, h_FC_dim=RNN_FC_dim, drop_p=RNN_dropout_p, \
                                                output_dim=final_output_length).to(device)
        
    # Find and then combine the learnable parameters
    crnn_params = list(rnn_decoder.parameters())
                  
    # Loss func and the optimizer    
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(crnn_params, lr=learning_rate)
    
    # 
    for idx_epoch in range(EPOCH):
        print("Epoch {}\n---------".format(idx_epoch+1))
        train_loss, train_loss_deg = train_model(device, train_loader, [cnn_encoder, rnn_decoder], loss_function, \
                                                 optimizer, dataset_sizes_dict)
        validation_loss, validation_loss_deg = validate_model(device, validation_loader, [cnn_encoder, rnn_decoder], \
                                                loss_function, dataset_sizes_dict)
        print()
        
        # Save the best model
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_cnn_dir=output_dir+'best_weight_cnn.pt'
            best_rnn_dir=output_dir+'best_weight_rnn.pt'
            best_cnn_wts = copy.deepcopy(cnn_encoder.state_dict())
            torch.save(best_cnn_wts, best_cnn_dir)
            best_rnn_wts = copy.deepcopy(rnn_decoder.state_dict())
            torch.save(best_rnn_wts, best_rnn_dir)
            #torch.save(optimizer.state_dict(), best_optimizer_dir)
            print('Save the model weights. Loss={:.2f}'.format(best_loss))
        
        # Save history
        csv_history.write('{},{},{},{},{}\n'.format((idx_epoch+1), train_loss, train_loss_deg, \
                                                    validation_loss, validation_loss_deg))
        
    #        
    csv_history.close()
    
    
