#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 10:43:12 2023

@author: jianxig
"""

import torch
import numpy as np
import pandas as pd
#from PIL import Image
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms, models
from torch.utils.data import DataLoader
import os
import glob

import DIY_data_loader_many_to_one
import network_many_to_one

# Wrap the results, then calculate the error for a specific batch
def batch_offset(np_outputs, np_labels):
    np_diff = np_outputs - np_labels
    bool_top_val=np_diff>=180
    bool_bottom_val=np_diff<=-180
    offset_larger_180=bool_top_val*(-360.0)
    offset_smaller_180=bool_bottom_val*(360.0)
    np_offset=offset_larger_180+offset_smaller_180
    
    return np_offset


def abbr_file_name(input_file_name):
    single_file_name_split=input_file_name.split('_')
    single_file_name_split_1=single_file_name_split[1].split('-')
    shorten_file_name=single_file_name_split_1[0][:4]+'-'+\
        single_file_name_split_1[4]+'-'+single_file_name_split_1[5]
    
    return shorten_file_name

# Test the network
# Note: It *only* supports batch_size=1
def test_model(cnn_encoder, rnn_decoder, data_loader_test, dataset_size, csv_dir_output, loss_function):
    total_loss, total_loss_deg=0,0
    total_nonzero_val=0
    
    # Predictions and ground truth will be saved within the csv file
    csv_output=open(csv_dir_output, 'w')
    csv_output.write('FN_from_bag,roll(GT),roll(Pred),sin(GT),cos(GT),sin(pred),cos(pred)\n')
    
    # Set the model to evaluation mode
    cnn_encoder.eval()
    rnn_decoder.eval()
    
    # Disable gradient calculation
    with torch.no_grad():
        
        # To show the progress bar
        with tqdm(data_loader_test, unit='batch') as tepoch:
            
            # Iterate over data
            for sample_batch in tepoch:
                inputs = sample_batch[0].to(device) #inputs.shape=(B,T,3,224,224)
                init_labels = sample_batch[1].to(device) #init_labels.shape=(B,2)
                current_frame_num=sample_batch[-1]
                # Cast the data type
                labels = init_labels.type(torch.float32)
                
                # Inference
                cnn_output=cnn_encoder(inputs) # Output size: (B, T, 64)
                rnn_output = rnn_decoder(cnn_output) # rnn_output.shape=(B,2)
                
                # statistics (after wraping the data)
                # Calculate the loss in degrees
                np_labels=labels.cpu().detach().numpy()
                np_labels_rad=np.arctan2(np_labels[:,0], np_labels[:,1])
                np_labels_deg=np_labels_rad*180/np.pi
                np_outputs=rnn_output.cpu().detach().numpy()
                np_outputs_rad=np.arctan2(np_outputs[:,0], np_outputs[:,1])
                np_outputs_deg=np_outputs_rad*180/np.pi
                #np_offset_deg=batch_offset_for_prediction(np_outputs_deg)
                np_offset_deg=batch_offset(np_outputs_deg, np_labels_deg)
                
                # (test) Check if offsets are used
                num_nonzero_val=np.count_nonzero(np_offset_deg)
                total_nonzero_val += num_nonzero_val
                    
                # save the predictions and ground truth to the csv file
                csv_output.write('{},{},{},{},{},{},{}\n'.format(current_frame_num[0], np_labels_deg[0], \
                    np_outputs_deg[0], np_labels[0,0], np_labels[0,1], np_outputs[0,0], np_outputs[0,1])) 
                
                # Calculate the loss
                loss = loss_function(rnn_output, labels)
                total_loss += loss.item()*labels.size(0)
                
                # Loss calculation (using MAE)
                total_loss_deg += np.sum(np.abs(np_outputs_deg - np_labels_deg + np_offset_deg))
                
        avg_loss = total_loss/dataset_size
        #print('Test loss: {:.4f}'.format(avg_loss))
                
        avg_loss_deg = total_loss_deg/dataset_size
        
    csv_output.close()
    
    return avg_loss, avg_loss_deg
                

if __name__ == '__main__':
    BATCH_SIZE=1#Note: BATCH_SIZE can only be 1
    NUM_WORKERS=4
    TIMESTEP=5
    res_size = 224        # ResNet image size
    # '7_subjects', 'GHS_2021_11_18', 'SCVS_2022_p1', '2_subjects', 'SAVS_2022', 'SCVS_2022_p2_train', 'SCVS_2022_p2'
    list_data_collection_location=['Rhodes_2023_12_4']#
    work_dir='/home/jianxig/CRNN/v2_1/l1/many-to-one/proposal/'
    cnn_weight_dir=work_dir+'best_weight_cnn.pt'
    rnn_weight_dir=work_dir+'best_weight_rnn.pt'
    data_dir='/project/regroff/jianxig/test_videos/github_video/' # '/project/regroff/jianxig/test_videos/dataset/'
    
    # Check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print('Using GPU\n')
    else:
        print('Using CPU\n')
        
    # Construct the network
    # EncoderCNN architecture
    cnn_encoder = network_many_to_one.ResCNNEncoder('', training=False)
    # DecoderRNN architecture
    CNN_embed_dim=128
    RNN_hidden_layers = 1
    RNN_hidden_nodes = 32
    RNN_FC_dim = 32
    RNN_dropout_p = 0.5       # dropout probability
    final_output_length=2
    rnn_decoder = network_many_to_one.DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, \
                                                 h_RNN=RNN_hidden_nodes, h_FC_dim=RNN_FC_dim, drop_p=RNN_dropout_p, \
                                                output_dim=final_output_length)
        
    # Load the weights
    cnn_encoder.load_state_dict(torch.load(cnn_weight_dir))
    rnn_decoder.load_state_dict(torch.load(rnn_weight_dir))
    
    # Upload the model to the device
    cnn_encoder = cnn_encoder.to(device)
    rnn_decoder = rnn_decoder.to(device)
    
    #
    for data_collection_location in list_data_collection_location:
        
        #
        print('============')
        print(data_collection_location)
        print('============')
    
        # Create directory for saving results
        output_dir=work_dir+data_collection_location+'/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else: # If the directory has exists, remove the content
            all_file_names=glob.glob(output_dir+'*')
            for single_name in all_file_names:
                os.remove(single_name)
                
        # Create a csv file to save the processing results
        output_csv=open(output_dir+'summary.csv', 'w')
        output_csv.write('Subject_id,Loss(cos-sin),Loss(deg)\n')
                
        # Read subject id
        subject_id_csv_dir=data_dir+data_collection_location+'.csv'
        df_name=pd.read_csv(subject_id_csv_dir, sep='\s*,\s*', usecols=['Subject_id', 'min(HSV)', 'max(HSV)'], \
                            engine='python', index_col=False)
        list_subject_id=df_name['Subject_id'].tolist()
        min_HSV_list=df_name['min(HSV)'].tolist()
        max_HSV_list=df_name['max(HSV)'].tolist()
        
        # Loss func 
        loss_function = nn.L1Loss()
        
        #
        result_dict, result_dict_deg={}, {}
        for i, file_name in enumerate(list_subject_id):
            
            # Get the HSV value
            min_HSV_str=min_HSV_list[i][1:-1]
            max_HSV_str=max_HSV_list[i][1:-1]
            split_min_HSV=min_HSV_str.split('-')
            split_max_HSV=max_HSV_str.split('-')
            
            # Ignore the trial if the HSV values are zero
            if int(split_min_HSV[0])==0 and int(split_min_HSV[1])==0 and int(split_min_HSV[2])==0:
                output_csv.write(f'{file_name},-1,-1\n')
                continue
            else:
                # For the videos used for training the algorithms
                if data_collection_location == 'SCVS_2022_p2_train':
                    csv_dir_test='/project/regroff/jianxig/img_dataset/'+file_name+'/d1.csv'
                    image_dir='/project/regroff/jianxig/img_dataset/'
                else:
                    csv_dir_test=data_dir+data_collection_location+'/'+file_name+'/d1.csv'
                    image_dir=data_dir+data_collection_location+'/'
                
                #
                shorten_file_name=abbr_file_name(file_name)
                csv_dir_output=output_dir+'test_results('+shorten_file_name+').csv'
                
                # Obtain the frame number
                gt_df=pd.read_csv(csv_dir_test, sep='\s*,\s*', header=[0], engine='python')
                sin_test=gt_df['sin'].to_numpy()
                cos_test=gt_df['cos'].to_numpy()
                area_num_test=gt_df['Area_Num'].to_numpy()
                image_name_test=gt_df['Image_name'].tolist()
                
                # 
                print('-'*10)
                print('Processing ', file_name)
                
                # 
                test_dataset = DIY_data_loader_many_to_one.SequenceDataset(sin_test, cos_test, area_num_test, \
                                                 image_name_test, image_dir, TIMESTEP, \
                                                 transform=transforms.Compose([
                                                 transforms.Resize((res_size,res_size),interpolation=transforms.InterpolationMode.BICUBIC),\
                                                 transforms.ToTensor(),\
                                                 # The mean and std are obtained from ImageNet data set.
                                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
                dataset_size=len(test_dataset)
                #print('Images in the test set: ', dataset_size)
                    
                # Data loader
                data_loader_test = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True,\
                                              num_workers=NUM_WORKERS, pin_memory=True)
                
                # Test the model
                avg_loss_single_file, avg_loss_deg_single_file=test_model(cnn_encoder, rnn_decoder, data_loader_test, dataset_size, \
                                                                          csv_dir_output, loss_function)
                
                # Save results
                output_csv.write('{},{},{}\n'.format(file_name, avg_loss_single_file, avg_loss_deg_single_file))
                output_csv.flush()
         
        #
        output_csv.close()