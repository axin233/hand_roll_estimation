#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 12:47:25 2023

@author: jianxig
"""

import torch
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms, models
from torch.utils.data import DataLoader
import os
import glob

import my_data_loader_encode

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


# Find out the extremely small bboxes
# This function returns a boolean array. Small bboxes (i.e., width <= 50 or bbox height <= 50,)
# would be labelled as 0. Other bboxes would be labelled as 1
def detect_small_bbox(np_bbox_coor, width_threshold=50, height_threshold=50):
    np_width = np_bbox_coor[:,2] - np_bbox_coor[:,0]
    np_height = np_bbox_coor[:,3] - np_bbox_coor[:,1]
    bool_proper_width = np_width>width_threshold
    bool_proper_height = np_height>height_threshold
    bool_proper_bbox = np.bitwise_and(bool_proper_width, bool_proper_height)
    
    return bool_proper_bbox


def test_model(model_ft, data_loader_test, dataset_size, csv_dir_output, list_FN_from_bag, criterion):
    total_loss, total_loss_deg=0, 0
    total_nonzero_val=0
    
    # Predictions and ground truth will be saved within the csv file
    csv_output=open(csv_dir_output, 'w')
    csv_output.write('FN_from_bag,roll(GT),roll(Pred),sin(GT),cos(GT),sin(pred),cos(pred)\n')
    
    # Set the model to evaluation mode
    model_ft.eval()
    
    # Lists for saving data
    list_labels_deg=[]
    list_output_deg=[]
    list_labels_0=[]
    list_labels_1=[]
    list_output_0=[]
    list_output_1=[]
    
    # Disable gradient calculation
    with torch.no_grad():
        
        # To show the progress bar
        with tqdm(data_loader_test, unit='batch') as tepoch:
            
            # Iterate over data
            for sample_batch in tepoch:
                inputs = sample_batch['image'].to(device) 
                init_labels = sample_batch['sin_cos'].to(device) 
                # If dim 1 of init_labels is 1, remove dim 1 of init_labels
                squeezed_labels=torch.squeeze(init_labels, 1)
                # Cast the data type
                labels = squeezed_labels.type(torch.float32)
                
                # Inference
                outputs = model_ft(inputs)
                
                # Calculate the loss
                loss = criterion(outputs, labels)
                total_loss += loss.item()*labels.size(0)
                
                # statistics (after wraping the data)
                # Calculate the loss in degrees
                np_labels=labels.cpu().detach().numpy()
                np_labels_rad=np.arctan2(np_labels[:,0], np_labels[:,1])
                np_labels_deg=np_labels_rad*180/np.pi
                np_outputs=outputs.cpu().detach().numpy()
                np_outputs_rad=np.arctan2(np_outputs[:,0], np_outputs[:,1])
                np_outputs_deg=np_outputs_rad*180/np.pi
                #np_offset_deg=batch_offset_for_prediction(np_outputs_deg)
                np_offset_deg=batch_offset(np_outputs_deg, np_labels_deg)
                
                # (test) Check if offsets are used
                num_nonzero_val=np.count_nonzero(np_offset_deg)
                total_nonzero_val += num_nonzero_val
                
                # Save results to lists
                for i in range(np_labels_deg.shape[0]):
                    list_labels_deg.append(np_labels_deg[i])
                    list_output_deg.append(np_outputs_deg[i])
                    list_labels_0.append(np_labels[i,0])
                    list_labels_1.append(np_labels[i,1])
                    list_output_0.append(np_outputs[i,0])
                    list_output_1.append(np_outputs[i,1])
                
                # Loss calculation (using MAE)
                total_loss_deg += np.sum(np.abs(np_outputs_deg - np_labels_deg + np_offset_deg))
        
        # save the predictions and ground truth to the csv file
        for i in range(len(list_labels_deg)):
            csv_output.write('{},{},{},{},{},{},{}\n'.format(list_FN_from_bag[i], list_labels_deg[i], \
            list_output_deg[i], list_labels_0[i], list_labels_1[i], list_output_0[i], list_output_1[i]))        
        
        avg_loss=total_loss/dataset_size            
        avg_loss_deg = total_loss_deg/dataset_size
        
    csv_output.close()
    
    return avg_loss, avg_loss_deg
                

if __name__ == '__main__':
    BATCH_SIZE=1
    NUM_WORKERS=4
    criterion=nn.L1Loss()
    # '2_subjects', 'SAVS_2022', '7_subjects', 'GHS_2021_11_18', 'SCVS_2022_p1', 'SCVS_2022_p2_train', 'SCVS_2022_p2'
    list_data_collection_location=['Rhodes_2023_12_4']#
    work_dir='/home/jianxig/CNN_roll/dataset_v11/ResNet_FineT/2023-3-1/l1_RC/proposal/'
    weight_dir=work_dir+'best_weight_1.pt'
    data_dir='/project/regroff/jianxig/test_videos/github_video/'# '/project/regroff/jianxig/test_videos/dataset/'
    
    # Check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print('Using GPU\n')
    else:
        print('Using CPU\n')
    
    # The pre-trained model
    # Reference: https://www.kaggle.com/code/pmigdal/transfer-learning-with-resnet-50-in-pytorch
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
    
    # Load the weights
    model_ft.load_state_dict(torch.load(weight_dir))
    
    # Upload the model to the device
    model_ft = model_ft.to(device)
    
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
    
        #
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
                list_FN_from_bag=gt_df['RGB_FN_from_bag'].tolist()
                
                # 
                print('-'*10)
                print('Processing ', file_name)
                
                # 
                
                hand_dataset_test = my_data_loader_encode.Hand_Orientation_Dataset(csv_file=csv_dir_test,\
                                            root_dir=image_dir,\
                                            transform=transforms.Compose([
                                        transforms.Resize((224,224),interpolation=transforms.InterpolationMode.BICUBIC),\
                                        transforms.ToTensor(),\
                                        # The mean and std are obtained from ImageNet data set.
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
                dataset_size=len(hand_dataset_test)
                #print('Images in the test set: ', dataset_size)
                    
                # Data loader
                data_loader_test = DataLoader(hand_dataset_test, batch_size=BATCH_SIZE, shuffle=False, drop_last=True,\
                                              num_workers=NUM_WORKERS, pin_memory=True)
                
                # Test the model
                avg_loss_single_file, avg_loss_deg_single_file=test_model(model_ft, data_loader_test, \
                                                dataset_size, csv_dir_output, list_FN_from_bag, criterion)
                
                # Save results
                output_csv.write('{},{},{}\n'.format(file_name, avg_loss_single_file, avg_loss_deg_single_file))
                output_csv.flush()
            
        #
        output_csv.close()