#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 18:12:47 2023

@author: jianxig

Note: Different from LSTM data generator, the CRNN data generator also using the current info to
predict the current output
"""
#import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os
import torch
#from torchvision.transforms.functional import pad

class SequenceDataset(Dataset):
    def __init__(self, np_sin, np_cos, area_num, list_image_name, input_img_dir, time_step=2, transform=None):
        # Note: Never use variables in __init__, or the data loader wouldn't support shuffle
        self.np_sin=np_sin
        self.np_cos=np_cos
        self.area_num=area_num
        self.list_image_name=list_image_name
        self.input_img_dir=input_img_dir
        self.time_step=time_step
        self.dataset_size=self.np_sin.shape[0] - self.time_step
        self.transform=transform
        
    def __len__(self):
        return self.dataset_size
    
    def read_images(self, list_img_dir):
        X = []
        for single_img_dir in list_img_dir:
            # Ignore the padded string '0'
            if single_img_dir=='0':
                image = torch.zeros((3,224,224), dtype=torch.uint8, device='cpu')
            else:
                image = Image.open(os.path.join(self.input_img_dir, single_img_dir))
    
                if self.transform:
                    image = self.transform(image)
    
            X.append(image)
            
        # The first dimension of X is timestep
        X = torch.stack(X, dim=0)
    
        return X
    
    def __getitem__(self, index):
        
        # If the input sequence and the target are in different area, insert 'black images'
        # at the beginning of network input
        if index<self.dataset_size and self.area_num[index]!=self.area_num[index+self.time_step-1]:
            
            # Find out and copy the data
            init_network_input=self.list_image_name[index:(index+self.time_step)].copy()
            init_network_target_sin=self.np_sin[index:(index+self.time_step)].copy()
            init_network_target_cos=self.np_cos[index:(index+self.time_step)].copy()                                      
            np_area_num=self.area_num[index:(index+self.time_step)].copy()
            
            # Create a mask
            # For the first sample of a suture
            np_mask = np_area_num==np_area_num[-1]
                    
            # Ignore values that belong to the previous area
            modified_network_input=\
            [single_name if np_mask[idx] else '0' for idx, single_name in enumerate(init_network_input)]
            modified_network_target_sin=init_network_target_sin*np_mask
            modified_network_target_cos=init_network_target_cos*np_mask
            
            # Note: 'modified_network_input' contains image names, while 'network_input' 
            # contains a stack of images
            network_input=self.read_images(modified_network_input)
            network_target_sin=modified_network_target_sin
            network_target_cos=modified_network_target_cos
            network_target=np.asarray([network_target_sin, network_target_cos]).transpose().astype('float')#Target
            current_area_num=self.area_num[index+self.time_step-1]#Area number
            split_current_img_name=modified_network_input[-1].split('/')
            current_frame_num=split_current_img_name[-1].split('_')[0]#Frame number
            
        # 
        else:
            # Note: 'modified_network_input' contains image names, while 'network_input' 
            # contains a stack of images
            modified_network_input=self.list_image_name[index:index+self.time_step]
            network_input=self.read_images(modified_network_input)
            network_target_sin=self.np_sin[index:(index+self.time_step)]
            network_target_cos=self.np_cos[index:(index+self.time_step)]
            network_target=np.asarray([network_target_sin, network_target_cos]).transpose().astype('float')#Target
            current_area_num=self.area_num[index+self.time_step-1]#Area number
            split_current_img_name=modified_network_input[-1].split('/')
            current_frame_num=split_current_img_name[-1].split('_')[0]#Frame number
        
        # network_target should be float, as the model is float
        return network_input, network_target, current_area_num, current_frame_num
    

        