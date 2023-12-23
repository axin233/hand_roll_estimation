#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 23:02:00 2022

@author: jianxig
"""

import os
import torch
#from skimage import io
from PIL import Image, ImageFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import utils
from torchvision.transforms.functional import pad

class Hand_Orientation_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file,sep='\s*,\s*',header=[0],engine='python')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.data_frame.iloc[idx, 13])
        image = Image.open(img_name)
        df_labels=self.data_frame.iloc[idx,5:7]
        np_labels=np.asarray(df_labels)
        np_labels_1=np_labels.astype('float').reshape(-1,2)

        if self.transform:
            image = self.transform(image)
            
        sample = {'image': image, 'sin_cos': np_labels_1}    

        return sample
    

# For visualizing a batch within data_loader    
def visualize_single_batch(data_loader):
    plt.figure()
    for batch_idx, sample_batch in enumerate(data_loader):
        
        # (torch tensor)
        print(batch_idx, sample_batch['image'].size(), len(sample_batch['sin_cos'])) 
        
        # (torch tensor) Separate images and labels
        image_batch, label_batch = sample_batch['image'], sample_batch['sin_cos']
        
        # (torch tensor)
        grid=utils.make_grid(image_batch)
        
        # (torch tensor) Reverse the normalization
        t_mean=torch.tensor([0.485, 0.456, 0.406])
        t_std=torch.tensor([0.229, 0.224, 0.225])
        t_reversed_img=grid * t_std[:,None,None] + t_mean[:,None,None]
        
        # (torch tensor) Convert torch.tensor to np.array
        np_img=t_reversed_img.numpy()
        
        # Change the order of dimension
        np_img=np_img.transpose(1,2,0)
        
        # Put the sin-cos pair to list
        list_sin_cos=[]
        for i in range(label_batch.size()[0]):
            list_sin_cos.append('({:.2f}, {:.2f})'.format(label_batch[i][0][0], label_batch[i][0][1]))
        
        plt.imshow(np_img)
        plt.title(list_sin_cos)
        plt.axis('off')
        plt.show()
        
        if batch_idx==0:
            break