#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 23:01:16 2022

@author: jianxig
"""
import torch
import copy
import numpy as np
#import random
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import my_data_loader_encode

# =============================================================================
# # Set random seed
# seed=0
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)  # Numpy module.
# random.seed(seed)  # Python random module.
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# =============================================================================

# Wrap the results, then calculate the error for a specific batch
def batch_offset(np_outputs, np_labels):
    np_diff = np_outputs - np_labels
    bool_top_val=np_diff>=180
    bool_bottom_val=np_diff<=-180
    offset_larger_180=bool_top_val*(-360.0)
    offset_smaller_180=bool_bottom_val*(360.0)
    np_offset=offset_larger_180+offset_smaller_180
    
    return np_offset


def train_model(model, data_loaders_dict, dataset_sizes_dict, best_model_dir, criterion, optimizer, \
                num_epochs=25):
    
    # For recording the loss
    list_train_loss_deg, list_val_loss_deg = [0]*num_epochs, [0]*num_epochs
    list_train_loss, list_val_loss = [0]*num_epochs, [0]*num_epochs

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e5

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode (The same as model.train(mode=False))

            running_loss = 0.0
            running_loss_deg=0.0
            
            # To show the progress bar
            with tqdm(data_loaders_dict[phase], unit='batch') as tepoch:

                # Iterate over data.
                for batch_idx, sample_batch in enumerate(tepoch):
                    inputs = sample_batch['image'].to(device)
                    init_labels = sample_batch['sin_cos'].to(device)
                    squeezed_labels=torch.squeeze(init_labels, 1)
                    # Cast the data type
                    labels = squeezed_labels.type(torch.float32)
    
                    # zero the parameter gradients
                    optimizer.zero_grad()
    
                    # forward
                    # track history if only in train
                    # torch.set_grad_enabled: enable or disable grads based on its argument mode
                    # If mode == True, enable grad. If mode == False, disable grad
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        #_, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
    
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
    
                    # statistics
                    running_loss += loss.item()*labels.size(0)
                    # Calculate the loss in degrees
                    np_labels=labels.cpu().detach().numpy()#np_labels.shape=(B, 2)
                    np_labels_rad=np.arctan2(np_labels[:,0], np_labels[:,1])#np_labels_rad.shape=(B,)
                    np_labels_deg=np_labels_rad*180/np.pi
                    np_outputs=outputs.cpu().detach().numpy()#np_outputs.shape=(B, 2)
                    np_outputs_rad=np.arctan2(np_outputs[:,0], np_outputs[:,1])#np_outputs_rad.shape=(B,)
                    np_outputs_deg=np_outputs_rad*180/np.pi
                    #np_offset_deg=batch_offset_for_prediction(np_outputs_deg)
                    np_offset_deg=batch_offset(np_outputs_deg, np_labels_deg)#np_offset_deg.shape=(B,)
                    
                    # MAE
                    running_loss_deg += np.sum(np.abs(np_outputs_deg - np_labels_deg + np_offset_deg))

            epoch_loss = running_loss / dataset_sizes_dict[phase]
            print(f'{phase} Loss: {epoch_loss:.4f}')
            epoch_loss_deg = running_loss_deg / dataset_sizes_dict[phase]
            print(f'{phase} Loss (deg): {epoch_loss_deg:.4f}')
            
            # For recording the loss
            if phase=='train':
                list_train_loss[epoch]=epoch_loss
                list_train_loss_deg[epoch]=epoch_loss_deg
            else:
                list_val_loss[epoch]=epoch_loss
                list_val_loss_deg[epoch]=epoch_loss_deg

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, best_model_dir)

        print()

    print(f'Best val Loss: {best_loss:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, list_train_loss_deg, list_val_loss_deg, list_train_loss, list_val_loss


if __name__=='__main__':
    TRIAL=1
    BATCH_SIZE=128
    EPOCH=20
    NUM_WORKERS=4
    img_dir='/zfs/cucat/jianxig/img_dataset/'
    csv_dir='/zfs/cucat/jianxig/dataset_v11/CNN/'
    output_dir='/home/jianxig/CNN_roll/dataset_v11/ResNet_FineT/2023-3-1/l1_RC/'
    csv_dir_train=csv_dir+'d1_train.csv'
    csv_dir_validation=csv_dir+'d1_develop.csv'
    csv_dir_history=output_dir+'history_'+str(TRIAL)+'.csv'
    best_model_dir=output_dir+'best_weight_'+str(TRIAL)+'.pt'
    
    # (Training set)
    hand_dataset_train = my_data_loader_encode.Hand_Orientation_Dataset(csv_file=csv_dir_train,\
                                root_dir=img_dir,\
                                transform=transforms.Compose([
                            transforms.ColorJitter(brightness=(0.7,1.2),contrast=(0.7,1.2),saturation=(0.8,1.3)),\
                            transforms.RandomResizedCrop((224,224), scale=(0.7,1.0), ratio=(1.0,1.0), interpolation=transforms.InterpolationMode.BICUBIC),\
                            transforms.ToTensor(),\
                            # The mean and std are obtained from ImageNet data set.
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
    
    # (Training set) Data loader
    data_loader_train = DataLoader(hand_dataset_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, \
                                   num_workers=NUM_WORKERS, pin_memory=True)
    
    # (Validation set)
    hand_dataset_validation = my_data_loader_encode.Hand_Orientation_Dataset(csv_file=csv_dir_validation,\
                                root_dir=img_dir,\
                                transform=transforms.Compose([
                            transforms.Resize((224,224), interpolation=transforms.InterpolationMode.BICUBIC),\
                            transforms.ToTensor(),\
                            # The mean and std are obtained from ImageNet data set.
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
        
    # (Validation set) Data loader
    data_loader_validation = DataLoader(hand_dataset_validation, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, \
                                        num_workers=NUM_WORKERS, pin_memory=True)
    
    # Check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print('Using GPU\n')
    else:
        print('Using CPU\n')
    
    # The pre-trained model
    # Reference: https://www.kaggle.com/code/pmigdal/transfer-learning-with-resnet-50-in-pytorch
    model_ft = models.resnet50(pretrained=True)
    
    # Freeze the a portion of the network
    # Reference: https://medium.com/udacity-pytorch-challengers/ideas-on-how-to-fine-tune-a-pre-trained-model-in-pytorch-184c47185a20
    for name, child in model_ft.named_children():
        if name not in ['layer3', 'layer4']:
            print('Freeze ', name)
            for param in child.parameters():
                param.requires_grad = False
        else:
            print('Unfreeze ', name)
            for param in child.parameters():
                param.requires_grad = True
        
    # Replace the output layer with other FC layers
    # For CRNN, batchNorm1d uses momentum=0.01
    model_ft.fc=nn.Sequential(
               nn.Linear(2048, 512),
               nn.BatchNorm1d(512),
               nn.ReLU(inplace=True),
               #nn.Dropout(p=0.5),#Dropout
               nn.Linear(512, 128),
               nn.BatchNorm1d(128),
               nn.ReLU(inplace=True),
               #nn.Dropout(p=0.5),#Dropout
               nn.Linear(128, 2))

    # Show the network
    #print(model_ft)
    
    # Upload the model to the device
    model_ft = model_ft.to(device)
    
    # Find the trainable layers
    # Reference: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    print('Trainable layers:')
    params_to_update=[]
    for name, param in model_ft.named_parameters():
        if param.requires_grad==True:
            params_to_update.append(param)
            print('\t',name)
    
    # The loss function 
    criterion = nn.L1Loss()
    
    # The optimizer (Note: The optimizer only includes the trainable params)
    optimizer_ft=optim.Adam(params_to_update, lr=1e-4)
    
    #
    data_loaders_dict={'train':data_loader_train, 'val':data_loader_validation}
    dataset_sizes_dict={'train':len(hand_dataset_train), 'val':len(hand_dataset_validation)}
    
    # Train the model
    model_ft, train_loss_deg, val_loss_deg, train_loss, val_loss = \
        train_model(model_ft, data_loaders_dict, dataset_sizes_dict, best_model_dir, \
                                                 criterion, optimizer_ft, num_epochs=EPOCH)
        
    # Save the history
    csv_history=open(csv_dir_history,'w')
    csv_history.write('Epoch,Loss(train),Loss(deg)(train),Loss(validation),Loss(deg)(validation)\n')
    for i in range(len(train_loss)):
        csv_history.write('{},{},{},{},{}\n'.format\
                         (i+1,train_loss[i],train_loss_deg[i],val_loss[i],val_loss_deg[i]))
    csv_history.close()
        
    
    
