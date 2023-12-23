# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 21:27:08 2022

@author: jianxig
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   

# Remove rows that have the same frame number
def csv_drop_duplicated_FN(csv_dir):
    df=pd.read_csv(csv_dir, sep='\s*,\s*', header=[0], engine='python')
    
    # Drop duplicated rows
    df=df.drop_duplicates(subset='RGB_FN_from_bag', keep='first')
    
    # Save results
    df.to_csv(csv_dir, index=False)
    
    return

# Generate a txt file that allows user to save the problematic frame numbers
def generate_problematic_FN_txt(output_txt_dir):
    output_txt=open(output_txt_dir, 'w')
    output_txt.write('# This file includes the frame number with incorrect bbox coordinates\n')
    output_txt.write('# Format: \n')
    output_txt.write('#	1: The bbox in Frame 1 will be adjusted\n')
    output_txt.write('#	1-100: The bbox from Frame 1 to 100 will be adjusted\n')
    output_txt.write('# (Note: In each row, the frame number must be in ascending order. \n')
    output_txt.write('# However, the frame numbers in different rows can be in random order)\n')
    output_txt.close()
    
    return
    

def read_EnEx_info(csv_name):
    df=pd.read_csv(csv_name,sep='\s*,\s*',header=[0],engine='python')
    np_en_time=df['entry time'].to_numpy()
    np_ex_time=df['exit time'].to_numpy()
    np_area_num=df['AreaNum'].to_numpy()
    return np_en_time, np_ex_time, np_area_num
    

def read_external_cam_data(csv_name):
    df=pd.read_csv(csv_name,sep='\s*,\s*',header=[0],engine='python')
    rgb_frameNum=df['frame_number_from_RGB'].to_numpy()
    external_cam_time=df['true_time_stamp (second)'].to_numpy()
    return rgb_frameNum, external_cam_time


def read_IMU_data(IMU_name):
    df=pd.read_csv(IMU_name,sep='\s*,\s*',header=[0],engine='python')
    return df


def time_for_first_frame(np_timeStamp,target_timeStamp, extra_time):
    # Binary search
    idx=np.searchsorted(np_timeStamp, (target_timeStamp-extra_time))
    timeStamp=np_timeStamp[idx]
    # Note: For the external camera, frame_num = index + 1
    #idx+=1 
    return timeStamp


def time_for_last_frame(np_timeStamp,target_timeStamp, extra_time=0):
    # Binary search
    idx=np.searchsorted(np_timeStamp, (target_timeStamp+extra_time)) 
    timeStamp=np_timeStamp[idx]
    # Note: For the external camera, frame_num = index + 1
    #idx+=1
    return timeStamp


def find_nearest_idx(np_timeStamp, target_timeStamp):
    # Binary search
    target_idx=np.searchsorted(np_timeStamp, target_timeStamp)
    
    return target_idx


def find_first_time_stamp_idx(df,timeStamp):
    
    # If timeStamp exists
    df_idx=df[df['time']==(timeStamp)]
    
    # If timeStamp does not exist, find the time stamp just below timeStamp
    if df_idx.empty:
        idx=df[df['time']>timeStamp]['time'].idxmin()
    else:
        idx=df_idx.index[0]
        
    return idx


def find_last_time_stamp_idx(df,timeStamp):
    
    # If timeStamp exists
    df_idx=df[df['time']==timeStamp]
    
    # If timeStamp does not exist, find the time stamp just above timeStamp
    if df_idx.empty:
        idx=df[df['time']<timeStamp]['time'].idxmax()
    else:
        idx=df_idx.index[0]
        
    return idx


def read_eluer_angle(df,location,first_idx,second_idx):
    
    # Read both the hand and wrist data
    df_hand_wrist=df.iloc[first_idx:second_idx]
    df_single_sensor=df_hand_wrist[df_hand_wrist['location']==location]
    time=df_single_sensor['time'].to_numpy()
    roll=df_single_sensor['roll'].to_numpy()
    pitch=df_single_sensor['pitch'].to_numpy()
    yaw=df_single_sensor['yaw'].to_numpy()
    return time, roll, pitch, yaw


# For drawing the figure
def summary_view(rgb_img, np_depth_img, boundRect, ex_cam_start_time, ex_cam_end_time, frame_num_from_bag, target_time, \
                 np_time_for_plot, IMU_idx_1, np_roll, np_pitch, np_yaw, current_roll_value, current_pitch_value, \
                 current_yaw_value, sin_value, cos_value):
    
    # Draw the bounding Rectangle on the images
    if boundRect[2]>0 and boundRect[3]>0:
        cv2.rectangle(rgb_img, (int(boundRect[0]), int(boundRect[1])), \
              (int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])), (0,255,0), 5)
        cv2.rectangle(np_depth_img, (int(boundRect[0]), int(boundRect[1])), \
              (int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])), (0,255,0), 5)
    
    # Note: Make sure the multiplication of figsize and dip is 
    # the same as the VideoWriter resolution
    fig = plt.figure()
    fig.set_figwidth(12.8)
    fig.set_figheight(7.2)
    fig.set_dpi(100)
    
    # Creating grid for subplots
    ax_img_1 = plt.subplot2grid(shape=(6, 3), loc=(0, 0), rowspan=3)
    ax_img_2 = plt.subplot2grid(shape=(6, 3), loc=(3, 0), rowspan=3)
    ax_roll = plt.subplot2grid(shape=(6, 3), loc=(0, 1), rowspan=2, colspan=2)
    ax_pitch = plt.subplot2grid(shape=(6, 3), loc=(2, 1), rowspan=2, colspan=2)
    ax_yaw = plt.subplot2grid(shape=(6, 3), loc=(4, 1), rowspan=2, colspan=2)
    
    # Draw the figure
    ax_img_1.imshow(rgb_img)
    ax_img_1.axis('off')
    ax_img_2.imshow(np_depth_img)
    ax_img_2.axis('off')
    ax_img_1.set_title('S:{:.2f} E:{:.2f} FN:{} T:{:.2f}'.format(ex_cam_start_time,ex_cam_end_time,frame_num_from_bag,target_time))
    ax_roll.plot(np_time_for_plot, np_roll, label='roll')
    ax_roll.axvline(np_time_for_plot[IMU_idx_1], color='r')
    #ax_roll.axvline(np_time_for_plot[IMU_idx_2], color='r')
    ax_roll.set_title('S: {:.2f} E: {:.2f} T: {:.2f} R: {:.2f} P: {:.2f} Y: {:.2f}'.\
        format(np_time_for_plot[0], np_time_for_plot[-1], np_time_for_plot[IMU_idx_1], current_roll_value, \
               current_pitch_value, current_yaw_value))
    ax_roll.set_xticklabels([])
    ax_roll.legend(loc='upper right')
    ax_pitch.plot(np_time_for_plot, np_pitch, label='pitch')
    ax_pitch.axvline(np_time_for_plot[IMU_idx_1], color='r')
    #ax_pitch.axvline(np_time_for_plot[IMU_idx_2], color='r')
    ax_pitch.set_xticklabels([])
    ax_pitch.legend(loc='upper right')
    ax_yaw.plot(np_time_for_plot, np_yaw, label='yaw')
    ax_yaw.axvline(np_time_for_plot[IMU_idx_1], color='r')
    #ax_yaw.axvline(np_time_for_plot[IMU_idx_2], color='r')
    #ax_yaw.set_xticklabels([])
    ax_yaw.legend(loc='upper right')
    
    # automatically adjust padding horizontally as well as vertically.
    plt.tight_layout()
                
    # Redraw the canvas
    fig.canvas.draw()
                
    # Convert matplotlib figure to opencv mat
    labeled_img=cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    
    # Close the current figure
    plt.close()
    
    return labeled_img