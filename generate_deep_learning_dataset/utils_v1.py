# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 21:27:08 2022

@author: jianxig
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   

def abbr_file_name(input_file_name):
    single_file_name_split=input_file_name.split('_')
    single_file_name_split_1=single_file_name_split[1].split('-')
    shorten_file_name=single_file_name_split_1[0][:4]+'-'+\
        single_file_name_split_1[4]+'-'+single_file_name_split_1[5]
    
    return shorten_file_name


def read_bbox_coordinates(input_bbox_csv_dir):
    df=pd.read_csv(input_bbox_csv_dir, sep='\s*,\s*',header=[0],engine='python')
    return df


def read_bbox_single_area(df, area_number):
    df_single_area=df[df['Area_Num']==area_number]
    return df_single_area

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


def find_nearest_idx_v1(np_timeStamp, target_timeStamp):
    # The absolute difference
    np_diff=np.abs(np_timeStamp-target_timeStamp)
    # 
    target_idx=np.argmin(np_diff)
    
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


def read_eluer_and_gyro(df,location,first_idx,second_idx):
    
    # Read both the hand and wrist data
    df_hand_wrist=df.iloc[first_idx:second_idx]
    df_single_sensor=df_hand_wrist[df_hand_wrist['location']==location]
    time=df_single_sensor['time'].to_numpy()
    roll=df_single_sensor['roll'].to_numpy()
    pitch=df_single_sensor['pitch'].to_numpy()
    yaw=df_single_sensor['yaw'].to_numpy()
    gyro_x=df_single_sensor['gyrX'].to_numpy()
    gyro_y=df_single_sensor['gyrY'].to_numpy()
    gyro_z=df_single_sensor['gyrZ'].to_numpy()
    return time, roll, pitch, yaw, gyro_x, gyro_y, gyro_z


# For drawing the figure
def summary_view(rgb_img, np_depth_img, boundRect, ex_cam_start_time, ex_cam_end_time, frame_num_from_bag, target_time, \
                 np_time_for_plot, IMU_idx_1, np_roll, np_pitch, np_yaw, current_roll_value, current_pitch_value, \
                 current_yaw_value, sin_value, cos_value, np_roll_for_plot, np_pitch_for_plot, np_yaw_for_plot):
    
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
    ax_roll.plot(np_time_for_plot, np_roll_for_plot, label='roll (original)')
    ax_roll.axvline(np_time_for_plot[IMU_idx_1], color='r')
    ax_roll.scatter([np_time_for_plot[IMU_idx_1]], [current_roll_value], color='g', zorder=5)
    #ax_roll.axvline(np_time_for_plot[IMU_idx_2], color='r')
    ax_roll.set_title('S: {:.2f} E: {:.2f} T: {:.2f} R: {:.2f} P: {:.2f} Y: {:.2f}'.\
        format(np_time_for_plot[0], np_time_for_plot[-1], np_time_for_plot[IMU_idx_1], current_roll_value, \
               current_pitch_value, current_yaw_value))
    ax_roll.set_xticklabels([])
    ax_roll.legend(loc='upper right')
    ax_pitch.plot(np_time_for_plot, np_pitch, label='pitch')
    ax_pitch.plot(np_time_for_plot, np_pitch_for_plot, label='pitch (original)')
    ax_pitch.axvline(np_time_for_plot[IMU_idx_1], color='r')
    ax_pitch.scatter([np_time_for_plot[IMU_idx_1]], [current_pitch_value], color='g', zorder=5)
    #ax_pitch.axvline(np_time_for_plot[IMU_idx_2], color='r')
    ax_pitch.set_xticklabels([])
    ax_pitch.legend(loc='upper right')
    ax_yaw.plot(np_time_for_plot, np_yaw, label='yaw')
    ax_yaw.plot(np_time_for_plot, np_yaw_for_plot, label='yaw (original)')
    ax_yaw.axvline(np_time_for_plot[IMU_idx_1], color='r')
    ax_yaw.scatter([np_time_for_plot[IMU_idx_1]], [current_yaw_value], color='g', zorder=5)
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