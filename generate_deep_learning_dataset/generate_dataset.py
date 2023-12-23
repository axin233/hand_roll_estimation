# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 15:39:13 2022

@author: jianxig
"""

import cv2
import os
import datetime
import pandas as pd
import pyrealsense2 as rs
import numpy as np
from utils_v1 import read_EnEx_info, read_external_cam_data, read_IMU_data, time_for_first_frame, time_for_last_frame, \
    find_nearest_idx_v1, find_first_time_stamp_idx, find_last_time_stamp_idx, read_eluer_and_gyro, summary_view, \
    abbr_file_name, read_bbox_coordinates, read_bbox_single_area
from dataset_func import generate_data_set

def convert_bag_video(log_txt, path_for_bag_file, path_for_avi_output, path_for_IMU_csv, \
                      path_for_external_cam_csv, path_for_EnExP, output_dataset_dir, \
                      output_bb_csv_dir, output_d1_csv_dir, output_d2_csv_dir, \
                      output_d3_csv_dir, input_bbox_csv_dir, hsv_min_tuple, hsv_max_tuple, \
                      GENERATE_VIDEO=False):
    IMU_idx_1=-1
    target_time=-1
    exit_the_program=False
    area_idx = 0
    number_of_recorded_frame = 1
    extra_time_start = 0 # Info in [needle_en_time - extra_time_start, needle_ex_time + extra_time_end] are shown
    extra_time_end = 0
    time_for_buffer=0.5 # It is used for avg roll, pitch, and yaw calculation
    location='hand' # Before SCVS (2022), use 'hand' rather than 'Hand'
    
    # Read data from csv files
    np_rgb_frameNum, external_cam_time = read_external_cam_data(path_for_external_cam_csv)
    df_IMU= read_IMU_data(path_for_IMU_csv)
    np_en_time, np_ex_time, np_area_num = read_EnEx_info(path_for_EnExP)
    df_bbox=read_bbox_coordinates(input_bbox_csv_dir)
    area_num=np_area_num[area_idx]
    
    # Read data for Area 1
    ex_cam_start_time = time_for_first_frame(external_cam_time, np_en_time[area_idx], extra_time_start)
    ex_cam_end_time = time_for_last_frame(external_cam_time, np_ex_time[area_idx], extra_time_end)
    start_idx = find_first_time_stamp_idx(df_IMU,ex_cam_start_time)
    end_idx = find_last_time_stamp_idx(df_IMU,ex_cam_end_time)
    np_time_for_plot, np_roll, np_pitch, np_yaw, gyro_x, gyro_y, gyro_z = \
        read_eluer_and_gyro(df_IMU, location, start_idx, end_idx)
    df_bbox_single_area=read_bbox_single_area(df_bbox, area_num)
    
    # Copy the data
    np_roll_for_plot=np.copy(np_roll)
    np_pitch_for_plot=np.copy(np_pitch)
    np_yaw_for_plot=np.copy(np_yaw)
    
    # Unwrap data (i.e., Make the adjacent differences are never greater than pi)
    np_roll_radian = np.unwrap((np_roll*np.pi/180))
    np_roll = np_roll_radian * 180/np.pi
    np_pitch_radian = np.unwrap((np_pitch*np.pi/180))
    np_pitch = np_pitch_radian * 180/np.pi
    np_yaw_radian = np.unwrap((np_yaw*np.pi/180))
    np_yaw = np_yaw_radian * 180/np.pi
    
    # Open a csv file for saving info
    bounding_box_csv=open(output_bb_csv_dir,'w')
    bounding_box_csv.write('Area_Num,RGB_FN_from_bag,Has_BB,endpoint_1_x,endpoint_1_y,endpoint_2_x,endpoint_2_y\n')
    d1_csv=open(output_d1_csv_dir,'w')
    d1_csv.write('Area_Num,RGB_FN_from_bag,roll,pitch,yaw,sin,cos,gyro_x,gyro_y,gyro_z,AVG_gyro_x,AVG_gyro_y,AVG_gyro_z,Image_name\n')
    d2_csv=open(output_d2_csv_dir,'w')
    d2_csv.write('Area_Num,RGB_FN_from_bag,roll,pitch,yaw,sin,cos,gyro_x,gyro_y,gyro_z,AVG_gyro_x,AVG_gyro_y,AVG_gyro_z,Image_name\n')
    d3_csv=open(output_d3_csv_dir,'w')
    d3_csv.write('Area_Num,RGB_FN_from_bag,roll,pitch,yaw,sin,cos,gyro_x,gyro_y,gyro_z,AVG_gyro_x,AVG_gyro_y,AVG_gyro_z,Image_name\n')
    
    # Define the codec and create VideoWriter object
    if GENERATE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        out_video = cv2.VideoWriter(path_for_avi_output, fourcc, 60.0, (1280, 720))
    
    # Create opencv window to render image in
    cv2.namedWindow("Converted image", cv2.WINDOW_NORMAL)
    
    # Create pipeline
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()

    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    # 'False' refers to 'no repeat_playback'
    rs.config.enable_device_from_file(config, path_for_bag_file, False)
    
    # Configure the pipeline
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
    
    # Start streaming from file
    profile = pipeline.start(config)
    
    # Get playback device
    playback=profile.get_device().as_playback()
    
    # Disable real-time playback
    playback.set_real_time(False)
    
    # Obtain depth scale to convert unit from pixel to meter
    depth_sensor=profile.get_device().first_depth_sensor()
    depth_scale=depth_sensor.get_depth_scale()
    
    # Create colorizer object
    colorizer = rs.colorizer()
    
    while(True):
        
        # Get frameset of depth
        frame_set = pipeline.try_wait_for_frames(10000)
        
        # Break the loop when reaching the end of the video
        if(frame_set[0]==False):
            print("Found the end of the video.")
            break
        
        # Align the rgb frame and the depth frame to the color viewport
        align = rs.align(rs.stream.color)
        aligned = align.process(frame_set[1])
        
        # Get RGB frame and depth frame
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        
        # Obtain the frame number from the bag file
        rgb_frame_num_from_bag=color_frame.get_frame_number()
        
        # Find out the time stamp for the RGB frame
        # Note: Since 'wait_for_frame()' is used, one RGB frame might have multiple time stamps
        RGB_idx=np.where(np_rgb_frameNum==rgb_frame_num_from_bag)
        if RGB_idx[0].shape[0] >= 1:
            prev_target_time=target_time
            target_time=external_cam_time[RGB_idx[0][0]]
        else:
            log_txt.write('rgb_frame_num_from_bag is not within the csv file.\n')
            log_txt.write('rgb_frame_num_from_bag: {}\n'.format(rgb_frame_num_from_bag))
            log_txt.write('Skipping this frame.\n')
            log_txt.flush()
            continue
        
        # Record information when suturing
        if target_time >= (ex_cam_start_time+time_for_buffer) and target_time < ex_cam_end_time:
            
            # Obtain the 1-channel depth image
            np_depth_img_1 = np.asanyarray(depth_frame.get_data())
            
            # Colorize depth frame to jet colormap
            depth_frame_colored = colorizer.colorize(depth_frame)
        
            # Convert color_frame and depth_frame to numpy array
            np_rgb_img = np.asanyarray(color_frame.get_data())
            np_depth_img = np.asanyarray(depth_frame_colored.get_data())
            
            # Find out the range of IMU data
            IMU_idx_0=find_nearest_idx_v1(np_time_for_plot, prev_target_time)
            IMU_idx_1=find_nearest_idx_v1(np_time_for_plot, target_time)
            
            # If IMU_idx_0 == IMU_idx_1, it indicates there is a duplicated frame,
            # so we should ignore the current frame.
            if IMU_idx_0 == IMU_idx_1:
                continue
            
            # Avoid the index is out of range
            #if IMU_idx_1>(np_roll.shape[0]-1) or IMU_idx_2>(np_roll.shape[0]-1):
            if IMU_idx_0<0:
                log_txt.write('Error! IMU_idx_0 out of range.\n')
                log_txt.write('np_roll.shape[0] - 1: {}'.format(np_roll.shape[0]-1))
                log_txt.write('IMU_idx_0: {}'.format(IMU_idx_0))
                IMU_idx_0=0
                log_txt.flush()    
            elif IMU_idx_1>(np_roll.shape[0]-1):
                log_txt.write('Error! IMU_idx_1 out of range.\n')
                log_txt.write('np_roll.shape[0] - 1: {}'.format(np_roll.shape[0]-1))
                log_txt.write('IMU_idx_1: {}'.format(IMU_idx_1))
                #print('IMU_idx_2: ', IMU_idx_2)
                IMU_idx_1=np_roll.shape[0]-1
                #IMU_idx_2=np_roll.shape[0]-1
                log_txt.flush()      
                
            # Find out the bbox coordinates for the current frame
            df_bbox_single_frame = df_bbox_single_area\
                [df_bbox_single_area['RGB_FN_from_bag']==rgb_frame_num_from_bag]
            if len(df_bbox_single_frame)==0:
                top_x, top_y, bottom_x, bottom_y = 0, 0, 0, 0
            else: 
                # Note: bbox.csv does not have duplicated rows, so the following method works
                top_x = int(df_bbox_single_frame['endpoint_1_x'])
                top_y = int(df_bbox_single_frame['endpoint_1_y'])
                bottom_x = int(df_bbox_single_frame['endpoint_2_x'])
                bottom_y = int(df_bbox_single_frame['endpoint_2_y'])
                
            # Generate data sets
            np_depth_img_1_cp=np.copy(np_depth_img_1)
            np_rgb_img_cp=np.copy(np_rgb_img)
            boundRect, current_roll_value, current_pitch_value, current_yaw_value, \
            sin_value, cos_value, np_bgr_img = generate_data_set(np_depth_img_1_cp, \
                                        np_rgb_img_cp, depth_scale, output_dataset_dir, \
                                        rgb_frame_num_from_bag, np_roll, np_pitch, np_yaw, \
                                        gyro_x, gyro_y, gyro_z, IMU_idx_0, IMU_idx_1, \
                                        bounding_box_csv, d1_csv, d2_csv, d3_csv, \
                                        hsv_min_tuple, hsv_max_tuple, area_num, target_time, \
                                        top_x, top_y, bottom_x, bottom_y, log_txt)
            
            # For generating the sychronized videos
            if GENERATE_VIDEO:
                # For checking the synchronization of video frames and IMU data
                np_depth_img_cp=np.copy(np_depth_img)
                np_rgb_img_cp=np.copy(np_rgb_img)
                labeled_img=summary_view(np_rgb_img_cp, np_depth_img_cp, boundRect, \
                                ex_cam_start_time, ex_cam_end_time, rgb_frame_num_from_bag, \
                                target_time, np_time_for_plot, IMU_idx_1, np_roll, np_pitch, np_yaw, \
                                current_roll_value, current_pitch_value, current_yaw_value,  \
                                sin_value, cos_value, np_roll_for_plot, np_pitch_for_plot, \
                                np_yaw_for_plot)
                 
                # Save all image to video file
                out_video.write(labeled_img)
            
            
            # Render image in opencv window
            cv2.imshow("Converted image", np_bgr_img)

            # Count the recorded frames
            number_of_recorded_frame = number_of_recorded_frame+1
            
        # After processing a single area, update info
        if target_time >= ex_cam_end_time:
           area_idx+=1
           
           if area_idx>=12:
               print('Finish 12 areas.')
               break
           
           if area_idx==np_en_time.shape[0]:
               print('The detection result does not include all 12 areas.')
               log_txt.write('The detection result does not include all 12 areas.\n')
               break
           
           area_num=np_area_num[area_idx]
           ex_cam_start_time = time_for_first_frame(external_cam_time, np_en_time[area_idx], extra_time_start)
           ex_cam_end_time = time_for_last_frame(external_cam_time, np_ex_time[area_idx], extra_time_end)
           start_idx = find_first_time_stamp_idx(df_IMU,ex_cam_start_time)
           end_idx = find_last_time_stamp_idx(df_IMU,ex_cam_end_time)
           np_time_for_plot, np_roll, np_pitch, np_yaw, gyro_x, gyro_y, gyro_z = \
               read_eluer_and_gyro(df_IMU, location, start_idx, end_idx) 
           df_bbox_single_area=read_bbox_single_area(df_bbox, area_num)
           
           # Copy the data
           np_roll_for_plot=np.copy(np_roll)
           np_pitch_for_plot=np.copy(np_pitch)
           np_yaw_for_plot=np.copy(np_yaw)
           
           # Unwrap data (i.e., Make the adjacent differences are never greater than pi)
           np_roll_radian = np.unwrap((np_roll*np.pi/180))
           np_roll = np_roll_radian * 180/np.pi
           np_pitch_radian = np.unwrap((np_pitch*np.pi/180))
           np_pitch = np_pitch_radian * 180/np.pi
           np_yaw_radian = np.unwrap((np_yaw*np.pi/180))
           np_yaw = np_yaw_radian * 180/np.pi
        
        # if pressed escape exit program
        if cv2.waitKey(1) == 27:
            exit_the_program=True
            break    
    
    bounding_box_csv.close()
    d1_csv.close()
    d2_csv.close()
    d3_csv.close()
    pipeline.stop()
    #out_video.release()
    cv2.destroyAllWindows()
    
    # test
    print('number_of_recorded_frame: ', number_of_recorded_frame)
    
    return exit_the_program


if __name__=='__main__':
    BOOL_GENERATE_VIDEO=False
    data_dir='D:/Novices/7 Subjects from Dec 6 to Dec 8/'
    post_process_data_dir='D:/Novices/post-process results(2023-2-12)/7_subjects(12-6-2021)/'
    root_directory='E:/Jianxin/test_videos/'
    input_bbox_dir=root_directory+'videos_with_bbox(7_subjects)/'
    output_file_dir=root_directory+'dataset/7_subjects/'
    hsv_csv_dir=root_directory+'7_subjects_1.csv'
    log_file_dir=root_directory+'event_log(7_subjects_1).txt'
    
    # Read info (Note: This function might use the first column in csv file as index)
    # df=pd.read_csv(hsv_csv_dir, sep='\s*,\s*', usecols=['Subject_id', 'min(HSV)', 'max(HSV)'],engine='python')
    # (A version that avoids using the first column in csv file as index)
    df=pd.read_csv(hsv_csv_dir, sep='\s*,\s*', usecols=['Subject_id', 'min(HSV)', 'max(HSV)'], \
                   engine='python', index_col=False)
    
    # For saving the info
    log_txt=open(log_file_dir, 'w')
    now=datetime.datetime.now()
    log_txt.write('Program start time: {}\n'.format(now))
    log_txt.flush()
    
    for i in range(len(df)):
        
        # Ignore the problematic videos
        if type(df.iloc[i,4])!=str:
            log_txt.write('------------\n')
            log_txt.write('Skip {}\n'.format(df.iloc[i,0]))
            log_txt.write('------------\n')
            log_txt.flush()
            continue
        
        # Note: sur.csv and depth.csv have different structures. subject_id is at col 1 in sur.csv, 
        # while is at col 0 in depth.csv
        subject_name=df.iloc[i,0]
        subject_name_split=subject_name.split('_')
        subject_name_vd=subject_name_split[0]+'-VD_'+subject_name_split[1]
        IMU_csv=data_dir + subject_name + '/' + subject_name + '_xsens.csv'
        external_cam_csv=data_dir + subject_name + '/' + subject_name + '_frame_rate(external_camera).csv'
        EnExP_csv=post_process_data_dir + subject_name + '/' + subject_name + '_needle_entry_and_exit_points.csv'
        bag_video_dir=data_dir + subject_name_vd + '/' + subject_name + '_External_Camera.bag'
        output_dataset_dir=output_file_dir+subject_name+'/'
        output_video_dir=output_dataset_dir+'video.avi'
        output_bb_csv_dir=output_dataset_dir+'bounding_box.csv'
        output_d1_csv_dir=output_dataset_dir+'d1.csv'
        output_d2_csv_dir=output_dataset_dir+'d2.csv'
        output_d3_csv_dir=output_dataset_dir+'d3.csv'
        # bbox dir
        shorten_subject_name=abbr_file_name(subject_name)
        input_bbox_csv_dir=input_bbox_dir+shorten_subject_name+'/bbox.csv'
        
        # Read hsv values
        hsv_min_str=df.iloc[i,1][1:-1]
        hsv_max_str=df.iloc[i,2][1:-1]
        hsv_min_str_split=hsv_min_str.split('-')
        hsv_max_str_split=hsv_max_str.split('-')
        # Skip the video if the min HSV values are all 0
        if int(hsv_min_str_split[0])==0 and int(hsv_min_str_split[1])==0 and int(hsv_min_str_split[2])==0:
            log_txt.write('------------\n')
            log_txt.write('Skip {}\n'.format(df.iloc[i,0]))
            log_txt.write('------------\n')
            log_txt.flush()
            continue
        
        hsv_min_tuple=(int(hsv_min_str_split[0]), int(hsv_min_str_split[1]), int(hsv_min_str_split[2]))
        hsv_max_tuple=(int(hsv_max_str_split[0]), int(hsv_max_str_split[1]), int(hsv_max_str_split[2]))
        
        # Create directories for data sets
        try:
            #os.makedirs(output_dataset_dir)
            os.makedirs(output_dataset_dir+'d0')
            os.makedirs(output_dataset_dir+'d1')
            os.makedirs(output_dataset_dir+'d2')
            os.makedirs(output_dataset_dir+'d3')
        except OSError:
            raise Exception('Fail to create directories.')
        
        # Create a video with IMU reading and external camera view 
        exit_the_program=convert_bag_video(log_txt, bag_video_dir, output_video_dir, IMU_csv, \
                                           external_cam_csv, EnExP_csv, output_dataset_dir, \
                                        output_bb_csv_dir, output_d1_csv_dir, output_d2_csv_dir, \
                                            output_d3_csv_dir, input_bbox_csv_dir, hsv_min_tuple, hsv_max_tuple, \
                                            BOOL_GENERATE_VIDEO)
            
        # Break the loop if 'Esc' is pressed
        if exit_the_program==True:
            log_txt.write('*Esc* is pressed by the user. Stopping the program.\n')
            log_txt.flush()
            break            
        
        #
        log_txt.write('------------\n')
        log_txt.write('Finish {}\n'.format(subject_name))
        log_txt.write('------------\n')
        log_txt.flush()
    
    #
    now=datetime.datetime.now()
    log_txt.write('Program end time: {}\n'.format(now))
    log_txt.flush()
    log_txt.close()