# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 10:44:43 2022

@author: jianxig
"""

import cv2
import os
import pyrealsense2 as rs
import pandas as pd
import numpy as np
import glob
from utils_v1 import read_EnEx_info, read_external_cam_data, time_for_first_frame, \
    time_for_last_frame, csv_drop_duplicated_FN, generate_problematic_FN_txt
from dataset_func import remove_noise_via_connectedComponent, figure_for_bbox_centroid


def abbr_file_name(input_file_name):
    single_file_name_split=input_file_name.split('_')
    single_file_name_split_1=single_file_name_split[1].split('-')
    shorten_file_name=single_file_name_split_1[0][:4]+'-'+\
        single_file_name_split_1[4]+'-'+single_file_name_split_1[5]
    
    return shorten_file_name

def check_HSV(path_for_bag_file, output_dir, path_for_external_cam_csv, path_for_EnExP, \
              hsv_min_tuple, hsv_max_tuple):
    bool_exit_program=False
    area_num = 1
    number_of_recorded_frame = 1
    extra_time_start = 0 # Info in [needle_en_time - extra_time_start, needle_ex_time + extra_time_end] are shown
    extra_time_end = 0
    #location='Hand'
    path_for_avi_output=output_dir+'bbox_video.avi'
    path_for_bbox_csv=output_dir+'bbox.csv'
    
    # Open a csv file for saving bbox coordinates
    bbox_csv=open(path_for_bbox_csv, 'w')
    bbox_csv.write('Area_Num,RGB_FN_from_bag,endpoint_1_x,endpoint_1_y,endpoint_2_x,endpoint_2_y\n')
    
    # Read data from csv files
    np_rgb_frameNum, external_cam_time = read_external_cam_data(path_for_external_cam_csv)
    #df_IMU= read_IMU_data(path_for_IMU_csv)
    np_en_time, np_ex_time, np_area_num = read_EnEx_info(path_for_EnExP)
    
    # Read data for Area 1
    ex_cam_start_time = time_for_first_frame(external_cam_time, np_en_time[area_num-1], extra_time_start)
    ex_cam_end_time = time_for_last_frame(external_cam_time, np_ex_time[area_num-1], extra_time_end)
    #start_idx = find_first_time_stamp_idx(df_IMU,ex_cam_start_time)
    #end_idx = find_last_time_stamp_idx(df_IMU,ex_cam_end_time)
    #np_time_for_plot, np_roll, np_pitch, np_yaw = read_eluer_angle(df_IMU, location, start_idx, end_idx)
    #IMU_data_idx=[i for i in range(0,np_roll.shape[0])]
    
# =============================================================================
#     # Unwrap data (i.e., Make the adjacent differences are never greater than pi)
#     np_roll_radian = np.unwrap((np_roll*np.pi/180))
#     np_roll = np_roll_radian * 180/np.pi
#     np_pitch_radian = np.unwrap((np_pitch*np.pi/180))
#     np_pitch = np_pitch_radian * 180/np.pi
#     np_yaw_radian = np.unwrap((np_yaw*np.pi/180))
#     np_yaw = np_yaw_radian * 180/np.pi
# =============================================================================
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    out_video = cv2.VideoWriter(path_for_avi_output, fourcc, 60.0, (848, 480))
    
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
    
    while(True):
        
        # Get frameset of depth
        frame_set = pipeline.try_wait_for_frames(1000)
        
        # Break the loop when reaching the end of the video
        if(frame_set[0]==False):
            print("Found the end of the video.")
            break
        
        # Align the rgb frame and the depth frame to the color viewport
        align = rs.align(rs.stream.color)
        aligned = align.process(frame_set[1])
        
        # Get RGB frame and depth frame
        color_frame = aligned.get_color_frame()
        
        # Obtain the frame number from the bag file
        rgb_frame_num_from_bag=color_frame.get_frame_number()
        
        # Find out the time stamp for the RGB frame
        # Note: Since 'wait_for_frame()' is used, one RGB frame might have multiple time stamps
        RGB_idx=np.where(np_rgb_frameNum==rgb_frame_num_from_bag)
        if RGB_idx[0].shape[0] >= 1:
            target_time=external_cam_time[RGB_idx[0][0]]
        else:
            print('rgb_frame_num_from_bag is not within the csv file.')
            print('rgb_frame_num_from_bag: ', rgb_frame_num_from_bag)
            print('Skipping this frame.')
            continue
        
        # Record information when suturing
        if target_time >= ex_cam_start_time and target_time < ex_cam_end_time:
        
            # Convert color_frame and depth_frame to numpy array
            np_rgb_img = np.asanyarray(color_frame.get_data())
            
            # Obtain the mask for glove 
            np_hsv_img=cv2.cvtColor(np_rgb_img, cv2.COLOR_RGB2HSV)
            glove_mask_temp=cv2.inRange(np_hsv_img,hsv_min_tuple,hsv_max_tuple)
            
            # Detect the bounding rectangle
            # glove_mask_filled, boundRect = glove_detection(glove_mask_temp) # (Deprecated)
            # (Note: glove_mask is in [0, 255] rather than [0(false), 1(true)])
            np_bgr_img=cv2.cvtColor(np_rgb_img, cv2.COLOR_RGB2BGR)
            glove_mask, boundRect = remove_noise_via_connectedComponent\
                (np_bgr_img, glove_mask_temp, '', visualize_idividual_cc=False)
            
            # Draw the bounding rectangle on the image. Also, save the coordinates to csv
            p_1_x=int(boundRect[0])
            p_1_y=int(boundRect[1])
            p_2_x=int(boundRect[0]+boundRect[2])
            p_2_y=int(boundRect[1]+boundRect[3])
            
            # 
            cv2.rectangle(np_bgr_img, (p_1_x, p_1_y), (p_2_x, p_2_y), (0,255,0), 5)
            
            # 
            bbox_csv.write(f'{area_num},{rgb_frame_num_from_bag},{p_1_x},{p_1_y},{p_2_x},{p_2_y}\n')
                
            # Write rgb_frame_num_from_bag on the image
            cv2.rectangle(np_bgr_img, (702, 460),(848, 480),(255,255,255),-1)
            cv2.putText(np_bgr_img, str(rgb_frame_num_from_bag), (702, 475), 1, 1, (0,0,0), 2, cv2.LINE_AA)
            
            # Render image in opencv window
            #cv2.imshow("result_img", result_img)
            cv2.imshow("Converted image", np_bgr_img)
            
            # Save all image to video file
            out_video.write(np_bgr_img)

            # Count the recorded frames
            number_of_recorded_frame = number_of_recorded_frame+1
            
        # After processing a single area, update info
        if target_time >= ex_cam_end_time:
           area_num+=1
           
           if area_num>12:
               print('Finish 12 areas.')
               break
           
           if area_num==(np_en_time.shape[0] + 1):
               print('The detection result does not include all 12 areas.')
               break
           
           ex_cam_start_time = time_for_first_frame(external_cam_time, np_en_time[area_num-1], extra_time_start)
           ex_cam_end_time = time_for_last_frame(external_cam_time, np_ex_time[area_num-1], extra_time_end)
           #start_idx = find_first_time_stamp_idx(df_IMU,ex_cam_start_time)
           #end_idx = find_last_time_stamp_idx(df_IMU,ex_cam_end_time)
           #np_time_for_plot, np_roll, np_pitch, np_yaw = read_eluer_angle(df_IMU, location, start_idx, end_idx) 
           #IMU_data_idx=[i for i in range(0,np_roll.shape[0])]
           
# =============================================================================
#            # Unwrap data (i.e., Make the adjacent differences are never greater than pi)
#            np_roll_radian = np.unwrap((np_roll*np.pi/180))
#            np_roll = np_roll_radian * 180/np.pi
#            np_pitch_radian = np.unwrap((np_pitch*np.pi/180))
#            np_pitch = np_pitch_radian * 180/np.pi
#            np_yaw_radian = np.unwrap((np_yaw*np.pi/180))
#            np_yaw = np_yaw_radian * 180/np.pi
# =============================================================================
        
        # if pressed escape exit program
        if cv2.waitKey(1) == 27:
            bool_exit_program=True
            break    
    
    pipeline.stop()
    out_video.release()
    cv2.destroyAllWindows()
    bbox_csv.close()
    
    # Drop csv rows that have the same frame number
    csv_drop_duplicated_FN(path_for_bbox_csv)
    
    # Generate a txt file that allows user to save the problematic frame numbers
    output_txt_dir=output_dir+'problematic_frame.txt'
    generate_problematic_FN_txt(output_txt_dir)
    
    # test
    print('number_of_recorded_frame: ', number_of_recorded_frame)
    
    # Generate figures for the change of bbox centroids
    figure_for_bbox_centroid(output_dir, path_for_bbox_csv)
    
    #
    return bool_exit_program


if __name__=='__main__':
    BOOL_PROCESS_CSV_W_HSV=True 
    exit_program=False
    data_dir='F:/experts/SAVS 2022 (Jan 18 to Jan 22)/23 Subjects from SAVS 2022/'
    post_process_data_dir='F:/experts/post-process results(2023-1-12)/SAVS(2022)/'
    root_output_dir='D:/test_videos/videos_with_bbox(SAVS_2022)/' # videos_with_bbox_one_per_person
            
    # For checking a csv file that contains both video id and the corresponding HSV value
    if BOOL_PROCESS_CSV_W_HSV:
        subject_name_dir='D:/test_videos/portion.csv'
        
        # Read subject id
        df_name=pd.read_csv(subject_name_dir, sep='\s*,\s*', usecols=['Subject_id', 'min(HSV)', 'max(HSV)'], engine='python')
        df_name=df_name.fillna('')
        subject_name_list=df_name['Subject_id'].tolist()
        min_HSV_list=df_name['min(HSV)'].tolist()
        max_HSV_list=df_name['max(HSV)'].tolist()
        
        #
        for i, subject_name in enumerate(subject_name_list):
            
            if min_HSV_list[i] != '':
                # Get the HSV value
                min_HSV_str=min_HSV_list[i][1:-1]
                max_HSV_str=max_HSV_list[i][1:-1]
                split_min_HSV=min_HSV_str.split('-')
                split_max_HSV=max_HSV_str.split('-')
                hsv_min_tuple=(int(split_min_HSV[0]), int(split_min_HSV[1]), int(split_min_HSV[2]))
                hsv_max_tuple=(int(split_max_HSV[0]), int(split_max_HSV[1]), int(split_max_HSV[2]))
                
                # Ignore the video if the HSV values are 0
                if int(split_min_HSV[0])==0 and int(split_min_HSV[1])==0 and int(split_min_HSV[2])==0:
                       continue
                
                # Get other info
                shorten_subject_name=abbr_file_name(subject_name)
                subject_name_split=subject_name.split('_')
                subject_name_vd=subject_name_split[0]+'-VD_'+subject_name_split[1]
                external_cam_csv=data_dir + subject_name + '/' + subject_name + '_frame_rate(external_camera).csv'
                EnExP_csv=post_process_data_dir + subject_name + '/' + subject_name + '_needle_entry_and_exit_points.csv'
                bag_video_dir=data_dir + subject_name_vd + '/' + subject_name + '_External_Camera.bag'
                 
                # Make a dir for each subject
                output_dir=root_output_dir+shorten_subject_name+'/' 
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                else:
                    all_file_names=glob.glob(output_dir+'*')
                    for single_name in all_file_names:
                        os.remove(single_name)
                
                # Create a video with IMU reading and external camera view 
                exit_program = check_HSV(bag_video_dir, output_dir, external_cam_csv, EnExP_csv, \
                          hsv_min_tuple, hsv_max_tuple)
            
                # If the key 'esc' is pressed, exit the program
                if exit_program:
                    print('Esc is pressed. Exitting the program')
                    break
            
                #
                print('Finish processing ', subject_name)
                

    
