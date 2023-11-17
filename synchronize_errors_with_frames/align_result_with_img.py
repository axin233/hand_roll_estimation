# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 17:49:09 2023

@author: jianxig
"""
import numpy as np
import cv2
import pyrealsense2 as rs
import pandas as pd
from utils import summary_view, abbr_file_name, extract_data_for_single_area, \
    extract_data_for_single_frame, calculate_error_roll, calculate_error_roll_vel, adjust_bbox


def align_result_with_img(gt_df, roll_mm_df, roll_mo_df, vel_mm_df, vel_mo_df, bbox_df, \
                      path_for_bag_file, path_for_avi_output, SHOW_BBOX=False):
    
    area_num=1
    
    # Extract data for Area 1
    start_FN, end_FN, np_FN, np_gt_roll, np_gt_pitch, np_gt_sin, np_gt_cos, np_gt_roll_vel, \
        np_mm_roll, np_mm_sin, np_mm_cos, np_mm_roll_vel, \
        np_mo_roll, np_mo_sin, np_mo_cos,np_mo_roll_vel = \
            extract_data_for_single_area(area_num, gt_df, roll_mm_df, roll_mo_df, vel_mm_df, vel_mo_df)
    # Calculate roll errors at Area 1
    np_mm_error, mm_avg_err=calculate_error_roll(np_mm_roll, np_gt_roll)
    np_mo_error, mo_avg_err=calculate_error_roll(np_mo_roll, np_gt_roll)
    # Calculate roll velocity errors at Area 1
    np_mm_vel_error, mm_avg_vel_err = calculate_error_roll_vel(np_mm_roll_vel, np_gt_roll_vel)
    np_mo_vel_error, mo_avg_vel_err = calculate_error_roll_vel(np_mo_roll_vel, np_gt_roll_vel)
    
    # Define the codec and create VideoWriter object
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
    #depth_sensor=profile.get_device().first_depth_sensor()
    #depth_scale=depth_sensor.get_depth_scale()
    
    # Create colorizer object
    #colorizer = rs.colorizer()
    
    while(True):
        
        # Get frameset of depth
        frame_set = pipeline.try_wait_for_frames(2000)
        
        # Break the loop when reaching the end of the video
        if(frame_set[0]==False):
            print("Found the end of the video.")
            break
        
        # Align the rgb frame and the depth frame to the color viewport
        align = rs.align(rs.stream.color)
        aligned = align.process(frame_set[1])
        
        # Get RGB frame and depth frame
        color_frame = aligned.get_color_frame()
        #depth_frame = aligned.get_depth_frame()
        
        # Obtain the frame number from the bag file
        rgb_frame_num_from_bag=color_frame.get_frame_number()
        
        # Obtain the 1-channel depth image
        #np_depth_img_1 = np.asanyarray(depth_frame.get_data())
        
        # Colorize depth frame to jet colormap
        #depth_frame_colored = colorizer.colorize(depth_frame)
    
        # Convert color_frame and depth_frame to numpy array
        np_rgb_img = np.asanyarray(color_frame.get_data())
        #np_depth_img = np.asanyarray(depth_frame_colored.get_data())
        
        # Visualize the result if start_FN < rgb_frame_num_from_bag < end_FN
        if start_FN<=rgb_frame_num_from_bag and end_FN>=rgb_frame_num_from_bag:
            
            # Extract bbox coordinates
            list_ori_bbox_xywh=[0,0,0,0]
            list_bbox_xyxy=[0,0,0,0]
            if SHOW_BBOX:
                bbox_df_single_frame=bbox_df[bbox_df['RGB_FN_from_bag']==rgb_frame_num_from_bag]
                if len(bbox_df_single_frame)>=1:
                    bbox_x1 = int(bbox_df_single_frame.iloc[0,:]['endpoint_1_x'])
                    bbox_y1 = int(bbox_df_single_frame.iloc[0,:]['endpoint_1_y'])
                    bbox_x2 = int(bbox_df_single_frame.iloc[0,:]['endpoint_2_x'])
                    bbox_y2 = int(bbox_df_single_frame.iloc[0,:]['endpoint_2_y'])
                        
                    # Make sure the bbox is a square
                    # Note: ori_bbox: [top_left_x, top_left_y, width, height]
                    list_ori_bbox_xywh[0], list_ori_bbox_xywh[1] = bbox_x1, bbox_y1
                    list_ori_bbox_xywh[2] = bbox_x2-bbox_x1
                    list_ori_bbox_xywh[3] = bbox_y2-bbox_y1
                    list_bbox_xywh=adjust_bbox(list_ori_bbox_xywh)
                    
                    # Convert the format from (top-left x, top-left y, width, height) to 
                    # (x_1, y_1, x_2, y_2)
                    list_bbox_xyxy[0]=list_bbox_xywh[0]
                    list_bbox_xyxy[1]=list_bbox_xywh[1]
                    list_bbox_xyxy[2]=list_bbox_xywh[0]+list_bbox_xywh[2]
                    list_bbox_xyxy[3]=list_bbox_xywh[1]+list_bbox_xywh[3]
            
            gt_roll, gt_pitch, gt_sin, gt_cos, gt_roll_vel, \
                mm_roll, mm_sin, mm_cos, mm_roll_vel,\
                mo_roll, mo_sin, mo_cos, mo_roll_vel,\
                mm_roll_err, mo_roll_err, mm_vel_err, mo_vel_err = extract_data_for_single_frame(\
                rgb_frame_num_from_bag, np_FN, np_gt_roll, np_gt_pitch, np_gt_sin, np_gt_cos, np_gt_roll_vel, \
                np_mm_roll, np_mm_sin, np_mm_cos, np_mm_roll_vel, \
                np_mo_roll, np_mo_sin, np_mo_cos, np_mo_roll_vel, \
                np_mm_error, np_mo_error, np_mm_vel_error, np_mo_vel_error)
                    
            labeled_img = summary_view(np_rgb_img, area_num, rgb_frame_num_from_bag, \
                        mm_avg_err, mo_avg_err, np_FN, \
                        np_gt_roll, np_gt_pitch, np_mm_roll, np_mo_roll, gt_roll, gt_pitch, mm_roll, mo_roll, \
                        np_mm_error, np_mo_error, gt_sin, gt_cos, mm_sin, mm_cos, mo_sin, mo_cos, \
                        np_gt_roll_vel, np_mm_roll_vel, np_mo_roll_vel, \
                        gt_roll_vel, mm_roll_vel, mo_roll_vel, np_mm_vel_error, \
                        np_mo_vel_error, mm_roll_err, mo_roll_err, mm_vel_err, mo_vel_err, \
                        mm_avg_vel_err, mo_avg_vel_err, list_bbox_xyxy)
        
            # Render image in opencv window
            cv2.imshow("Converted image", labeled_img)
            
            # Save all image to video file
            out_video.write(labeled_img)
            
            # if pressed escape exit program
            if cv2.waitKey(1) == 27:
                print('Esc is pressed by user. Exitting')
                break
            
        # Update the data when finishing an area
        if end_FN<rgb_frame_num_from_bag:
            area_num+=1
            
            # After finishing 12 areas, exit the program
            if area_num==13:
                print('Finish processing a video. Exitting')
                break
            else:
                # Extract data for a specific area
                start_FN, end_FN, np_FN, np_gt_roll, np_gt_pitch, np_gt_sin, np_gt_cos, np_gt_roll_vel, \
                    np_mm_roll, np_mm_sin, np_mm_cos, np_mm_roll_vel, \
                    np_mo_roll, np_mo_sin, np_mo_cos,np_mo_roll_vel = \
                        extract_data_for_single_area(area_num, gt_df, roll_mm_df, roll_mo_df, vel_mm_df, vel_mo_df)
                # Calculate roll errors at Area 1
                np_mm_error, mm_avg_err=calculate_error_roll(np_mm_roll, np_gt_roll)
                np_mo_error, mo_avg_err=calculate_error_roll(np_mo_roll, np_gt_roll)
                # Calculate roll velocity errors at Area 1
                np_mm_vel_error, _ = calculate_error_roll_vel(np_mm_roll_vel, np_gt_roll_vel)
                np_mo_vel_error, _ = calculate_error_roll_vel(np_mo_roll_vel, np_gt_roll_vel)
            
    pipeline.stop()
    #out_video.release()
    cv2.destroyAllWindows()
    
    return
    

if __name__=='__main__':
    #subject_id='Subject_10096897566064790785-3-22-22-15-18-sur-sh' 
    subject_id='Subject_6188649480280766488-3-21-22-9-32-dep-sh'
    shorten_subject_id=abbr_file_name(subject_id)
    TIMESTEP_FOR_ROLL=5
    TIMESTEP_FOR_VEL=10
    BOOL_SHOW_BBOX=True
    
    # 
    pred_result_dir='E:/Jianxin/analyze_result/2023-8-26/'
    path_for_avi_output=pred_result_dir+shorten_subject_id+'.avi' #Output_dir
    #
    gt_dir='E:/Jianxin/test_set(SCVS_p1)/test_set(SCVS_p1)/csv/'
    gt_csv_dir=gt_dir+subject_id+'_d1.csv'
    #
    pred_roll_dir=pred_result_dir+'roll/v2_1/'
    roll_mm_dir=pred_roll_dir+'many-to-many/'
    roll_mo_dir=pred_roll_dir+'many-to-one/'
    roll_mm_csv_dir=roll_mm_dir+'test_results('+subject_id+'_d1).csv'
    roll_mo_csv_dir=roll_mo_dir+'test_results('+subject_id+'_d1).csv'
    #
    pred_vel_dir=pred_result_dir+'roll_velocity/v2_2_three_layer_10ts_v2/'
    vel_mm_dir=pred_vel_dir+'many-to-many/'
    vel_mo_dir=pred_vel_dir+'many-to-one/'
    vel_mm_csv_dir=vel_mm_dir+'test_results('+subject_id+'_d1).csv'
    vel_mo_csv_dir=vel_mo_dir+'test_results('+subject_id+'_d1).csv'
    # 
    dataset_dir='D:/experts/SCVS 2022 (Mar 21 to Mar 22)/Platform 1/'
    subject_id_split=subject_id.split('_')
    path_for_bag_file=dataset_dir+subject_id_split[0]+'-VD_'+subject_id_split[1]+'/'+\
        subject_id+'_External_Camera.bag'
        
    #
    bb_dir='E:/Jianxin/test_set(SCVS_p1)/test_set(SCVS_p1)/'
    if BOOL_SHOW_BBOX:
        bbox_df=pd.read_csv(bb_dir+subject_id+'/bounding_box.csv', sep='\s*,\s*', \
                            header=[0], engine='python')
    else:
        bbox_df=0    
            
    # Read data        
    gt_df=pd.read_csv(gt_csv_dir, sep='\s*,\s*', header=[0], engine='python')
    gt_df=gt_df.iloc[TIMESTEP_FOR_VEL-1:-1, :]#Ignore frames at the beginning of a trial
    roll_mm_df=pd.read_csv(roll_mm_csv_dir, sep='\s*,\s*', header=[0], engine='python')
    roll_mm_df=roll_mm_df.iloc[(TIMESTEP_FOR_VEL-TIMESTEP_FOR_ROLL-1):, :]
    roll_mo_df=pd.read_csv(roll_mo_csv_dir, sep='\s*,\s*', header=[0], engine='python')
    roll_mo_df=roll_mo_df.iloc[(TIMESTEP_FOR_VEL-TIMESTEP_FOR_ROLL-1):, :]
    vel_mm_df=pd.read_csv(vel_mm_csv_dir, sep='\s*,\s*', header=[0], engine='python')
    vel_mo_df=pd.read_csv(vel_mo_csv_dir, sep='\s*,\s*', header=[0], engine='python')
    
    #
    align_result_with_img(gt_df, roll_mm_df, roll_mo_df, vel_mm_df, vel_mo_df, bbox_df, \
                          path_for_bag_file, path_for_avi_output, SHOW_BBOX=BOOL_SHOW_BBOX)
    
    
    