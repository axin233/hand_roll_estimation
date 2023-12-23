# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 17:49:09 2023

@author: jianxig
"""
import numpy as np
import cv2
import pyrealsense2 as rs
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from utils import summary_view, abbr_file_name, abbr_file_name_with_sur_dep, \
    extract_data_for_single_area, extract_data_for_single_frame, calculate_error_roll, \
    adjust_bbox, show_roll_vs_time, wrap_data, find_entry_exit_FN, find_problematic_areas


def align_result_with_img(gt_df, mm_df, mo_df, bbox_df, cnn_df, enex_df, external_cam_df, \
                          path_for_bag_file, path_for_avi_output, path_for_output_img, \
                          output_csv, list_ignored_areas, SHOW_BBOX=False, \
                          SHOW_ROLL_IMG=False, SAVE_VIDEO=False, SHOW_ENTRY_EXIT=False):
    
    area_num=1
    
    # Find out the first area that is not in list_ignored_areas
    for i in range(1, 14):
        if i not in list_ignored_areas:
            area_num=i
            break
        elif i==13:
            raise ValueError('All areas have been ignored.')
            
    # Test
    print('At Area ', area_num)
    
    # Extract data for Area 1
    start_FN, end_FN, np_FN, np_external_cam_time, np_external_cam_FN, \
        np_gt_roll, np_gt_pitch, np_gt_sin, np_gt_cos, \
        np_mm_roll, np_mm_sin, np_mm_cos, \
        np_mo_roll, np_mo_sin, np_mo_cos, \
        np_cnn_roll, entry_time, exit_time = \
            extract_data_for_single_area(area_num, gt_df, mm_df, mo_df, cnn_df, enex_df, external_cam_df)
    np_gt_roll=wrap_data(np_gt_roll)
    # Get the frame number for needle entry and needle exit
    entry_FN, exit_FN = find_entry_exit_FN(np_external_cam_FN, np_external_cam_time, entry_time, exit_time)
    # Calculate roll errors at Area 1
    np_mm_error, mm_avg_err=calculate_error_roll(np_mm_roll, np_gt_roll)
    np_mo_error, mo_avg_err=calculate_error_roll(np_mo_roll, np_gt_roll)
    np_cnn_error, cnn_avg_err=calculate_error_roll(np_cnn_roll, np_gt_roll)
    
    # Save errors to csv file
    output_csv.write('{},{:.2f},{:.2f},{:.2f}\n'.format(area_num, cnn_avg_err, \
                                                        mo_avg_err, mm_avg_err))
    
    # Save figures
    if SHOW_ROLL_IMG:
        show_roll_vs_time(np_FN, np_gt_roll, np_cnn_roll, np_mm_roll, np_mo_roll, \
                          path_for_output_img, area_num, cnn_avg_err, mo_avg_err, mm_avg_err, \
                          entry_FN, exit_FN, SHOW_EN_EX=SHOW_ENTRY_EXIT)
    
    # Define the codec and create VideoWriter object
    if SAVE_VIDEO:
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
        frame_set = pipeline.try_wait_for_frames(5000)
        
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
        
        # Save the result to videos if SAVE_VIDEO = True
        if SAVE_VIDEO:
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
                
                gt_roll, gt_pitch, mm_roll, mo_roll, cnn_roll,\
                    mm_roll_err, mo_roll_err, cnn_roll_err = extract_data_for_single_frame(\
                                rgb_frame_num_from_bag, np_FN, np_gt_roll, np_gt_pitch, \
                                 np_mm_roll, np_mo_roll, np_cnn_roll,\
                                np_mm_error, np_mo_error, np_cnn_error)
                        
                labeled_img = summary_view(np_rgb_img, area_num, rgb_frame_num_from_bag, np_FN, \
                                           np_gt_roll, np_gt_pitch, np_mm_roll, np_mo_roll, np_cnn_roll,\
                                           gt_roll, gt_pitch, mm_roll, mo_roll, cnn_roll,\
                                           np_mm_error, np_mo_error, np_cnn_error,\
                                           mm_roll_err, mo_roll_err, cnn_roll_err,\
                                           mm_avg_err, mo_avg_err, cnn_avg_err, \
                                           list_bbox_xyxy)
            
                # Render image in opencv window
                cv2.imshow("Converted image", labeled_img)
                
                # Save all image to video file
                out_video.write(labeled_img)
        else:
            # Convert rgb to bgr 
            np_bgr_img=cv2.cvtColor(np_rgb_img, cv2.COLOR_RGB2BGR)
            
            # Render image in opencv window
            cv2.imshow("Converted image", np_bgr_img)
            
        # if pressed escape exit program
        if cv2.waitKey(1) == 27:
            print('Esc is pressed by user. Exitting')
            break
            
        # Update the data when finishing an area
        if end_FN<rgb_frame_num_from_bag:
            area_num+=1
            
            # Make sure area_num is not in list_ignored_areas
            start_area=area_num
            for i in range(start_area, 13):
                if i not in list_ignored_areas:
                    area_num=i
                    break
    
            # test
            print('At Area ', area_num)
            
            # After finishing 12 areas, exit the program
            if area_num==13:
                print('Finish processing a video. Exitting')
                break
            else:
                
                # Extract data for a specific area
                start_FN, end_FN, np_FN, np_external_cam_time, np_external_cam_FN, \
                    np_gt_roll, np_gt_pitch, np_gt_sin, np_gt_cos, \
                    np_mm_roll, np_mm_sin, np_mm_cos, \
                    np_mo_roll, np_mo_sin, np_mo_cos, \
                    np_cnn_roll, entry_time, exit_time = \
                        extract_data_for_single_area(area_num, gt_df, mm_df, mo_df, cnn_df, enex_df, external_cam_df)
                np_gt_roll=wrap_data(np_gt_roll)
                # Get the frame number for needle entry and needle exit
                entry_FN, exit_FN = find_entry_exit_FN(np_external_cam_FN, np_external_cam_time, entry_time, exit_time)
                # Calculate roll errors at Area 1
                np_mm_error, mm_avg_err=calculate_error_roll(np_mm_roll, np_gt_roll)
                np_mo_error, mo_avg_err=calculate_error_roll(np_mo_roll, np_gt_roll)
                np_cnn_error, cnn_avg_err=calculate_error_roll(np_cnn_roll, np_gt_roll)
                
                # Save errors to csv file
                output_csv.write('{},{:.2f},{:.2f},{:.2f}\n'.format(area_num, cnn_avg_err, \
                                                                    mo_avg_err, mm_avg_err))
                
                # Save figures
                if SHOW_ROLL_IMG:
                    show_roll_vs_time(np_FN, np_gt_roll, np_cnn_roll, np_mm_roll, \
                                      np_mo_roll, path_for_output_img, area_num, cnn_avg_err, \
                                      mo_avg_err, mm_avg_err, entry_FN, exit_FN, \
                                      SHOW_EN_EX=SHOW_ENTRY_EXIT)
            
    pipeline.stop()
    #out_video.release()
    cv2.destroyAllWindows()
    
    return
    

if __name__=='__main__':
    # 'Subject_7281502457159821573-3-4-22-16-15-dep-sh'
    event_name='Rhodes_2023_12_4'
    subject_id= 'Subject_12638199295183552317-12-3-23-23-59-sur-sh'
    shorten_subject_id=abbr_file_name(subject_id)
    shorten_subject_id_with_sur_dep=abbr_file_name_with_sur_dep(subject_id)
    TIMESTEP=5
    BOOL_SHOW_BBOX=True
    BOOL_SHOW_ROLL_IMG=False
    BOOL_SHOW_ENTRY_EXIT=False
    BOOL_SAVE_VIDEO=True
    
    # Read the problematic areas
    info_df=pd.read_csv('D:/github_video/for_roll_estimation/'+event_name+'.csv', sep='\s*,\s*', \
                        usecols=['Subject_id', 'min(HSV)', 'max(HSV)', 'Problematic_areas'], \
                        engine='python',\
                        index_col=False)
    list_subject_id=info_df['Subject_id'].tolist()
    list_problematic_areas=info_df['Problematic_areas'].fillna('').tolist()
    list_problematic_areas=[str(i) for i in list_problematic_areas] # Make sure all elements are strings
    list_ignored_areas=find_problematic_areas(list_subject_id, list_problematic_areas, subject_id)
    
    #
    if event_name=='2_subjects_p1' or event_name=='2_subjects_p2':
        folder_name='2_subjects'
    else:
        folder_name=event_name
    
    # 
    gt_dir='D:/github_video/for_roll_estimation/dataset/'+folder_name+'/'
    estimated_result_dir='D:/github_video/for_roll_estimation/results/'
    mm_dir=estimated_result_dir + 'CRNN(angle)(mm)/'+folder_name+'/'
    mo_dir=estimated_result_dir + 'CRNN(angle)(mo)/'+folder_name+'/'
    cnn_dir=estimated_result_dir + 'CNN(conv+FC)/'+folder_name+'/'
    gt_csv_dir=gt_dir+subject_id+'/d1.csv'
    mm_csv_dir=mm_dir+'test_results('+shorten_subject_id+').csv'
    mo_csv_dir=mo_dir+'test_results('+shorten_subject_id+').csv'
    cnn_csv_dir=cnn_dir+'test_results('+shorten_subject_id+').csv'
    
    # Read bag file
    dataset_dir='D:/github_video/for_roll_estimation/data/'
    subject_id_split=subject_id.split('_')
    path_for_bag_file=dataset_dir+subject_id_split[0]+'-VD_'+subject_id_split[1]+'/'+\
        subject_id+'_External_Camera.bag'
        
    # Read external camera info
    path_for_external_camera_csv=dataset_dir+subject_id+'/'+subject_id+'_frame_rate(external_camera).csv'
    external_cam_df=pd.read_csv(path_for_external_camera_csv, sep='\s*,\s*', header=[0], engine='python')
        
    # Read entry/exit time
    post_process_data_dir='D:/github_video/for_roll_estimation/data/'
    path_for_en_ex_time=post_process_data_dir+subject_id+'/'+subject_id+'_needle_entry_and_exit_points.csv'
    enex_df=pd.read_csv(path_for_en_ex_time, sep='\s*,\s*', header=[0], engine='python')
        
    # Read bounding box locations
    if BOOL_SHOW_BBOX:
        bbox_df=pd.read_csv(gt_dir+subject_id+'/bounding_box.csv', sep='\s*,\s*', \
                            header=[0], engine='python')
    else:
        bbox_df=0    
        
    #
    output_dir='D:/github_video/for_roll_estimation/'
    path_for_output_img=''
    path_for_output_csv=output_dir+shorten_subject_id_with_sur_dep+'_error_log.csv'
    path_for_avi_output=output_dir+shorten_subject_id_with_sur_dep+'_roll.avi'
        
    # (Deprecated) The roll images are generated by visualize_roll_angle_err.py at
    # E:\python_code\nn\test_videos\visualize_roll_angle_err\
    # Create a directory to save results if BOOL_SHOW_ROLL_IMG=True
    path_for_output_img=''
    if BOOL_SHOW_ROLL_IMG:
        path_for_output_img=output_dir+shorten_subject_id+'-roll/'
        if not os.path.exists(path_for_output_img):
            os.makedirs(path_for_output_img)
        else: # Otherwise, delete content in the folder
            all_file_names=glob.glob(path_for_output_img+'*')
            for single_name in all_file_names:
                os.remove(single_name)
    
    # Open a csv file for saving results
    output_csv=open(path_for_output_csv, 'w')
    output_csv.write('Area_num,Error(CNN)(deg),Error(MO)(deg),Error(MM)(deg)\n')
            
    # Read data        
    gt_df=pd.read_csv(gt_csv_dir, sep='\s*,\s*', header=[0], engine='python')
    gt_df=gt_df.iloc[TIMESTEP-1:-1, :]#Ignore frames at the beginning of a trial
    mm_df=pd.read_csv(mm_csv_dir, sep='\s*,\s*', header=[0], engine='python')
    mo_df=pd.read_csv(mo_csv_dir, sep='\s*,\s*', header=[0], engine='python')
    cnn_df=pd.read_csv(cnn_csv_dir, sep='\s*,\s*', header=[0], engine='python')
    cnn_df=cnn_df.iloc[TIMESTEP-1:-1, :]#Ignore frames at the beginning of a trial
    
    #
    align_result_with_img(gt_df, mm_df, mo_df, bbox_df, cnn_df, enex_df, external_cam_df, 
                          path_for_bag_file, path_for_avi_output, path_for_output_img, \
                          output_csv, list_ignored_areas, SHOW_BBOX=BOOL_SHOW_BBOX, \
                          SHOW_ROLL_IMG=BOOL_SHOW_ROLL_IMG, \
                          SAVE_VIDEO=BOOL_SAVE_VIDEO, SHOW_ENTRY_EXIT=BOOL_SHOW_ENTRY_EXIT)
        
    #
    output_csv.close()
    
    
    