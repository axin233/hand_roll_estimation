# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 12:51:55 2022

@author: jianxig
"""

import pyrealsense2 as rs
import cv2
import numpy as np
import time
import os

def abbr_file_name(input_file_name):
    single_file_name_split=input_file_name.split('_')
    single_file_name_split_1=single_file_name_split[1].split('-')
    shorten_file_name=single_file_name_split_1[0][:4]+'-'+\
        single_file_name_split_1[4]+'-'+single_file_name_split_1[5]
    
    return shorten_file_name


def convert_bag_video(path_for_bag_file, path_for_output, list_target_frame_number):
    
    successfully_save_img=False
    
    # The frame number
    frame_num = 1
    
    # Create pipeline
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()

    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    rs.config.enable_device_from_file(config, path_for_bag_file, False)
    
    # Configure the pipeline
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
    
    # Start streaming from file
    profile=pipeline.start(config)
    
    # Get playback device
    playback=profile.get_device().as_playback()
    
    # Disable real-time playback
    playback.set_real_time(False)
    
    # Create opencv window to render image in
    cv2.namedWindow("Color Stream", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth Stream", cv2.WINDOW_NORMAL)
    
    # Create colorizer object
    colorizer = rs.colorizer()
    
    while(True):
        
        # Get frameset of depth
        frame_set = pipeline.try_wait_for_frames(1000)
        
        # Break the loop when reaching the end of the video
        if(frame_set[0]==False):
            print("Found the end of the video.")
            break
        
        # Get depth frame
        frames = frame_set[1]
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        # Get frame number from bag file
        rgb_frame_number_from_bag = color_frame.get_frame_number()
        
        # Colorize depth frame to jet colormap
        depth_frame_colored = colorizer.colorize(depth_frame)
        
        # Convert depth_frame to numpy array to render image in opencv
        cv_depth_img = np.asanyarray(depth_frame_colored.get_data())
        cv_rgb_img = np.asanyarray(color_frame.get_data())
        
        # Convert RGB to BGR
        cv_bgr_img = cv2.cvtColor(cv_rgb_img, cv2.COLOR_RGB2BGR)
        
        # Write the frame number on the bgr image
        cv2.rectangle(cv_bgr_img, (760, 460), (848, 480), (255, 255, 255), -1)
        cv2.putText(cv_bgr_img, str(rgb_frame_number_from_bag), (760, 476), 1, 1, (0, 0, 0), 2)
        
        # Accumulate the frame number
        frame_num += 1
        
        # Render image in opencv window
        cv2.imshow("Color Stream", cv_bgr_img)
        cv2.imshow("Depth Stream", cv_depth_img)

        # Save specific images
        if rgb_frame_number_from_bag in list_target_frame_number:
            successfully_save_img=cv2.imwrite(path_for_output+'/'+str(rgb_frame_number_from_bag)+'.png', \
                                              cv_bgr_img, [int(cv2.IMWRITE_PNG_COMPRESSION),0])
            if successfully_save_img==False:
                print('Error! Fail to save the images.')
                
        # Exit the program after saving the last target frame
        if rgb_frame_number_from_bag > list_target_frame_number[-1]:
            print('Finish extracting frames. Exitting.')
            break
        
        # if pressed escape exit program
        if cv2.waitKey(1) == 27:
            break
    
    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    file_dir='F:/experts/SAVS 2022 (Jan 18 to Jan 22)/23 Subjects from SAVS 2022/'
    subject_name='Subject_7175637201969199672-1-20-22-16-49-sur-sh'
    shorten_subject_name=abbr_file_name(subject_name)
    subject_name_split=subject_name.split('_')
    subject_name_vd=subject_name_split[0]+'-VD_'+subject_name_split[1]
    path_for_bag_file = file_dir+subject_name_vd+'/'+subject_name+"_External_Camera.bag"
    path_for_output = "D:/test_videos/extracted_frame/"+shorten_subject_name
    list_target_frame_number=[1390, 1969, 1980, 3130, 3528, 3534]
    
    # Create the directory for data sets if the directory does not exist
    if os.path.exists(path_for_output)==False:
        try:
            os.makedirs(path_for_output)
        except OSError:
            raise Exception('Fail to create directories.')
    
    start_time = time.time()
    convert_bag_video(path_for_bag_file, path_for_output, list_target_frame_number)
    end_time = time.time()
    
    # Elapsed time (in second)
    print("Elapsed time: ",(end_time-start_time)),