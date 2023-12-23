# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 15:56:01 2023

@author: jianxig
"""
import os
import cv2
import numpy as np
import pyrealsense2 as rs
#import pandas as pd
from utils import get_problematic_frame_numbers, abbr_file_name, update_csv, generate_new_bbox_video

# Global variables
g_drawing = False
g_img_cp=np.zeros((480,640,3), np.uint8)
g_top_x, g_top_y, g_bottom_x, g_bottom_y = -1, -1, -1, -1


# define mouse callback function to draw circle
def draw_rectangle(event, x, y, flags, param):
    # 'global' makes the local variables become global variables 
    global g_top_x, g_top_y, g_bottom_x, g_bottom_y, g_drawing 
    if event == cv2.EVENT_LBUTTONDOWN:
        g_drawing = True
        g_top_x = x
        g_top_y = y
    elif event == cv2.EVENT_MOUSEMOVE:
        if g_drawing == True:
            cv2.rectangle(g_img_cp, (g_top_x, g_top_y), (x, y),(0, 0, 255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        g_drawing = False
        g_bottom_x=x
        g_bottom_y=y
        cv2.rectangle(g_img_cp, (g_top_x, g_top_y), (g_bottom_x, g_bottom_y), (0, 0, 255), 2)
        

def adjust_bbox(bag_video_dir, list_error_frames, input_dir, output_dir):
    
    # 'global' makes the local variables become global variables 
    global g_top_x, g_top_y, g_bottom_x, g_bottom_y, g_img_cp
    
    bool_quit_program=False# To quit the program when adjusting the bboxes
    next_traget_frame=0# To inform the user when reaching a specific frame
    prev_labeled_frame=0# Avoid multiple labels for the frames with identical frame number
    
    # Create a window and bind the function to window
    cv2.namedWindow("Rectangle Window (Esc: escape)", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Rectangle Window (Esc: escape)", 10, 10)
    cv2.namedWindow("Result (f: forward)(q: quit)", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Result (f: forward)(q: quit)", 800, 10)
    
    # Connect the mouse button to our callback function
    cv2.setMouseCallback("Rectangle Window (Esc: escape)", draw_rectangle)
    
    # Create pipeline
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()

    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    # 'False' refers to 'no repeat_playback'
    rs.config.enable_device_from_file(config, bag_video_dir, False)
    
    # Configure the pipeline
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
    
    # Start streaming from file
    profile = pipeline.start(config)
    
    # Get playback device
    playback=profile.get_device().as_playback()
    
    # Disable real-time playback
    playback.set_real_time(False)
    
    # List for saving adjusted bbox coordinates
    list_top_x, list_top_y, list_bottom_x, list_bottom_y=[], [], [], []
    
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
        
        # Adjust bbox when meet the errorous frames
        # This if statement also avoid multiple labels for the frames with identical frame number
        if (rgb_frame_num_from_bag in list_error_frames) and (rgb_frame_num_from_bag!=prev_labeled_frame):
            
            # Notify the user when reaching a specific frame
            if rgb_frame_num_from_bag==next_traget_frame:
                print('Done.')
            
            # Convert color_frame and depth_frame to numpy array
            np_rgb_img = np.asanyarray(color_frame.get_data())
            
            # OpenCV requires images in bgr format
            np_bgr_img=cv2.cvtColor(np_rgb_img, cv2.COLOR_RGB2BGR)
            
            # Write rgb_frame_num_from_bag on the image
            cv2.rectangle(np_bgr_img, (702, 460),(848, 480),(255,255,255),-1)
            cv2.putText(np_bgr_img, str(rgb_frame_num_from_bag), (702, 475), 1, 1, (0,0,0), 2, cv2.LINE_AA)
            
            # For each image, reset bboxes coordinates before drawing
            g_top_x, g_top_y, g_bottom_x, g_bottom_y=0, 0, 0, 0
            
            # For double checking the bboxes
            while True:
                
                # Copy the image
                g_img_cp=np.copy(np_bgr_img)
                img_cp_1=np.copy(g_img_cp)
        
                # For drawing bboxes using the mouse
                while True:
                   cv2.imshow("Rectangle Window (Esc: escape)", g_img_cp)
                   if cv2.waitKey(10) == 27: # 'esc'
                      break
                  
                # For inspecting the bboxes
                cv2.rectangle(img_cp_1, (g_top_x, g_top_y), (g_bottom_x, g_bottom_y), (255, 0, 0), 2)
                cv2.imshow("Result (f: forward)(q: quit)", img_cp_1)
                key_input=cv2.waitKey(0)
                if key_input==102: # 'f' stands for 'forward' 
                    break
                elif key_input==113: # 'q' stands for 'quit'
                    bool_quit_program=True
                    break
                else: # Otherwise, let the user adjust the bbox again
                    continue
                
            # Save results
            list_top_x.append(g_top_x)
            list_top_y.append(g_top_y)
            list_bottom_x.append(g_bottom_x)
            list_bottom_y.append(g_bottom_y)
            
            # To avoid multiple labels for the frames with identical frame number
            prev_labeled_frame=rgb_frame_num_from_bag
            
            # Notify the operator if the next frame is far away from the current frame
            cur_frame_idx=list_error_frames.index(rgb_frame_num_from_bag)
            if cur_frame_idx < (len(list_error_frames)-1):
                if list_error_frames[cur_frame_idx+1] - list_error_frames[cur_frame_idx]>10:
                    next_traget_frame=list_error_frames[cur_frame_idx+1]
                    print('The next frame is far away from the current frame. Please wait...')
        
        # Break the loop if user quits the program
        if bool_quit_program==True:
            print('Quiting the program. BBoxes have not been updated.')
            break
        
        # Break the loop after adjusting the bbox in the last errorous frames
        if bool_quit_program==False and rgb_frame_num_from_bag>list_error_frames[-1]:
            print('Finish adjusting errorous frames: ')
            print(list_error_frames)
            
            # Update the bbox coordinates
            input_csv_dir=input_dir+'bbox.csv'
            output_csv_dir=output_dir+'bbox.csv'
            update_csv(input_csv_dir, output_csv_dir, list_error_frames, list_top_x, \
                       list_top_y, list_bottom_x, list_bottom_y)
            print('Finished updating bbox csv')
            
            break
        
    pipeline.stop()
    cv2.destroyAllWindows()
    
    return bool_quit_program
    

if __name__=='__main__':
    work_dir='D:/test_videos/videos_with_bbox(SAVS_2022)/'
    subject_name='Subject_7175637201969199672-1-20-22-17-7-dep-sh'
    shorten_subject_name=abbr_file_name(subject_name)
    
    # dir for videos
    data_dir='F:/experts/SAVS 2022 (Jan 18 to Jan 22)/23 Subjects from SAVS 2022/'
    subject_name_split=subject_name.split('_')
    subject_name_vd=subject_name_split[0]+'-VD_'+subject_name_split[1]
    bag_video_dir=data_dir + subject_name_vd + '/' + subject_name + '_External_Camera.bag'
    
    # Rename the original dir
    src_dir=work_dir+shorten_subject_name+'/'
    dst_dir=work_dir+'original_'+shorten_subject_name+'/'
    os.rename(src_dir, dst_dir)
    # Update the names, so the code is easier to interpret
    input_dir=dst_dir
    output_dir=src_dir
    
    # Make a directory again. It will be used to save results
    if os.path.exists(output_dir):
        raise Exception('Error! Fail to rename the original directory')
    else:
        os.makedirs(output_dir)
        
    # (Step 1) Read the frame numbers that require bbox adjustment
    FN_txt_dir=input_dir+'problematic_frame.txt'
    list_error_frames=get_problematic_frame_numbers(FN_txt_dir)
    
    # (Step 2) Adjust the bbox
    bool_quit_program=adjust_bbox(bag_video_dir, list_error_frames, input_dir, output_dir)
    
    # (Step 3) Generate a new bbox video
    if not bool_quit_program:
        print('Generating a new bbox video. Please wait...')
        output_video_dir=output_dir+'bbox_video.avi'
        output_csv_dir=output_dir+'bbox.csv'
        generate_new_bbox_video(bag_video_dir, output_dir, output_csv_dir, output_video_dir)
        print('Finishing generating the new video.')