# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 17:05:53 2023

@author: jianxig
"""
import pandas as pd
import cv2
import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import pickle

def abbr_file_name(input_file_name):
    single_file_name_split=input_file_name.split('_')
    single_file_name_split_1=single_file_name_split[1].split('-')
    shorten_file_name=single_file_name_split_1[0][:4]+'-'+\
        single_file_name_split_1[4]+'-'+single_file_name_split_1[5]
    
    return shorten_file_name

# Get the problematic frame number
def get_problematic_frame_numbers(FN_txt_dir):
    list_error_frames=[]
    
    with open(FN_txt_dir, 'r') as f:
        
        # Inspect each row in the file, from top to bottom.
        for line_number, line in enumerate(f):
            
            # '#' denotes comments, so we ignore them 
            if '#' in line:
                continue
            else: 
                split_line=line.split('-')
                # For a sequence of frames
                if len(split_line)==2:
                    list_temp=list(range(int(split_line[0]), int(split_line[1])+1))
                    list_error_frames.extend(list_temp)
                else: # For a single frame
                    list_error_frames.append(int(split_line[0]))
                    
    # Sort the frame numbers so they are in ascending order
    list_error_frames.sort()
                    
    return list_error_frames


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


# Generate figures for the change of bbox centroids
def figure_for_bbox_centroid(output_dir, path_for_bbox_csv, distance_threshold=25):
    df=pd.read_csv(path_for_bbox_csv, sep='\s*,\s*', header=[0], engine='python')
    np_area_num=df['Area_Num'].to_numpy()
    np_RGB_FN=df['RGB_FN_from_bag'].to_numpy()
    np_p_1_x=df['endpoint_1_x'].to_numpy()
    np_p_1_y=df['endpoint_1_y'].to_numpy()
    np_p_2_x=df['endpoint_2_x'].to_numpy()
    np_p_2_y=df['endpoint_2_y'].to_numpy()
    
    # Find out the boundary between areas
    area_boundary_mask=((np_area_num[1:]-np_area_num[:-1])>0)
    #Add one more value at the beginning
    area_boundary_mask=np.concatenate((np.array([False]), area_boundary_mask), axis=0)
    np_first_frame_per_area=np_RGB_FN[area_boundary_mask]
    
    # Calculate the centroid
    np_c_x=(np_p_1_x+np_p_2_x)/2
    np_c_y=(np_p_1_y+np_p_2_y)/2
    
    # Obtain the distance between adjacent centroids
    np_distance=np.sqrt((np_c_x[1:]-np_c_x[:-1])**2 + (np_c_y[1:]-np_c_y[:-1])**2)
    #Add one more value at the beginning
    np_distance=np.concatenate((np.array([0]), np_distance), axis=0)
    
    # Draw figures
    fig, ax = plt.subplots()
    # Set the size of the figure
    fig.set_figwidth(12.8)
    fig.set_figheight(7.2)
    fig.set_dpi(100)
    ax.plot(np_RGB_FN, np_distance, '.')#Distance between centroids
    y_bottom, y_top = ax.get_ylim()#The range of y axis
    for i in range(np_first_frame_per_area.shape[0]):
        # Set area number (ax.text(x, y, string))
        if i==0: # Area 1
            ax.text(int(np_first_frame_per_area[i]/2), int(y_top-0.2*y_top), str(i+1))
        else: # Other areas
            ax.text(int((np_first_frame_per_area[i]+np_first_frame_per_area[i-1])/2), \
                    int(y_top-0.2*y_top), str(i+1))
        ax.axvline(np_first_frame_per_area[i], color='r', linestyle='--')
    # points above the threshold imply inaccurate detection
    ax.axhline(distance_threshold, color='g', linestyle='--')
    
    # Save the figure
    plt.show()
    plt.savefig(output_dir+'centroid_distance.png')
    pickle.dump(fig, open(output_dir+'centroid_distance.pickle', 'wb'))
    
    # Close the current figure
    plt.close()


# Update bbox coordinates in csv
def update_csv(input_dir, output_dir, list_FN, list_top_x, list_top_y, list_bottom_x, list_bottom_y):
    
    # Read the original csv
    df=pd.read_csv(input_dir, sep='\s*,\s*', header=[0], engine='python')
    
    # Sanity check
    assert len(list_FN)==len(list_top_x), 'len(list_FN) != len(list_top_x)'
    assert len(list_FN)==len(list_top_y), 'len(list_FN) != len(list_top_y)'
    assert len(list_FN)==len(list_bottom_x), 'len(list_FN) != len(list_bottom_x)'
    assert len(list_FN)==len(list_bottom_y), 'len(list_FN) != len(list_bottom_y)'
    
    
    # Update csv content
    for i, single_FN in enumerate(list_FN):
        
        df.loc[df['RGB_FN_from_bag']==single_FN, 'endpoint_1_x']=list_top_x[i]
        df.loc[df['RGB_FN_from_bag']==single_FN, 'endpoint_1_y']=list_top_y[i]
        df.loc[df['RGB_FN_from_bag']==single_FN, 'endpoint_2_x']=list_bottom_x[i]
        df.loc[df['RGB_FN_from_bag']==single_FN, 'endpoint_2_y']=list_bottom_y[i]
        
    # Save the updated csv
    df.to_csv(output_dir, index=False)
    
    return


# Generate a new bbox video
def generate_new_bbox_video(bag_video_dir, output_dir, output_csv_dir, output_video_dir):
    
    csv_row_idx=0
    # Read the bounding box coordinate 
    df=pd.read_csv(output_csv_dir, sep='\s*,\s*', header=[0], engine='python')
    list_RGB_FN=df['RGB_FN_from_bag'].tolist()
    list_top_x=df['endpoint_1_x'].tolist()
    list_top_y=df['endpoint_1_y'].tolist()
    list_bottom_x=df['endpoint_2_x'].tolist()
    list_bottom_y=df['endpoint_2_y'].tolist()
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    out_video = cv2.VideoWriter(output_video_dir, fourcc, 60.0, (848, 480))
    
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
                
        # Draw bbox on the video frame
        # Note that the csv file does not have rows with duplicated frame number,
        # so we can move to the next targeted frame by csv_row_idx+=1
        if rgb_frame_num_from_bag<=list_RGB_FN[-1] and csv_row_idx<len(list_RGB_FN): # Avoid the 'index out of range' 
            if rgb_frame_num_from_bag==list_RGB_FN[csv_row_idx]:
                
                # Convert color_frame and depth_frame to numpy array
                np_rgb_img = np.asanyarray(color_frame.get_data())
                
                # Convert from RGB format to BGR format
                np_bgr_img=cv2.cvtColor(np_rgb_img, cv2.COLOR_RGB2BGR)
                
                # Draw the adjusted bboxes
                cv2.rectangle(np_bgr_img, (int(list_top_x[csv_row_idx]), int(list_top_y[csv_row_idx])), \
                              (int(list_bottom_x[csv_row_idx]), int(list_bottom_y[csv_row_idx])), (0,255,0), 5)
                
                # Write rgb_frame_num_from_bag on the image
                cv2.rectangle(np_bgr_img, (702, 460),(848, 480),(255,255,255),-1)
                cv2.putText(np_bgr_img, str(rgb_frame_num_from_bag), (702, 475), 1, 1, (0,0,0), 2, cv2.LINE_AA)
                
                # Save all image to video file
                out_video.write(np_bgr_img)
                
                #
                csv_row_idx+=1
        else:#Break the loop after drawing all bboxes
            break
      
    # 
    pipeline.stop()
    out_video.release()
    
    # Generate figures for the change of bbox centroids
    figure_for_bbox_centroid(output_dir, output_csv_dir)
    
    # Generate a txt file that allows user to save the problematic frame numbers
    output_txt_dir=output_dir+'problematic_frame.txt'
    generate_problematic_FN_txt(output_txt_dir)
        
    return