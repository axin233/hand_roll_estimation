# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 20:34:02 2023

@author: jianxig
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

def abbr_file_name(input_file_name):
    single_file_name_split=input_file_name.split('_')
    single_file_name_split_1=single_file_name_split[1].split('-')
    shorten_file_name=single_file_name_split_1[0][:4]+'-'+\
        single_file_name_split_1[4]+'-'+single_file_name_split_1[5]
    
    return shorten_file_name


# Wrap the results, then calculate the error for a specific batch
def batch_offset_for_prediction(np_outputs, np_labels):
    np_diff = np_outputs - np_labels
    bool_top_val=np_diff>=180
    bool_bottom_val=np_diff<=-180
    offset_larger_180=bool_top_val*(-360.0)
    offset_smaller_180=bool_bottom_val*(360.0)
    np_offset=offset_larger_180+offset_smaller_180
    
    return np_offset


# Check if the compensated output is in [-pi, pi]
def check_range_of_compensated_output(np_arr):
    
    # Find out the elements that is <-pi or >pi
    bool_mask_1=np_arr<-180
    bool_mask_2=np_arr>180
    bool_mask=bool_mask_1+bool_mask_2
    
    assert np.count_nonzero(bool_mask)==0, 'Compensated output range is not in [-pi, pi]'
    
    return


def extract_data_for_single_area(area_num, gt_df, roll_mm_df, roll_mo_df, vel_mm_df, vel_mo_df):
    
    # Get data from groundtruth csv
    gt_df_single_area=gt_df[gt_df['Area_Num']==area_num]
    
    # Get ground-truth
    if len(gt_df_single_area)==0:
        np_gt_roll=np.array([])
        np_gt_pitch=np.array([])
        np_gt_sin=np.array([])
        np_gt_cos=np.array([])
        np_gt_roll_vel=np.array([])
        np_FN=np.array([])
    else:        
        np_gt_roll=gt_df_single_area['roll'].to_numpy()
        np_gt_pitch=gt_df_single_area['pitch'].to_numpy()
        np_gt_sin=gt_df_single_area['sin'].to_numpy()
        np_gt_cos=gt_df_single_area['cos'].to_numpy()
        np_gt_roll_vel=gt_df_single_area['AVG_gyro_x'].to_numpy()
        np_FN=gt_df_single_area['RGB_FN_from_bag'].to_numpy()
        
    # (mm) Get prediction result 
    if np_FN.shape[0]==0:
        start_FN=0
        end_FN=0
        np_mm_roll=np.array([])
        np_mm_sin=np.array([])
        np_mm_cos=np.array([])
        np_mm_roll_vel=np.array([])
    else:
        # Find out the range of data
        start_FN=np.min(np_FN)
        end_FN=np.max(np_FN)
        roll_mm_df_single_area=roll_mm_df.loc[(roll_mm_df['FN_from_bag']>=start_FN) & \
                                              (roll_mm_df['FN_from_bag']<=(end_FN+1))]
        vel_mm_df_single_area=vel_mm_df.loc[(vel_mm_df['FN_from_bag']>=start_FN) & \
                                            (vel_mm_df['FN_from_bag']<=(end_FN+1))]
        np_mm_roll=roll_mm_df_single_area['roll(Pred)'].to_numpy()
        np_mm_sin=roll_mm_df_single_area['sin(pred)'].to_numpy()
        np_mm_cos=roll_mm_df_single_area['cos(pred)'].to_numpy()
        np_mm_roll_vel=vel_mm_df_single_area['roll vel(Pred)'].to_numpy()
        
    # (mo) Get prediction result
    if np_FN.shape[0]==0:
        np_mo_roll=np.array([])
        np_mo_sin=np.array([])
        np_mo_cos=np.array([])
        np_mo_roll_vel=np.array([])
    else:
        roll_mo_df_single_area=roll_mo_df.loc[(roll_mo_df['FN_from_bag']>=start_FN) & \
                                              (roll_mo_df['FN_from_bag']<=(end_FN+1))]
        vel_mo_df_single_area=vel_mo_df.loc[(vel_mo_df['FN_from_bag']>=start_FN) & \
                                            (vel_mo_df['FN_from_bag']<=(end_FN+1))]
        np_mo_roll=roll_mo_df_single_area['roll(Pred)'].to_numpy()
        np_mo_sin=roll_mo_df_single_area['sin(pred)'].to_numpy()
        np_mo_cos=roll_mo_df_single_area['cos(pred)'].to_numpy()
        np_mo_roll_vel=vel_mo_df_single_area['roll vel(Pred)'].to_numpy()

    return start_FN, end_FN, np_FN, np_gt_roll,np_gt_pitch, np_gt_sin, np_gt_cos, np_gt_roll_vel, \
        np_mm_roll, np_mm_sin, np_mm_cos, np_mm_roll_vel, \
        np_mo_roll, np_mo_sin, np_mo_cos, np_mo_roll_vel
        

def extract_data_for_single_frame(rgb_frame_num_from_bag, np_FN, np_gt_roll, np_gt_pitch, np_gt_sin, \
            np_gt_cos, np_gt_roll_vel, np_mm_roll, np_mm_sin, np_mm_cos, np_mm_roll_vel, \
            np_mo_roll, np_mo_sin, np_mo_cos, np_mo_roll_vel,\
            np_mm_error, np_mo_error, np_mm_vel_error, np_mo_vel_error):
    
    # Find out the index for the current RGB frame
    # Note: Since 'wait_for_frame()' is used, one RGB frame might have multiple time stamps
    temp_target_idx=np.where(np_FN==rgb_frame_num_from_bag)
    if temp_target_idx[0].shape[0] >= 1:
            target_idx=temp_target_idx[0][0]
            
            #
            gt_roll=np_gt_roll[target_idx]
            gt_pitch=np_gt_pitch[target_idx]
            gt_sin=np_gt_sin[target_idx]
            gt_cos=np_gt_cos[target_idx]
            gt_roll_vel=np_gt_roll_vel[target_idx]
            #
            mm_roll=np_mm_roll[target_idx]
            mm_sin=np_mm_sin[target_idx]
            mm_cos=np_mm_cos[target_idx]
            mm_roll_vel=np_mm_roll_vel[target_idx]
            #
            mo_roll=np_mo_roll[target_idx]
            mo_sin=np_mo_sin[target_idx]
            mo_cos=np_mo_cos[target_idx]
            mo_roll_vel=np_mo_roll_vel[target_idx]
            #
            mm_roll_err=np_mm_error[target_idx]
            mo_roll_err=np_mo_error[target_idx]
            mm_vel_err=np_mm_vel_error[target_idx]
            mo_vel_err=np_mo_vel_error[target_idx]
    else:
            print('rgb_frame_num_from_bag is not within the gt csv file.\n')
            print('rgb_frame_num_from_bag: {}\n'.format(rgb_frame_num_from_bag))
            print('Skipping this frame.\n')
            
            # 
            gt_roll, gt_pitch, gt_sin, gt_cos, gt_roll_vel = 0, 0, 0, 0, 0
            mm_roll, mm_sin, mm_cos, mm_roll_vel = 0, 0, 0, 0
            mo_roll, mo_sin, mo_cos, mo_roll_vel = 0, 0, 0, 0
            mm_roll_err, mo_roll_err, mm_vel_err, mo_vel_err = 0, 0, 0, 0
            
    return gt_roll, gt_pitch, gt_sin, gt_cos, gt_roll_vel,\
        mm_roll, mm_sin, mm_cos, mm_roll_vel,\
        mo_roll, mo_sin, mo_cos, mo_roll_vel,\
        mm_roll_err, mo_roll_err, mm_vel_err, mo_vel_err
        
        
def calculate_error_roll(np_pred_roll, np_gt_roll):        
    np_offset=batch_offset_for_prediction(np_pred_roll, np_gt_roll)
    np_err_with_offset = np_pred_roll - np_gt_roll + np_offset
    # (mm) Sanity check
    check_range_of_compensated_output(np_err_with_offset)
    # (mm) Calculate errors
    avg_err=np.sum(np.abs(np_err_with_offset))/np_err_with_offset.shape[0]
    
    return np_err_with_offset, avg_err


def calculate_error_roll_vel(np_pred_roll_vel, np_gt_roll_vel):
    
    # Calculate the error
    # (Note: The prediction has been de-normalized when testing the algorithm)
    np_err=np_pred_roll_vel-np_gt_roll_vel
    # Calculate the avg error
    avg_err=np.sum(np.abs(np_err))/np_pred_roll_vel.shape[0]
    
    return np_err, avg_err


# Adust the bbox so that it is a square
# Note: boundRect: [top_left_x, top_left_y, width, height]
def adjust_bbox(boundRect, img_width=848, img_height=480):
    
    # Make sure the bbox is a suqare
    # Find out the width, height of the original bbox
    width=boundRect[2]
    height=boundRect[3]
    # The longest edge would be the edge of the square
    square_dim=width if width>=height else height 
    # Calculate the center of the original bbox
    ori_center_x=boundRect[0]+boundRect[2]//2
    ori_center_y=boundRect[1]+boundRect[3]//2
    # Save the modified bounding box
    modified_boundRect=[0]*4
    modified_boundRect[0]=ori_center_x-square_dim//2# top-left x
    modified_boundRect[1]=ori_center_y-square_dim//2# top-left y
    modified_boundRect[2]=square_dim# width
    modified_boundRect[3]=square_dim# height
    
    # Adjust the bbox, so that it is within the image
    if modified_boundRect[0]<0:
        modified_boundRect[0]=0
    if modified_boundRect[1]<0:
        modified_boundRect[1]=0
    if (modified_boundRect[0]+modified_boundRect[2])>=img_width:
        modified_boundRect[0]=img_width-square_dim-1
    if (modified_boundRect[1]+modified_boundRect[3])>=img_height:
        modified_boundRect[1]=img_height-square_dim-1
        
    # Make sure the numbers are integers
    modified_boundRect = list(map(int, modified_boundRect))
    
    return modified_boundRect


# For drawing the figure
def summary_view(rgb_img, area_num, frame_num_from_bag, \
                 mm_avg_err, mo_avg_err, np_FN_single_area, \
                 np_gt_roll, np_gt_pitch, np_mm_roll, np_mo_roll, \
                gt_roll, gt_pitch, mm_roll, mo_roll, \
                np_mm_error, np_mo_error,\
                gt_sin, gt_cos, mm_sin, mm_cos, mo_sin, mo_cos, \
                np_gt_roll_vel, np_mm_roll_vel, np_mo_roll_vel, \
                gt_roll_vel, mm_roll_vel, mo_roll_vel, \
                np_mm_vel_error, np_mo_vel_error, mm_roll_err, mo_roll_err, \
                mm_vel_err, mo_vel_err,  mm_avg_vel_err, mo_avg_vel_err, list_bbox=[0,0,0,0]):
    
    # AVG error
    # mm_avg_err, mo_avg_err
    # mm_avg_vel_err, mo_avg_vel_err, 
    
    # Draw the bounding Rectangle on the images
    if (list_bbox[2]-list_bbox[0])>0 and (list_bbox[3]-list_bbox[1])>0:
        cv2.rectangle(rgb_img, (list_bbox[0], list_bbox[1]), \
              (list_bbox[2], list_bbox[3]), (0,255,0), 5)
        
    
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
    ax_roll_err = plt.subplot2grid(shape=(6, 3), loc=(2, 1), rowspan=1, colspan=2)
    ax_roll_vel = plt.subplot2grid(shape=(6, 3), loc=(3, 1), rowspan=1, colspan=2)
    ax_roll_vel_err = plt.subplot2grid(shape=(6, 3), loc=(4, 1), rowspan=1, colspan=2)
    ax_pitch = plt.subplot2grid(shape=(6, 3), loc=(5, 1), rowspan=1, colspan=2)
    
    # Draw the figure
    # (Top-left image)
    ax_img_1.imshow(rgb_img)
    ax_img_1.axis('off')
    #ax_img_2.imshow(np_depth_img)
    #ax_img_2.axis('off')
    ax_img_1.set_title('Area:{}    Frame number:{}'.format(area_num, frame_num_from_bag))
# =============================================================================
#     ax_img_1.set_title('Area:{} FN:{} MM_error:{:.2f} MO_error:{:.2f}'.\
#                        format(area_num, frame_num_from_bag, mm_avg_err, mo_avg_err))
# =============================================================================
    # (Middle image)
    ax_img_2.plot([0,10], [0,10], 'w')# Create a blank blackground
    ax_img_2.axis('off')
    ax_img_2.text(0, 10, 'Avg angle error (MM): {:.2f}'.format(mm_avg_err))
    ax_img_2.text(0, 9, 'Avg angle error (MO): {:.2f}'.format(mo_avg_err))
    ax_img_2.text(0, 8, 'Avg velocity error (MM): {:.2f}'.format(mm_avg_vel_err))
    ax_img_2.text(0, 7, 'Avg velocity error (MO): {:.2f}'.format(mo_avg_vel_err))
    # (Right figures)
    ax_roll.plot(np_FN_single_area, np_gt_roll, color='b', label='GT')
    ax_roll.plot(np_FN_single_area, np_mm_roll, color='g', label='MM')
    ax_roll.plot(np_FN_single_area, np_mo_roll, color='r', label='MO')
    ax_roll.axvline(frame_num_from_bag, color='m', linestyle='--')
    #ax_roll.scatter([np_time_for_plot[IMU_idx_1]], [current_roll_value], color='g', zorder=5)
    #ax_roll.axvline(np_time_for_plot[IMU_idx_2], color='r')
    ax_roll.set_title('Roll angle (GT: {:.2f} MM: {:.2f} MO: {:.2f})'.format(gt_roll, mm_roll, mo_roll))
    ax_roll.set_xticklabels([])
    ax_roll.legend(loc='upper right')
    ax_roll_err.plot(np_FN_single_area, np_mm_error, color='g', label='MM')
    ax_roll_err.plot(np_FN_single_area, np_mo_error, color='r', label='MO')
    ax_roll_err.axvline(frame_num_from_bag, color='m', linestyle='--')
# =============================================================================
#     ax_roll_err.set_title('GT(sin,cos): ({:.2f},{:.2f}) MM(sin,cos): ({:.2f},{:.2f}) MO(sin,cos): ({:.2f},{:.2f})'.\
#                           format(gt_sin, gt_cos, mm_sin, mm_cos, mo_sin, mo_cos))
# =============================================================================
    ax_roll_err.set_title('Errors for roll angle (MM: {:.2f} MO:{:.2f})'.format(mm_roll_err, mo_roll_err))
    ax_roll_err.set_xticklabels([])
    ax_roll_err.legend(loc='upper right')
    ax_roll_vel.plot(np_FN_single_area, np_gt_roll_vel, color='b', label='GT')
    ax_roll_vel.plot(np_FN_single_area, np_mm_roll_vel, color='g', label='MM')
    ax_roll_vel.plot(np_FN_single_area, np_mo_roll_vel, color='r', label='MO')
    ax_roll_vel.axvline(frame_num_from_bag, color='m', linestyle='--')
    ax_roll_vel.set_title('Roll velocity (GT: {:.4f} MM: {:.4f} MO: {:.4f})'.format(gt_roll_vel, mm_roll_vel, mo_roll_vel))
    ax_roll_vel.set_xticklabels([])
    ax_roll_vel.legend(loc='upper right')
    ax_roll_vel_err.plot(np_FN_single_area, np_mm_vel_error, color='g', label='MM')
    ax_roll_vel_err.plot(np_FN_single_area, np_mo_vel_error, color='r', label='MO')
    ax_roll_vel_err.set_title('Errors for roll velocity (MM: {:.2f} MO:{:.2f})'.format(mm_vel_err, mo_vel_err))
    ax_roll_vel_err.set_xticklabels([])
    ax_roll_vel_err.axvline(frame_num_from_bag, color='m', linestyle='--')
    ax_roll_vel_err.legend(loc='upper right')
    ax_pitch.plot(np_FN_single_area, np_gt_pitch)
    ax_pitch.set_title('Pitch (GT: {:.2f})'.format(gt_pitch))
    ax_pitch.axvline(frame_num_from_bag, color='m', linestyle='--')
    
    # automatically adjust padding horizontally
    # as well as vertically.
    plt.tight_layout()
                
    # Redraw the canvas
    fig.canvas.draw()
                
    # Convert matplotlib figure to opencv mat
    labeled_img=cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    
    # Close the current figure
    plt.close()
    
    return labeled_img