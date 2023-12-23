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


def abbr_file_name_with_sur_dep(input_file_name):
    single_file_name_split=input_file_name.split('_')
    single_file_name_split_1=single_file_name_split[1].split('-')
    shorten_file_name=single_file_name_split_1[0][:4]+'-'+\
        single_file_name_split_1[4]+'-'+single_file_name_split_1[5]+'-'\
        +single_file_name_split_1[6]
    
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

# Wrap the data, so the range of the data is in [-180, 180]
def wrap_data(np_input):
# =============================================================================
#     # Find out the values that are larger/smaller than the upper/lower bounds
#     mask_upper_bound=np_input>180
#     mask_lower_bound=np_input<(-180)
#     
#     # Calculate the offsets
#     offset_upper_bound = mask_upper_bound*(-360)
#     offset_lower_bound = mask_lower_bound*360
#     np_offset=offset_upper_bound+offset_lower_bound
#     
#     # 
#     np_output=np_input+np_offset
# =============================================================================

    # https://stackoverflow.com/questions/15927755/opposite-of-numpy-unwrap
    np_output = (np_input + 180) % (360) - 180
    
    return np_output
    
    return np_output


def find_problematic_areas(list_subject_id, list_problematic_areas, target_subject_id):
    list_ignored_areas=[]
    bool_id_found=False
    
    for i, single_id in enumerate(list_subject_id):
        if single_id==target_subject_id:
            bool_id_found=True
            if list_problematic_areas[i] != '':
                # For the problematic areas that are not continuous
                if '&' in list_problematic_areas[i]:
                    split_prob_areas=list_problematic_areas[i].split('&')
                    list_ignored_areas=[int(j) for j in split_prob_areas]
                # For the problematic areas that are continuous
                elif '_' in list_problematic_areas[i]:
                    # Note: len(split_prob_areas)==2, which are [min_area, max_area]
                    split_prob_areas=list_problematic_areas[i].split('_')
                    list_ignored_areas=[j for j in range(int(split_prob_areas[0]), int(split_prob_areas[1])+1)]
                # If only one problematic areas (Note: convert a string to float first, then to integer)
                else:
                    list_ignored_areas.append(int(float(list_problematic_areas[i])))
                    
        # Sanity check
        if bool_id_found==False and i == len(list_subject_id)-1:
            print('Error! Fail to find the problematic areas for ', target_subject_id)
                    
    return list_ignored_areas


def extract_data_for_single_area(area_num, gt_df, mm_df, mo_df, cnn_df, enex_df, external_cam_df):
    
    # Get data from groundtruth csv
    gt_df_single_area=gt_df[gt_df['Area_Num']==area_num]
    
    # Get ground-truth
    if len(gt_df_single_area)==0:
        np_gt_roll=np.array([])
        np_gt_pitch=np.array([])
        np_gt_sin=np.array([])
        np_gt_cos=np.array([])
        np_FN=np.array([])
    else:        
        np_gt_roll=gt_df_single_area['roll'].to_numpy()
        np_gt_pitch=gt_df_single_area['pitch'].to_numpy()
        np_gt_sin=gt_df_single_area['sin'].to_numpy()
        np_gt_cos=gt_df_single_area['cos'].to_numpy()
        np_FN=gt_df_single_area['RGB_FN_from_bag'].to_numpy()
        
    # Get data for the external camera
    np_external_cam_time=external_cam_df['true_time_stamp (second)'].to_numpy()
    np_external_cam_FN=external_cam_df['frame_number_from_RGB'].to_numpy()
        
    # Get enty/exit time
    entry_time, exit_time=0, 0
    gt_df_enex_single_area=enex_df[enex_df['AreaNum']==area_num]
    if len(gt_df_enex_single_area)!=0:
        gt_df_enex_single_area=gt_df_enex_single_area.iloc[0] #Only get the first row
        entry_time=float(gt_df_enex_single_area['entry time'])
        exit_time=float(gt_df_enex_single_area['exit time'])
        
    # (mm) Get prediction result 
    if np_FN.shape[0]==0:
        start_FN=0
        end_FN=0
        np_mm_roll=np.array([])
        np_mm_sin=np.array([])
        np_mm_cos=np.array([])
    else:
        # Find out the range of data
        start_FN=np.min(np_FN)
        end_FN=np.max(np_FN)
        mm_df_single_area=mm_df.loc[(mm_df['FN_from_bag']>=start_FN) & (mm_df['FN_from_bag']<=(end_FN+1))]
        np_mm_roll=mm_df_single_area['roll(Pred)'].to_numpy()
        np_mm_sin=mm_df_single_area['sin(pred)'].to_numpy()
        np_mm_cos=mm_df_single_area['cos(pred)'].to_numpy()
        
    # (mo) Get prediction result
    if np_FN.shape[0]==0:
        np_mo_roll=np.array([])
        np_mo_sin=np.array([])
        np_mo_cos=np.array([])
    else:
        mo_df_single_area=mo_df.loc[(mo_df['FN_from_bag']>=start_FN) & (mo_df['FN_from_bag']<=(end_FN+1))]
        np_mo_roll=mo_df_single_area['roll(Pred)'].to_numpy()
        np_mo_sin=mo_df_single_area['sin(pred)'].to_numpy()
        np_mo_cos=mo_df_single_area['cos(pred)'].to_numpy()
        
    # (cnn) Get prediction result
    if np_FN.shape[0]==0:
        np_cnn_roll=np.array([])
    else:
        cnn_df_single_area=cnn_df.loc[(cnn_df['FN_from_bag']>=start_FN) & (cnn_df['FN_from_bag']<=(end_FN+1))]
        np_cnn_roll=cnn_df_single_area['roll(Pred)'].to_numpy()

    return start_FN, end_FN, np_FN, np_external_cam_time, np_external_cam_FN, \
        np_gt_roll, np_gt_pitch, np_gt_sin, np_gt_cos, \
        np_mm_roll, np_mm_sin, np_mm_cos, \
        np_mo_roll, np_mo_sin, np_mo_cos, \
        np_cnn_roll, entry_time, exit_time
        
        
def find_entry_exit_FN(np_FN, np_time, entry_time, exit_time):
    # (entry_FN) Find out the frame number closest to the entry time
    entry_time_idx=np.argmin(np.abs(np_time - entry_time))
    entry_FN=np_FN[entry_time_idx]
    
    # (exit_FN) Find out the frame number closest to the exit time
    exit_time_idx=np.argmin(np.abs(np_time - exit_time))
    exit_FN=np_FN[exit_time_idx]
    
    return entry_FN, exit_FN        


def extract_data_for_single_frame(rgb_frame_num_from_bag, np_FN, np_gt_roll, np_gt_pitch, \
             np_mm_roll, np_mo_roll, np_cnn_roll,\
            np_mm_error, np_mo_error, np_cnn_error):
    
    # Find out the index for the current RGB frame
    # Note: Since 'wait_for_frame()' is used, one RGB frame might have multiple time stamps
    temp_target_idx=np.where(np_FN==rgb_frame_num_from_bag)
    if temp_target_idx[0].shape[0] >= 1:
            target_idx=temp_target_idx[0][0]
            
            #
            gt_roll=np_gt_roll[target_idx]
            gt_pitch=np_gt_pitch[target_idx]
            #
            mm_roll=np_mm_roll[target_idx]
            #
            mo_roll=np_mo_roll[target_idx]
            #
            cnn_roll=np_cnn_roll[target_idx]
            #
            mm_roll_err=np_mm_error[target_idx]
            mo_roll_err=np_mo_error[target_idx]
            cnn_roll_err=np_cnn_error[target_idx]
            
    else:
            print('rgb_frame_num_from_bag is not within the gt csv file.\n')
            print('rgb_frame_num_from_bag: {}\n'.format(rgb_frame_num_from_bag))
            print('Skipping this frame.\n')
            
            # 
            gt_roll, gt_pitch = 0, 0
            mm_roll, mo_roll, cnn_roll = 0, 0, 0
            mm_roll_err, mo_roll_err, cnn_roll_err = 0, 0, 0
            
    return gt_roll, gt_pitch, mm_roll, mo_roll, cnn_roll,\
        mm_roll_err, mo_roll_err, cnn_roll_err
        
        
def calculate_error_roll(np_pred_roll, np_gt_roll):        
    np_offset=batch_offset_for_prediction(np_pred_roll, np_gt_roll)
    np_err_with_offset = np_pred_roll - np_gt_roll + np_offset
    # (mm) Sanity check
    check_range_of_compensated_output(np_err_with_offset)
    # (mm) Calculate errors
    avg_err=np.sum(np.abs(np_err_with_offset))/np_err_with_offset.shape[0]
    
    return np_err_with_offset, avg_err


def calculate_error_roll_vel(np_pred_roll_vel, np_gt_roll_vel):
    # Normalize the ground-truth data
    velocity_mean=0.071182
    velocity_std=1.301622
    # De-normalize prediction results
    np_pred_roll_vel=np_pred_roll_vel*velocity_std+velocity_mean
    # Calculate the error
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
def summary_view(rgb_img, area_num, frame_num_from_bag, np_FN_single_area, \
                 np_gt_roll, np_gt_pitch, np_mm_roll, np_mo_roll, np_cnn_roll,\
                gt_roll, gt_pitch, mm_roll, mo_roll, cnn_roll,\
                np_mm_error, np_mo_error, np_cnn_error,\
                mm_roll_err, mo_roll_err, cnn_roll_err,\
                mm_avg_err, mo_avg_err, cnn_avg_err, \
                list_bbox=[0,0,0,0]):
    
    #
    MARKER_SIZE=10
    
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
    #ax_img_2 = plt.subplot2grid(shape=(6, 3), loc=(3, 0), rowspan=3)
    ax_roll = plt.subplot2grid(shape=(6, 3), loc=(0, 1), rowspan=3, colspan=2)
    #ax_roll_err = plt.subplot2grid(shape=(6, 3), loc=(2, 1), rowspan=1, colspan=2)
    #ax_roll_vel = plt.subplot2grid(shape=(6, 3), loc=(3, 1), rowspan=1, colspan=2)
    #ax_roll_vel_err = plt.subplot2grid(shape=(6, 3), loc=(4, 1), rowspan=1, colspan=2)
    #ax_pitch = plt.subplot2grid(shape=(6, 3), loc=(5, 1), rowspan=1, colspan=2)
    
    # Draw the figure
    # (Top-left image)
    ax_img_1.imshow(rgb_img)
    ax_img_1.axis('off')
    ax_img_1.set_title('Suture:{}    Frame number:{}'.format(area_num, frame_num_from_bag))
    # (Right figures)
    ax_roll.plot(np_FN_single_area, np_gt_roll, color='b', label='IMU')
    ax_roll.plot(np_FN_single_area, np_cnn_roll, color='orange', label='CNN (MAE={:.2f})'.format(cnn_avg_err))
    ax_roll.plot(np_FN_single_area, np_mm_roll, color='g', label='MM (MAE={:.2f})'.format(mm_avg_err))
    ax_roll.plot(np_FN_single_area, np_mo_roll, color='r', label='MO (MAE={:.2f})'.format(mo_avg_err))
    ax_roll.axvline(frame_num_from_bag, color='m', linestyle='--')
    ax_roll.set_xlabel('Frame number')
    ax_roll.set_ylabel('Roll angle (degree)')
    #min_FN, max_FN, min_roll, max_roll = ax_roll.axis()
    #ax_roll.plot([frame_num_from_bag], [max_roll], 'rv', markersize=MARKER_SIZE)#Indicating the current frame
    ax_roll.grid()
    ax_roll.set_title('Roll angle (degree) (IMU: {:.2f} CNN: {:.2f} MM: {:.2f} MO: {:.2f})'.\
                      format(gt_roll, cnn_roll, mm_roll, mo_roll))
    #ax_roll.set_xticklabels([])
    ax_roll.legend(loc='upper left')
# =============================================================================
#     ax_roll_err.plot(np_FN_single_area, np_cnn_error, color='orange', label='CNN')
#     ax_roll_err.plot(np_FN_single_area, np_mm_error, color='g', label='MM')
#     ax_roll_err.plot(np_FN_single_area, np_mo_error, color='r', label='MO')
#     _, _, min_err, max_err = ax_roll_err.axis()
#     ax_roll_err.plot([frame_num_from_bag], [max_err], 'rv', markersize=MARKER_SIZE)#Indicating the current frame
#     ax_roll_err.grid()
#     #ax_roll_err.axvline(frame_num_from_bag, color='m', linestyle='--')
#     ax_roll_err.set_title('Roll angle (degree) (CNN: {:.2f} MM: {:.2f} MO:{:.2f})'.\
#                           format(cnn_roll_err, mm_roll_err, mo_roll_err))
#     ax_roll_err.legend(loc='upper left')
#     ax_pitch.plot(np_FN_single_area, np_gt_pitch)
#     _, _, min_pitch, max_pitch = ax_pitch.axis()
#     ax_pitch.plot([frame_num_from_bag], [max_pitch], 'rv', markersize=MARKER_SIZE)#Indicating the current frame
#     ax_pitch.grid()
#     ax_pitch.set_title('Pitch (degree) (IMU: {:.2f})'.format(gt_pitch))
# =============================================================================
    
    
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


def show_roll_vs_time(np_FN_single_area, np_gt_roll, np_cnn_roll, np_mm_roll, np_mo_roll, \
                      path_for_output_img, area_num, cnn_avg_err, mo_avg_err, \
                      mm_avg_err, entry_FN, exit_FN, SHOW_EN_EX=True):
    
    # Note: Make sure the multiplication of figsize and dip is 
    # the same as the VideoWriter resolution
    fig = plt.figure()
    fig.set_figwidth(12.8)
    fig.set_figheight(7.2)
    fig.set_dpi(100)
    
    # Creating grid for subplots
    #ax_img_1 = plt.subplot2grid(shape=(6, 3), loc=(0, 0), rowspan=3)
    #ax_img_2 = plt.subplot2grid(shape=(6, 3), loc=(3, 0), rowspan=3)
    ax_roll = plt.subplot2grid(shape=(6, 3), loc=(0, 1), rowspan=3, colspan=2)
    #ax_roll_err = plt.subplot2grid(shape=(6, 3), loc=(2, 1), rowspan=1, colspan=2)
    #ax_roll_vel = plt.subplot2grid(shape=(6, 3), loc=(3, 1), rowspan=1, colspan=2)
    #ax_roll_vel_err = plt.subplot2grid(shape=(6, 3), loc=(4, 1), rowspan=1, colspan=2)
    #ax_pitch = plt.subplot2grid(shape=(6, 3), loc=(5, 1), rowspan=1, colspan=2)
    
    # (Right figures)
    ax_roll.plot(np_FN_single_area, np_gt_roll, color='b', label='GT')
    ax_roll.plot(np_FN_single_area, np_cnn_roll, color='orange', label='CNN (MAE={:.2f} degrees)'.format(cnn_avg_err))
    ax_roll.plot(np_FN_single_area, np_mm_roll, color='g', label='MM (MAE={:.2f} degrees)'.format(mm_avg_err))
    ax_roll.plot(np_FN_single_area, np_mo_roll, color='r', label='MO (MAE={:.2f} degrees)'.format(mo_avg_err)) 
    if SHOW_EN_EX:
        ax_roll.axvline(entry_FN, color='c', linestyle='--')
        ax_roll.axvline(exit_FN, color='c', linestyle='--')
    ax_roll.set_xlabel('Frame number')
    ax_roll.set_ylabel('Roll angle (degree)')
    ax_roll.grid()
    ax_roll.set_title('Roll angle')
    ax_roll.legend(loc='upper left')
    
    # automatically adjust padding horizontally
    # as well as vertically.
    plt.tight_layout()
    
    # Save the SVG image
    plt.savefig(path_for_output_img+str(area_num)+'.svg')
                
    # Redraw the canvas
    fig.canvas.draw()
                
    # Convert matplotlib figure to opencv mat
    labeled_img=cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    
    # Save the PNG image
    bool_png_img_saved=cv2.imwrite(path_for_output_img+str(area_num)+'.png', labeled_img)
    assert bool_png_img_saved==True, 'Fail to save the PNG images'
    
    # Close the current figure
    plt.close()
    
    return 