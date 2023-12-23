# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 21:34:43 2022

@author: jianxig
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Global variables
IMG_HEIGHT=480
IMG_WIDTH=848

# Fill the hole within the mask image. The hole is due to the IMU color 
# different from the glove color
def glove_detection(input_mask):
    contours,_=cv2.findContours(input_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the radius of each contours
    contour_poly=[None]*len(contours)
    centers=[None]*len(contours)
    radius=[None]*len(contours)
    for i,single_contour in enumerate(contours):
        contour_poly[i]=cv2.approxPolyDP(single_contour, 3, True)
        centers[i], radius[i]=cv2.minEnclosingCircle(contour_poly[i])
            
    # Fill the contours
    drawing=np.zeros((input_mask.shape[0], input_mask.shape[1]), dtype=np.uint8)
    for i in range(0,len(contours)):
        if radius[i]>5:    
            cv2.drawContours(drawing, contours, i, (255,255,255), -1)
            
    # Find the bounding rectangle for the glove
    boundRect=cv2.boundingRect(drawing)    
    
    return drawing, boundRect


# Removing noise by morphology operation and connected components
def remove_noise_via_connectedComponent(ori_bgr_img, binary_img, output_dir, visualize_idividual_cc=True):
    
    # Remove noise by morphology operations
    # (Dilation)
    dilate_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)) #Circle with diameter=7
    dilated_img = cv2.dilate(binary_img, dilate_kernel, iterations=2)
    # (Erosion)
    erode_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)) #Circle with diameter=7
    eroded_img = cv2.erode(dilated_img, erode_kernel, iterations=2)
    
    # Remove noise by connected component calculation
    # (Find out the connected components in the image)
    cc_output = cv2.connectedComponentsWithStats(eroded_img, 8, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = cc_output
    # (Get the detection results)
    # ('-1' denotes 'background')
    list_top_x, list_top_y, list_h, list_w, list_area = [-1], [-1], [-1], [-1], [-1]
    # (Note: numLabels=0 denotes 'background', which shold be ignored)
    for idx in range(1, numLabels):
        
        # Get info for a cc
        single_cc_top_x=stats[idx, cv2.CC_STAT_LEFT]
        single_cc_top_y=stats[idx, cv2.CC_STAT_TOP]
        single_cc_h=stats[idx, cv2.CC_STAT_HEIGHT]
        single_cc_w=stats[idx, cv2.CC_STAT_WIDTH]
        single_cc_area=stats[idx, cv2.CC_STAT_AREA]
        
        # If both the height and the width of cc are < 5 pixels, the cc is
        # considered as noise
        if single_cc_h<5 and single_cc_w<5:
            list_top_x.append(0)# '0' denotes noise
            list_top_y.append(0)
            list_h.append(0)
            list_w.append(0)
            list_area.append(0)
        else:
            list_top_x.append(single_cc_top_x)
            list_top_y.append(single_cc_top_y)
            list_h.append(single_cc_h)
            list_w.append(single_cc_w)
            list_area.append(single_cc_area)
            
            # Visualize the connected component
            # (single_cc_mask is in [0, 255] rather than [0(false), 1(true)])
            if visualize_idividual_cc:
                single_cc_mask_255 = (labels==idx).astype('uint8')*255 
                stacked_img=np.stack((single_cc_mask_255,)*3, axis=-1)
                ori_rgb_img=cv2.cvtColor(ori_bgr_img,  cv2.COLOR_BGR2RGB)
                cat_img=np.concatenate((ori_rgb_img, stacked_img), axis=0)
                # (Save results)
                fig, ax = plt.subplots()
                ax.imshow(cat_img)
                ax.axis('off')
                ax.set_title('cc num: {}, H: {}, W: {}, Area: {:.2f}'.\
                             format(idx, single_cc_h, single_cc_w, list_area[-1]))
                plt.savefig(output_dir+str(idx)+'.png')
                plt.close()
        
    # Find out the connected component that has the largest area
    np_area=np.array(list_area)
    max_idx=np.argmax(np_area)
    # (single_cc_mask is in [0, 255] rather than [0(false), 1(true)])
    target_mask_255 = (labels==max_idx).astype('uint8')*255 
    # (Find out the corresponding bbox)(bbox=[top_x, top_y, h, w])
    target_bbox = [0] *4
    target_bbox[0]=list_top_x[max_idx]
    target_bbox[1]=list_top_y[max_idx]
    target_bbox[2]=list_w[max_idx]
    target_bbox[3]=list_h[max_idx]
    
    return target_mask_255, target_bbox


# (Data set 0) Remove the background using distance 
def remove_background_by_distance(np_depth_img_1, depth_scale, np_rgb_img):
    # Segment object using depth infomation
    # The unit for distance_mat is meter
    np_depth_float_1=np_depth_img_1.astype(float)
    distance_mat=np_depth_float_1*depth_scale
    # Note: If img_mask=(distance_mat<0.65), then the undefined pixels would be detected by rgb_img_segment
    img_mask=(distance_mat<0.75) #*(distance_mat!=0)
    img_segment=np_rgb_img*img_mask[:,:,None]
    
    return img_segment


# (Data set 1) Crop the glove in rgb image (using bounding rectangle)
# Note: boundRect: [top_left_x, top_left_y, width, height]
def crop_color_img_br(np_bgr_img, boundRect):
    
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
    modified_boundRect=np.zeros(4, dtype=np.int64)
    modified_boundRect[0]=ori_center_x-square_dim//2# top-left x
    modified_boundRect[1]=ori_center_y-square_dim//2# top-left y
    modified_boundRect[2]=square_dim# width
    modified_boundRect[3]=square_dim# height
    
    # Adjust the bbox, so that it is within the image
    if modified_boundRect[0]<0:
        modified_boundRect[0]=0
    if modified_boundRect[1]<0:
        modified_boundRect[1]=0
    if (modified_boundRect[0]+modified_boundRect[2])>=IMG_WIDTH:
        modified_boundRect[0]=IMG_WIDTH-square_dim-1
    if (modified_boundRect[1]+modified_boundRect[3])>=IMG_HEIGHT:
        modified_boundRect[1]=IMG_HEIGHT-square_dim-1

	# Note: boundRect[1] and [3] refer to rows, while boundRect[0] and [2] refer to columns
    cropped_bgr_img=np.copy(np_bgr_img[modified_boundRect[1]:modified_boundRect[1]+modified_boundRect[3], \
                                       modified_boundRect[0]:modified_boundRect[0]+modified_boundRect[2]])
    
    return cropped_bgr_img, modified_boundRect


# Calculate the threshold using Otsu method
def modified_Otsu_threshold(input_img):
    # find normalized_histogram, and its cumulative distribution function
    hist = cv2.calcHist([input_img],[0],None,[256],[0,256])
    
    # Remove the undefined pixels (ignore the pixel if its value = 0)
    # Note: A potential error: hist.sum() == 0, which will cause zero
    # division warning
    hist[0][0]=0
    
    # To avoid hist.sum()==0, which will cause zero division warning
    hist_sum=hist.sum()
    hist_sum=hist_sum if hist_sum>0 else 1e-4
    
    hist_norm = hist.ravel()/hist_sum
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1
    for i in range(1,256):
        p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
        if q1 < 1.e-6 or q2 < 1.e-6:
            continue
        b1,b2 = np.hsplit(bins,[i]) # weights
        # finding means and variances
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
            
    return thresh

# To display the depth images
def visualize_depth_image(depth_img_uint16):
    depth_img_f=depth_img_uint16.astype(float)
    pixel_range=np.max(depth_img_f) - np.min(depth_img_f)
    pixel_range=1e-9 if pixel_range==0 else pixel_range # Prevent the denominator being 0
    depth_img_normalized=(depth_img_f-np.min(depth_img_f))*255/pixel_range
    depth_img_uint8=depth_img_normalized.astype(np.uint8)
    depth_img_merged=cv2.merge([depth_img_uint8, depth_img_uint8, depth_img_uint8])
    
    return depth_img_merged

# Calculate the average 
def calculate_avg(input_arr, idx_0, idx_1):
    arr_segment=input_arr[idx_0:(idx_1+1)]
    avg=np.mean(arr_segment)
    
    return avg

# To generate the data sets
def generate_data_set(np_depth_img_1, np_rgb_img, depth_scale, output_dataset_dir, \
                      rgb_frame_num_from_bag, np_roll, np_pitch, np_yaw, gyro_x, gyro_y, gyro_z, \
                      IMU_idx_0, IMU_idx_1, bounding_box_csv, d1_csv, d2_csv, d3_csv, hsv_min_tuple, \
                      hsv_max_tuple, area_num, target_time, top_x, top_y, bottom_x, bottom_y, \
                      log_txt,visualize_result=False):
    
    # Generate the file name
    instant_gyro_x=gyro_x[IMU_idx_1]
    instant_gyro_y=gyro_y[IMU_idx_1]
    instant_gyro_z=gyro_z[IMU_idx_1]
    avg_gyro_x=calculate_avg(gyro_x, IMU_idx_0, IMU_idx_1)
    avg_gyro_y=calculate_avg(gyro_y, IMU_idx_0, IMU_idx_1)
    avg_gyro_z=calculate_avg(gyro_z, IMU_idx_0, IMU_idx_1)
    
    # test
    current_roll_value=calculate_avg(np_roll, IMU_idx_0, IMU_idx_1)
    current_pitch_value=calculate_avg(np_pitch, IMU_idx_0, IMU_idx_1)
    current_yaw_value=calculate_avg(np_yaw, IMU_idx_0, IMU_idx_1)
    sin_value=np.sin(current_roll_value*np.pi/180.0) # Using sin and cos to encode roll value
    cos_value=np.cos(current_roll_value*np.pi/180.0) # Using sin and cos to encode roll value
    str_roll_value=str(current_roll_value)
    split_roll_value=str_roll_value.split('.')
    if len(split_roll_value)==2:
        img_name=str(rgb_frame_num_from_bag)+'_'+split_roll_value[0]+'_'+split_roll_value[1][:3]+'.png'
    else:
        img_name=str(rgb_frame_num_from_bag)+'_'+split_roll_value[0]+'.png'
    
    # Convert rgb image to bgr image
    np_bgr_img=cv2.cvtColor(np_rgb_img, cv2.COLOR_RGB2BGR)

    # Record the bbox info
    boundRect=(top_x, top_y, bottom_x-top_x, bottom_y-top_y)
    if boundRect[2]<=50 or boundRect[3]<=50:
             d0_has_BB=0
    else:
             d0_has_BB=1
    bounding_box_csv.write('{},{},{},{},{},{},{}\n'.format(area_num, rgb_frame_num_from_bag, d0_has_BB, \
                         top_x, top_y, bottom_x, bottom_y))
    
    # Crop the glove in depth image (using bounding rectangle)
    if boundRect[2]>0 and boundRect[3]>0:
                
        # (Data set 1) Crop the glove in rgb image using color and depth info
        # *Note: This function will modify boundRect*
        d1_bgr_img, boundRect=crop_color_img_br(np_bgr_img, boundRect)
                  
        # Avoid Saving cropped images that have improper bounding box
        if boundRect[0]>=0 and boundRect[1]>=0 and boundRect[2]<IMG_WIDTH and boundRect[3]<IMG_HEIGHT:
        
            # (Data set 1) Save the images
            d1_path=output_dataset_dir+'d1/'+img_name
            d1_is_saved=cv2.imwrite(d1_path,d1_bgr_img,[cv2.IMWRITE_PNG_COMPRESSION, 0])
            if d1_is_saved == False:
                print('Error! Cannot save images for data set 1.')
                
            # (Data set 1) Save the results to csv file
            output_dataset_dir_split=output_dataset_dir.split('/')
            subject_name=output_dataset_dir_split[-2]
            d1_path_for_csv=subject_name+'/d1/'+img_name
            d1_csv.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(area_num, rgb_frame_num_from_bag, \
                        current_roll_value, current_pitch_value, current_yaw_value, sin_value, cos_value, \
                        instant_gyro_x, instant_gyro_y, instant_gyro_z, avg_gyro_x, avg_gyro_y, avg_gyro_z, \
                        d1_path_for_csv))
        else:
            log_txt.write('Error! Invalid bbox at Frame {} at Area {}\n'.format(rgb_frame_num_from_bag, area_num))
            log_txt.write('bbox location (topLeft_x, topLeft_y, width, height): ({}, {}, {}, {})\n'.\
                          format(boundRect[0], boundRect[1], boundRect[2], boundRect[3]))
            log_txt.write('Skip the frame...\n')
            log_txt.flush()
            boundRect[0], boundRect[1], boundRect[2], boundRect[3] = 0, 0, 0, 0 # Reset parameters
    
    return boundRect, current_roll_value, current_pitch_value, current_yaw_value,\
        sin_value, cos_value, np_bgr_img