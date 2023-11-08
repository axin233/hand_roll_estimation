# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 21:34:43 2022

@author: jianxig
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

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
    
    return


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
def crop_color_img_br(np_bgr_img, boundRect):

	# Note: boundRect[1] and [3] refer to rows, while boundRect[0] and [2] refer to columns
    cropped_bgr_img=np.copy(np_bgr_img[boundRect[1]:boundRect[1]+boundRect[3], \
                                       boundRect[0]:boundRect[0]+boundRect[2]])
    
    return cropped_bgr_img


# (Data set 2) Crop the glove in depth image (using bounding rectangle)
def crop_depth_img_br(np_depth_img_1, boundRect):
    cropped_depth_img_uint16\
        =np.copy(np_depth_img_1[boundRect[1]:boundRect[1]+boundRect[3], \
                                boundRect[0]:boundRect[0]+boundRect[2]])
    
    return cropped_depth_img_uint16


# Calculate the threshold using Otsu method
def modified_Otsu_threshold(input_img):
    # find normalized_histogram, and its cumulative distribution function
    hist = cv2.calcHist([input_img],[0],None,[256],[0,256])
    
    # Remove the undefined pixels (ignore the pixel if its value = 0)
    # Note: A potential error: hist.sum() would be zero, which will cause zero
    # division warning
    hist[0][0]=0
    
    hist_norm = hist.ravel()/hist.sum()
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


# To generate the data sets
def generate_data_set(np_depth_img_1, np_rgb_img, depth_scale, output_dataset_dir, \
                      rgb_frame_num_from_bag, np_roll, np_pitch, np_yaw, IMU_idx_1, bounding_box_csv,\
                         d1_csv, d2_csv, d3_csv, hsv_min_tuple, hsv_max_tuple, area_num,\
                             visualize_result=False):
    
    # Generate the file name
    current_roll_value=np_roll[IMU_idx_1]
    current_pitch_value=np_pitch[IMU_idx_1]
    current_yaw_value=np_yaw[IMU_idx_1]
    sin_value=np.sin(current_roll_value*np.pi/180.0) # Using sin and cos to encode roll value
    cos_value=np.cos(current_roll_value*np.pi/180.0) # Using sin and cos to encode roll value
    str_roll_value=str(current_roll_value)
    split_roll_value=str_roll_value.split('.')
    img_name=str(rgb_frame_num_from_bag)+'_'+split_roll_value[0]+'_'+split_roll_value[1]+'.png'
    
    # Convert rgb image to bgr image
    np_bgr_img=cv2.cvtColor(np_rgb_img, cv2.COLOR_RGB2BGR)
    
    # (Data set 0) Remove the background by distance
    rm_bg_img_bgr_full_size=remove_background_by_distance(np_depth_img_1, depth_scale, np_bgr_img)
        
    # (Data set 0) Save the results
    d0_path=output_dataset_dir+'d0/'+img_name
    d0_is_saved=cv2.imwrite(d0_path,rm_bg_img_bgr_full_size,[cv2.IMWRITE_PNG_COMPRESSION, 0])
    if d0_is_saved == False:
        print('Error! Cannot save images for data set 0.')
    
    # Obtain the mask for glove 
    np_hsv_img=cv2.cvtColor(np_rgb_img, cv2.COLOR_RGB2HSV)
    lower_bound=hsv_min_tuple
    upper_bound=hsv_max_tuple
    glove_mask_temp=cv2.inRange(np_hsv_img,lower_bound,upper_bound)
    
# =============================================================================
#     # test (For Subject_13860513672029808805-3-21-22-16-52-sur-sh)
#     if rgb_frame_num_from_bag>=2619 and rgb_frame_num_from_bag<=2630:
#         up_mask=np.ones((100, 848)).astype('uint8')
#         down_mask=np.zeros((380, 848)).astype('uint8')
#         mask=np.concatenate((up_mask*255, down_mask), axis=0)
#         glove_mask_temp=cv2.subtract(glove_mask_temp, mask)
#         cv2.imshow('glove_mask_temp', glove_mask_temp)
# =============================================================================
    
    glove_mask_filled, boundRect = glove_detection(glove_mask_temp)
    
    # (Data set 0) Record info
    # Convert the bounding box representation from *top-left corner + width and height* to 
    # *top-left corner + bottom-right corner*
	# Note: boundRect[3] refers to rows (i.e., height), 
	# while boundRect[2] refers to columns (i.e., width)
    endpoint_1_x=int(boundRect[0])
    endpoint_1_y=int(boundRect[1])
    endpoint_2_x=int(boundRect[0]+boundRect[2])
    endpoint_2_y=int(boundRect[1]+boundRect[3])
    output_dataset_dir_split=output_dataset_dir.split('/')
    subject_name=output_dataset_dir_split[-2]
    d0_path_for_csv=subject_name+'/d0/'+img_name
    if boundRect[2]<=50 or boundRect[3]<=50:
        d0_has_BB=0
    else:
        d0_has_BB=1
    bounding_box_csv.write('{},{},{},{},{},{},{},{}\n'.format(area_num, rgb_frame_num_from_bag, d0_has_BB, \
                    endpoint_1_x, endpoint_1_y, endpoint_2_x, endpoint_2_y, d0_path_for_csv))
    
    # Crop the glove in depth image (using bounding rectangle)
    if boundRect[2]>0 and boundRect[3]>0:
                
        # (Data set 1) Crop the glove in rgb image using color and depth info
        d1_bgr_img=crop_color_img_br(np_bgr_img, boundRect)
        
        # (Data set 1) Save the images
        d1_path=output_dataset_dir+'d1/'+img_name
        d1_is_saved=cv2.imwrite(d1_path,d1_bgr_img,[cv2.IMWRITE_PNG_COMPRESSION, 0])
        if d1_is_saved == False:
            print('Error! Cannot save images for data set 1.')
            
        # (Data set 1) Save the results to csv file
        d1_path_for_csv=subject_name+'/d1/'+img_name
        d1_csv.write('{},{},{},{},{},{},{},{}\n'.format(area_num, rgb_frame_num_from_bag, current_roll_value, \
                                                     current_pitch_value, current_yaw_value,\
                                                     sin_value, cos_value, d1_path_for_csv))
        
        # (Data set 2) Crop the glove in depth image using bounding rectangle
        d2_depth_img_uint16=crop_depth_img_br(np_depth_img_1, boundRect)
        d2_depth_img_uint16_cp=np.copy(d2_depth_img_uint16)
        
        # (Data set 3) Save the images
        d3_depth_img_uint16=np.copy(d2_depth_img_uint16)
        d3_path=output_dataset_dir+'d3/'+img_name
        d3_is_saved=cv2.imwrite(d3_path,d3_depth_img_uint16,[cv2.IMWRITE_PNG_COMPRESSION, 0])
        if d3_is_saved == False:
            print('Error! Cannot save images for data set 3.')
            
        # (Data set 3) Save the results to csv file
        d3_path_for_csv=subject_name+'/d3/'+img_name
        d3_csv.write('{},{},{},{},{},{},{},{}\n'.format(area_num, rgb_frame_num_from_bag, current_roll_value, \
                                                     current_pitch_value, current_yaw_value, \
                                                    sin_value, cos_value, d3_path_for_csv))
        
        # (Data set 2) Convert the datatype from uint16 to uint8
        cropped_depth_img_f=d2_depth_img_uint16_cp.astype(float)
        pixel_range=np.max(cropped_depth_img_f) - np.min(cropped_depth_img_f)
        pixel_range=1e-9 if pixel_range==0 else pixel_range # Prevent the denominator being 0
        depth_img_normalized=(cropped_depth_img_f-np.min(cropped_depth_img_f))*255/pixel_range
        depth_img_uint8=depth_img_normalized.astype(np.uint8)
        
        # (Data set 2) Obtain the threshold value via the modified Otsu method
        otsu_threshold=modified_Otsu_threshold(depth_img_uint8)
        
        # (Data set 2) Threshold the image
        actual_th, thre_img=cv2.threshold(depth_img_uint8,otsu_threshold,255,cv2.THRESH_BINARY)
        
        # (Data set 2) Convert the datatype from uint8 to uint16
        thre_img_f=thre_img.astype(float)
        pixel_range_1=np.max(thre_img_f) - np.min(thre_img_f)
        pixel_range_1=1e-9 if pixel_range_1==0 else pixel_range_1
        thre_img_normalized=(thre_img-np.min(thre_img_f))*65535/pixel_range_1
        thre_img_uint16=thre_img_normalized.astype(np.uint16)
        
        # (Data set 2) Remove the background
        d2_depth_img_final=cv2.subtract(d2_depth_img_uint16, thre_img_uint16)
        
        if visualize_result==True:
            d2_img=visualize_depth_image(d2_depth_img_final)
            cv2.namedWindow('RGB cropped image', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('Depth cropped image', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RGB cropped image',d1_bgr_img)
            cv2.imshow('Depth cropped image',d2_img) #Note: waitKey() is at the other function
            
        # (Data set 2) Save the images
        d2_path=output_dataset_dir+'d2/'+img_name            
        d2_is_saved=cv2.imwrite(d2_path,d2_depth_img_final,[cv2.IMWRITE_PNG_COMPRESSION, 0])
        if d2_is_saved == False:
            print('Error! Cannot save images for data set 2.')
            
        # (Data set 2) Save the results to csv file
        d2_path_for_csv=subject_name+'/d2/'+img_name
        d2_csv.write('{},{},{},{},{},{},{},{}\n'.format(area_num, rgb_frame_num_from_bag, current_roll_value, \
                                                     current_pitch_value, current_yaw_value,\
                                                    sin_value, cos_value, d2_path_for_csv))
    
    return boundRect, current_roll_value, current_pitch_value, current_yaw_value,\
        sin_value, cos_value, np_bgr_img