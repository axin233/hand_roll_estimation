# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 11:33:22 2023

@author: jianxig
"""
import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt


# For checking if the track bar has been dragged
def checkChanged(x):
    pass

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
    

def adjust_threshold_slider_bar(original_bgr_img, output_data_dir, tuple_HSV_min, tuple_HSV_max):
    
    # Open a file to save the results
    output_csv=open(output_data_dir, 'w')
    output_csv.write('h_min,s_min,v_min,h_max,s_max,v_max\n')
    
    # Define the initial hsv value
    init_H_min, init_S_min, init_V_min=tuple_HSV_min[0], tuple_HSV_min[1], tuple_HSV_min[2]
    init_H_max, init_S_max, init_V_max=tuple_HSV_max[0], tuple_HSV_max[1], tuple_HSV_max[2]
    
    #
    cv2.namedWindow("Original image", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Thresholded image", cv2.WINDOW_AUTOSIZE)
    
    # The window for slider bars
    cv2.namedWindow('HSV adjustment')
    cv2.createTrackbar("H_min", "HSV adjustment", init_H_min, 180, checkChanged)
    cv2.createTrackbar("S_min", "HSV adjustment", init_S_min, 255, checkChanged)
    cv2.createTrackbar("V_min", "HSV adjustment", init_V_min, 255, checkChanged)
    cv2.createTrackbar("H_max", "HSV adjustment", init_H_max, 180, checkChanged)
    cv2.createTrackbar("S_max", "HSV adjustment", init_S_max, 255, checkChanged)
    cv2.createTrackbar("V_max", "HSV adjustment", init_V_max, 255, checkChanged)
    
    # Convert the image from bgr to hsv
    # Note: For cv.COLOR_BGR2HSV, H=[0,180]
    hsv_img=cv2.cvtColor(original_bgr_img, cv2.COLOR_BGR2HSV)
    
    # Loop for adjusting threshold
    while(True):
        
        # Get the value from the slider bar
        H_min=cv2.getTrackbarPos("H_min", "HSV adjustment")
        S_min=cv2.getTrackbarPos("S_min", "HSV adjustment")
        V_min=cv2.getTrackbarPos("V_min", "HSV adjustment")
        H_max=cv2.getTrackbarPos("H_max", "HSV adjustment")
        S_max=cv2.getTrackbarPos("S_max", "HSV adjustment")
        V_max=cv2.getTrackbarPos("V_max", "HSV adjustment")
        
        # Adjust threshold
        threshold_img=cv2.inRange(hsv_img, (H_min, S_min, V_min), (H_max, S_max, V_max))
        
        # Show the original and the thresholded image
        cv2.imshow('Original image', original_bgr_img)
        cv2.imshow('Thresholded image', threshold_img)
        
        # if pressed escape exit program
        if cv2.waitKey(1) == 27:
            print('Key Esc is pressed. Exitting...')
            break
        
    # Save the hsv value
    output_csv.write('{},{},{},{},{},{}\n'.format(H_min,S_min,V_min,H_max,S_max,V_max))
    tuple_hsv_min=(H_min,S_min,V_min)
    tuple_hsv_max=(H_max,S_max,V_max)
        
    #    
    cv2.destroyAllWindows()  
    output_csv.close()
    
    return threshold_img, tuple_hsv_min, tuple_hsv_max
        

if __name__=='__main__':
    list_frame_number=[3130, 1390, 1969, 1980, 3528, 3534]#
    tuple_HSV_min=(98, 175, 132)
    tuple_HSV_max=(102, 255, 255)
    frame_number=list_frame_number[0]
    work_dir='D:/test_videos/extracted_frame/7175-16-49/'
    input_img_dir=work_dir+str(frame_number)+'.png'
    output_data_dir=work_dir+'hsv.csv'
    BOOL_SHOW_INDIVIDUAL_CC=True
    
    # Read images
    original_bgr_img=cv2.imread(input_img_dir, cv2.IMREAD_UNCHANGED)
    
    # Adjust hsv threshold
    threshold_img, tuple_hsv_min, tuple_hsv_max = adjust_threshold_slider_bar\
        (original_bgr_img, output_data_dir, tuple_HSV_min, tuple_HSV_max)
        
    # Threshold the images using the returned hsv values
    for frame_number in list_frame_number:
        single_png_dir=work_dir+str(frame_number)+'.png'
        single_bgr_img=cv2.imread(single_png_dir, cv2.IMREAD_UNCHANGED)
        # Change the color space from bgr to hsv
        single_hsv_img=cv2.cvtColor(single_bgr_img, cv2.COLOR_BGR2HSV)
        # Threshold the image
        single_threshold_img=cv2.inRange(single_hsv_img, tuple_hsv_min, tuple_hsv_max)
        # Detect the connected components (cc) in each image. The cc for each
        # image are saved to the corresponding folder
        cc_folder_dir=''
        if BOOL_SHOW_INDIVIDUAL_CC:
            cc_folder_dir=work_dir+str(frame_number)+'/'
            if not os.path.exists(cc_folder_dir):#Create dir if it does not exist
                os.makedirs(cc_folder_dir)
            else: # Otherwise, delete content in the folder
                all_file_names=glob.glob(cc_folder_dir+'*')
                for single_name in all_file_names:
                    os.remove(single_name)
        
        # Remove noise using morphology and connected component calculation
        modified_threshold_img, boundRect = remove_noise_via_connectedComponent\
            (single_bgr_img, single_threshold_img, cc_folder_dir, \
             visualize_idividual_cc=BOOL_SHOW_INDIVIDUAL_CC)
            
        # Draw the bounding box on image
        if boundRect[2]>0 and boundRect[3]>0:
            cv2.rectangle(single_bgr_img, (int(boundRect[0]), int(boundRect[1])), \
                (int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])), (0,255,0), 5)
        
        # Save the final result
        single_stacked_img=np.stack((modified_threshold_img,)*3, axis=-1)
        single_cat_img=np.concatenate((single_bgr_img, single_stacked_img), axis=0)
        # Save the result
        output_img_dir=work_dir+str(frame_number)+'_threshold.png'
        saved_img=cv2.imwrite(output_img_dir, single_cat_img)
        if saved_img==False:
            print('Failed to save the thresholded image')
    