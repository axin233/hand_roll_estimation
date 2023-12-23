check_hsv.py, dataset_func.py, and utils_v1.py are used to check if the global threshold works or not. The output of the program includes:
- A video with the detected bounding box. (Tips: This video also contains frame numbers, so it can be used to identify which video frame
  has inaccurate bounding box. Those problematic video frames can then be adjusted via the program in the adjust_bbox folder)
- A scatter plot (i..e. centroid_distance.png) for showing the distance between the bounding box centroids in adjacent video frames. 
  It helps to identify which video has problematic bounding box detection. Note that an problematic bounding box detection usually causes
  a spike in the scatter plot (i.e., large distance between the bounding box centroids in adjacent video frames)
  
pickle_show_single_img.py
- It is for showing centroid_distance.pickle, which is the vector image of centroid_distance.png