This foder contains some helpful programs for determining the global threshold for each videos.

- extract_frame.py: It is used to extract video frames that has inaccurate glove detection (or inaccurate bounding box detection). 
  These video frames suggest the global threshold should be adjusted.
  
- HSV_adjustment_via_slider_bar.py: This program loads the video frames extracted by extract_frame.py, and then allow the user to re-define 
  the global threshold using the slider bar.
  
(Tips: Using the global threshold for bounding box detection is much faster than manually drawing the bounding box for each frame.)