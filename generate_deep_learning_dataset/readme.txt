This program is for generating the deep-learning dataset.

Required input:
1. The videos with participants' hands. (Specified by data_dir)
2. The hand IMU measurements. (Specified by data_dir)
2. The needle entry/exit time for each suture (Specified by post_process_data_dir)
3. The bounding box coordinates (Specified by input_bbox_dir)
4. a csv file with the video names (Specified by hsv_csv_dir)

Outputs:
1. Cropped images that only contains participants' hands (It is in the folder 'd1')
2. A csv file containing the directories for cropped images and the corresponding IMU measurements. (It is called d1.csv)