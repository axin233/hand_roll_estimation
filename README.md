# Hand roll angle estimation via deep-learning computer vision algorithms
This is the repository for the hand roll angle estimation algorithms. 

For a surgical suturing simulator aiming for the clock-face model in FVS, Shayan et al. [^Mehdi_paper] show that participants' skill levels can be identified by their hand rotation at the roll axis. The approach, however, requires participants to wear IMUs, which might interfere with their suturing performance. Thus, this repository proposes 3 deep-learning computer vision algorithms for contact-free hand roll angle estimation.
- CNN: A convolutional neural network that receives a single image and then estimates the corresponding hand roll angle.
- Many-to-one (MO): A convolutional recurrent neural network that receives a sequence of images and then estimates the hand roll angle for the last image (i.e., many images -> one estimation).
- Many-to-many (MM): A convolutional recurrent neural network that receives a sequence of images and then estimates the hand roll angle for each image (i.e., many images -> many estimations).


[^Mehdi_paper]:
    Shayan, Amir Mehdi, Simar Singh, Jianxin Gao, Richard E. Groff, Joe Bible, John F. Eidt, Malachi Sheahan, Sagar S. Gandhi, Joseph V. Blas, and Ravikiran Singapogu. "Measuring hand movement for suturing skill assessment: A simulation-based study." Surgery 174, no. 5 (2023): 1184-1192.


## Demo
This demo is the output of the 3 algorithms when processing a graduate student's video. This participant has zero experience in suturing. The demo includes two portions:
- The left portion is a video with hand motion. Its corresponding frame number is denoted by the magenta vertical line in the right figure, and its corresponding IMU angle and estimated angles are shown in the title of the right figure.
- The right figure shows IMU angles and estimated angles from the 3 algorithms. Its legend includes the mean square error (MAE) for the 3 algorithms per suture. 

https://github.com/axin233/hand_roll_estimation/assets/59490151/0c84766d-c1a0-482e-a4a7-b8eb7deb8044

As shown in the demo, the estimated angles have similar patterns as the IMU angles, indicating the 3 algorithms can estimate hand roll angles based on the input video. Also, notice that CNN estimated angles have a large spike at Suture 7, but the spike disappear at MO and MM. One of the reasons is that hand motion rarely occurs in the training dataset, and CNN is sensitive to unseen hand motion.

## Performance evaluation

### Robustness

To examine the algorithms' robustness, videos collected at 4 locations are used. Notice that the 4 locations have different backgrounds and different camera orientations. 
