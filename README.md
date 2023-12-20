# Hand roll angle estimation via deep-learning computer vision algorithms
This is the repository for the hand roll angle estimation algorithms. 

For a surgical suturing simulator aiming for the clock-face model in FVS, Shayan et al. [^Mehdi_paper] show that participants' skill levels can be identified by their hand rotation at the roll axis. The approach, however, requires participants to wear IMUs, which might interfere with their suturing performance. Thus, this repository proposes 3 deep-learning computer vision algorithms for contact-free hand roll angle estimation.
- CNN: A convolutional neural network that receives a single image and then estimates the corresponding hand roll angle.
- CRNN (MO): A convolutional recurrent neural network that receives a sequence of images and then estimates the hand roll angle for the last image.
- CRNN (MM): A convolutional recurrent neural network that receives a sequence of images and then estimates the hand roll angle for each image.


[^Mehdi_paper]:
    Shayan, Amir Mehdi, Simar Singh, Jianxin Gao, Richard E. Groff, Joe Bible, John F. Eidt, Malachi Sheahan, Sagar S. Gandhi, Joseph V. Blas, and Ravikiran Singapogu. "Measuring hand movement for suturing skill assessment: A simulation-based study." Surgery 174, no. 5 (2023): 1184-1192.


# Demo
https://github.com/axin233/hand_roll_estimation/assets/59490151/a5e2b88a-0d58-43e9-a641-7c2dd1df8b93

