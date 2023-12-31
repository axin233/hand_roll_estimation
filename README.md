# Hand roll angle estimation via deep-learning computer vision algorithms

## Introduction
For a surgical suturing simulator designed for the FVS clock-face model (see Fig. 1(a)), Shayan et al. [^Mehdi_paper] show that participants' skill levels can be identified by their hand rotation at the roll axis (see Fig. 1(d)). The approach, however, requires participants to wear IMUs (see Fig. 1(d)), which might interfere with their suturing performance. Thus, this repository proposes 3 deep-learning computer vision algorithms for contact-free hand roll angle estimation.
- CNN: A convolutional neural network that receives a single image and then estimates the corresponding hand roll angle [^CNN_paper].
- Many-to-one (MO): A convolutional recurrent neural network that receives a sequence of images and then estimates the hand roll angle for the last image (i.e., many images -> one estimation).
- Many-to-many (MM): A convolutional recurrent neural network that receives a sequence of images and then estimates the hand roll angle for each image (i.e., many images -> many estimations).

[^CNN_paper]:
    Jianxin Gao, Shayan, Amir Mehdi, Simar Singh, Ravikiran Singapogu and Richard E. Groff. "Deep-learning computer vision algorithm for hand roll angle estimation and motion-based surgical suturing skill assessment" (Unpublished)

<!--
![simulator_v1](https://github.com/axin233/hand_roll_estimation/assets/59490151/4eac4c05-d48b-4527-999a-c96100a86ed6)
-->

<p align="center">
  <img width="661" height="373" src="https://github.com/axin233/hand_roll_estimation/assets/59490151/4eac4c05-d48b-4527-999a-c96100a86ed6">
</p>

> Fig. 1. The suturing simulator: (a) Front view; (b) Top view of the membrane housing; (c) The internal structure of the membrane housing; (d) Surface condition; (e) Depth condition.

[^Mehdi_paper]:
    Shayan, Amir Mehdi, Simar Singh, Jianxin Gao, Richard E. Groff, Joe Bible, John F. Eidt, Malachi Sheahan, Sagar S. Gandhi, Joseph V. Blas, and Ravikiran Singapogu. "Measuring hand movement for suturing skill assessment: A simulation-based study." Surgery 174, no. 5 (2023): 1184-1192.


## Demo
This demo is the output of the 3 algorithms when processing a graduate student's video. This participant has zero experience in suturing, and his videos are excluded when training the algorithms. 

The demo includes two portions:
- The left portion is a video from Camera 2 (see Fig. 1(a)). Its corresponding frame number is denoted by the magenta vertical line in the right figure, and its corresponding IMU angle and estimated angles are shown in the title of the right figure.
- The right figure shows IMU angles and estimated angles from the 3 algorithms. Its legend includes the mean absolute error (MAE) for the 3 algorithms per suture. 

https://github.com/axin233/hand_roll_estimation/assets/59490151/0c84766d-c1a0-482e-a4a7-b8eb7deb8044

As shown in the demo, the estimated angles have similar patterns as the IMU angles, indicating the 3 algorithms can estimate hand roll angles based on the input video. Also, notice that CNN estimated angles have a large spike at Suture 7, but the spike disappear at MO and MM. One of the reasons is that hand motion rarely occurs in the training dataset, and CNN is sensitive to unseen hand motion.

## Performance evaluation

### Results of processing videos collected at different locations

To examine the algorithms' robustness, videos collected at 4 locations are used. Notice that the 4 locations have different backgrounds and different camera orientations. 

![location_img_1](https://github.com/axin233/hand_roll_estimation/assets/59490151/7be46c6a-8521-43f0-a2fc-6283e100c16f)

> Fig. 2 Example video frames from the 4 data collection locations

This table summarizes the roll angle estimation error (unit: degrees) for the 3 algorithms at the 4 locations.

|          | Number of video frames | CNN | MO | MM |
| :------: | :------: | :------: | :------: | :------: |
| Location A | 75k | 10.25 | 9.6 | 8.77 |
| Location B | 235k | 9.52 | 9.45 | 9.08 |
| Location C | 200k | 7.27 | 7.47 | 7.37 |
| Location D | 57k | 8.39 | 8.46 | 9.03 |

> Note that the number of sutures consists of sutures at surface condition (see Fig. 1(d)) and sutures at depth condition (see Fig. 1(e)).

### Inference speed

This table shows the inference speed of the 3 algorithms on an NVidia V100 GPU.

|          | CNN | MO | MM |
| :------: | :------: | :------: | :------: |
| Speed (frames/sec) | 127.8 | 105.2 | 106 |

> Note that the results are the averaged inference speed of 5 trials. To accelerate MO and MM, the sliding window approch is used. That is, when a new image sequence arrives, the algorithms only process its last image.
 
## How does it work

<!--
![dataset_code](https://github.com/axin233/hand_roll_estimation/assets/59490151/795213f7-1eee-49f2-8545-b6f58d5d4450)
-->

### Generating the custom deep-learning dataset

Fig. 3 demonstrates the process of generating the deep-learning dataset. A detailed explanation of the process can be found in our paper.

<p align="center">
  <img width="600" height="200" src="https://github.com/axin233/hand_roll_estimation/assets/59490151/795213f7-1eee-49f2-8545-b6f58d5d4450">
</p>

> Fig. 3 The process for generating the custom deep-learning dataset

<!--
![network](https://github.com/axin233/hand_roll_estimation/assets/59490151/0ff6bb9a-a3ff-4265-add1-736cd2ac9750)
-->

### Network structures

Fig. 4 shows the structures of CNN, MO, and MM. A detailed description of the structures can be found in our paper.

<p align="center">
  <img width="863" height="250" src="https://github.com/axin233/hand_roll_estimation/assets/59490151/0ff6bb9a-a3ff-4265-add1-736cd2ac9750">
</p>

> Fig. 4 The structure of (a) CNN and (b) MO and MM. Notice that MO and MM have the same structure, as MM uses identical output branches at each time step.

## Reproducing the results

To reproduce the results in the demo, please download the [data](https://drive.google.com/drive/folders/1SZlVx9E_UTGEork87cag5zlajOH2bFDY?usp=sharing) and the [algorithms' weights](https://drive.google.com/drive/folders/1GKsiPBIkfuXwhrxvSPqSASQpL3ghfv8j?usp=sharing). The code is located at `<algorithm_name>/sliding_window/test_video.py`. If you have any questions, please feel free to email me at jianxig@g.clemson.edu :smile:

## System requirements

The code is written in Python. It requires the following packages.

- Pytorch 1.10.0
- torchvision 0.11.0
- Pandas 1.4.3
- Numpy 1.23.5
- tqdm 4.64.1

<!--
|          | CNN (surface) | MO (surface) | MM (surface) | CNN (depth) | MO (depth) | MM (depth) |
| :------: | :------: | :------: | :------: | :------: | :------: | :------: |
| Location A | 9.79 | 9.08 | 8.42 | 10.71 | 10.11 | 9.12 |
| Location B | 7.04 | 7.4 | 6.99 | 12.07 | 11.54 | 11.21 |
| Location C | 6.35 | 6.45 | 6.26 | 8.21 | 8.52 | 8.52 |
| Location D | 6.73 | 6.42 | 6.95 | 10.79 | 11.4 | 12.05 |
-->
