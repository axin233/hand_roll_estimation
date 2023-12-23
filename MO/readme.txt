This folder contains the code for training and testing MO

For training MO:
- train_many_to_one.py
- DIY_data_loader_many_to_one.py
- network_many_to_one.py

For testing MO:
- test_video.py
- DIY_data_loader_many_to_one.py
- network_many_to_one.py

For testing MO (using the sliding window approach)
- The code is in the folder called 'sliding_window'
(Note: The sliding window approach significantly increases the inference speed, but it cannot be used during training. It is because
the data loader shuffles the image sequence, causing the adjacent sequences are not in order.)