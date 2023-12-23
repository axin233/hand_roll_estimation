# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 11:49:42 2022

@author: jianxig
"""

import pickle
#import matplotlib.pyplot as plt


if __name__=='__main__':
    pred_dir='D:/test_videos/videos_with_bbox(SAVS_2022)/7175-17-7/'
    
    # Load pickle file, so the figure can be viewed interactively
    fig = pickle.load(open(pred_dir+'centroid_distance.pickle', 'rb'))
    fig.show()