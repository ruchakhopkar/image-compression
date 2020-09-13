# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 12:52:59 2020

@author: rucha
"""

#24 bit coloured representation, each pixel is represented as a 3 8bit numbers (0 to 255(R, G, B)). Therefore, total colors is 256^3. 
#We want to reduce these number of colors to 16 or 64.
from __future__ import print_function
%matplotlib inline
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
plt.style.use('ggplot')
from skimage import io
from sklearn.cluster import KMeans
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual, IntSlider
plt.rcParams['figure.figsize']=(20,12)

img=io.imread('images/1-Saint-Basils-Cathedral.jpg')
ax=plt.axes(xticks=[], yticks=[])
ax.imshow(img)

print(img.shape)
img_data=(img/255.).reshape(-1,3)

from plot_utils import plot_utils
x=plot_utils(img_data, title='Input color space: Over 16 million possible colors')
x.colorSpace()

from sklearn.cluster import MiniBatchKMeans
kmeans=MiniBatchKMeans(16).fit(img_data)
k_colors=kmeans.cluster_centers_[kmeans.predict(img_data)]
y=plot_utils(img_data, title='Reduced map with 16 colors', colors=k_colors)
y.colorSpace()

img_dir='images/'

@interact
def color_compression(image=os.listdir(img_dir), k=IntSlider(min=1, max=10000, step=1, value=10, continuous_update=False, layout=dict(width='100%'))):
    input_img=io.imread(img_dir+image)
    img_data=(input_img/255.).reshape(-1,3)
    kmeans=MiniBatchKMeans(k).fit(img_data)
    k_colors=kmeans.cluster_centers_[kmeans.predict(img_data)]
    k_img=np.reshape(k_colors, (input_img.shape))
    
    fig, (ax1,ax2)= plt.subplots(1,2)
    fig.suptitle('K-Means Image Compression', fontsize=20)
    
    ax1.set_title('Compressed')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.imshow(k_img)
    
    ax2.set_title('Original')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.imshow(img_data)
    
    plt.subplots_adjust(top=0.85)