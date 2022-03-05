from pickle import FALSE, TRUE
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from matplotlib import image
from skimage.transform import resize,rescale, rotate
from skimage.filters import gaussian, unsharp_mask
from skimage.morphology import dilation,erosion,selem
import scipy.signal as sig
from skimage.transform import warp, PiecewiseAffineTransform
from skimage.registration import optical_flow_tvl1
import time

import deconv_func as func

'''
This was an experimental script file used in order to determine what were the
best hyperparameters for high pass filtering something after the image super-resolution
pipeline.
'''

# Get img, convert to grayscale
filepath = "../log/revised_algo/shift_linear_deconv_cosmarium_revised_algo_3.npy"
img = np.load(filepath)


# low_pass = gaussian(img, sigma=3)
# high_pass = np.subtract(img,(200*low_pass))
# print(func.get_spatial_snr(high_pass))
# mean, std = (np.mean(high_pass), np.std(high_pass))
# plt.imshow(high_pass[165:325,165:325],cmap='Greys')
# plt.colorbar()
# plt.show()

fig, ax = plt.subplots(2,5)
for i in range(5):
    low_pass = gaussian(img, sigma=7*i)
    high_pass = np.subtract(img,(0.8*low_pass))
    mean, std = (np.mean(high_pass), np.std(high_pass))
    ax[0,i].imshow(high_pass,cmap='Greys', vmin=mean-(8*std), vmax=mean+(8*std))
    ax[1,i].imshow(low_pass)
    print(func.get_spatial_snr(high_pass))

plt.show()

