from pickle import FALSE, TRUE
from cv2 import norm
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
This is the function that plots the four different composite images composed of
different distance offsets and plots them nicely. This was NOT used in the 
initial submission of the paper due to the lack of clarity in showing how the 
different distances affect things. It was not as clear as we wanted it to be.
'''

def highpass_and_crop(img, sigma, crop_x, crop_y):
    # High Pass
    low_pass = gaussian(img, sigma=sigma)
    high_pass = np.subtract(img,(0.8*low_pass))

    # CROP
    cropped = high_pass[crop_y[0]:crop_y[1],crop_x[0]:crop_x[1]]

    mean, std = (np.mean(cropped), np.std(cropped))

    return cropped, mean, std

# Load in all images

filepath_microscope = "../data/super_res/minerva_low_salt/Microscope/zoom_2.bmp"
filepath_all = "../log/shift_linear_deconv_cosmarium_all_take2.npy"
filepath_0_2 = "../log/shift_linear_deconv_cosmarium_0_2_take2.npy"
filepath_2_4 = "../log/shift_linear_deconv_cosmarium_2_4_take2.npy"
filepath_4_6 = "../log/shift_linear_deconv_cosmarium_4_6_take2.npy"

m_img = plt.imread(filepath_microscope)

# Open into numpy arrays
sigma = 20
# crop_x = (600, 1100)
# crop_y = (300, 1100)

crop_x = (0, 2048)
crop_y = (0, 2048)


img_all, mean_all, std_all = highpass_and_crop(np.load(filepath_all), sigma, crop_x, crop_y)
img_0_2, mean_0_2, std_0_2 = highpass_and_crop(np.load(filepath_0_2), sigma, crop_x, crop_y)
img_2_4, mean_2_4, std_2_4 = highpass_and_crop(np.load(filepath_2_4), sigma, crop_x, crop_y)
img_4_6, mean_4_6, std_4_6 = highpass_and_crop(np.load(filepath_4_6), sigma, crop_x, crop_y)


std_num = 8

# Initial Plots
fig, ax = plt.subplots(1,5)
ax[0].imshow((m_img[100:500,1000:1250]*2)-50)
ax[0].set_title("Optical")

ax[1].imshow(img_all, cmap='Greys', vmin=mean_all-(std_all*std_num), vmax=mean_all+(std_all*(std_num)))
ax[1].set_title("0 < R")

ax[2].imshow(img_0_2, cmap='Greys', vmin=mean_0_2-(std_0_2*std_num), vmax=mean_0_2+(std_0_2*std_num))
ax[2].set_title("0 < R < 2")

ax[3].imshow(img_2_4, cmap='Greys', vmin=mean_2_4-(std_2_4*std_num), vmax=mean_2_4+(std_2_4*std_num))
ax[3].set_title("2 < R < 4")

ax[4].imshow(img_4_6, cmap='Greys', vmin=mean_4_6-(std_4_6*(std_num-2)), vmax=mean_4_6+(std_4_6*(std_num-2)))
ax[4].set_title("4 < R < 6")

ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[2].set_xticks([])
ax[2].set_yticks([])
ax[3].set_xticks([])
ax[3].set_yticks([])
ax[4].set_xticks([])
ax[4].set_yticks([])


# slice = np.zeros(img_4_6.shape[1])
# normalized = func.minmaxnorm(img_4_6)

# # print(normalized[150,451])
# # for i in range(img_4_6.shape[0]):
# #     normalized[i, 450:img_4_6.shape[0]] = normalized[i, 450:img_4_6.shape[0]] +0.05
# # print(normalized[150,451])


# plt.figure(3)
# plt.title("Unfixed?")
# plt.imshow(normalized,cmap='Greys')

# slice = np.average(normalized[200:500,:], axis=0)
# factor = 1.5
# middle = 0.35

# reflect = slice * -1 * factor
# reflect_offset = middle - np.mean(reflect)
# reflect += reflect_offset

# multiplied = reflect * slice

# for i in range(img_4_6.shape[0]):
#     normalized[i,:] = normalized[i,:] * reflect 

# mean_n, std_n = (np.mean(normalized), np.std(normalized))
# std_num = 7

# plt.figure(4)
# plt.title("4 < R < 6 Fixed?")
# plt.imshow(normalized,cmap='Greys', vmin=mean_n-(std_n*(std_num)), vmax=mean_n+(std_n*(std_num))) 



# plt.figure(2)
# plt.plot(slice)
# plt.plot(reflect)
# plt.plot(multiplied)



plt.show()

