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

# Get img, convert to grayscale
filepath = "../log/shift_linear_deconv_pediastrum_multicell_2.png"
img = cv2.imread(filepath)
print(img.shape)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
# kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
# im = cv2.filter2D(img, -1, kernel)

# unsharped = unsharp_mask(img, radius=5, amount=3)


# ax[0].set_title("Original")
# ax[0].imshow(img, cmap='Greys')
# ax[1].set_title("Sharpness Kernel")
# ax[1].imshow(im,cmap='Greys')
# ax[2].set_title("Unsharp Mask")
# ax[2].imshow(unsharped,cmap='Greys')

# img = cv2.equalizeHist(img)

# NOTE: These are all settings for PEDIASTRUM
low_pass = gaussian(img, sigma=12)
high_pass = np.subtract(img,(200*low_pass))
print(func.get_spatial_snr(high_pass))
mean, std = (np.mean(high_pass), np.std(high_pass))
plt.imshow(high_pass[165:325,165:325],cmap='Greys', vmin=mean-(8*std), vmax=mean+(8*std))
plt.colorbar()
plt.show()



fig, ax = plt.subplots(2,5)
for i in range(5):
    low_pass = gaussian(img, sigma=(4*i))
    high_pass = np.subtract(img,(200*low_pass))
    mean, std = (np.mean(high_pass), np.std(high_pass))
    ax[0,i].imshow(high_pass,cmap='Greys', vmin=mean-(8*std), vmax=mean+(8*std))
    ax[1,i].imshow(low_pass)
    print(func.get_spatial_snr(high_pass))

plt.show()

