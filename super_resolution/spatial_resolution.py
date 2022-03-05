import h5py
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.cm as cm
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib import image
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import os
import cv2
from datetime import datetime
import time
from skimage.transform import resize,rescale, rotate
from skimage.filters import gaussian
from skimage import color, data, restoration
from skimage.morphology import dilation,erosion,selem
import scipy.signal as sig
import imageio
from skimage.transform import warp, PiecewiseAffineTransform
from skimage.registration import optical_flow_tvl1
import deconv_func as func

'''
TODO:

This script was a script that was never finished. It primarily existed
as a way to try and compare our spatial resolution compared to the other
algorithms that existed previously. Not much work got done here as we 
ran out of time to try and implement it.
'''

shift_sum = cv2.cvtColor(cv2.imread("../log/beads/shift_sum_beads.png"), cv2.COLOR_BGR2GRAY)
deconv = cv2.cvtColor(cv2.imread("../log/beads/linear_deconvolution.png"), cv2.COLOR_BGR2GRAY)
mirroring = cv2.cvtColor(cv2.imread("../log/beads/mirroring_beads.png"), cv2.COLOR_BGR2GRAY)

shift_sum = func.normalize_img(shift_sum)
picture_mean = np.mean(shift_sum)
std = np.std(shift_sum)

mask = shift_sum < (picture_mean - (2 * std))

plt.imshow(mask * shift_sum)
plt.show()



fig, ax = plt.subplots(1,3)
ax[0].imshow(shift_sum)
ax[0].set_title("Shift Sum")

ax[1].imshow(deconv)
ax[1].set_title("Deconvolution")

ax[2].imshow(mirroring)
ax[2].set_title("Mirroring Beads")


plt.show()