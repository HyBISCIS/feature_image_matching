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
This was an initial file that was used to generate some general figure images before
we got any real raw data or information. This was not used in the paper submission to
my knowledge and is not that useful to us.
'''

# FOR PLOTTING PEDIASTRUM OR COSMARIUM AND THE LINE PLOT GOING THROUGH IT

def interpolate(img, row, col):
    # Row 502, col 22 for low salt, corrupted, so just do easy interpolation
    im_size = img.shape

    # Fix Row
    for i in range(im_size[1]):
        img[row, i] = (img[row+1,i] + img[row-1,i]) / 2

    # Fix Col
    for j in range(im_size[0]):
        img[j, col] = (img[j,col+1] + img[j, col-1]) / 2

    return img
        


# MAIN VARIABLES
CHIP_NAME = "MINERVA" 
BLOCK_SIZE = (11,11)
UPSAMPLE_RATIO = 1         # FIXME: DO NOT CHANGE. WINDOW IS HARD CODED RIGHT NOW
FREQ = 3125                 #6250  # in kHz
#FREQ = 6250
SAVE_IMAGES = True
LOW_SALT = True
INTERPOLATE_ORDER = 2

# Cropping Parameters (In order of Importance)
CROP = True
RAD = 8                                # Radius of cropped square # In original pixels rather than upsampled pixels, need to be factor of 2?
CROPPED_LENGTH = 2*RAD*UPSAMPLE_RATIO

# single_lobe = (2944,3287)       # Isolated Single Lobe 
# two_lobe = (4413, 1830)         # Isolated Double Lobe
single_lobe = (185, 205)
two_lobe = (275, 114)

NUM_LOBES = 1

if NUM_LOBES == 2:
    INTEREST_POINT = two_lobe
else:
    INTEREST_POINT = single_lobe


# Row and Column Offset Names
if (CHIP_NAME == "LILLIPUT"):
    im_size = (100,100)
    row = "R"
    col = "C"
    f_name = "sw_freq"
    FREQ = FREQ
    im = 'image'

    logdir = r"../data/bead_20um"
    logfile = r"phase2_sweep_11x11_block.h5"
    microscope_img = 'Microscope_bead_20u_0639pm.bmp'
    CENTER = (5,5)
    # Note: Freq in kHz
else:
    im_size = (512,256)
    row = "row_offset"
    col = "col_offset"
    f_name = "f_sw"
    FREQ = FREQ * 1000
    im = 'image_2d_ph2'
    # Note: Freq in Hz

    logdir = r"../data/super_res"
    logfile = r"ECT_block11x11_Mix_Cosmarium_Pediastrum_6p25M_set_1.h5"

    microscope_img = None
    CENTER = (0,0)
    
    if (LOW_SALT):
        logfile = r"minerva_low_salt/ECT_block11x11_Mix_Cosmarium_Pediastrum_3p125M_VCM_500_VSTBY_300_set_1.h5"

    center_img_file = r"minerva_low_salt/impedance_single_phase_3p125_set_3.h5"

# ======================================================


# Get img, convert to grayscale
filepath = "../log/shift_linear_deconv_cosmarium.png"
img = cv2.imread(filepath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
low_pass = gaussian(img, sigma=12)
high_pass = np.subtract(img,(200*low_pass))
print(func.get_spatial_snr(high_pass))
mean, std = (np.mean(high_pass), np.std(high_pass))

# Get Raw Image
mydata = h5py.File(os.path.join(logdir,logfile),'r')
sortedkeys = sorted(mydata.keys(), key=lambda k: int(mydata[k].attrs[row])*100+int(mydata[k].attrs[col]))
sortedkeys[:] = [x for x in sortedkeys if int(mydata[x].attrs[f_name]) == FREQ]

if NUM_LOBES == 2:
    CENTER = (CENTER[0]-3, CENTER[1]-1)
else:
    CENTER = (CENTER[0], CENTER[1]+1)

    # (0,0) not worth looking
    # (0,1) not bad
    # (1,0) probably best one, but marginally better than (1,1)
    # (-1,0) not good
    # (0,-1) drab. Contrast not as good
    # (-1,-1) not good
    # (-1, 1) not good
    # (1, -1) not good
    # (1,1) okay...


# Adjust what reference we use
k_reference = [x for x in sortedkeys if int(mydata[x].attrs[f_name]) == FREQ and int(mydata[x].attrs[row])==CENTER[0] and int(mydata[x].attrs[col])==CENTER[1]][0]

myimage = mydata[k_reference][im][:]
myimage = interpolate(myimage, 501, 245)
# plt.figure(3)
# plt.imshow(myimage, cmap='Greys')


myimage = func.minerva_channel_shift(myimage)
myimage = func.channel_norm(myimage)
# myimage = func.cropimage(myimage, BLOCK_SIZE)
whole_mean = np.mean(myimage)
whole_std = np.std(myimage)


np.save("../log/raw_impedance_array_image_full_image", myimage)
plt.figure(1)
plt.imshow(myimage, cmap='Greys', vmin=(whole_mean-(5*whole_std)), vmax=(whole_mean+(5*whole_std)))
plt.xlabel("Column [px]")
plt.ylabel("Row [px]")



myimage = func.get_area_around(myimage, INTEREST_POINT, RAD, UPSAMPLE_RATIO)



# # Get subplots for this
# high_pass_norm = func.minmaxnorm(high_pass)
# myimage_norm = func.minmaxnorm(myimage)


# diag = np.diagonal(high_pass)
# other_diag = np.diagonal(myimage)
# expand_diag = np.zeros(diag.shape)
# index_count = 0

# for i in range(diag.shape[0]):
#     if (i % 16) == 0 and i != 0:
#         index_count += 1

#     expand_diag[i] = other_diag[index_count]

# diag = func.minmaxnorm(diag)
# expand_diag = func.minmaxnorm(expand_diag)


# dist_per_pixel = np.sqrt(200)
# num_pixels = 300 / 16
# tot_dist = dist_per_pixel * num_pixels
# position_arr = np.linspace(0, tot_dist, 300)

# plt.figure(2)
# plt.plot(position_arr,diag[100:400], color="r", label="Composite")
# plt.plot(position_arr,expand_diag[100:400], color='b', label='Single')
# plt.xlabel(r"Position [$\mu$m]")
# plt.ylabel("Normalized Intensity")
# plt.legend()
# plt.ylim([-0.05, 1.05])

# plt.figure(2)
# plt.imshow(myimage, cmap='Greys')
# plt.xticks([])
# plt.yticks([])



# fig, ax = plt.subplots(2,1)
# ax[1].imshow(high_pass, cmap='Greys', vmin=mean-(12*std), vmax=mean+(20*std))
# ax[1].set_xlabel("Upsampled Column [px]")
# ax[1].set_ylabel("Upsampled Row [px]")
# ax[0].imshow(myimage, cmap='Greys')
# ax[0].set_ylabel("Row [px]")
# ax[0].set_xlabel("Column [px]")

# ax[0].yaxis.set_label_position("right")
# ax[0].yaxis.tick_right()
# ax[1].yaxis.set_label_position("right")
# ax[1].yaxis.tick_right()

# ax[2].plot(position_arr,diag[100:400], color="r", label="Composite")
# ax[2].plot(position_arr,expand_diag[100:400], color='b', label='Single')
# ax[2].set_xlabel(r"Position [$\mu$m]")
# ax[2].set_ylabel("Normalized Intensity")
# ax[2].legend()
# ax[2].set_ylim([-0.05, 1.05])

plt.show()
