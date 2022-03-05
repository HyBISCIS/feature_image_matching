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
Script used to create nice looking line profiles and figures for the
paper submission. 

We note the odd disparity of the pediastrum, in which the polarity flips
in the middle of the cell during the super-resolution reconstruction. We
are not sure why, but we are looking into it.
'''

# FOR PLOTTING PEDIASTRUM OR COSMARIUM AND THE LINE PLOT GOING THROUGH IT
def fetch_cropped(data, ref_key,interest_point):
    myimage = data[ref_key][im][:]
    myimage = func.low_salt_interpolate(myimage)
    myimage = func.minerva_channel_shift(myimage)
    myimage = func.channel_norm(myimage)
    myimage = func.cropimage(myimage, BLOCK_SIZE)
    whole_mean = np.mean(myimage)
    whole_std = np.std(myimage)
    myimage = func.get_area_around(myimage, interest_point, RAD, UPSAMPLE_RATIO)
    return myimage, whole_mean, whole_std

def fetch_and_highpass(filepath, sigma):
    # High Pass
    img = np.load(filepath)
    low_pass = gaussian(img, sigma=sigma)
    high_pass = np.subtract(img,(0.8*low_pass))
    mean, std = (np.mean(high_pass), np.std(high_pass))

    return high_pass, mean, std


# MAIN VARIABLES
CHIP_NAME = "MINERVA" 
BLOCK_SIZE = (11,11)
UPSAMPLE_RATIO = 1      
FREQ = 3125         

# Cropping Parameters (In order of Importance)
CROP = True
RAD = 8                                # Radius of cropped square # In original pixels rather than upsampled pixels, need to be factor of 2?
CROPPED_LENGTH = 2*RAD*UPSAMPLE_RATIO

single_lobe = (183, 205)
two_lobe = (275, 114)

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
    logfile = r"minerva_low_salt/ECT_block11x11_Mix_Cosmarium_Pediastrum_3p125M_VCM_500_VSTBY_300_set_3.h5"

    CENTER = (0,0)
    
    


# ======================================================


# Get Images Ready
filepath_cos = "../log/revised_algo/shift_linear_deconv_cosmarium_revised_algo_1.npy"
filepath_ped = "../log/revised_algo/shift_linear_deconv_pediastrum_revised_algo_1.npy"
img_1, mean_1, std_1 = fetch_and_highpass(filepath_cos,20)
top = 128 
bot = 384 
img_1 = img_1[top-5:bot-5,top-5:bot-5]
img_2, mean_2, std_2 = fetch_and_highpass(filepath_ped,20)
img_2 = img_2[top:bot,top:bot]

# ======================================================

# Get Raw Image
mydata = h5py.File(os.path.join(logdir,logfile),'r')
sortedkeys = sorted(mydata.keys(), key=lambda k: int(mydata[k].attrs[row])*100+int(mydata[k].attrs[col]))
sortedkeys[:] = [x for x in sortedkeys if int(mydata[x].attrs[f_name]) == FREQ]

CENTER_cos = (CENTER[0]-2, CENTER[1])
CENTER_ped = (CENTER[0]+1, CENTER[1]+1)

# Adjust what reference we use
k_cos = [x for x in sortedkeys if int(mydata[x].attrs[f_name]) == FREQ and int(mydata[x].attrs[row])==CENTER_cos[0] and int(mydata[x].attrs[col])==CENTER_cos[1]][0]
k_ped = [x for x in sortedkeys if int(mydata[x].attrs[f_name]) == FREQ and int(mydata[x].attrs[row])==CENTER_ped[0] and int(mydata[x].attrs[col])==CENTER_ped[1]][0]


cos, mean_c, std_c = fetch_cropped(mydata, k_cos, two_lobe)
ped, mean_p, std_p = fetch_cropped(mydata, k_ped, single_lobe)


# Get subplots for this
img_1 = func.minmaxnorm(img_1)
cos = func.minmaxnorm(cos)
img_2 = func.minmaxnorm(img_2)
ped = func.minmaxnorm(ped)


diag_1 = np.diagonal(img_1)
other_diag_1 = np.diagonal(cos)
expand_diag_1 = np.zeros(diag_1.shape)

diag_2 = np.diagonal(img_2)
print(diag_2.shape)
other_diag_2 = np.diagonal(ped)
expand_diag_2 = np.zeros(diag_2.shape)

index_count = 0

for i in range(diag_1.shape[0]):
    if (i % 16) == 0 and i != 0:
        index_count += 1

    expand_diag_1[i] = other_diag_1[index_count]

index_count = 0

for j in range(diag_2.shape[0]):
    if (j % 16) == 0 and j != 0:
        index_count += 1

    expand_diag_2[j] = other_diag_2[index_count]


diag_1 = func.minmaxnorm(diag_1)
expand_diag_1 = func.minmaxnorm(expand_diag_1)
diag_2 = func.minmaxnorm(diag_2)
expand_diag_2 = func.minmaxnorm(expand_diag_2)

dist_per_pixel = np.sqrt(200)
num_pixels = 256 / 16
tot_dist = dist_per_pixel * num_pixels
position_arr = np.linspace(0, tot_dist, diag_1.shape[0])

fig, ax = plt.subplots(2, 1)
ax[0].plot(position_arr,diag_1, color="r", label="Composite")
ax[0].plot(position_arr,expand_diag_1, color='b', label='Single')
ax[0].set_xlabel(r"Position [$\mu$m]")
ax[0].set_ylabel("Normalized Intensity")
ax[0].legend()
ax[0].set_ylim([-0.05, 1.05])

ax[1].plot(position_arr,diag_2, color="r", label="Composite")
ax[1].plot(position_arr,expand_diag_2, color='b', label='Single')
ax[1].set_xlabel(r"Position [$\mu$m]")
ax[1].set_ylabel("Normalized Intensity")
ax[1].legend()
ax[1].set_ylim([-0.05, 1.05])


plt.show()


