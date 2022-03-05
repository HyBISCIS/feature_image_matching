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
This was a script used in order to display the three separate cosmarium raw images
alongside their respective super-resolution reconstruction. This was a figure in
the paper and a lot of this script exists in order to make things look pretty.
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

def highpass(filepath, sigma):
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
LOW_SALT = True
INTERPOLATE_ORDER = 2

# Cropping Parameters (In order of Importance)
CROP = True
RAD = 6                                # Radius of cropped square # In original pixels rather than upsampled pixels, need to be factor of 2?
CROPPED_LENGTH = 2*RAD*UPSAMPLE_RATIO

two_lobe_1 = (276, 114)   # Good 1
two_lobe_2 = (85, 49)                  
two_lobe_3 = (296, 102)

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
        logfile = r"minerva_low_salt/ECT_block11x11_Mix_Cosmarium_Pediastrum_3p125M_VCM_500_VSTBY_300_set_3.h5"
        #logfile = r"minerva_low_salt\ECT_block11x11_Mix_Cosmarium_Pediastrum_6p25M_VCM_500_VSTBY_300_set_2.h5"

    center_img_file = r"minerva_low_salt/impedance_single_phase_3p125_set_3.h5"
    #center_img_file = r"minerva_low_salt/impedance_single_phase_6p25_set_2.h5"

# ============= Super Resolution Image Loading =========================


filepath_1 = "../log/revised_algo/shift_linear_deconv_cosmarium_revised_algo_1.npy"
filepath_2 = "../log/revised_algo/shift_linear_deconv_cosmarium_revised_algo_2.npy"
filepath_3 = "../log/revised_algo/shift_linear_deconv_cosmarium_revised_algo_3.npy"

img_h_1, mean_h_1, std_h_1 = highpass(filepath_1, 20)
img_h_2, mean_h_2, std_h_2 = highpass(filepath_2, 20)
img_h_3, mean_h_3, std_h_3 = highpass(filepath_3, 20)



# ============ Load Raw Images =================


# Open h5 File Descriptors and sort keys
mydata = h5py.File(os.path.join(logdir,logfile),'r')
sortedkeys = sorted(mydata.keys(), key=lambda k: int(mydata[k].attrs[row])*100+int(mydata[k].attrs[col]))
sortedkeys[:] = [x for x in sortedkeys if int(mydata[x].attrs[f_name]) == FREQ]


CENTER_1 = (CENTER[0]-2, CENTER[1])
CENTER_2 = (CENTER[0]-2, CENTER[1]-1)
CENTER_3 = (CENTER[0]-3, CENTER[1]-1)


# Get Keys
key_1 = [x for x in sortedkeys if int(mydata[x].attrs[f_name]) == FREQ and int(mydata[x].attrs[row])==CENTER_1[0] and int(mydata[x].attrs[col])==CENTER_1[1]][0]
key_2 = [x for x in sortedkeys if int(mydata[x].attrs[f_name]) == FREQ and int(mydata[x].attrs[row])==CENTER_2[0] and int(mydata[x].attrs[col])==CENTER_2[1]][0]
key_3 = [x for x in sortedkeys if int(mydata[x].attrs[f_name]) == FREQ and int(mydata[x].attrs[row])==CENTER_3[0] and int(mydata[x].attrs[col])==CENTER_3[1]][0]

# Load Images
img_1, mean_1, std_1 = fetch_cropped(mydata, key_1, two_lobe_1)
img_2, mean_2, std_2 = fetch_cropped(mydata, key_2, two_lobe_2)
img_3, mean_3, std_3 = fetch_cropped(mydata, key_3, two_lobe_3)

fig, ax = plt.subplots(2,3, figsize=(10,7))
ax[0,0].imshow(img_1, cmap='Greys')
ax[0,1].imshow(img_2, cmap='Greys')
ax[0,2].imshow(img_3, cmap='Greys')

ax[0,0].set_xlabel("Column [px]")
ax[0,0].set_ylabel("Row [px]")
ax[0,0].set_xticks([0,2,4,6,8,10])
ax[0,1].set_xlabel("Column [px]")
ax[0,1].set_xticks([0,2,4,6,8,10])
ax[0,2].set_xlabel("Column [px]")
ax[0,2].set_xticks([0,2,4,6,8,10])

top = 168
bot = 344

ax[1,0].imshow(img_h_1[top+5:bot, top+5:bot+5], cmap='Greys', vmin=mean_h_1-(8*std_h_1), vmax=mean_h_1+(8*std_h_1))
ax[1,1].imshow(img_h_2[top:bot, top:bot], cmap='Greys', vmin=mean_h_2-(8*std_h_2), vmax=mean_h_2+(8*std_h_2))
ax[1,2].imshow(img_h_3[top:bot, top:bot], cmap='Greys', vmin=mean_h_3-(8*std_h_3), vmax=mean_h_3+(8*std_h_3))

ax[1,0].set_xlabel("Upsampled Column [px]")
ax[1,0].set_ylabel("Upsampled Row [px]")
ax[1,1].set_xlabel("Upsampled Column [px]")
ax[1,2].set_xlabel("Upsampled Column [px]")

plt.show()