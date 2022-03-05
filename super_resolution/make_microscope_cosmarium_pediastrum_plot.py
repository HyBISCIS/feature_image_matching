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
This was the script that created the microscope, reference, and impedance
image comparison figure. It was used for that purpose, but other than that,
nothing was required of it.
'''

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
RAD = 6                                # Radius of cropped square # In original pixels rather than upsampled pixels, need to be factor of 2?
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
    logfile = r"ECT_block11x11_Mix_Cosmarium_Pediastrum_6p25M_set_1.h5"

    microscope_img = None
    CENTER = (0,0)
    
    if (LOW_SALT):
        logfile = r"minerva_low_salt/ECT_block11x11_Mix_Cosmarium_Pediastrum_3p125M_VCM_500_VSTBY_300_set_3.h5"
        #logfile = r"minerva_low_salt\ECT_block11x11_Mix_Cosmarium_Pediastrum_6p25M_VCM_500_VSTBY_300_set_2.h5"

    center_img_file = r"minerva_low_salt/impedance_single_phase_3p125_set_3.h5"
    #center_img_file = r"minerva_low_salt/impedance_single_phase_6p25_set_2.h5"

# ======================================================

# Get Raw Impedance Image
mydata = h5py.File(os.path.join(logdir,logfile),'r')
sortedkeys = sorted(mydata.keys(), key=lambda k: int(mydata[k].attrs[row])*100+int(mydata[k].attrs[col]))
sortedkeys[:] = [x for x in sortedkeys if int(mydata[x].attrs[f_name]) == FREQ]

CENTER_1 = (CENTER[0]+1, CENTER[1]+1)       # (1,1) is much better offset!
CENTER_2 = (CENTER[0]-2, CENTER[1])

# Get Keys
key_1 = [x for x in sortedkeys if int(mydata[x].attrs[f_name]) == FREQ and int(mydata[x].attrs[row])==CENTER_1[0] and int(mydata[x].attrs[col])==CENTER_1[1]][0]
key_2 = [x for x in sortedkeys if int(mydata[x].attrs[f_name]) == FREQ and int(mydata[x].attrs[row])==CENTER_2[0] and int(mydata[x].attrs[col])==CENTER_2[1]][0]

img_1, mean_1, std_1 = fetch_cropped(mydata, key_1, single_lobe)
img_2, mean_2, std_2 = fetch_cropped(mydata, key_2, two_lobe)


# Load Bitmap Images
ped_img = plt.imread("../data/super_res/minerva_low_salt/Microscope/zoom_3.bmp")
cos_img = plt.imread("../data/super_res/minerva_low_salt/Microscope/zoom_2.bmp")

cos_m_cropped = (func.get_area_around(cos_img, (465,1218), 32, 1) * 2) - 50
ped_m_cropped = (func.get_area_around(ped_img, (570, 32), 32, 1) * 2) - 50

# Load Wikipedia Algae Images
cosmarium = plt.imread("../data/super_res/Cosmarium201512081550.JPG")
pediastrum = plt.imread("../data/super_res/Pediastrum_duplex_wagner.jpg")

# Plot Things
fig, ax = plt.subplots(2,3,figsize=(6,4))
ax[0,0].imshow(cosmarium)
ax[0,1].imshow(cos_m_cropped)
ax[0,2].imshow(img_2, cmap='Greys')


ax[1,0].imshow(pediastrum)
ax[1,1].imshow(ped_m_cropped)
ax[1,2].imshow(img_1,cmap='Greys')

for a in ax.reshape(-1):
    a.set_yticks([])
    a.set_xticks([])


plt.show()
