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

# ===================   Settings     ===================

# MAIN VARIABLES
CHIP_NAME = "MINERVA" 
BLOCK_SIZE = (11,11)
UPSAMPLE_RATIO = 16         # FIXME: DO NOT CHANGE. WINDOW IS HARD CODED RIGHT NOW
FREQ = 3125                 #6250  # in kHz
#FREQ = 6250
SAVE_IMAGES = False
LOW_SALT = True

# Cropping Parameters (In order of Importance)
CROP = True
RAD = 16                                # Radius of cropped square # In original pixels rather than upsampled pixels, need to be factor of 2?
CROPPED_LENGTH = 2*RAD*UPSAMPLE_RATIO
INTERPOLATE_ORDER=0


# Defining Interest Points
single_lobe = (452, 3081)
single_lobe = (2944,3287)
# single_lobe = (6897, 1949)
# single_lobe = (6837, 1324)
# single_lobe = (6459, 2352)

two_lobe = (4413, 1830)
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
        logfile = r"minerva_low_salt/ECT_block11x11_Mix_Cosmarium_Pediastrum_3p125M_VCM_500_VSTBY_300_set_3.h5"
        #logfile = r"minerva_low_salt\ECT_block11x11_Mix_Cosmarium_Pediastrum_6p25M_VCM_500_VSTBY_300_set_2.h5"

    center_img_file = r"minerva_low_salt/impedance_single_phase_3p125_set_3.h5"
    #center_img_file = r"minerva_low_salt/impedance_single_phase_6p25_set_2.h5"

# ======================================================

# ------------------------ Beginning of Script -----------------------------

# Load microscope image
myphoto = image.imread("../data/super_res/minerva_low_salt/Microscope/cropped.bmp")

fig, ax = plt.subplots(1,3)

# Load data and sort keys
mydata = h5py.File(os.path.join(logdir,logfile),'r')
sortedkeys = sorted(mydata.keys(), key=lambda k: int(mydata[k].attrs[row])*100+int(mydata[k].attrs[col]))
sortedkeys[:] = [x for x in sortedkeys if int(mydata[x].attrs[f_name]) == FREQ]

# Obtain Reference Images
k_reference = [x for x in sortedkeys if int(mydata[x].attrs[f_name]) == FREQ and int(mydata[x].attrs[row])==CENTER[0]-2 and int(mydata[x].attrs[col])==CENTER[1]][0]
reference_image,ref_shifted = func.getimage(mydata, k_reference, UPSAMPLE_RATIO,im,BLOCK_SIZE[0],CHIP_NAME, LOW_SALT,INTERPOLATE_ORDER)

ax[0].imshow(myphoto[60:-60,60:-60])
ax[0].set_title("Microscope Image")
ax[1].imshow(reference_image)
ax[1].set_title("Reference Image")

if CROP:
    cropped_ref = func.get_area_around(reference_image, INTEREST_POINT, RAD, UPSAMPLE_RATIO)

ax[2].imshow(cropped_ref)
ax[2].set_title("Cropped Reference Image")

plt.show()