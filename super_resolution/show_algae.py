from pickle import FALSE
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from matplotlib import image
from skimage.transform import resize,rescale, rotate
from skimage.filters import gaussian
from skimage.morphology import dilation,erosion,selem
import scipy.signal as sig
from skimage.transform import warp, PiecewiseAffineTransform
from skimage.registration import optical_flow_tvl1
import deconv_func as func

'''
This script was used to display the 11x11 raw image impedance images
with separate kernel offsets along with all the things needed to make
the plot like nice and pretty.
'''

def get_area_around(img, interest_point, radius):
    x = (interest_point[0] - radius, interest_point[0] + radius)
    y = (interest_point[1] - radius, interest_point[1] + radius)

    return img[x[0]:x[1], y[0]:y[1]]

# Quick Script to create block diagrams to show off two different algaes in images

# ===================       SETTINGS        ==============================

# # First Cosmarium Data Set
# two_lobe = (183,139)
# two_lobe_2 = (130,160)
# two_lobe_3 = (317, 80)
# one_lobe = (327,134)

# Low Salt Concentration Data
# two_lobe = (139, 203)
# one_lobe = (132, 203)

one_lobe = (185, 205)
two_lobe = (274, 114)

feature = one_lobe
label = "Pediastrum"

# MAIN VARIABLES
CHIP_NAME = "MINERVA" 
BLOCK_SIZE = (11,11)
UPSAMPLE_RATIO = 16
FREQ = 6250  # in kHz
#FREQ = 3125 # in kHz
WHOLE_PICTURE = False

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
    # logfile = r"ECT_block11x11_Mix_Cosmarium_Pediastrum_6p25M_set_2.h5"
    # logfile= r"minerva_low_salt/ECT_block11x11_Mix_Cosmarium_Pediastrum_3p125M_VCM_400_VSTBY_100_set_2.h5"
    # logfile= r"minerva_low_salt\ECT_block11x11_Mix_Cosmarium_Pediastrum_3p125M_VCM_500_VSTBY_300_set_3.h5"
    logfile= r"minerva_low_salt\ECT_block11x11_Mix_Cosmarium_Pediastrum_6p25M_VCM_500_VSTBY_300_set_2.h5"

    microscope_img = None
    CENTER = (0,0)

# ================================================================




mydata = h5py.File(os.path.join(logdir,logfile),'r')
sortedkeys = sorted(mydata.keys(), key=lambda k: int(mydata[k].attrs[row])*100+int(mydata[k].attrs[col]))

# Setting so that we can take a look at the whole picture to easily find coordinates for things
if WHOLE_PICTURE:
    for i in sortedkeys:
        myrow = int(mydata[i].attrs[row]) + 5 # In order to remedy (0,0) being the middle
        mycol = int(mydata[i].attrs[col]) + 5

        if (myrow == 4 and mycol == 4):
            myimage = mydata[i][im][:]
            myimage = func.low_salt_interpolate(myimage)
            myimage = func.minerva_channel_shift(myimage)
            myimage = func.channel_norm(myimage)

            myimage = func.cropimage(myimage, (11,11))

            plt.imshow(myimage, cmap='Greys')
            plt.show()

            exit(0)

# =======================================================================================

# Get rid of everything except for the frequencies that we are dealing with
sortedkeys[:] = [x for x in sortedkeys if int(mydata[x].attrs[f_name]) == FREQ]

# Plot Information
fig, ax = plt.subplots(11, 11, figsize=(8,8))
fig.text(0.5, 0.04, 'Column Offset', ha='center', va='center', fontsize=22)
fig.text(0.06, 0.5, 'Row Offset', ha='center', va='center', rotation='vertical', fontsize=22)
#fig.suptitle(label)

for i in sortedkeys:

    myrow = int(mydata[i].attrs[row]) + 5 # In order to remedy (0,0) being the middle
    mycol = int(mydata[i].attrs[col]) + 5

    if (myrow == 5 and mycol == 5):
        ax[myrow,mycol].set_xticks([])
        ax[myrow,mycol].set_yticks([])
        ax[myrow,mycol].axis('off') 
        continue

    myimage = mydata[i][im][:]
    myimage = func.low_salt_interpolate(myimage)
    myimage = func.minerva_channel_shift(myimage)
    myimage = func.channel_norm(myimage)
    myimage = func.cropimage(myimage, (11,11))

    # Get Image of Cosmarium that we want
    mycropped = get_area_around(myimage, feature, 5)
    median = np.median(mycropped)
    std = np.std(mycropped)

    # Determine vmin and vmax
    std_devs = 3
    min = median - (std_devs * std)
    max = median + (std_devs * std)

    ax[myrow, mycol].imshow(mycropped, cmap='Greys', vmin=min, vmax=max)

    # Pretty Plot Settings for figure
    ax[myrow,mycol].set_xticks([])
    ax[myrow,mycol].set_yticks([])

    if (myrow == 10):
        string =str(mycol - 5)
        if mycol-5 > 0:
            string = "+" + string

        ax[myrow, mycol].set_xlabel(string, fontsize=18)
        ax[myrow, mycol].xaxis.set_label_coords(0.5,-0.3)

    if (mycol == 0):
        string = str(myrow-5)
        if myrow-5 > 0:
            string = "+" + string
        
        if myrow-5 == 0:
            string = " " + string

        ax[myrow, mycol].set_ylabel(string, rotation=0, loc='center', fontsize=18)
        ax[myrow, mycol].yaxis.set_label_coords(-0.5,0.25)


plt.show()

# ====================================================
