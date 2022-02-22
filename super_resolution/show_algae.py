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

# FIXME: On Minerva, row col is from -5 to 5, rather than from 0 to 11 (on lilliput)

def get_area_around(img, interest_point, radius):
    x = (interest_point[0] - radius, interest_point[0] + radius)
    y = (interest_point[1] - radius, interest_point[1] + radius)

    return img[x[0]:x[1], y[0]:y[1]]

# Quick Script to create block diagrams to show off two different algaes in images

# This is for file one
two_lobe = (183,139)
two_lobe_2 = (130,160)
two_lobe_3 = (317, 80)
one_lobe = (327,134)

# Low Salt Concentration Stuff
two_lobe = (139, 171+22)
one_lobe = (132, 170+22)



feature = one_lobe

# MAIN VARIABLES
CHIP_NAME = "MINERVA" 
BLOCK_SIZE = (11,11)
UPSAMPLE_RATIO = 16
FREQ = 6250  # in kHz
FREQ = 3125 # in kHz
WHOLE_PICTURE = True

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
    #logfile = r"ECT_block11x11_Mix_Cosmarium_Pediastrum_6p25M_set_2.h5"
    logfile= r"minerva_low_salt/ECT_block11x11_Mix_Cosmarium_Pediastrum_3p125M_VCM_400_VSTBY_100_set_2.h5"
    logfile= r"minerva_low_salt\ECT_block11x11_Mix_Cosmarium_Pediastrum_3p125M_VCM_500_VSTBY_300_set_3.h5"

    microscope_img = None
    CENTER = (0,0)

mydata = h5py.File(os.path.join(logdir,logfile),'r')
sortedkeys = sorted(mydata.keys(), key=lambda k: int(mydata[k].attrs[row])*100+int(mydata[k].attrs[col]))
k_reference = [x for x in sortedkeys if int(mydata[x].attrs[f_name]) == FREQ and int(mydata[x].attrs[row])==CENTER[0] and int(mydata[x].attrs[col])==CENTER[1]][0]
reference_image,refmedian,refstd = func.getimage(mydata, k_reference, UPSAMPLE_RATIO,im,BLOCK_SIZE[0],CHIP_NAME)

if WHOLE_PICTURE:
    for i in sortedkeys:
        myrow = int(mydata[i].attrs[row]) + 5 # In order to remedy (0,0) being the middle
        mycol = int(mydata[i].attrs[col]) + 5
        print(myrow,mycol)

        if (myrow == 4 and mycol == 4):
            myimage = mydata[i][im][:]
            myimage = func.low_salt_interpolate(myimage)
        
            normrows = range(200,300)
            coeffs = np.ones(8)

            # normalize by channel
            for ch in range(8):
                coeffs[ch] = 1 / np.mean(myimage[normrows, ch*32:(ch+1)*32])
            myimage = func.apply_cal(myimage,coeffs) 

            #myimage = func.cropimage(myimage, (11,11))

            plt.imshow(myimage, cmap='Greys')
            plt.show()

            exit(0)

# Get rid of everything except for the frequencies that we are dealing with
sortedkeys[:] = [x for x in sortedkeys if int(mydata[x].attrs[f_name]) == FREQ]
count = 0
fig, ax = plt.subplots(11, 11, figsize=(8,8))
fig.text(0.5, 0.04, 'Column Offset', ha='center', va='center', fontsize=20)
fig.text(0.06, 0.5, 'Row Offset', ha='center', va='center', rotation='vertical', fontsize=20)
fig.suptitle("Pediastrum Raw Data")

min = np.inf
max = -np.inf

# # Find Min and Max
# for i in sortedkeys:
#     myimage = mydata[i][im][:]
#     normrows = range(200,300)
#     coeffs = np.ones(8)

#     # normalize by channel
#     for ch in range(8):
#         coeffs[ch] = 1 / np.mean(myimage[normrows, ch*32:(ch+1)*32])
#     myimage = func.apply_cal(myimage,coeffs) 

#     # Crop and Normalize
#     myimage = func.cropimage(myimage, (11,11))

#     # Get Image of Cosmarium that we want
#     mycropped = get_area_around(myimage, feature, 5)

#     min_i = np.min(mycropped)
#     max_i = np.max(mycropped)

#     if (min_i < min):
#         min = min_i

#     if (max_i > max):
#         max = max_i


# Put into the subplots
for i in sortedkeys:

    myrow = int(mydata[i].attrs[row]) + 5 # In order to remedy (0,0) being the middle
    mycol = int(mydata[i].attrs[col]) + 5

    if (myrow == 5 and mycol == 5):
        ax[myrow,mycol].set_xticks([])
        ax[myrow,mycol].set_yticks([])
        continue

    myimage = mydata[i][im][:]
    normrows = range(200,300)
    coeffs = np.ones(8)

    # normalize by channel
    for ch in range(8):
        coeffs[ch] = 1 / np.mean(myimage[normrows, ch*32:(ch+1)*32])
    myimage = func.apply_cal(myimage,coeffs) 

    # Crop and Normalize
    myimage = func.cropimage(myimage, (11,11))
    

    # Get Image of Cosmarium that we want
    mycropped = get_area_around(myimage, feature, 5)
    median = np.median(mycropped)
    std = np.std(mycropped)

    std_devs = 3
    min = median - (std_devs * std)
    max = median + (std_devs * std)

    # min = np.min(mycropped)
    # max = np.max(mycropped) 
    #mycropped = np.nan_to_num((mycropped-min)/(max-min))

    ax[myrow, mycol].imshow(mycropped, cmap='Greys', vmin=min, vmax=max)
    ax[myrow,mycol].set_xticks([])
    ax[myrow,mycol].set_yticks([])

    if (myrow == 10):
        ax[myrow, mycol].set_xlabel(str(mycol - 5), fontsize=12)
        ax[myrow, mycol].xaxis.set_label_coords(0.5,-0.3)

    if (mycol == 0):
        ax[myrow, mycol].set_ylabel(str(myrow - 5), rotation=0, loc='center', fontsize=12)
        ax[myrow, mycol].yaxis.set_label_coords(-0.5,0.25)

    count += 1

# labels = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
# for i in range(11):
#     plt.setp(ax[-1, i], xlabel=str(labels[i]))
#     plt.setp(ax[i, 0], ylabel=str(labels[i]))
    

plt.show()

# ====================================================
