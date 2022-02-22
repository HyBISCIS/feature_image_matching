from pickle import FALSE, TRUE
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
import time

import deconv_func as func


# FIXME: Shifting seeming to not work correctly given everything that is happenging.
#        I believe it has something to do with the weird shifting caused by getimage? Comes from weird shift in geti
single_lobe = (140,140)

# MAIN VARIABLES
CHIP_NAME = "MINERVA" 
BLOCK_SIZE = (11,11)
UPSAMPLE_RATIO = 16 # FIXME: DO NOT CHANGE. WINDOW IS HARD CODED RIGHT NOW
FREQ = 3125 #6250  # in kHz
SAVE_IMAGES = True
CROP = False
RAD = 16    # Radius of cropped square # In original pixels rather than upsampled pixels, need to be factor of 2?
INTEREST_POINT = (single_lobe[0]*UPSAMPLE_RATIO, single_lobe[1]*UPSAMPLE_RATIO)
CROPPED_LENGTH = 2*RAD*UPSAMPLE_RATIO
LOW_SALT = True

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

# --------------------------------------------------------------------------

mydata = h5py.File(os.path.join(logdir,logfile),'r')

# # Load microscope image
# if microscope_img != None:
#     myphoto = image.imread(os.path.join(logdir,'Microscope_bead_20u_0639pm.bmp'))
#     myphoto=myphoto[200:1130,1100:2020,:]
#     norm = np.zeros(myphoto.shape)

#     micro_snr = func.get_spatial_snr(myphoto)
#     print("Reference Microscope Image SNR dB: {}\n".format(micro_snr))

if CROP:
    window1d = np.abs(np.hanning(CROPPED_LENGTH))
    window2d = np.sqrt(np.outer(window1d, window1d))
else:
    window2d = func.create_window(CHIP_NAME, BLOCK_SIZE[0], UPSAMPLE_RATIO)

compositeimage = None
compositeimage2 = None

sortedkeys = sorted(mydata.keys(), key=lambda k: int(mydata[k].attrs[row])*100+int(mydata[k].attrs[col]))
k_reference = [x for x in sortedkeys if int(mydata[x].attrs[f_name]) == FREQ and int(mydata[x].attrs[row])==CENTER[0] and int(mydata[x].attrs[col])==CENTER[1]][0]

reference_image,refmedian,refstd = func.getimage(mydata, k_reference, UPSAMPLE_RATIO,im,BLOCK_SIZE[0],CHIP_NAME, LOW_SALT)

if CROP:
    reference_image = func.get_area_around(reference_image, INTEREST_POINT, RAD, UPSAMPLE_RATIO)

# Get rid of everything except for the frequencies that we are dealing with
sortedkeys[:] = [x for x in sortedkeys if int(mydata[x].attrs[f_name]) == FREQ]
count = 0

for i in sortedkeys:
    myrow = int(mydata[i].attrs[row])
    mycol = int(mydata[i].attrs[col])

    if (CHIP_NAME == "MINERVA"):
        # Dirty Fix for Minerva indexing from -5 to 5, rather than the 0 to 11 we expected previously.
        myrow += 5
        mycol += 5
    
    myimage,mymedian,mystd = func.getimage(mydata,i,UPSAMPLE_RATIO,im,BLOCK_SIZE[0],CHIP_NAME,LOW_SALT,shiftrow=myrow,shiftcol=mycol)

    if CROP:
        myimage = func.get_area_around(myimage, INTEREST_POINT, RAD, UPSAMPLE_RATIO)

    start = time.time()
    myimage_filtered,kernel_smoothed = func.linear_filter(myimage, reference_image, UPSAMPLE_RATIO, window2d)
    end = time.time()
    print("Linear Filter Elapsed Time (sec):", end-start)


    if compositeimage is None:
        compositeimage = np.zeros_like(myimage)
        compositeimage2 = np.zeros_like(myimage)

    compositeimage = compositeimage + myimage
    compositeimage2 = compositeimage2 + myimage_filtered

    count += 1
    print("Count: ", count)


# ====================================================

# Save Images
# FIXME: Switch to .SVG later
if SAVE_IMAGES:
    plt.imsave("../log/shift_linear_deconv.png", compositeimage2)
    plt.imsave("../log/shift_sum_one.png", compositeimage)

# ====================================================

# Normalize both images
compositeimage = func.normalize_img(compositeimage)
compositeimage2 = func.normalize_img(compositeimage2)

# SNRs
c_snr = func.get_spatial_snr(compositeimage)
c2_snr = func.get_spatial_snr(compositeimage2)
ref_snr = func.get_spatial_snr(reference_image)

print("Reference Image SNR dB: {}\n".format(ref_snr))
print("Composite Image 1 (Naive Shift Sum) SNR dB: {}\n".format(c_snr))
print("Composite Image 2 (Deconvolution) SNR dB: {}\n".format(c2_snr))

plt.figure(1)
plt.title("Naive Shift Sum")
plt.imshow(compositeimage)

plt.figure(2)
plt.title("Linear Deconvolution Method")
plt.imshow(compositeimage2)

plt.show()