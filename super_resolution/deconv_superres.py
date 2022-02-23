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


# TODO List
# TODO: Need to find cosmarium and pediastrum isolated, so can get better results for super_resolution
# TODO: Try and get smoother linear deconvolutions.
# TODO: Why do the show_algae.py plots not line up with the ones that I created for
# TODO: Need to get spatial resolution stuff 




# FIXME: Shifting seeming to not work correctly given everything that is happening.
#        I believe it has something to do with the weird shifting caused by getimage? Comes from weird shift in get


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

two_lobe = (139, 171+22)
single_lobe = (123, 170+15)
single_lobe = (2262, 3240)
INTEREST_POINT = (single_lobe[0]*UPSAMPLE_RATIO, single_lobe[1]*UPSAMPLE_RATIO)
INTEREST_POINT = single_lobe

# FIXME: when we crop something, it is different for the shiftsum and for the linear deconvolution...



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

# Initialize Composite Images
compositeimage = None
compositeimage2 = None
compositeimage3 = None
count = 0

# Load data and sort keys
mydata = h5py.File(os.path.join(logdir,logfile),'r')
sortedkeys = sorted(mydata.keys(), key=lambda k: int(mydata[k].attrs[row])*100+int(mydata[k].attrs[col]))
sortedkeys[:] = [x for x in sortedkeys if int(mydata[x].attrs[f_name]) == FREQ]

if (CHIP_NAME == "MINERVA"):    
    center_data = h5py.File(os.path.join(logdir,center_img_file))



# # Load microscope image
# if microscope_img != None:
#     myphoto = image.imread(os.path.join(logdir,'Microscope_bead_20u_0639pm.bmp'))
#     myphoto=myphoto[200:1130,1100:2020,:]
#     norm = np.zeros(myphoto.shape)

#     micro_snr = func.get_spatial_snr(myphoto)
#     print("Reference Microscope Image SNR dB: {}\n".format(micro_snr))



# Creating Window for Deconvolution Superresolution Technique
if CROP:
    window_scale = 0.4
    window_length = int(CROPPED_LENGTH * window_scale)
    pad_length = int(CROPPED_LENGTH * ((1 - window_scale) / 2))
    window1d = np.abs(np.hanning(window_length))
    window2d = np.sqrt(np.outer(window1d, window1d))
    window2d = np.pad(window2d, pad_length+1)
else:
    window2d = func.create_window(CHIP_NAME, BLOCK_SIZE[0], UPSAMPLE_RATIO)

print("Cropped Size:", CROPPED_LENGTH)
print("Window Size:", window2d.shape)
plt.imshow(window2d)
plt.show()



# Obtain Reference Images
k_reference = [x for x in sortedkeys if int(mydata[x].attrs[f_name]) == FREQ and int(mydata[x].attrs[row])==CENTER[0]-2 and int(mydata[x].attrs[col])==CENTER[1]][0]

if (CHIP_NAME == "MINERVA"):
    key = list(center_data.keys())[0]
    reference_image,ref_shifted = func.getimage(center_data, key, UPSAMPLE_RATIO,im,BLOCK_SIZE[0],CHIP_NAME, False)
else:
    reference_image,ref_shifted = func.getimage(mydata, k_reference, UPSAMPLE_RATIO,im,BLOCK_SIZE[0],CHIP_NAME, LOW_SALT)

reference_image,ref_shifted = func.getimage(mydata, k_reference, UPSAMPLE_RATIO,im,BLOCK_SIZE[0],CHIP_NAME, LOW_SALT)

if CROP:
    # FIXME: For some reason, the interest point is wacky and different than the ones in the photos. Not sure why.
    # Kangping said I can probably get rid of that weird shifting thing, so maybe that will help a little?
    reference_image = func.get_area_around(reference_image, INTEREST_POINT, RAD, UPSAMPLE_RATIO)

plt.imshow(reference_image, cmap='Greys')
plt.show()



# Iterate through image offsets
for i in sortedkeys:
    # Obtain Row and Col Offsets of Particular Image
    myrow = int(mydata[i].attrs[row])
    mycol = int(mydata[i].attrs[col])

    if (CHIP_NAME == "MINERVA"):
        # Dirty Fix for Minerva indexing from -5 to 5, rather than the 0 to 11 we expected previously.
        myrow += 5
        mycol += 5
    
    myimage,myimage_shift = func.getimage(mydata,i,UPSAMPLE_RATIO,im,BLOCK_SIZE[0],CHIP_NAME,LOW_SALT,shiftrow=myrow,shiftcol=mycol)

    if CROP:
        myimage = func.get_area_around(myimage, INTEREST_POINT, RAD, UPSAMPLE_RATIO)
        myimage_shift = func.get_area_around(myimage_shift, INTEREST_POINT, RAD, UPSAMPLE_RATIO)

    # Perform linear filter
    # NOTE: We are no longer using shifted of anything. Up to the linear deconvolution to fix all of that
    #       which is what we should have done previously. I am not sure why I had not fixed that before. 
    start = time.time() 
    myimage_filtered,kernel_smoothed = func.linear_filter(myimage, reference_image, UPSAMPLE_RATIO, window2d)
    end = time.time()
    print("Linear Filter Elapsed Time (sec):", end-start)

    # Sum into the composite images
    if compositeimage is None:
        compositeimage = np.zeros_like(myimage)
        compositeimage2 = np.zeros_like(myimage)
        compositeimage3 = np.zeros_like(myimage)

    compositeimage += myimage_shift
    compositeimage2 += myimage_filtered
    compositeimage3 += myimage

    count += 1
    print("Count: ", count)

    # fig, ax = plt.subplots(1,4)
    # ax[0].imshow(reference_image, cmap='Greys')
    # ax[0].set_title("Reference Image")
    # ax[1].imshow(myimage, cmap='Greys')
    # ax[1].set_title("My Image")
    # ax[2].imshow(myimage_shift, cmap='Greys')
    # ax[2].set_title("My Image Shifted")
    # ax[3].imshow(myimage_filtered, cmap='Greys')
    # ax[3].set_title("My Image Filtered")
    # plt.suptitle("Row Offset: {}, Col Offset: {}".format(myrow, mycol))
    # plt.show()




# Normalize at the end?
if not CROP:
    compositeimage = func.channel_norm(compositeimage)
    compositeimage = compositeimage[:7000,1500:]





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
compositeimage3 = func.normalize_img(compositeimage3)

# SNRs
c_snr = func.get_spatial_snr(compositeimage)
c2_snr = func.get_spatial_snr(compositeimage2)
c3_snr = func.get_spatial_snr(compositeimage3)
ref_snr = func.get_spatial_snr(reference_image)

print("Reference Image SNR dB: {}\n".format(ref_snr))
print("Composite Image 1 (Naive Shift Sum) SNR dB: {}\n".format(c_snr))
print("Composite Image 2 (Deconvolution) SNR dB: {}\n".format(c2_snr))
print("Composite Image 3 (Just Sum) SNR dB: {}\n".format(c3_snr))

plt.figure(1)
plt.title("Naive Shift Sum")
plt.imshow(compositeimage)

plt.figure(2)
plt.title("Linear Deconvolution Method")
plt.imshow(compositeimage2)

plt.figure(3)
plt.title("Just Sum")
plt.imshow(compositeimage3)

plt.show()