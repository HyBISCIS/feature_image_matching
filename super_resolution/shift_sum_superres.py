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
Script that only runs shift sum
'''

# ===================   Settings     ===================

# MAIN VARIABLES
CHIP_NAME = "MINERVA" 
BLOCK_SIZE = (11,11)
UPSAMPLE_RATIO = 16         # FIXME: DO NOT CHANGE. WINDOW IS HARD CODED RIGHT NOW
FREQ = 3125                 #6250  # in kHz
#FREQ = 6250
SAVE_IMAGES = True
LOW_SALT = True
INTERPOLATE_ORDER = 0

# Cropping Parameters (In order of Importance)
CROP = True
RAD = 16                                # Radius of cropped square # In original pixels rather than upsampled pixels, need to be factor of 2?
CROPPED_LENGTH = 2*RAD*UPSAMPLE_RATIO

single_lobe = (2262, 3240)      # Single Lobe and Double Lobe near
single_lobe = (2944,3287)       # Isolated Single Lobe 
single_lobe = (6897, 1949)
single_lobe = (6837, 1324)
single_lobe = (6459, 2352)

two_lobe = (2048,617)           # Isolated Double Lobe
two_lobe = (3231, 3367)         # Isolated Double Lobe
two_lobe = (5804, 408)          # Isolated Double Lobe
two_lobe = (4413, 1830)         # Isolated Double Lobe
#INTEREST_POINT = (single_lobe[0]*UPSAMPLE_RATIO, single_lobe[1]*UPSAMPLE_RATIO)
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

# Initialize Composite Images
compositeimage = None
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
    window_scale = 0.5
    window_length = int(CROPPED_LENGTH * window_scale)
    pad_length = int(CROPPED_LENGTH * ((1 - window_scale) / 2))
    window1d = np.abs(np.hanning(window_length))
    window2d = np.sqrt(np.outer(window1d, window1d))
    window2d = np.pad(window2d, pad_length)
else:
    window2d = func.create_window(CHIP_NAME, BLOCK_SIZE[0], UPSAMPLE_RATIO)

print("Cropped Size:", CROPPED_LENGTH)
print("Window Size:", window2d.shape)
# plt.imshow(window2d)
# plt.show()



# Obtain Reference Images
k_reference = [x for x in sortedkeys if int(mydata[x].attrs[f_name]) == FREQ and int(mydata[x].attrs[row])==CENTER[0]-2 and int(mydata[x].attrs[col])==CENTER[1]][0]
reference_image,ref_shifted = func.getimage(mydata, k_reference, UPSAMPLE_RATIO,im,BLOCK_SIZE[0],CHIP_NAME, LOW_SALT,INTERPOLATE_ORDER)

# plt.imshow(reference_image)
# plt.show()
# exit(0)

if CROP:
    # FIXME: For some reason, the interest point is wacky and different than the ones in the photos. Not sure why.
    # Kangping said I can probably get rid of that weird shifting thing, so maybe that will help a little?
    reference_image = func.get_area_around(reference_image, INTEREST_POINT, RAD, UPSAMPLE_RATIO)

# plt.imshow(reference_image)
# plt.show()
# exit(1)

# Iterate through image offsets
for i in sortedkeys:
    start = time.time() 
    # Obtain Row and Col Offsets of Particular Image
    myrow = int(mydata[i].attrs[row])
    mycol = int(mydata[i].attrs[col])

    if (CHIP_NAME == "MINERVA"):
        # Dirty Fix for Minerva indexing from -5 to 5, rather than the 0 to 11 we expected previously.
        myrow += 5
        mycol += 5

    if (myrow == 5 and mycol == 5):
        # Skip middle as it is not useful to us and just adds noise
        continue
    
    myimage,myimage_shift = func.getimage(mydata,i,UPSAMPLE_RATIO,im,BLOCK_SIZE[0],CHIP_NAME,LOW_SALT,INTERPOLATE_ORDER,shiftrow=myrow,shiftcol=mycol)

    if CROP:
        myimage = func.get_area_around(myimage, INTEREST_POINT, RAD, UPSAMPLE_RATIO)
        new_interest = (INTEREST_POINT[0]-250, INTEREST_POINT[1]-100)
        myimage_shift = func.get_area_around(myimage_shift, new_interest, RAD, UPSAMPLE_RATIO)

    # Sum into the composite images
    if compositeimage is None:
        compositeimage = np.zeros_like(myimage)

    compositeimage += myimage_shift

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

    end = time.time()
    print("Image Elapsed Time (sec):", end-start)

# Normalize at the end?
if not CROP:
    compositeimage = func.channel_norm(compositeimage)
    compositeimage = compositeimage[:7000,1500:]


# compositeimage2 = unsharp_mask(compositeimage2, radius=5, amount=3)
compositeimage = func.normalize_img(compositeimage)

# ====================================================

# Save Images
# FIXME: Switch to .SVG later
if SAVE_IMAGES:
    plt.imsave("../log/shift_sum_pediastrum_multicell_2.png", compositeimage)

# ====================================================

print("Composite Mean:", np.mean(compositeimage))
print("Composite STD:", np.std(compositeimage))


# SNRs
c_snr = func.get_spatial_snr(compositeimage)
ref_snr = func.get_spatial_snr(reference_image)

print("Reference Image SNR dB: {}".format(ref_snr))
print("Composite Image 1 (Naive Shift Sum) SNR dB: {}".format(c_snr))

plt.figure(1)
plt.title("Naive Shift Sum")
plt.imshow(compositeimage)
plt.colorbar()
plt.show()