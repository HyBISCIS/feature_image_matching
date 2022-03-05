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
import multiprocessing

import deconv_func as func

'''
Old deprecated code that ran out of memory quickly. I had initially
thought that I could just run everything in parallel, but then my
computer ran out of memory and yelled at me. Perhaps this would work
on a computer with much more ram allocation than I have on my computer (8 gigs)
'''

# MAIN VARIABLES
CHIP_NAME = "MINERVA" 
BLOCK_SIZE = (11,11)
UPSAMPLE_RATIO = 16
FREQ = 6250  # in kHz

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
    im = 'image_2d_ph1'
    # Note: Freq in Hz

    logdir = r"../data/super_res"
    logfile = r"ECT_block11x11_Mix_Cosmarium_Pediastrum_6p25M_set_1.h5"
    microscope_img = None
    CENTER = (0,0)

image_list = []
thread_list = []

def thread_helper(key):
    myrow = int(mydata[i].attrs[row])
    mycol = int(mydata[i].attrs[col])

    if (CHIP_NAME == "MINERVA"):
        # Dirty Fix for Minerva indexing from -5 to 5, rather than the 0 to 11 we expected previously.
        myrow += 5
        mycol += 5
    
    myimage,mymedian,mystd = func.getimage(mydata,key,UPSAMPLE_RATIO,im,BLOCK_SIZE[0],CHIP_NAME,shiftrow=myrow,shiftcol=mycol)
    #myimage,kernel_smoothed = func.linear_filter(myimage, reference_image, UPSAMPLE_RATIO, window2d)

    image_list.append(myimage)

# --------------------------------------------------------------------------

mydata = h5py.File(os.path.join(logdir,logfile),'r')

# Load microscope image
if microscope_img != None:
    myphoto = image.imread(os.path.join(logdir,'Microscope_bead_20u_0639pm.bmp'))
    myphoto=myphoto[200:1130,1100:2020,:]
    norm = np.zeros(myphoto.shape)

    micro_snr = func.get_spatial_snr(myphoto)
    print("Reference Microscope Image SNR dB: {}\n".format(micro_snr))

window2d = func.create_window(CHIP_NAME, BLOCK_SIZE[0], UPSAMPLE_RATIO)

compositeimage = None
compositeimage2 = None

sortedkeys = sorted(mydata.keys(), key=lambda k: int(mydata[k].attrs[row])*100+int(mydata[k].attrs[col]))
k_reference = [x for x in sortedkeys if int(mydata[x].attrs[f_name]) == FREQ and int(mydata[x].attrs[row])==CENTER[0] and int(mydata[x].attrs[col])==CENTER[1]][0]

reference_image,refmedian,refstd = func.getimage(mydata, k_reference, UPSAMPLE_RATIO,im,BLOCK_SIZE[0],CHIP_NAME)

# Get rid of everything except for the frequencies that we are dealing with
sortedkeys[:] = [x for x in sortedkeys if int(mydata[x].attrs[f_name]) == FREQ]
count = 0

# Spin all threads
for i in sortedkeys:
    t_i = multiprocessing.Process(target=thread_helper, args=(i,))
    thread_list.append(t_i)
    t_i.start()

# Join with all threads
for t_i in thread_list:
    t_i.join()

# Add everything Together
compositeimage = None

for im in image_list:
    if compositeimage == None:
        compositeimage = im
    else:
        compositeimage += im

# ====================================================

# Save Images
# FIXME: Switch to .SVG later
plt.imsave("../log/linear_deconvolution.png", compositeimage)

# ====================================================

# Normalize both images
compositeimage = func.normalize_img(compositeimage)

# SNRs
c_snr = func.get_spatial_snr(compositeimage)
print("Composite Image 2 (Deconvolution) SNR dB: {}\n".format(c_snr))

plt.title("Linear Deconvolution Method")
plt.imshow(compositeimage)
plt.show()