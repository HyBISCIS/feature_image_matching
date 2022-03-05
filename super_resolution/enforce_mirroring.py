import h5py
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.cm as cm
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib import image
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import os
import cv2
from datetime import datetime
import time
from skimage.transform import resize,rescale, rotate
from skimage.filters import gaussian
from skimage import color, data, restoration
from skimage.morphology import dilation,erosion,selem
import scipy.signal as sig
import imageio
from skimage.transform import warp, PiecewiseAffineTransform
from skimage.registration import optical_flow_tvl1

'''
Old script to try and enforce mirroring.

TODO: SCRIPT NEEDS TO BE UPDATED TO RUN ON MINERVA
'''

# Note: Most of the code is taken from Professor Rosenstein's "plot_lilliput_ECT_4_ifft.py" code"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def getimage(mydata,k,shiftrow=5,shiftcol=5):

    myimage = mydata[k]['image'][:]
    myimage = cropimage(myimage)
    #myimage = minmaxnorm(myimage)
    
    myreference = mydata[k_reference]['image'][:]
    myreference = cropimage(myreference)
    #myreference = minmaxnorm(myreference)
    
    #%myimage = myimage - myreference
    #myimage = np.divide(myimage,myreference) - 1
    myimage[np.isinf(myimage)] = 0
    myimage = np.nan_to_num(myimage)
    
    myimage = myimage - np.mean(myimage)


    myimagereference=mydata[k_reference]['image']
    myimagereference=cropimage(myimagereference)
    

    myimage = mydata[k]['image'][:]
    myimage = cropimage(myimage)
    myreference = mydata[k_reference]['image'][:]
    myreference =cropimage(myreference)

    # remove outlier pixels
    for rep in range(2):
        mymedian=np.median(myimage)
        mymean=np.mean(myimage)
        mystd=np.std(myimage)
        myimage[np.abs(myimage-mymean)>4*mystd] = mymean
        mystd=np.std(myimage)   #[np.abs(myimage-mymedian)<4*mystd])
        
    myimage = rescale(myimage,upsample_ratio,order=0)

    myimage = myimage[myshift(shiftrow):(myshift(shiftrow)-82),
                                  myshift(shiftcol):(myshift(shiftcol)-82)]

    return myimage,mymedian,mystd
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def linear_filter(myimage, reference_image, upsample_ratio, window2d):
    outputimage = reference_image - gaussian(reference_image,sigma=10*upsample_ratio)
    inputimage = myimage - gaussian(myimage,sigma=10*upsample_ratio)

    #outputimage = rescale(outputimage,upsample_ratio)#,order=0)
    #inputimage = rescale(inputimage,upsample_ratio)#,order=0)

    output_f = np.fft.rfft2(outputimage)
    input_f = np.fft.rfft2(inputimage)
    
    kernel_f = output_f / input_f         # do the deconvolution
    kernel = np.fft.irfft2(kernel_f)      # inverse Fourier transform
    kernel = np.fft.fftshift(kernel)      # shift origin to center of image
    kernel *= window2d
    kernel /= kernel.max()             # normalize gray levels
    
    kernel_smoothed = gaussian(kernel,sigma=0.5*upsample_ratio)
    kernel_smoothed /= kernel_smoothed.max()
    kernel_smoothed = kernel

    myimage_filtered = np.fft.fftshift(np.fft.irfft2(input_f * np.fft.rfft2(kernel)))

    return myimage_filtered, kernel_smoothed

def linear_filter_help(myimage, upsample_ratio, kernel, num):
    inputimage = myimage - gaussian(myimage,sigma=10*upsample_ratio)
    input_f = np.fft.rfft2(inputimage)
    mirrored_kernel = flip_kernel(num, kernel)
    myimage_filtered = np.fft.fftshift(np.fft.irfft2(input_f * np.fft.rfft2(mirrored_kernel)))

    return myimage_filtered

def flip_kernel(num, kernel):
    if num==0:
        # Top right, flip horizontal
        return np.fliplr(kernel)
    elif num==1:
        # Bottom Left, flip vertical
        return np.flipud(kernel)

    elif num==2:
        # Bottom Right, flip both
        return np.flipud(np.fliplr(kernel))
    else:
        print("Something went wrong with flipping kernel")

def minmaxnorm(x):
    return np.nan_to_num((x-np.min(x))/(np.max(x)-np.min(x)))

def power_snr(amp):
    return (20 * np.log10(amp**2))

def cropimage(im):
    return im[10:90,10:90]

def myshift(x):
    return (40+(5-x)*8)

def find_mirrored(pt, center):
    # Names assume given from point from (0,0) to (5,5)

    # Assume that maxrow and maxcol will always be odd, so that we can find center
    row_offset = pt[0] - center[0]
    col_offset = pt[1] - center[1]

    top_right = (center[0]+row_offset, center[1]-col_offset)
    bot_left = (center[0]-row_offset, center[1]+col_offset)
    bot_right = (center[0]-row_offset, center[1]-col_offset)

    return [top_right, bot_left, bot_right]

def ax_helper(num):
    if num==0:
        return(0,0)
    elif num==1:
        return(0,1)
    elif num==2:
        return(1,0)
    elif num==3:
        return(1,1)
    else:
        return -1

# --------------------------------------------------------------------------

logdir = r"../data/bead_20um"
logfile = r"phase2_sweep_11x11_block.h5"
#freq = '6250kHz'
freq = '1562kHz'

mydata = h5py.File(os.path.join(logdir,logfile),'r')

myphoto = image.imread(os.path.join(logdir,'Microscope_bead_20u_0639pm.bmp'))
myphoto=myphoto[200:1130,1100:2020,:]

# Microscope img SNR
norm = np.zeros(myphoto.shape)
myphoto = cv2.normalize(myphoto, norm, 0, 255, cv2.NORM_MINMAX)
micro_snr = power_snr(np.mean(myphoto) / np.std(myphoto))

print("Reference Microscope Image SNR dB: {}\n".format(micro_snr))

upsample_ratio=16
window1d = np.abs(np.hanning(30*upsample_ratio))
window2d = np.sqrt(np.outer(window1d,window1d)) # Regular window
window2d = np.pad(window2d,25*upsample_ratio)  # Why do we pad?
window2d = window2d[myshift(5):(myshift(5)-82), # What are we doing here?
                    myshift(5):(myshift(5)-82)]

compositeimage = None

center = (5,5)
sortedkeys = sorted(mydata.keys(), key=lambda k: int(mydata[k].attrs['R'])*100+int(mydata[k].attrs['C']))
k_reference = [x for x in sortedkeys if freq in x and int(mydata[x].attrs['R'])==center[0] and int(mydata[x].attrs['C'])==center[1]][0]
reference_image,refmedian,refstd = getimage(mydata, k_reference)

# Get rid of everything except for the frequencies that we are dealing with
sortedkeys[:] = [x for x in sortedkeys if freq in x]

# Create dictionary with row, col as the key to this sorted key list for easier lookup later
dict = {}
count = 0

for i in sortedkeys:
    row = int(mydata[i].attrs['R'])
    col = int(mydata[i].attrs['C'])

    dict[(row,col)] = i

# Enumerate through 1/4 of the keys, or from (0,0) -> (5,5) in both directions
for i in sortedkeys:
    myrow = int(mydata[i].attrs['R'])
    mycol = int(mydata[i].attrs['C'])

    if (myrow > 5 or mycol > 5):
        continue

    # Break once reach 5,5 as finished quarter
    if (myrow==5 and mycol==5):
        break

    # Get the other things I want to compare against
    lst = find_mirrored((myrow,mycol), center)

    key = dict[(myrow,mycol)]
    myimage,mymedian,mystd = getimage(mydata,key,shiftrow=myrow,shiftcol=mycol)
    myimage_filtered,kernel_smoothed = linear_filter(myimage, reference_image, upsample_ratio, window2d)

    if compositeimage is None:
        compositeimage = np.zeros_like(myimage)

    compositeimage += myimage_filtered

    for c,j in enumerate(lst):
        # Get image
        key = dict[j]
        myimage,mymedian,mystd = getimage(mydata,key,shiftrow=j[0],shiftcol=j[1])
        myimage_filtered = linear_filter_help(myimage,upsample_ratio,kernel_smoothed, c)
        compositeimage += myimage_filtered
        
    count += 1
    print("Iteration: ", count)

# ==============================================

# Normalize both images
norm_img1 = np.zeros(compositeimage.shape)
norm_ref = np.zeros(reference_image.shape)
compositeimage = cv2.normalize(compositeimage, norm_img1, 0, 255, cv2.NORM_MINMAX)

plt.imsave("../log/beads/mirroring_beads.png", compositeimage)

# Determine STD, MEAN
mean1 = np.mean(compositeimage)
std1 = np.std(compositeimage)
e_snr_1 = power_snr(mean1 / std1)

print("Composite Image 1 (Naive Shift Sum) SNR dB: {}\n".format(e_snr_1))

plt.figure(1)
plt.title("Enforcing Mirroring")
plt.imshow(compositeimage)

plt.show()
