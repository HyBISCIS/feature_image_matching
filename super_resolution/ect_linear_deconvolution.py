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

CHIP_NAME = "MINERVA" # "LILLIPUT"
BLOCK_SIZE = (11,11)
CENTER = (5,5)

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

    myimage_filtered = np.fft.fftshift(np.fft.irfft2(input_f * np.fft.rfft2(kernel)))

    return myimage_filtered, kernel_smoothed

def minmaxnorm(x):
    return np.nan_to_num((x-np.min(x))/(np.max(x)-np.min(x)))

def cropimage(im):
    if (CHIP_NAME == "LILLIPUT"):
        cropped = im[10:90,10:90]
    elif (CHIP_NAME == "MINERVA"):
        cropped = im
    return cropped

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

logdir = r"data/bead_20um"
logfile = r"phase2_sweep_11x11_block.h5"
freq = '6250kHz'

mydata = h5py.File(os.path.join(logdir,logfile),'r')

myphoto = image.imread(os.path.join(logdir,'Microscope_bead_20u_0639pm.bmp'))
myphoto=myphoto[200:1130,1100:2020,:]

upsample_ratio=16
window1d = np.abs(np.hanning(30*upsample_ratio))
window2d = np.sqrt(np.outer(window1d,window1d)) # Regular window
window2d = np.pad(window2d,25*upsample_ratio)  # Why do we pad?
window2d = window2d[myshift(5):(myshift(5)-82), # What are we doing here?
                    myshift(5):(myshift(5)-82)]

sortedkeys = sorted(mydata.keys(), key=lambda k: int(mydata[k].attrs['R'])*100+int(mydata[k].attrs['C']))
k_reference = [x for x in sortedkeys if freq in x and int(mydata[x].attrs['R'])==CENTER[0] and int(data[x].attrs['C'])==CENTER[1]][0]
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

    # Break once reach 5,5 as finished quarter
    if (myrow==5 and mycol==5):
        break

    # Get the other things I want to compare against
    lst = find_mirrored((myrow,mycol), CENTER)
    lst.insert(0, (myrow,mycol))

    images = []
    fig, ax = plt.subplots(2,2)
    fig.suptitle("Mirrored Kernels in Frequency Domain")

    for c,j in enumerate(lst):
        # Get image
        key = dict[j]
        myimage,mymedian,mystd = getimage(mydata,key,shiftrow=j[0],shiftcol=j[1])

        # Get kernel
        myimage_filtered,kernel_smoothed = linear_filter(myimage, reference_image, upsample_ratio, window2d)
        images.append(kernel_smoothed[350:850,350:850])

        title = "Row: {} Col: {} Kernel Cropped".format(j[0], j[1])
        ax_index = ax_helper(c)
        ax[ax_index].imshow(kernel_smoothed[350:850,350:850], cmap='Blues_r')
        ax[ax_index].set_title(title, fontsize=10)
        ax[ax_index].set_xticks([])
        ax[ax_index].set_yticks([])

    specific_title = "row{}col{}_mirrored_kernels".format(myrow,mycol)
    #fig.savefig("log/kernels/"+specific_title)

    fig2, ax2 = plt.subplots(1,3)
    fig2.suptitle("Correlations between Row: {} Col: {} and Mirrored Kernels".format(lst[0][0], lst[0][1]))

    # # Do Cross Correlation
    for i in range(1,4):
        # Do correlation for 0 against 1,2,3
        correlate = sig.correlate2d(images[0], images[i], mode='same')
        ax2[i-1].imshow(correlate, cmap='Blues_r')
        ax2[i-1].set_xticks([])
        ax2[i-1].set_yticks([])
        ax2[i-1].title.set_text("Row: {}, Col: {}".format(lst[i][0], lst[i][1], fontsize=10))

    other_title = "row{}col{}_mirrored_correlations".format(myrow,mycol)
    #fig2.savefig("log/cross_correlations/"+other_title)

    count += 1
    print("Iteration: ", count)
