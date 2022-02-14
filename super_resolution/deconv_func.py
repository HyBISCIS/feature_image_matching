from pickle import FALSE
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import image
from skimage.transform import resize,rescale, rotate
from skimage.filters import gaussian
from skimage.morphology import dilation,erosion,selem
import scipy.signal as sig
from skimage.transform import warp, PiecewiseAffineTransform
from skimage.registration import optical_flow_tvl1




def apply_cal(image,coeffs):
    assert(len(coeffs)==8)
    image_cal = image.copy()

    for ch in range(8):
        image_cal[:,ch*32:(ch+1)*32] = image_cal[:,ch*32:(ch+1)*32] * coeffs[ch]
    return image_cal
        

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def getimage(mydata,k,upsample_ratio,im,block_length,chip_name,shiftrow=5,shiftcol=5):
    # myimage = mydata[k][im][:] 
    # myimage = cropimage(myimage)
    # #myimage = minmaxnorm(myimage)
    
    # #%myimage = myimage - myreference
    # #myimage = np.divide(myimage,myreference) - 1
    # myimage[np.isinf(myimage)] = 0
    # myimage = np.nan_to_num(myimage)
    
    # myimage = myimage - np.mean(myimage)
    
    myimage = mydata[k][im][:]

    if chip_name == "MINERVA":
        normrows = range(200,300)
        coeffs = np.ones(8)

        # normalize by channel
        for ch in range(8):
            coeffs[ch] = 1 / np.mean(myimage[normrows, ch*32:(ch+1)*32])
        myimage = apply_cal(myimage,coeffs)  

    myimage = cropimage(myimage, (block_length, block_length))
    block_center = int((block_length - 1) / 2)

    # remove outlier pixels
    for rep in range(2):
        mymedian=np.median(myimage)
        mymean=np.mean(myimage)
        mystd=np.std(myimage)
        myimage[np.abs(myimage-mymean)>4*mystd] = mymean
        mystd=np.std(myimage)   #[np.abs(myimage-mymedian)<4*mystd])

    # plt.imshow(myimage)
    # plt.show()
        
    myimage = rescale(myimage,upsample_ratio,order=0)

    if chip_name == "LILLIPUT":
        # FIXME: Ask Rosenstein if we should window by image lengths rather than do this wacky thing
        shift_row = myshift(shiftrow, 80, block_center, upsample_ratio)
        shift_col = myshift(shiftcol, 80, block_center, upsample_ratio)
        myimage = myimage[shift_row:(shift_row-82), shift_col:(shift_col-82)]
    else:
        # This is going to be 512 by 256, or cropped to 492 : 236
        shift_row = myshift(shiftrow, 492, block_center, upsample_ratio)
        shift_col = myshift(shiftcol, 236, block_center, upsample_ratio)
        myimage = myimage[shift_row:(shift_row-492), shift_col:(shift_col-236)] #FIXME: Weird fix to get 492 , 236 here

    return myimage,mymedian,mystd
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def linear_filter(myimage, reference_image, upsample_ratio, window2d):

    outputimage = reference_image - gaussian(reference_image,sigma=10*upsample_ratio)
    inputimage = myimage - gaussian(myimage,sigma=10*upsample_ratio)
    print(outputimage.shape)
    print(inputimage.shape)
    print(window2d.shape)

    #outputimage = rescale(outputimage,upsample_ratio)#,order=0)
    #inputimage = rescale(inputimage,upsample_ratio)#,order=0)

    output_f = np.fft.rfft2(outputimage)
    input_f = np.fft.rfft2(inputimage)
    
    kernel_f = output_f / input_f         # do the deconvolution
    kernel = np.fft.irfft2(kernel_f)      # inverse Fourier transform
    kernel = np.fft.fftshift(kernel)      # shift origin to center of image
    #kernel *= window2d // FIXME: Get rid of for now so can run stuff
    kernel /= kernel.max()             # normalize gray levels
    
    kernel_smoothed = gaussian(kernel,sigma=0.5*upsample_ratio)
    kernel_smoothed /= kernel_smoothed.max()

    myimage_filtered = np.fft.fftshift(np.fft.irfft2(input_f * np.fft.rfft2(kernel)))

    return myimage_filtered, kernel_smoothed

def minmaxnorm(x):
    return np.nan_to_num((x-np.min(x))/(np.max(x)-np.min(x)))

def cropimage(im, block_sz):
    crop_step = (block_sz[0]-1, block_sz[1]-1)
    im_size = im.shape

    # Obtain crop bounds
    x_bound = (crop_step[0], (im_size[0]-crop_step[0]))
    y_bound = (crop_step[1], (im_size[1]-crop_step[1]))

    cropped = im[x_bound[0]:x_bound[1],y_bound[0]:y_bound[1]]

    return cropped

def myshift(x, crop_length, block_center, upsample_ratio):
    half_upsample = upsample_ratio / 2
    half_crop_length = crop_length / 2

    shifted_index = half_crop_length + ((block_center - x) * half_upsample)

    return int(shifted_index)

def find_mirrored(pt, center):
    # Names assume given from point from (0,0) to (5,5)

    # Assume that maxrow and maxcol will always be odd, so that we can find center
    row_offset = pt[0] - center[0]
    col_offset = pt[1] - center[1]

    top_right = (center[0]+row_offset, center[1]-col_offset)
    bot_left = (center[0]-row_offset, center[1]+col_offset)
    bot_right = (center[0]-row_offset, center[1]-col_offset)

    return [top_right, bot_left, bot_right]

def create_window(chip_name, block_length, upsample_ratio):
    block_center = int((block_length - 1) / 2)

    if chip_name == "LILLIPUT":
        # FIXME: Ask Rosenstein if we should window by image lengths rather than do this wacky thing
        shift = myshift(5, 80, block_center, upsample_ratio)
        window1d = np.abs(np.hanning(30*upsample_ratio))
        window2d = np.sqrt(np.outer(window1d,window1d)) 
        window2d = np.pad(window2d,25*upsample_ratio)  
        window2d = window2d[shift:(shift-82), shift:(shift-82)] # TODO: ASK ROSENSTEIN WHAT THESE 82 MEAN
        return window2d
    else:
        # This is going to be 512 by 256, or cropped to 492 : 236
        shift_row = myshift(5, 492, block_center, upsample_ratio)
        shift_col = myshift(5, 236, block_center, upsample_ratio)

        window_row = np.abs(np.hanning(150 * upsample_ratio))
        window_col = np.abs(np.hanning(70 * upsample_ratio))

        window2d = np.sqrt(np.outer(window_row, window_col))
        window2d = np.pad(window2d, 100*upsample_ratio)          # TODO: Is there a better way that we can do this?
        window2d = window2d[shift_row:(shift_row-492), shift_col:(shift_col-236)] # TODO: ASK ROSENSTEIN WHAT THESE 82 MEAN
        print("Window Shape:", window2d.shape)

        return window2d

def power_snr(amp):
    return (20 * np.log10(amp**2))

def get_spatial_snr(img):
    # Normalize 
    norm_ref = np.zeros(img.shape)
    norm_img = cv2.normalize(img, norm_ref, 0, 255, cv2.NORM_MINMAX)
    
    # Calculate SNR from mean and std deviation
    snr = power_snr(np.mean(norm_img) / np.std(norm_img))

    return snr

def normalize_img(img):
    norm_ref = np.zeros(img.shape)
    norm_img = cv2.normalize(img, norm_ref, 0, 255, cv2.NORM_MINMAX)

    return norm_img

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