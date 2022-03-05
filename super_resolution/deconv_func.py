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

'''
analyze_h5 is a quick function to just take an h5py object
and output what the attributes are in the h5 file as well 
as display what the attributes are equal to. 

This was primarily used in order to display attribute names
for the change from Lilliput to Minerva and everything changed
names.

Input: data, an h5 file
Output: No outputs
'''
def analyze_h5(data):
    # Print all of the keys
    print(list(data.keys()))   

    # Loop through one of the keys
    for i in data.keys():
        # Print list of attributes
        a = list(data[i].attrs)
        print(a)

        # Print list of attributes and what they equal
        for j in a:
            print(j, ":", data[i].attrs[j])


        exit(0)


'''
minerva_channel_shift is a function that is used in order to fix some 
of the weird misalignments happening across channels in the raw image.

Run this function before running channel_norm as channel_norm will change
how things work.

Input: image, np array-like representing an image
Output: image_shift, np array-like that is rolled in correct way.
'''
def minerva_channel_shift(image):
    image_shift = np.zeros_like(image)

    # Roll image over by 10 pixels for each of the channels
    for s in range(8):
        image_shift[:,s*32:(s+1)*32] = np.roll(image[:,s*32:(s+1)*32], 10)

    return image_shift

'''
channel_norm is a function that takes in an image and that normalizes
the channels so that we don't get that weird banding happening across 
channels.

At the moment, it is not as good as it could be as there is still apparent
banding happening kernels with large distances.

Input: myimage, np array-like
Output: image_cal, np array-like that is fixed 
'''
def channel_norm(myimage):
    normrows = range(200,300)
    image_cal = myimage.copy()
    im_dim = image_cal.shape

    # Normalize by Channel
    for ch in range(8):
        # Define bounds for channel and get coeff
        bot = (ch*32)
        top = ((ch+1) * 32)
        coeff = 1 / np.mean(myimage[:, bot:top])

        # Break if out of bounds
        if (top > im_dim[1]):
            break
        
        # Apply Coefficient
        image_cal[:,bot:top] = image_cal[:,bot:top] * coeff

    return image_cal 

'''
get_area_around is a function that takes in an image and interest point and 
gets the number of pixels to each side of the interest point * upsample_ratio
and then outputs the cropped image.

Input: img, np array-like representing image
Input: interest_point, tuple representing (row, col)
Input: radius, integer of number of pixels
Input: upsample_ratio, integer representing upsampling ratio
'''
def get_area_around(img, interest_point, radius, upsample_ratio):
    new_rad = radius * upsample_ratio
    x = (interest_point[0] - new_rad, interest_point[0] + new_rad)
    y = (interest_point[1] - new_rad, interest_point[1] + new_rad)

    return img[x[0]:x[1], y[0]:y[1]]

'''
low_salt_interpolate is a quick fix function for one row and one 
col that were messed up in the low salt dataset for Minerva. 

Input: img, an np array-like representing the image
Output: img, np array-like representing fixed image
'''
def low_salt_interpolate(img):
    # Row 502, col 22 for low salt, corrupted, so just do easy interpolation
    im_size = img.shape

    # Fix Row
    for i in range(im_size[1]):
        img[502, i] = (img[503,i] + img[501,i]) / 2

    # Fix Col
    for j in range(im_size[0]):
        img[j, 22] = (img[j,21] + img[j, 23]) / 2

    return img
        

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
getimage is the bread and butter function which loads images, gets rid 
of weird outliers, and shifts things if necessary.

Input: mydata, an h5py file descriptor
Input: k, a string representing the key to the image
Input: upsample_ratio, integer representing upsample ratio to apply to image
Input: im, a string representing the image field attribute in h5 file
Input: block_length, the number of impedance images in one column
Input: chip_name, string representing chip name: "MINERVA" or "LILLIPUT"
Input: low_salt, boolean representing if working on low_salt concentration images and need interpolation
Input: int_order, interpolation order of rescaling function call
Input: shiftrow and shiftcol required for shift sum calculations.

Output: myimage, an np array-like of image
Output: myimage_shifted, np array-like shifted for shift_sum
'''
def getimage(mydata,k,upsample_ratio,im,block_length,chip_name,low_salt,int_order,shiftrow=5,shiftcol=5):
    myimage = mydata[k][im][:]

    if low_salt:
        myimage = low_salt_interpolate(myimage)
        myimage = minerva_channel_shift(myimage)

    if chip_name == "MINERVA":
        myimage = channel_norm(myimage)

    myimage = cropimage(myimage, (block_length, block_length))
    block_center = int((block_length - 1) / 2)

    # remove outlier pixels
    for rep in range(2):
        mymedian=np.median(myimage)
        mymean=np.mean(myimage)
        mystd=np.std(myimage)
        myimage[np.abs(myimage-mymean)>4*mystd] = mymean
        mystd=np.std(myimage)   #[np.abs(myimage-mymedian)<4*mystd])
        
    # Rescaling, and optional interpolation
    myimage = rescale(myimage,upsample_ratio,order=int_order)       

    # Shift if doing shift sum
    if chip_name == "LILLIPUT":
        print(shiftrow, shiftcol)
        shift_row = myshift(shiftrow, 80, block_center, upsample_ratio)
        shift_col = myshift(shiftcol, 80, block_center, upsample_ratio)
        print("Image Shape:", myimage.shape)
        print("Shift Range: {} , {}".format(shift_row, shift_row-82))
        myimage_shift = myimage[shift_row:(shift_row-82), shift_col:(shift_col-82)]
        print("Shifted Image Shape:", myimage.shape)
    elif chip_name == "MINERVA":
        # This is going to be 512 by 256, or cropped to 492 : 236
        shift_row = myshift(shiftrow, 492, block_center, upsample_ratio)
        shift_col = myshift(shiftcol, 236, block_center, upsample_ratio)

        myimage_shift = myimage[shift_row:(shift_row-492), shift_col:(shift_col-236)] #FIXME: Weird fix to get 492 , 236 here


    return myimage, myimage_shift
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''
Bread and butter function for the linear deconvolution method. 
This is the function that takes in the raw kernel offset capacitance
image, a reference image, and the windowing function and determines 
the kernel to get from the raw image to the reference image as well 
as applies it to the raw image and windows. This is discussed in 
more detail in the super_resolution reconstruction section of the 
paper.

Input: myimage, np array-like of raw image
Input: output_f, reference image that is highpassed and brought into
       the frequency domain.
Input: upsample_ratio, an integer representing upsampling ratio from raw img
Input: 2D windowing function to apply to the kernel.

Output: myimage_filtered, np array-like of raw img with filter applied

TODO: See whether or not I can pass the window2d in the frequency domain. This
uses approximately three extra fft/ifft functions, which would significantly
increase the amount of time things take to run as it is needed for every single
linear_filter
'''
def linear_filter(myimage, output_f, upsample_ratio, window2d):
    inputimage = myimage - gaussian(myimage,sigma=10*upsample_ratio)
    input_f = np.fft.rfft2(inputimage)
    
    kernel_f = output_f / input_f         # do the deconvolution
    kernel = np.fft.irfft2(kernel_f)      # inverse Fourier transform
    kernel = np.fft.fftshift(kernel)      # shift origin to center of image

    kernel *= window2d 
    
    kernel_smoothed = gaussian(kernel,sigma=0.5*upsample_ratio)

    myimage_filtered = np.fft.fftshift(np.fft.irfft2(input_f * np.fft.rfft2(kernel)))

    return myimage_filtered, kernel_smoothed

'''
minmaxnorm is a function to apply min-max normalization on any array-like
and get things in terms of 0 - 1.0 in float64

Input: x, an np array-like
Output: min-maxed normalization applied on x
'''
def minmaxnorm(x):
    return np.nan_to_num((x-np.min(x))/(np.max(x)-np.min(x)))

'''
cropimage is a function that crops images based off of the
block size. That is, when we increase the block size, the amount 
of pixels useful to us decreases at the edges of the image since
we are limited by the distance we are measuring across. We crop
all images in order to accomodate that disparity

Input: im, an np-array like raw image
Input: block_sz, a tuple representing the (num_rows, num_cols) of
       different impedance images that we are dealing with.
'''
def cropimage(im, block_sz):
    crop_step = (block_sz[0]-1, block_sz[1]-1)
    im_size = im.shape

    # Obtain crop bounds
    x_bound = (crop_step[0], (im_size[0]-crop_step[0]))
    y_bound = (crop_step[1], (im_size[1]-crop_step[1]))

    cropped = im[x_bound[0]:x_bound[1],y_bound[0]:y_bound[1]]

    return cropped

'''
myshift is a function that applies a shift based off of the size of the
image, the upsample rate, and the crop length. It is used in order to 
determine how to shift things when we have upsampled things.

This function is only used for the shift-sum functions

Input: x, row or col offset of an image
Input: crop_length, an integer representing crop length
Input: block_center, center of all impedance images.
Input: upsample_ratio, upsample ratio integer

Output: shifted_index, an integer representing the amount to shift an image
'''
def myshift(x, crop_length, block_center, upsample_ratio):
    half_upsample = upsample_ratio / 2
    half_crop_length = crop_length / 2

    shifted_index = half_crop_length + ((block_center - x) * half_upsample)

    return int(shifted_index)

'''
find_mirrored is a function used when trying to enforce mirrored 
kernels in the linear deconvolution method. Essentially, given a row 
and col offset, we are able to determine the three other mirrored 
complementary tuples representing the other kernels in (row, col)
offset.

Input: pt, represents top left quarter row,col offsets
Input: center, represents center of impedance array. Typically (0,0) on 
       MINERVA, or (5,5) on Lilliput chip
Output: list containing the three corresponding mirrored images
'''
def find_mirrored(pt, center):
    # Names assume given from point from (0,0) to (5,5)

    # Assume that maxrow and maxcol will always be odd, so that we can find center
    row_offset = pt[0] - center[0]
    col_offset = pt[1] - center[1]

    top_right = (center[0]+row_offset, center[1]-col_offset)
    bot_left = (center[0]-row_offset, center[1]+col_offset)
    bot_right = (center[0]-row_offset, center[1]-col_offset)

    return [top_right, bot_left, bot_right]

'''
create_window is a function that takes in the chip parameters and
creates a full size window for the image. This is a 2D window.

Input: chip_name, string representing chip name. At the moment, there
       is just "MINERVA" and "LILLIPUT".
Input: block_length, int representing num of rows / cols (assume square)
       of impedance image offsets. In our case with 120 images, it is 11
Input: upsample_ratio, int representing upsamplee ratio.

TODO: I have been having issues with this function for Minerva,
      meaning that full-chip reconstructions are still not there.
      I am sure that it won't be that bad to fix, but that is 
      something to think about.

'''
def create_window(chip_name, block_length, upsample_ratio):
    block_center = int((block_length - 1) / 2)

    if chip_name == "LILLIPUT":
        shift = myshift(5, 80, block_center, upsample_ratio)
        window1d = np.abs(np.hanning(30*upsample_ratio))
        print("Window 1D Length:", window1d.shape)
        window2d = np.sqrt(np.outer(window1d,window1d)) 
        print("Window 2D Shape:", window2d.shape)
        window2d = np.pad(window2d,25*upsample_ratio)  
        print("Window 2D Padded Shape:", window2d.shape)
        window2d = window2d[shift:(shift-82), shift:(shift-82)] 
        print("Window Shifted Shape:", window2d.shape)
        return window2d
    else:
        # This is going to be 512 by 256, or cropped to 492 : 236
        shift_row = myshift(5, 492, block_center, upsample_ratio)
        shift_col = myshift(5, 236, block_center, upsample_ratio)
        # (7600, 6240)
        # (7872, 3776)

        window_row = np.abs(np.hanning(175 * upsample_ratio))
        window_col = np.abs(np.hanning(90 * upsample_ratio))

        window2d = np.sqrt(np.outer(window_row, window_col))

        padding_width_row = 160*upsample_ratio - 24
        padding_width_col = 75*upsample_ratio - 32
        padding = ((padding_width_row,padding_width_row),(padding_width_col,padding_width_col))

        window2d = np.pad(window2d, padding)          
        window2d = window2d[shift_row:(shift_row-492), shift_col:(shift_col-236)] 

        return window2d

'''
Function to take amplitude and convert it to power_snr

Input: amp, a float representing the amplitude of something
'''
def power_snr(amp):
    return (20 * np.log10(amp**2))

'''
From image, normalize to 0 - 255 and then calculate the power SNR
based off of (mean / std) that is used for typical images.

Input: img, an np array-like representing the image at hand
'''
def get_spatial_snr(img):
    # Normalize 
    norm_ref = np.zeros(img.shape)
    norm_img = cv2.normalize(img, norm_ref, 0, 255, cv2.NORM_MINMAX)
    
    # Calculate SNR from mean and std deviation
    snr = power_snr(np.mean(norm_img) / np.std(norm_img))

    return snr

'''
normalize_img is a function that normalizes an image to 0 - 255 range.
'''
def normalize_img(img):
    norm_ref = np.zeros(img.shape)
    norm_img = cv2.normalize(img, norm_ref, 0, 255, cv2.NORM_MINMAX)

    return norm_img

'''
Quick and dirty function that is a lookup table to easily plot
iterable things of 4.
'''
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