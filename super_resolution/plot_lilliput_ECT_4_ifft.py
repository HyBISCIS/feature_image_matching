#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 16:24:05 2021
@author: jacobrosenstein
"""


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
import scipy.signal
import imageio
from skimage.transform import warp, PiecewiseAffineTransform
from skimage.registration import optical_flow_tvl1


logdir = r"data/bead_20um"
logfile = r"phase2_sweep_11x11_block.h5"

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def Get_Data(logname, exp_name, dataname='image'):
    hf = h5py.File(logname, 'r')
    grp_data = hf.get(exp_name)
    image = grp_data[dataname][:]
    return image    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
def Get_Attr(logname, exp_name, attrname):
    hf = h5py.File(logname, 'r')
    grp_data = hf.get(exp_name)
    return grp_data.attrs[attrname]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
def Get_Time(logname, exp_name):
    return datetime.strptime(Get_Attr(logname, exp_name, 'timestamp'), "%Y%m%d_%H%M%S")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
def Get_List(logname,filterstring=None,sortby=None):
    hf = h5py.File(logname, 'r')
    base_items = list(hf.items())
    grp_list = []
    for i in range(len(base_items)):
        grp = base_items[i]
        grp_list.append(grp[0])
    if filterstring is not None:
        grp_list = [x for x in grp_list if filterstring in x]
    if sortby is 'time':
        grp_list = sorted(grp_list,key=lambda x: Get_Time(logname,x))
    return grp_list	
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    

mydata = h5py.File(os.path.join(logdir,logfile),'r')

myphoto = image.imread(os.path.join(logdir,'Microscope_bead_20u_0639pm.bmp'))
myphoto=myphoto[200:1130,1100:2020,:]

def minmaxnorm(x):
    return np.nan_to_num((x-np.min(x))/(np.max(x)-np.min(x)))

def cropimage(im):
    return im[10:90,10:90]

def myshift(x):
    return (40+(5-x)*8)

upsample_ratio=16
window1d = np.abs(np.hanning(30*upsample_ratio))
window2d = np.sqrt(np.outer(window1d,window1d))
window2d = np.pad(window2d,25*upsample_ratio)
window2d = window2d[myshift(5):(myshift(5)-82), 
                    myshift(5):(myshift(5)-82)]
print(window2d.shape)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def getimage(k,shiftrow=5,shiftcol=5):

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




compositeimage = None
compositeimage2 = None

fig1,ax1=plt.subplots(11,11,figsize=(8,8))

myframes=[]

sortedkeys = sorted(mydata.keys(), key=lambda k: int(mydata[k].attrs['R'])*100+int(mydata[k].attrs['C']))

k_reference = [x for x in sortedkeys if '6250kHz' in x and int(mydata[x].attrs['R'])==3 and int(mydata[x].attrs['C'])==5][0]
reference_image,refmedian,refstd = getimage(k_reference)

mymedian,mystd=None,None


allimages=[]

for ik,k in enumerate(sortedkeys):
    
    if('6250kHz' not in k):
        continue
    
    print(ik, ' ', k)
    
    myrow = int(mydata[k].attrs['R'])
    mycol = int(mydata[k].attrs['C'])
    
    r = np.sqrt((5-myrow)**2 + (5-mycol)**2)
    theta = np.angle((5-mycol)+(5-myrow)*1j)
    print('row/col',myrow,mycol,'theta',theta)


    # get the image
    myimage,mymedian,mystd = getimage(k,shiftrow=myrow,shiftcol=mycol)
    
    ################
    # grid of images
    im=ax1[myrow,mycol].imshow(myimage,
          vmin=mymedian-5*mystd,vmax=mymedian+5*mystd,
          cmap='viridis')
    ax1[myrow,mycol].set_xticks([])
    ax1[myrow,mycol].set_yticks([])
    
    
    ################
    # animation, in single frame
    fig2,ax2=plt.subplots(nrows=2,ncols=4,figsize=(18,9))
    
    im0 = ax2[0,0].imshow(myphoto)
    ax2[0,0].set_xticks([])
    ax2[0,0].set_yticks([])
    ax2[0,0].set_title('microscope')

    im=ax2[0,1].imshow(myimage,
          vmin=mymedian-5*mystd,vmax=mymedian+5*mystd,
          cmap='Blues')
    ax2[0,1].set_title("impedance R%u C%u" % (myrow,mycol))
    
    
    im=ax2[0,2].imshow(myimage,
          vmin=mymedian-5*mystd,vmax=mymedian+5*mystd,
          cmap='Blues')
    ax2[0,2].set_title('shift/re-center')
    print(myrow,mycol,theta*180.0/np.pi)



    # ------------ BEGINLINEAR FILTER ---------------------------
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

    ax2[1,0].imshow(outputimage,
                      #vmin=np.median(outputimage)-2*np.std(outputimage),
                      #vmax=np.median(outputimage)+3*np.std(outputimage),
                      cmap='Blues_r')
    ax2[1,0].set_title('reference image')

    ax2[1,1].imshow(inputimage,
                      #vmin=np.median(inputimage)-2*np.std(inputimage),
                      #vmax=np.median(inputimage)+3*np.std(inputimage),
                      cmap='Blues_r')
    ax2[1,1].set_title('offset ECT image')

    ax2[1,2].imshow(myimage_filtered,
                      #vmin=np.median(myimage_filtered)-2*np.std(myimage_filtered),
                      #vmax=np.median(myimage_filtered)+3*np.std(myimage_filtered),
                      cmap='Blues_r')
    ax2[1,2].set_title('after linear filter')


    
    # ------------ END LINEAR FILTER ---------------------------

    if compositeimage is None:
        compositeimage = np.zeros_like(myimage)
        compositeimage2 = np.zeros_like(myimage)

    compositeimage = compositeimage + myimage
    compositeimage2 = compositeimage2 + myimage_filtered
    
    im=ax2[0,3].imshow(compositeimage,
                      #vmin=np.median(compositeimage)-2*np.std(compositeimage),
                      #vmax=np.median(compositeimage)+3*np.std(compositeimage),
                      cmap='Blues_r')
    ax2[0,3].set_title('composite of shifted raw')

    im=ax2[1,3].imshow(compositeimage2,
                      #vmin=np.median(compositeimage2)-2*np.std(compositeimage2),
                      #vmax=np.median(compositeimage2)+3*np.std(compositeimage2),
                      cmap='Blues_r')
    ax2[1,3].set_title('composite of inverse filtered')
    
    # ~~~~~~~~~~~~~~~~~~~~~~~        
    fig2.canvas.draw()       # draw the canvas, cache the renderer
    imframe = np.frombuffer(fig2.canvas.tostring_rgb(), dtype='uint8')
    imframe  = imframe.reshape(fig2.canvas.get_width_height()[::-1] + (3,))
    myframes.append(imframe)
    # ~~~~~~~~~~~~~~~~~~~~~~~        

            
    plt.show()


    
    # ~~~~~~~~~~~~~~~~~~~~~~~
    allimages.append(myimage)
    # ~~~~~~~~~~~~~~~~~~~~~~~
    
    
    
    # create animation
    if 0:
        imageio.mimsave("%s_composite.gif" % (os.path.join(logdir,'plots',logfile),), myframes, fps=4)
    
    if 0:
        # create .avi video file
        imageio.mimsave("%s_5a.mp4" % (os.path.join(logdir,logfile),), myframes, fps=10)        
        

plt.show()




#outputimage = allimages[62]
#outputimage = outputimage - gaussian(outputimage,sigma=5)
fig,ax=plt.subplots(nrows=1,ncols=3,figsize=(18,6))
ax[0].imshow(reference_image,cmap='Blues_r')    
ax[0].set_title('single reference image')
ax[1].imshow(compositeimage,cmap='Blues_r')    
ax[1].set_title('super-resolution 1 (simple shift+add)')
ax[2].imshow(compositeimage2,cmap='Blues_r')    
ax[2].set_title('super-resolution 2 (inverse linear filter)')


####################################################################

# # ~~~~~~~~~~~~~~~~~~~~~~~~~~
# #from scipy import misc
# from numpy import fft

# superres=None
# upsample_ratio=16
# window1d = np.abs(np.hanning(30*upsample_ratio))
# window2d = np.sqrt(np.outer(window1d,window1d))
# window2d = np.pad(window2d,25*upsample_ratio)

# for myimage in allimages: 
#     outputimage = allimages[62]
#     inputimage = myimage
    
#     outputimage = outputimage - gaussian(outputimage,sigma=5)
#     inputimage = inputimage - gaussian(inputimage,sigma=5)

#     outputimage = rescale(outputimage,upsample_ratio)#,order=0)
#     inputimage = rescale(inputimage,upsample_ratio)#,order=0)
    
#     output_f = fft.rfft2(outputimage)
#     input_f = fft.rfft2(inputimage)
    
#     kernel_f = output_f / input_f         # do the deconvolution
#     kernel = fft.irfft2(kernel_f)      # inverse Fourier transform
#     kernel = fft.fftshift(kernel)      # shift origin to center of image
#     kernel *= window2d
#     kernel /= kernel.max()             # normalize gray levels
    
    
#     kernel_smoothed = gaussian(kernel,sigma=0.5*upsample_ratio)
#     kernel_smoothed /= kernel_smoothed.max()
    
#     #crosscorr = scipy.signal.correlate2d(inputimage,outputimage,mode='same')
#     #crosscorr = np.flip(crosscorr)
#     #crosscorr /= crosscorr.max()
#     #crosscorr *= window2d
    
    
#     fig,ax=plt.subplots(nrows=3,ncols=3,figsize=(12,12))
#     ax[0,0].imshow(inputimage,cmap='Blues_r')     
#     ax[0,0].set_title('offset ECT image')
    
#     ax[0,1].imshow(outputimage,cmap='Blues_r')    
#     ax[0,1].set_title('reference image')
    
#     #ax[0,2].imshow(crosscorr[20:60,20:60])
#     #ax[0,2].set_title('ECT image')
    
#     ax[1,0].imshow(kernel[(20*upsample_ratio):(60*upsample_ratio),(20*upsample_ratio):(60*upsample_ratio)],cmap='Greys_r')
#     ax[1,0].set_title('kernel (raw)')
#     #ax[1,1].imshow(kernel_smoothed[(20*upsample_ratio):(60*upsample_ratio),(20*upsample_ratio):(60*upsample_ratio)],cmap='Greys_r')
#     ax[1,1].imshow(kernel_smoothed,cmap='Greys_r')
#     ax[1,1].set_title('kernel (smoothed)')
#     #ax[1,2].plot(kernel_smoothed[40,:])
#     #ax[1,2].plot(crosscorr[40,:])
    
#     #image_destretched = scipy.signal.convolve2d(inputimage,kernel,mode='same')
#     image_destretched = fft.fftshift(fft.irfft2(input_f * fft.rfft2(kernel)))
#     if superres is None:
#         superres = np.zeros_like(inputimage)

#     superres = superres + image_destretched
        
    
    
#     ax[2,0].imshow(image_destretched,cmap='Blues_r')
#     ax[2,0].set_title('image de-stretched with kernel')
#     ax[2,1].imshow(superres,cmap='Blues_r')    
#     ax[2,1].set_title('super-resolution image')
    
#     #ax[2,1].imshow(scipy.signal.convolve2d(inputimage,crosscorr,mode='same'),cmap='Blues_r')


    
#     #ax[2,2].imshow(scipy.signal.convolve2d(outputimage,crosscorr,mode='same'),cmap='Blues_r')
    
#     plt.show()


# outputimage = allimages[62]
# outputimage = outputimage - gaussian(outputimage,sigma=5)
# fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(12,6))
# ax[0].imshow(outputimage,cmap='Blues_r')    
# ax[0].set_title('reference image')
# ax[1].imshow(compositeimage2,cmap='Blues_r')    
# ax[1].set_title('super-resolution image')

# # ~~~~~~~~~~~~~~~~~~~~~~~~~~