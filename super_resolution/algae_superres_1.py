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
from skimage.transform import resize,rescale
from skimage.filters import gaussian
from skimage import color, data, restoration
import scipy.signal
import imageio
from skimage.transform import warp, PiecewiseAffineTransform
from skimage.registration import optical_flow_tvl1

logdir = r"super_resolution"
logfile = r"cosmarium_and_micrasterias_wet_phase2_sweep_11x11_block_multiple_freq.h5"

mydata = h5py.File(os.path.join(logdir,logfile),'r')

myphoto = image.imread(os.path.join(logdir,'cosmarium_and_micrasterias_wet_phase2_sweep_11x11_block_4.bmp'))
myphoto=myphoto[900:1900,1100:2100,:]

def minmaxnorm(x):
    return np.nan_to_num((x-np.min(x))/(np.max(x)-np.min(x)))

def rotate(point, origin, radians):
    #radians = np.deg2rad(degrees)
    x,y = point
    offset_x, offset_y = origin
    cos_rad = np.cos(radians)
    sin_rad = np.sin(radians)
    qx = offset_x + cos_rad*(x-offset_x) + sin_rad*(y-offset_y)
    qy = offset_y + -sin_rad*(x-offset_x) + cos_rad*(y-offset_y)
    return qx, qy

def cropimage(im):
    return im[10:90,10:90]

allzooms = [[0,80,0,80],
            [35,55,35,55],
            [45,75,40,70],
            [0,70,40,70],]

#allzooms = [[45,75,40,70],]
allzooms = [[40,80,40,80],]
#allzooms = [[0,70,0,80],]

for xmin,xmax,ymin,ymax in allzooms:
    
    compositeimage = None
    compositeimage2 = None
    
    print('zoom',xmin,xmax,ymin,ymax)
    fig1,ax1=plt.subplots(11,11,figsize=(16,16))
    
    myframes=[]
    
    sortedkeys = sorted(mydata.keys(), key=lambda k: int(mydata[k].attrs['R'])*100+int(mydata[k].attrs['C']))
    
    k_reference = [x for x in sortedkeys if int(mydata[x].attrs['R'])==5 and int(mydata[x].attrs['C'])==5][0]
    
    mymedian,mystd=None,None
    
    allimages=[]
    
    for ik,k in enumerate(sortedkeys):
        
        #if(mydata[k].attrs['V_STDBY'] != 250):
        #    continue
    
        if('6250kHz' not in k):
            continue
        
        print(ik, ' ', k)
        
        myrow = int(mydata[k].attrs['R'])
        mycol = int(mydata[k].attrs['C'])
        
        r = np.sqrt((5-myrow)**2 + (5-mycol)**2)
        theta = np.angle((5-mycol)+(5-myrow)*1j)
        if not (r>2 and r<5):
            continue

        print('row/col',myrow,mycol,'theta',theta)


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
        #myimage = myimage - myreference
        #myimage = np.divide(myimage,myreference) - 1
        #myimage[np.isinf(myimage)] = 0
        #myimage = np.nan_to_num(myimage)
        
        mymedian=np.median(myimage)
        mystd=np.std(myimage)
        myimage[np.abs(myimage-mymedian)>5*mystd] = mymedian
        mystd=np.std(myimage)   #[np.abs(myimage-mymedian)<4*mystd])
        
        ################
        # grid of images
        im=ax1[myrow,mycol].imshow(myimage[xmin:xmax,ymin:ymax],
              vmin=mymedian-5*mystd,vmax=mymedian+5*mystd,
              cmap='viridis')
        ax1[myrow,mycol].set_xticks([])
        ax1[myrow,mycol].set_yticks([])
        rect = patches.Rectangle((mycol+0.5,myrow+0.5),
                              1,1,
                              linewidth=1,
                              edgecolor='r',
                              facecolor='none')
        #ax_z1[myrow,mycol].add_patch(rect)
        arr = patches.Arrow((xmax-xmin)*4,(ymax-ymin)*4,
                            (mycol-5)*8,(myrow-5)*8,
                            width=10.0,facecolor='r',edgecolor='r')
        #ax1[myrow,mycol].add_patch(arr)
        
        
        ################
        # animation, in single frame
        fig2,ax2=plt.subplots(nrows=2,ncols=4,figsize=(18,9))
        
        im0 = ax2[0,0].imshow(myphoto)
        ax2[0,0].set_xticks([])
        ax2[0,0].set_yticks([])
        ax2[0,0].set_title('microscope')
        rect = patches.Rectangle((200+(ymin)*8.25,200+(xmin)*8.25),
                      (ymax-ymin)*8.25,(xmax-xmin)*8.25,
                      linewidth=1,
                      edgecolor='r',
                      facecolor='none')
        ax2[0,0].add_patch(rect)

        imrescaled = rescale(myimage[xmin:xmax,ymin:ymax],16,order=0)

        im=ax2[0,1].imshow(imrescaled,
              vmin=mymedian-5*mystd,vmax=mymedian+5*mystd,
              cmap='Blues')
        ax2[0,1].set_title("impedance R%u C%u" % (myrow,mycol))
        arr = patches.Arrow((xmax-xmin)*8,(ymax-ymin)*8,
                            (mycol-5)*16,(myrow-5)*16,
                            width=20.0,facecolor='r',edgecolor='r')
        ax2[0,1].add_patch(arr)
        
        def myshift(x):
            return (40+(5-x)*8)
        imrescale_and_shift = imrescaled[myshift(myrow):(myshift(myrow)-81),
                                         myshift(mycol):(myshift(mycol)-81)]
        im=ax2[0,2].imshow(imrescale_and_shift,
              vmin=mymedian-5*mystd,vmax=mymedian+5*mystd,
              cmap='Blues')
        ax2[0,2].set_title('shift/re-center')
        print(myrow,mycol)


        # affine warp transform
        rows, cols = imrescale_and_shift.shape[0], imrescale_and_shift.shape[1]

        
        src = [[0,0],[0,cols-1],[rows-1,0],[rows-1,cols-1]]
        dst = [[0,0],[0,cols-1],[rows-1,0],[rows-1,cols-1]]
        
        obj_ctr = [150,240]
        obj_w = 200
        obj_bb = [[obj_ctr[0]-obj_w/2,obj_ctr[1]-obj_w/2],
                    [obj_ctr[0]-obj_w/2,obj_ctr[1]+obj_w/2],
                    [obj_ctr[0]+obj_w/2,obj_ctr[1]-obj_w/2],
                    [obj_ctr[0]+obj_w/2,obj_ctr[1]+obj_w/2]]
        
        src.extend(obj_bb)
        
        stretch_scale = +20
        obj_bb_stretch=obj_bb.copy()



        #rotateby=0
        rotateby=(np.mod(np.pi/4-theta-0.01,np.pi/2)-np.pi/4)/2
        print('theta',theta,'rotate by',rotateby)

        obj_bb_stretch[0] = rotate(obj_bb_stretch[0],obj_ctr,rotateby)
        obj_bb_stretch[1] = rotate(obj_bb_stretch[1],obj_ctr,rotateby)
        obj_bb_stretch[2] = rotate(obj_bb_stretch[2],obj_ctr,rotateby)
        obj_bb_stretch[3] = rotate(obj_bb_stretch[3],obj_ctr,rotateby)
         
        dst.extend(obj_bb_stretch)
                

        src = np.array(src)
        dst = np.array(dst)

        tform = PiecewiseAffineTransform()
        tform.estimate(src, dst)
        
        out_rows = rows
        out_cols = cols
        warp_out = warp(imrescale_and_shift, tform, output_shape=(out_rows, out_cols))
        
        ax2[1,1].imshow(imrescale_and_shift,
                        vmin=mymedian-5*mystd,vmax=mymedian+5*mystd,
                        cmap='Blues')
        ax2[1,1].plot(src[:, 0], src[:, 1], 'xr')
        ax2[1,1].plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], 'ok',markersize=10)
        ax2[1,1].axis((0, out_cols, out_rows, 0))
        ax2[1,1].set_title('registration red:original black:warp')

        ax2[1,2].imshow(warp_out,
                        vmin=mymedian-5*mystd,vmax=mymedian+5*mystd,
                        cmap='Blues')
        #ax2[1,2].plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.k')
        ax2[1,2].axis((0, out_cols, out_rows, 0))
        ax2[1,2].set_title('after warp')



        if compositeimage is None:
            compositeimage = imrescale_and_shift
            compositeimage2 = warp_out
        else:
            compositeimage = compositeimage + imrescale_and_shift
            compositeimage2 = compositeimage2 + warp_out
            
        im=ax2[0,3].imshow(compositeimage,
                          vmin=np.median(compositeimage)-2*np.std(compositeimage),
                          vmax=np.median(compositeimage)+3*np.std(compositeimage),
                          cmap='Blues')
        ax2[0,3].set_title('composite of shifted raw')

        im=ax2[1,3].imshow(compositeimage2,
                          vmin=np.median(compositeimage2)-2*np.std(compositeimage2),
                          vmax=np.median(compositeimage2)+3*np.std(compositeimage2),
                          cmap='Blues')
        ax2[1,3].set_title('composite of warped')
        
        fig2.canvas.draw()       # draw the canvas, cache the renderer
        imframe = np.frombuffer(fig2.canvas.tostring_rgb(), dtype='uint8')
        imframe  = imframe.reshape(fig2.canvas.get_width_height()[::-1] + (3,))
        myframes.append(imframe)
                
        plt.show()
        
        # ~~~~~~~~~~~~~~~~~~~~~~~
        allimages.append(myimage)
        # ~~~~~~~~~~~~~~~~~~~~~~~
    
    #repeat the last frame
    myframes.append(imframe)
    myframes.append(imframe)
    myframes.append(imframe)
    myframes.append(imframe)
    myframes.append(imframe)
    myframes.append(imframe)
    myframes.append(imframe)
    myframes.append(imframe)
    myframes.append(imframe)
    myframes.append(imframe)
    
    # create animation
    if 0:
        imageio.mimsave("%s_%u_%u_%u_%u_warp1e.gif" % (os.path.join(logdir,logfile),
                                                    xmin,
                                                    xmax,
                                                    ymin,
                                                    ymax), myframes, fps=1)
    
    if 0:
        # create .avi video file
        imageio.mimsave("%s_%u_%u_%u_%u_51.mp4" % (os.path.join(logdir,logfile),
                                                    xmin,
                                                    xmax,
                                                    ymin,
                                                    ymax), myframes, fps=4)        
        

plt.show()