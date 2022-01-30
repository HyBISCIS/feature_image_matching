import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import re
import stat

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

### Parameters and Options ###
SHOW_MICRO_IMAGE = False
SHOW_SAMPLE_IMAGE = False
SW_FREQUENCY_LIST = ['1562kHz','25000kHz','3125kHz','50000kHz','6250kHz']
SW_FREQUENCY = SW_FREQUENCY_LIST[2]
ARRAY_PITCH = 1e-05

### Declaration of Functions

# Print out the attributes assigned to a particular image in the .h5 file
def pretty_print(h5_ob, key):
    attributes = list(h5_ob.attrs)

    all_strings = str(h5_ob) + "\n"

    # Iterate through all attributes and get their values
    for attr in attributes:
        val = h5_ob.attrs[attr]
        string = str(attr) + ": " + str(val) + "\n"
        all_strings += string

    print(all_strings)

# Determine pixel distance based off of offset
def pixel_dist(row_offset, col_offset, center, pitch):
    # Convert row, col offsets to real units
    row_real = (row_offset - center[0]) * pitch
    col_real = (col_offset - center[1]) * pitch

    # Determine distance
    dist = np.sqrt(np.square(row_real) + np.square(col_real))
    return dist
    
# Determine function that maps amplitude to distance
def determine_dist_to_amp_func(imp_data, sortedkeys, center, sw_freq, pitch):
    distances = []
    amp = []
    
    # For each offset image, gather average amplitude of the entire Image
    for k in sortedkeys:
        # Check if correct switching frequency
        if(sw_freq not in k):
            continue

        # Get image
        image = imp_data[k]['image'][:]
        row_offset = int(imp_data[k].attrs['R'])
        col_offset = int(imp_data[k].attrs['C'])

        # Do not include middle point
        if ((row_offset, col_offset)==center):
            continue

        # Average Image over both axes
        avg_amp = np.mean(image)

        # Get Distasnce
        dist = pixel_dist(row_offset, col_offset, center, pitch)

        # Deposit in list 
        distances.append(dist)
        amp.append(avg_amp)

    # Run Polynomial Regression to Fit Line
    regression = np.poly1d(np.polyfit(distances, amp, 4))
    sample_x = np.linspace(1e-5, 7e-5, 50)
    print("Polynomial Coefficients: ", regression.c)

    # Plot both 
    plt.figure(10)
    plt.plot(sample_x, regression(sample_x), color='red')
    plt.scatter(distances, amp)
    plt.title("Average Amplitude of Pixels Vs. Distances for Switching Frequency: {}".format(sw_freq))
    plt.xlabel("Distance (m)")
    plt.ylabel("Average Amplitude")

    return regression 

# ---------------------------------------------------------------------

# Define Directory and Filename of Raw EIS Images
logdir = r"data/super_res"
logfile_imp = r"cosmarium_and_micrasterias_wet_phase2_sweep_11x11_block_multiple_freq.h5"
logfile_micro = r"cosmarium_and_micrasterias_wet_phase2_sweep_11x11_block_4.bmp"
actual_log_filepath = r"log/raw_names_sorted.txt"

# Load in Data
imp_data = h5py.File(os.path.join(logdir,logfile_imp),'r')
microscope_im = cv2.imread(os.path.join(logdir, logfile_micro))
microscope_im_crop = microscope_im[900:1900,1100:2100,:]

if SHOW_MICRO_IMAGE:
    plt.figure(1)
    plt.imshow(microscope_im)
    plt.title("Raw Microscope Image")

    plt.figure(2)
    plt.imshow(microscope_im_crop)
    plt.title("Cropped Micrscope Image")

# Sort Keys and Gather Row and Column Data
sortedkeys = sorted(imp_data.keys(), key=lambda k: int(imp_data[k].attrs['R'])*100+int(imp_data[k].attrs['C']))
num_rows_off = int(imp_data[sortedkeys[len(sortedkeys) - 1]].attrs['R']) + 1
num_cols_off = int(imp_data[sortedkeys[len(sortedkeys) - 1]].attrs['C']) + 1
center_indices = (int((num_rows_off - 1) / 2), int((num_cols_off - 1) / 2)) # row, col

# Determine how big of array we will need
(im_rows, im_cols) = imp_data[sortedkeys[0]]['image'][:].shape
new_r = im_rows * num_rows_off
new_c = im_cols * num_cols_off
upscale_size = (new_r, new_c)
upscaled_img = np.ones(upscale_size, dtype='float64')



poly_model = determine_dist_to_amp_func(imp_data, sortedkeys, center_indices, SW_FREQUENCY, ARRAY_PITCH)

if SHOW_SAMPLE_IMAGE:
    img = imp_data[sortedkeys[30]]['image'][:]
    plt.figure(3)
    plt.imshow(img, cmap='gray')

for k in sortedkeys:
    # Check if correct switching frequency
    if(SW_FREQUENCY not in k):
        continue

    # Get image
    image = imp_data[k]['image'][:]
    row_offset = int(imp_data[k].attrs['R'])
    col_offset = int(imp_data[k].attrs['C'])

    # Normalize Image
    dist = pixel_dist(row_offset, col_offset, center_indices, ARRAY_PITCH)

    if (row_offset, col_offset != center_indices):
        image = image / poly_model(dist)
    else:
        image = image + 1.0

    for i in range(im_rows):
        for j in range(im_cols):
            amplitude = image[i][j]
            new_row = (i * num_rows_off) + row_offset
            new_col = (j * num_cols_off) + col_offset
            upscaled_img[new_row][new_col] = amplitude

# Try and get rid of the edge things by cropping to 1000x1000 for now
upscaled_img_crop = upscaled_img[0:1000, 0:1000]

# Normalize Image
kernel_size = (3,3)
upscaled_img_crop = cv2.GaussianBlur(upscaled_img_crop, kernel_size, 0)
norm_image = cv2.normalize(upscaled_img_crop, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) + 100

# Look at Frequnecy Domain
f = np.fft.fft2(upscaled_img_crop)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
plt.figure(4)
plt.subplot(121),plt.imshow(upscaled_img_crop, cmap = 'gray')
plt.title('Input Image')
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum')


plt.figure(5)
plt.imshow(norm_image, cmap='gray')

plt.figure(6)
plt.imshow(microscope_im_crop, cmap='gray')
plt.show()



#print(sortedkeys)


#prettyprint(imp_data[sortedkeys[0]], sortedkeys[0])