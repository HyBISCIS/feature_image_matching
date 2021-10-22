import csv
import sys
import argparse
import numpy as np
import scipy.io as scio

# Import matplotlib
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Import skimage packages
from skimage import io, filters, feature, img_as_float32
from skimage.transform import rescale
from skimage.color import rgb2gray

# Import Open CV2
import cv2

# Import other python packages
import feature_match as fm
 
filepath = 'data/f0113/F0113_10012021_initial_BF.tif'

imp_c = cv2.imread(filepath)

# Convert to grayscale and normalize
imp = cv2.cvtColor(imp_c, cv2.COLOR_BGR2GRAY)

mask = cv2.inRange(imp, 80, 120)

cv2.bitwise_and(imp, imp, mask=mask)

cv2.normalize(imp, imp, 0, 255, cv2.NORM_MINMAX)
imp = 255 - imp

plt.imshow(imp)
plt.show()




# Blur 
imp = cv2.GaussianBlur(imp, (7,7), 0)
imp = 255 - imp

# Adaptive Thresholding for due to different lighting regions and biofilm
imp = cv2.adaptiveThreshold(imp, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 3)
imp = 255 - imp

# Morphological Filtering
k = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
imp = cv2.morphologyEx(imp, cv2.MORPH_OPEN, k)
imp = cv2.morphologyEx(imp, cv2.MORPH_CLOSE, k)

# Erode to get thinner lines
k1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
imp = cv2.morphologyEx(imp, cv2.MORPH_ERODE, k1)

# Find Contours in order to bound rectangles
contours, hierarchy = cv2.findContours(imp, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Sort contours by their size
num_contours = len(contours)
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Take top 5 contours
new_contours = sorted_contours[1:5]


cv2.drawContours(imp_c, new_contours, -1, (0,255,0), 3)

# Show IMage
plt.imshow(imp_c)
plt.show()
