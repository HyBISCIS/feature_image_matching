import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from brightfield_helper import show_hist, show_equal

# MACROS FOR SHOWING IMAGES OF WHAT IS HAPPENING
show_histograms = False
show_equalized = False
draw_matches = True

# Import Clean and TestBrightfield Image
main_bf_c = cv2.imread("data/brightfield_main_minerva.tif")
test_bf_c = cv2.imread("data/f0113/F0113_10012021_initial_BF.tif")

# Convert to grayscale
main_bf = cv2.cvtColor(main_bf_c, cv2.COLOR_BGR2GRAY)
test_bf = cv2.cvtColor(test_bf_c, cv2.COLOR_BGR2GRAY)

# f = np.fft.fft2(test_bf)
# fshift = np.fft.fftshift(f)
# magnitude_spectrum = 20*np.log(np.abs(fshift))
# plt.subplot(121),plt.imshow(test_bf, cmap = 'gray')
# plt.title('Input Image')
# plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('Magnitude Spectrum')
# plt.show()

# f = np.fft.fft2(main_bf)
# fshift = np.fft.fftshift(f)
# magnitude_spectrum = 20*np.log(np.abs(fshift))
# plt.subplot(121),plt.imshow(main_bf, cmap = 'gray')
# plt.title('Input Image')
# plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('Magnitude Spectrum')
# plt.show()

kernel_size = 5
main_bf = cv2.GaussianBlur(main_bf, (kernel_size, kernel_size), 0)
#main_bf = main_bf - main_bf_blur
test_bf = cv2.GaussianBlur(test_bf, (kernel_size, kernel_size), 0)
#test_bf = test_bf - test_bf_blur

# Equalize Histogram to get Same Brightness and Contrast of Both Images
show_hist(main_bf, test_bf, show_histograms)
equ_main = cv2.equalizeHist(main_bf)
equ_test = cv2.equalizeHist(test_bf)
show_equal(main_bf, equ_main, test_bf, equ_test, show_equalized)

# Apply Feature Match Algorithm between Brightfield Images
sift_detect = cv2.SIFT_create()
kp1,d1 = sift_detect.detectAndCompute(equ_main, None)
kp2,d2 = sift_detect.detectAndCompute(equ_test, None)

FLAN_INDEX_KDTREE = 0
index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(d1, d2, k=2)

# Determine Uniqueness of Flann Matches using ratio test
RATIO = 0.6
unique_matches = []
for a,b in matches:
    if a.distance < RATIO * b.distance:
        unique_matches.append(a)

if draw_matches:
    unique_match2 = []

    for a in unique_matches:
        unique_match2.append([a])

    img3 = cv2.drawMatchesKnn(main_bf_c,kp1,test_bf_c,kp2,unique_match2,None,flags=2)
    plt.imshow(img3)
    plt.show()

# Define Homography Transform using RANSAC to get from one image to other
# Note: From https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
src_pts = np.float32([kp1[m.queryIdx].pt for m in unique_matches]).reshape(-1,1,2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in unique_matches]).reshape(-1,1,2)

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

# All this stuff is just to have
matchesMask = mask.ravel().tolist()
h,w = main_bf.shape
pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)
img2 = cv2.polylines(test_bf,[np.int32(dst)],True,255,3, cv2.LINE_AA)

fig = plt.figure(2)
fig.add_subplot(1,2,1)
plt.imshow(main_bf)
fig.add_subplot(1,2,2)
plt.imshow(img2)
plt.show()

# With Matrix, we can find chip corner, knowing where it is on clean brightfield
# TLC, TRC, BLC, BRC
CLEAN_BRIGHTFIELD_CORNERS = [(241, 765), (1827, 802), (224, 1561), (1810, 1596)]
NEW_CORNERS = [(0,0), (1600, 0), (0, 800), (1600, 800)]

# Identify chip corners on experimental setup
bf_test_corners = cv2.perspectiveTransform(np.float32(CLEAN_BRIGHTFIELD_CORNERS).reshape(-1,1,2), M)
bf_test_corners = sum(np.int32(bf_test_corners.round()).tolist(), [])

print(bf_test_corners)

for (x,y) in bf_test_corners:
    test_bf_c = cv2.circle(test_bf_c, (x,y), radius=5, color=(255, 0, 0), thickness=-1)

plt.imshow(test_bf_c)
plt.show()

for (x,y) in CLEAN_BRIGHTFIELD_CORNERS:
    main_bf_c = cv2.circle(main_bf_c, (x,y), radius=0, color=(255, 0, 0), thickness=-1)

plt.imshow(main_bf_c)
plt.show()

M_clean_to_sense = cv2.getPerspectiveTransform(np.float32(CLEAN_BRIGHTFIELD_CORNERS),np.float32(NEW_CORNERS))
dst = cv2.warpPerspective(main_bf_c,M_clean_to_sense,(1600,800))

M_test_to_clean = cv2.getPerspectiveTransform(np.float32(bf_test_corners), np.float32(NEW_CORNERS))
dst2 = cv2.warpPerspective(test_bf_c, M_test_to_clean, (1600,800))

plt.imshow(dst2)
plt.show()  