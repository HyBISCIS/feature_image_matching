import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

# MACROS FOR SHOWING IMAGES OF WHAT IS HAPPENING
show_histograms = False
show_equalized = False
draw_matches = True

# Import Clean and TestBrightfield Image
main_bf_c = cv2.imread("data/brightfield_main_minerva.tif")
test_bf_c = cv2.imread("data/f0113/F0113_10012021_initial_BF.tif")
#main_bf_c = cv2.imread("data/EGaudi_1.jpg")
#test_bf_c = cv2.imread("data/EGaudi_2.jpg")

# Convert to grayscale
main_bf = cv2.cvtColor(main_bf_c, cv2.COLOR_BGR2GRAY)
test_bf = cv2.cvtColor(test_bf_c, cv2.COLOR_BGR2GRAY)

kernel_size = 5
main_bf = cv2.GaussianBlur(main_bf, (kernel_size, kernel_size), 0)
test_bf = cv2.GaussianBlur(test_bf, (kernel_size, kernel_size), 0)
# plt.imshow(main_bf)
# plt.show()

# dst = cv2.cornerHarris(main_bf, 15, 15, 0.03)
# dst = cv2.dilate(dst, None) 
# main_bf_c[dst > 0.01 * dst.max()]=[255,0,0]
# plt.imshow(main_bf_c)
# plt.show()

# Equalize Histogram to get Same Brightness and Contrast of Both Images
if show_histograms:
    # TODO: Put this in a function and another file so that it is easier to read
    # Main Brightfield Image 
    hist,bins = np.histogram(main_bf.flatten(),256,[0,256])
    cdf_normalized = hist.cumsum() * float(hist.max()) / hist.cumsum().max()
    fig, ax = plt.subplots(1,2)
    ax[0].plot(cdf_normalized, color = 'b')
    ax[0].hist(main_bf.flatten(),256,[0,256], color = 'r')
    ax[0].legend(('cdf','histogram'), loc = 'upper left')
    ax[0].set_title("Main BF Histogram")

    # Test Brightfield Image
    hist,bins = np.histogram(test_bf.flatten(),256,[0,256])
    cdf_normalized = hist.cumsum() * float(hist.max()) / hist.cumsum().max()
    ax[1].plot(cdf_normalized, color = 'b')
    ax[1].hist(test_bf.flatten(),256,[0,256], color = 'r')
    ax[1].legend(('cdf','histogram'), loc = 'upper left')
    ax[1].set_title("Test BF Histogram")

    plt.show()

equ_main = cv2.equalizeHist(main_bf)
equ_test = cv2.equalizeHist(test_bf)

if show_equalized:
    # TODO: Put this in a function and another file so that it is easier to read
    fig = plt.figure(2)
    fig.add_subplot(2,2,1)
    plt.title("Main Brightfield")
    plt.imshow(main_bf)

    fig.add_subplot(2,2,2)
    plt.title("Equalized Main Brightfield")
    plt.imshow(equ_main)
    
    fig.add_subplot(2,2,3)
    plt.title("Test Brightfield")
    plt.imshow(test_bf)

    fig.add_subplot(2,2,4)
    plt.title("Equalized Test Brightfield")
    plt.imshow(equ_test)
    plt.show()

# Apply Feature Match Algorithm between Brightfield Images
sift_detect = cv2.SIFT_create()
kp1,d1 = sift_detect.detectAndCompute(equ_main, None)
kp2,d2 = sift_detect.detectAndCompute(equ_test, None)

FLAN_INDEX_KDTREE = 0
index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
search_params = dict (checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(d1, d2, k=2)

# Determine Uniqueness of Flann Matches using ratio test
unique_matches = []
for a,b in matches:
    if a.distance < 0.6 * b.distance:
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
print(M)

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

# Multiply Affine Matrix with clean brightfield image to get to square pixels

# Apply Transforms and Crop Image