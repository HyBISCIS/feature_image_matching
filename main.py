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

'''
Parts of code obtained from: 
https://opencv24-python-tutorials.readthedocs.io/en/stable/py_tutorials/py_feature2d/py_matcher/py_matcher.html
'''

def get_features(sift, img):
    '''
    Use feature detector to produce SIFT keypoints and descriptors
    '''
    k, d = sift.detectAndCompute(img, None)

    return k, d

def flann_match(flann, d1, d2):
    '''
    Use built in OpenCV Flann to match descriptors

    Returns set of tuples of descriptors that have passed ratio test
    '''
    match_out = flann.knnMatch(d1, d2,k=2)
    matches = []

    # Go through matches and use ratio test to pick best ones
    for a, b in match_out:
        if ((a.distance / b.distance) < 0.78):
            matches.append([a])
    
    return matches






def load_data(num):
    '''
    TODO:
    '''
    # TODO: Get filepath for image1_file and image2_file
    image1_file = ''
    image2_file = ''

    image1 = img_as_float32(io.imread(image1_file))
    image2 = img_as_float32(io.imread(image2_file))

    return image1, image2

def load_impedance_img():
    '''
    Use Rosenstein's code that already exists that runs to use impedance
    data to create impedance image.
    TODO: 
    '''
    filepath = 'data/F0088_imp_match_w_fluor.tif'

    # imp = plt.imread(filepath)
    imp = io.imread(filepath)

    print(imp.shape)

def main():
    # Get filepath
    FILEPATH = ''
    EXPERIMENT_NAME = ''
    # TODO:

    # Construct filepath for impedance stack and flouresence stack
    # TODO:

    # Adjust flouresence Image Color 
    # TODO:

    # Load Images
    # TODO: 
    # imp_img = load_impedance_img()
    # flour_img = load_data()

    imp_img_c = cv2.imread('data/impedance.PNG')
    flour_img_c = cv2.imread('data/scarlett.PNG')
    # imp_img = cv2.imread('data/EGaudi_1.jpg')
    # flour_img = cv2.imread('data/EGaudi_2.jpg')

    # Convert to grayscale
    imp_img = cv2.cvtColor(imp_img_c, cv2.COLOR_BGR2GRAY)
    flour_img = cv2.cvtColor(flour_img_c, cv2.COLOR_BGR2GRAY)

    cv2.normalize(imp_img, imp_img, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(flour_img, flour_img, 0, 255, cv2.NORM_MINMAX)

    # Try and Blur both to spatially low pass filter
    blur_size = 5
    gaussian = True

    if gaussian:
        imp_img = cv2.blur(imp_img, (blur_size,blur_size))
        flour_img = cv2.blur(flour_img, (blur_size,blur_size))
    else:
        # Try median blur instead
        imp_img = cv2.medianBlur(imp_img, blur_size)
        flour_img = cv2.medianBlur(flour_img, blur_size)

    # Edge filtering 
    sobel_64 = cv2.Sobel(imp_img,cv2.CV_64F,1,0,ksize=3)
    abs_64 = np.absolute(sobel_64)
    imp_img = np.uint8(abs_64) 

    sobel_64 = cv2.Sobel(flour_img,cv2.CV_64F,1,0,ksize=3)
    abs_64 = np.absolute(sobel_64)
    flour_img = np.uint8(abs_64) 

    cv2.normalize(imp_img, imp_img, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(flour_img, flour_img, 0, 255, cv2.NORM_MINMAX)

    # Clamp Images that are less than some value to 0 to get nice edges
    # lower_clamp = 100
    # imp_img[imp_img > lower_clamp] = 255
    # flour_img[flour_img > 20] = 255
    
    # Brighten Flourescence image edges


    # TODO: Need to get rid of high frequency noise in the imp_img

    # Brighten flour_img
    # TODO: 

    print("Images Loaded!")

    # Perform SIFT Feature Detection
    sift = cv2.SIFT_create()

    k1, d1 = get_features(sift, imp_img)
    k2, d2 = get_features(sift, flour_img)

    print("Num Features: ", len(k1))
    print("Got SIFT Features!")


    # Pass descriptors into FLANN Matcher
    flann = True

    if flann:
        FLAN_INDEX_KDTREE = 0
        index_params = dict (algorithm = FLAN_INDEX_KDTREE, trees=5)
        search_params = dict (checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann_match(flann, d1, d2)
    else:
        # create BFMatcher object
        bf = cv2.BFMatcher()

        # Match descriptors.
        matches = bf.match(d1,d2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)

        
        
    print("Matches: ", len(matches))
    print("Matching Descriptors between Images!")

    # Graph and display images
    if flann:
        flann_matches =cv2.drawMatchesKnn(imp_img, k1, flour_img, k2, matches[:30], None, flags=2)
        cv2.imwrite('flann_matches.jpg', flann_matches)
    else:
        # Draw first 10 matches.
        bf_matches = cv2.drawMatches(imp_img,k1,flour_img,k2,matches[:10], None, flags=2)
        cv2.imwrite('bf_matches.jpg', bf_matches)



if __name__ == '__main__':
    main()
    #load_impedance_img()

