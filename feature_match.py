import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from scipy import ndimage
from skimage.measure import regionprops
import sys

def get_interest_points(image, feature_width):
    '''
    Returns a set of interest points for the input image

    Note: Uses harris corner detection and I wonder whether or not other feature
    point matching algorithms would be a lot better

    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:

    :returns:
    :xs: an np array of the x coordinates of the interest points
    :ys: an np array of the y coordinates of the interest points
    '''
    ALPHA = 0.058 # Choose from 0.04 to 0.06

    # Get gradients of x and y
    grad_x = ndimage.sobel(image, 1)
    grad_y = ndimage.sobel(image, 0)

    # Find corner matrix
    squared_x = grad_x * grad_x
    squared_y = grad_y * grad_y 
    x_y = grad_x * grad_y

    gauss_square_x = filters.gaussian(squared_x)
    gauss_square_y = filters.gaussian(squared_y)
    gauss_x_y = filters.gaussian(x_y)

    alpha_value = ALPHA * np.square(gauss_square_x + gauss_square_y)
    corner_matrix = (gauss_square_x * gauss_square_y) - np.square(gauss_x_y) - alpha_value

    # Threshold the corner matrix
    thresh = np.mean(corner_matrix) / np.std(corner_matrix)
    corner_matrix[corner_matrix < thresh] = 0

    # Determine local max points
    half_square = int(feature_width / 2)
    non_max_supress = feature.peak_local_max(corner_matrix, min_distance=9, exclude_border=half_square)

    xs = non_max_supress[:, 1]
    ys = non_max_supress[:, 0]

    return xs, ys


def get_features(image, x, y, feature_width):
    '''
    Returns a set of feature descriptors for a given set of interest points by
    implementing SIFT (Shift Invariant Feature Transform)
 
    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width

    :returns:
    :features: np array of computed features.(for standard SIFT feature
            dimensionality is 128)
    '''

    features = []

    # Get image derivative information
    grad_x = ndimage.sobel(image, 1)
    grad_y = ndimage.sobel(image, 0)
    grad_mag = np.sqrt((np.square(grad_x)) + (np.square(grad_y)))
    grad_ori = np.arctan2(grad_y, grad_x)

    # Check to make sure that the feature_width is divisible by 4
    if feature_width % 4 != 0:
        print('Inputted feature_width not divisible by 4')
        return 0
    
    # Get the width of each of the small squares
    small_square_width = int(feature_width / 4)
    square_half = int(feature_width / 2)
    
    # Loop through each of the provided interest points
    for counter in range(len(x)):
        x_num = int(x[counter])
        y_num = int(y[counter])
        
        # Determine if the x_num and y_num don't overflow 
        if (x_num <= square_half) or (x_num + square_half > image.shape[1]) \
            or (y_num <= square_half) or (y_num + square_half > image.shape[0]):
            continue

        # Get feature_width * feature_width patch around point provided
        descriptor = np.zeros((1,128))
        feature_mag = grad_mag[y_num-square_half:y_num+square_half, \
                               x_num-square_half:x_num+square_half]
        feature_ori = grad_ori[y_num-square_half:y_num+square_half, \
                               x_num-square_half:x_num+square_half]
        
        # Loop through each of the squares created
        for i in range(4): 
            for j in range(4):
                smaller_mag = feature_mag[i*small_square_width:((i*small_square_width)+small_square_width), \
                                          j*small_square_width:((j*small_square_width)+small_square_width)]
                smaller_ori = feature_ori[i*small_square_width:((i*small_square_width)+small_square_width), \
                                          j*small_square_width:((j*small_square_width)+small_square_width)]
                bins = np.zeros((1,8))

                # Compute histogram bins for each of the small squares
                for k in range(small_square_width): 
                    for l in range(small_square_width): 
                         # Compute where bins are stored and store magnitude
                        bin_number = int((smaller_ori[k][l] + np.pi) / (np.pi / 4))
                        if bin_number == 8:
                            bin_number = 0

                        bins[0][bin_number] += smaller_mag[k][l]
                
                # Put bin information into descriptor
                for a in range(8):
                    # For each i, 8 bins * 4 vectors; For each j jump 8 bins
                    descriptor[0][(32 * i) + (8 * j) + a] = bins[0][a]

        # Put into feature matrix and normalize 
        descriptor = descriptor ** (0.35)
        norm = np.linalg.norm(descriptor)
        normed_descriptor = descriptor / norm

        # Clamp values and renormalize
        normed_descriptor[normed_descriptor > 0.2] = 0.2
        norm_again = np.linalg.norm(normed_descriptor)
        clamped = normed_descriptor / norm_again
        features.append(clamped)
    
    # Turn the features list into a features np array
    features = np.vstack(features)

    return features


def match_features(im1_features, im2_features):
    '''
    Implements the Nearest Neighbor Distance Ratio Test (NNDR) to assign matches between interest points
    in two images. 
    
    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    '''
    # Initialize match lists so that we can create np arrays later on
    confidence_list = []
    match_list = []
    counter = 0

    # Get B for the distance 
    im2_transpose = np.transpose(im2_features)
    im2_dot = im1_features.dot(im2_transpose)
    B = im2_dot * 2

    # Get the squared im1_features
    im1_feature_squared = im1_features * im1_features
    im1_sum = np.sum(im1_feature_squared, axis=1)
    im1_sum.shape = (len(im1_sum), 1)

    # Get the squared im2_features
    im2_feature_squared = im2_features * im2_features
    im2_sum = np.sum(im2_feature_squared, axis=1)
    im2_sum.shape = (len(im2_sum), 1)
    im2_sum_transpose = np.transpose(im2_sum)

    # Produce the singleton expansion and produce the distance matrix
    A = im1_sum + im2_sum_transpose

    distances = np.sqrt(A - B)
    
    # Sort the distances so that the minimum is the first entry
    index_sorted = np.argsort(distances, axis=1)
    mag_sorted = np.sort(distances, axis=1)

    # Get the first and second columns of the min to find ratio and confidence
    min_col = mag_sorted[:, 0]
    second_min_col = mag_sorted[:, 1]
    ratios = min_col / second_min_col
    confidence = second_min_col / min_col

    # Loop through ratios to filter out low 
    for ratio in ratios:
        if ratio < 0.96:
            match_list.append((counter, index_sorted[counter][0]))
            confidence_list.append(confidence[counter])

        counter += 1
    
    # Convert the matches and the confidences to np.arrays to return
    matches = np.asarray(match_list)
    confidences = np.asarray(confidence_list)

    return matches, confidences
