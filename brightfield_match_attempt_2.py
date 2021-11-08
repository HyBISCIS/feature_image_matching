import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Import Clean Brightfield Image

# Import Test Brightfield Image

# Apply Feature Match Algorithm between Brightfield Images
    # Perhaps we can use SIFT, ORB, etc.

# Use RANSAC to get more robust set of feature matches between two

# Define Homography Transform to get from one image to other

# With Matrix, we can find chip corner, knowing where it is on clean brightfield

# Multiply Affine Matrix with clean brightfield image to get to square pixels

# Apply Transforms and Crop Image