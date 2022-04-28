# feature_image_matching
A library of tools utilizing computer vision techniques to align impedance images to flourescence imaging. This work by taking a clean brightfield image of the microscope setup along with the 
experimental brightfield image taken before the start of an experiment using fluorescence.

## Dependencies
- Python
- sys 
- numpy
- matplotlib
- cv2

## Files
- brightfield_helper.py: A file that includes helpers to equalize the histograms between different
  microscope images. This is important to help alleviate contrast issues and such.

- brightfield_match.py: This is the first try at performing brightfield matching. This 
  program tries to bounding box the impedance array by directly trying to match and create
  a rectangular contour.

- brightfield_match_attempt_2.py: This is the second attempt at performing feature matching.
  We introduce histogram equalization, SIFT feature detectors, and estimation of the homography
  matrix for rectifying the two images. We then use that to estimate the corners of the 
  experimental array and then enforce that it is rectangular with a particular height to width
  ratio.
  
