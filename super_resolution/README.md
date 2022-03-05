# EIS Super-Resolution Reconstruction Algorithms

This repo contains code used for the Minerva super-resolution
paper submitted to TBioCas in March 2022. Many of the scripts
denoted in the repo are no longer used and are no longer needed. 
These have been noted below and each file is explained in further
detail.

## Package Requirements
In order to use this repo, there are certain packages required in order
to make everything work correctly. They have been listed below:
- pickle
- h5py
- matplotlib
- numpy
- cv2
- skimage
- scipy

## Necessary Data
- Before use of this repo, it is necessary to create a data folder that 
  containins the h5 files that someone may want to run.

## Files
### Main Files for Linear Deconvolution
- deconv_superres.py is the main work horse script that all other super
  resolution scripts is based off of. It is for linear deconvolution
- shift_sum_superres.py is the shift-sum equivalent to the linear
  deconvolution method mentioned above.
- deconv_func.py includes all functions used for linear deconvolution and
  shift sum scripts.
- super_res_at_distances.py was the script used to generate the different
  distance composite images.

### Figure Scripts
- line_plots.py is the script used to make the line profiles through
  algae.
- make_distance_plots.py was the script used to make the different distance
  composite images. It was not used for the paper submission.
- make_microscope_cosmarium_pediastrum_plot.py was used to make the figure
  comparing the microscope, reference, and the impedance image.
- make_three_cosmarium_plots.py is the script used to make the three raw images and three super-resolution cosmarium figure.
- make_three_pediastrum_plots.py is the script used to make the three raw images and three super-resolution pediastrum figure.
- show_algae.py was a script used to show off the array of 120 impedance
images used in that big 11x11 array. This is for the figure for pediastrum and
for cosmarium

### Helpful Scripts
- find_algae.py is a script used to find algae by comparing the optical
  microscope image against the full raw impedance image to look for
  certain algae to do super-resolution on.
- high_pass_filter.py is a script used to determine which parameters were
  best for the high pass filtering after linear-deconvolution method
- spatial_resolution.py is a script that was NOT FINISHED for Minerva, but
hopes to demonstrate the spatial-resolution superiority of the linear-deconvolution method over the shift sum method.

### Deprecated Scripts
- DEPRECATED: algae_superres_1.py, algae_superres_jason.py, ect_linear_deconvolution.py, and plot_lilliput_ECT_4_ifft.py are all old and deprecated scripts that were used to construct the majority of the scripts used for the paper submission. They can be effectively ignored
- DEPRECATED: Similarly, we can ignore enforce_mirroring.py for the time being since it has yet to be updated to work with the MINERVA chip. 
- make_super_resolution_plots.py was a script that was not used at all in 
the making of the paper. It is old code that is not that useful to us.

