import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

def show_hist(main_bf, test_bf, show_histograms):
    if show_histograms:
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

def show_equal(main_bf, equ_main, test_bf, equ_test, show_equalized):
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