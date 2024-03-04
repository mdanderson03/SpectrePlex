import numpy as np
import skimage.util
from skimage import io, morphology, filters, measure
from skimage.restoration import rolling_ball
from matplotlib import pyplot as plt
import tifffile as tf
import os
import cv2 as cv
import time
import scipy.optimize as opt

folder = r'C:\Users\mike\Desktop\focused'
file = r'x2_y_1_c_DAPI.tif'
os.chdir(folder)
image = io.imread(file)


def image_percentile_level( image, cut_off_threshold=0.99):
    '''
    Takes in image and cut off threshold and finds pixel value that exists at that threshold point.

    :param numpy array image: numpy array image
    :param float cut_off_threshold: percentile for cut off. For example a 0.99 would disregaurd the top 1% of pixels from calculations
    :return: intensity og pixel that resides at the cut off fraction that was entered in the image
    :rtype: int
    '''

    pixel_values = np.sort(image, axis=None)
    indicies = np.nonzero(pixel_values)[0]
    # thresh = filters.threshold_otsu(pixel_values[indicies])
    # indicies = np.where(pixel_values > thresh)[0]
    pixel_values = pixel_values[indicies]

    pixel_count = int(np.size(pixel_values))
    cut_off_index = int(pixel_count * cut_off_threshold)
    min_cut_off_index = int(pixel_count - cut_off_index)
    print(cut_off_index, min_cut_off_index)
    tail_intensity = pixel_values[cut_off_index]
    min_intensity = pixel_values[min_cut_off_index]

    return tail_intensity, min_intensity


print(image_percentile_level(image))