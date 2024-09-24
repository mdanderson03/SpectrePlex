import os
from skimage import io, util
import tifffile
import cv2
import numpy as np
from matplotlib import pyplot as plt
from microscope_bright import *

microscope = cycif()

#experiment_directory = r'D:\Images\AutoCyPlex\test_set'
experiment_directory = r'E:\30-8-24 gutage'
os.chdir(experiment_directory)


for cycle in range(10, 11):

#microscope.illumination_flattening(experiment_directory, cycle_number=7)
    microscope.brightness_uniformer(experiment_directory, cycle, 1)
    microscope.brightness_uniformer(experiment_directory, cycle, 2)
    microscope.brightness_uniformer(experiment_directory, cycle, 3)
    microscope.brightness_uniformer(experiment_directory, cycle, 4)
    microscope.brightness_uniformer(experiment_directory, cycle, 5)
    #microscope.brightness_uniformer(experiment_directory, cycle, 6)

    microscope.stage_placement(experiment_directory, cycle, 2960, down_sample_factor=4)



'''
# Reading the image from the present directory
image = cv2.imread("DAPI_cy_1_Stain_placed_16b.tif")
# Resizing the image for compatibility
#image = cv2.resize(image, (500, 600))

# The initial processing of the image
# image = cv2.medianBlur(image, 3)
image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# The declaration of CLAHE
# clipLimit -> Threshold for contrast limiting
clahe = cv2.createCLAHE(clipLimit=5)
final_img = clahe.apply(image_bw) + 30

# Ordinary thresholding the same image
_, ordinary_img = cv2.threshold(image_bw, 155, 255, cv2.THRESH_BINARY)

# Showing the two images
cv2.imshow("ordinary threshold", ordinary_img)
cv2.imshow("CLAHE image", final_img)
cv2.waitKey(0)
'''