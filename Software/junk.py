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


filenames = os.listdir(folder)
for x in range(0, len(filenames)):
    im2 = io.imread(filenames[x])
    im2 = np.nan_to_num(im2, posinf=65500)
    io.imsave(filenames[x], im2)



