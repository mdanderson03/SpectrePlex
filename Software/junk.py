import numpy as np
import skimage.util
from skimage import io, morphology, filters, measure
from matplotlib import pyplot as plt
import tifffile as tf
import os
import cv2 as cv
import time


a = np.array([[[1,2], [3,4]], [[11,20], [13,10]]])

print(np.average(a, axis = 0))