import numpy as np
import os
from skimage import io, filters, morphology, transform
from skimage.filters import rank
from matplotlib import pyplot as plt
import cv2
import math


from skimage.morphology import disk
#from stardist.models import StarDist2D
#from csbdeep.utils import normalize



os.chdir(r'E:\9-10-24 gutage\Tissue_Binary')


#filename = 'x6_y_13_c_A647.tif'
filename = 'labelled_tissue_filtered.tif'
im = io.imread(filename)
io.imshow(im)
io.show()

