import numpy as np
import os
from skimage import io, morphology
from matplotlib import pyplot as plt
import os


os.chdir(r'C:\Users\CyCIF PC\Desktop')
binary_image = io.imread('x1_y_2_c_A555_binary.tif')
tissue_binary_image = io.imread('x1_y_2_c_A555_tissue_binary.tif')
nucs_binary = io.imread('nucs_binary.tif')
#er = binary_image
footprint=np.ones((2, 2))
er = morphology.binary_erosion(nucs_binary)
dilate = morphology.binary_erosion(nucs_binary)
er = morphology.binary_erosion(er)
er = morphology.binary_erosion(er)
er = morphology.binary_erosion(er)
er = morphology.binary_erosion(er)
er = morphology.binary_erosion(er)

io.imshow(er)
io.show()