import numpy as np
import os
from skimage import io, morphology, filters
from matplotlib import pyplot as plt
import os
from tifffile import imread

folder_path = r'C:\Users\CyCIF PC\Desktop'
os.chdir(folder_path)
filename = 'ear_isabelle.tif'
image = imread(filename)
#add all threee RGB channels together
image = image[0] + image[1] + image[2]
#make sat pixels equal zero
image[image > 65000] = 0
#find indicies of nonzero pxiels
indicies = np.nonzero(image)
#apply otsu to nonzero pixels
thresh = filters.threshold_otsu(image[indicies])

#find pixels under threshold
image[indicies] = image[indicies] < thresh

'''
#dilate mask to fill holes in
footprint = morphology.disk(20)
image = morphology.binary_erosion(image, footprint)
image = morphology.binary_dilation(image, footprint)
image = morphology.binary_dilation(image, footprint)
image = morphology.binary_erosion(image, footprint)
#erode mask to get rid of small objects
'''


io.imshow(image)
io.show()