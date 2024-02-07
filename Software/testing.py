import numpy as np
import os
from skimage import io, morphology
from matplotlib import pyplot as plt
import os

experiment_directory = r'E:\tissue_identify\Tissue_Binary'
os.chdir(experiment_directory)
file_name = 'x3_y_0_tissue.tif'
image = io.imread(file_name)
image = image.astype('bool')

image_2 = morphology.remove_small_objects(image, min_size=80000, connectivity=1)
image_2 = image_2.astype('int8')

io.imshow(image_2)
io.show()
