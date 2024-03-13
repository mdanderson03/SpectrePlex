import numpy as np
import os
from skimage import io, morphology, restoration, filters
from matplotlib import pyplot as plt
import os
import psfmodels as psfm
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

file_directory = r'C:\Users\CyCIF PC\Desktop'
os.chdir(file_directory)
image = io.imread(r'MAX_Stack-1.tif')


nx = .45
nz = 5

# generate centered psf with a point source at `pz` microns from coverslip
# shape will be (127, 127, 127)
psf = psfm.make_psf(5, 5, dxy=0.206, dz=.05, pz=0)

#plt.imshow(psf[nz//2], norm=PowerNorm(gamma=0.4))
#plt.show()



filtered_image = filters.butterworth(image, cutoff_frequency_ratio=0.01, high_pass=True)
filtered_image[filtered_image < 0] = 0
io.imsave('filter_max_1.tif', filtered_image)
io.imshow(filtered_image)
io.show()

