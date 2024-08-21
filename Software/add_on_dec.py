import numpy as np
from skimage import io, util
import os



os.chdir(r'E:\20-8-24 gutage\Tissue_Binary')


im = io.imread(r'labelled_tissue_filtered.tif')

im -= np.min(im)

io.imsave(r'labelled_tissue_filtered.tif', im)
