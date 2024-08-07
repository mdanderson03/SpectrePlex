import numpy as np
from skimage import io
import os


os.chdir(r'E:\7-8-24 gutage\np_arrays')
fm_array = np.load('fm_array.npy', allow_pickle=False)
fm_array[10] = 2
#np.save('fm_array.npy', fm_array)

os.chdir(r'E:\7-8-24 gutage\Tissue_Binary')
im = io.imread(r'labelled_tissue_filtered.tif')
print(im[30280][22512])
io.imsave(r'labelled_tissue_filtered.tif', im)

