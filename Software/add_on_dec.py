import numpy as np
from skimage import io, util
import os
import math
from skimage import transform, morphology
import microscope


os.chdir(r'E:\14-11-24_integrity_testing\np_arrays')
fm_array = np.load('fm_array.npy', allow_pickle=False)
fm_array[2][::,::] = -97
print(fm_array[2][::,::])

np.save('fm_array.npy', fm_array)