import numpy as np
from skimage import io
import os

os.chdir(r'E:\29-7-24 gutage\np_arrays')
fm_array = np.load('fm_array.npy', allow_pickle=False)
fm_array[2] -= 4
np.save('fm_array.npy', fm_array)
print(fm_array[2][0][0], fm_array[3][0][0])


