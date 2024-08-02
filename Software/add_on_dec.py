import numpy as np
from skimage import io
import os


os.chdir(r'E:\1-8-24 gutage\np_arrays')
fm_array = np.load('fm_array.npy', allow_pickle=False)

print(fm_array[10][0][7], fm_array[3][0][0])



