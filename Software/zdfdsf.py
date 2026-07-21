import numpy as np
from skimage import io
import os

experiment_directory = r'D:\20-7-26_Kate_SP24_9928_A1'


numpy_path = experiment_directory + '/' + 'np_arrays'
os.chdir(numpy_path)
full_array = np.load('fm_array.npy', allow_pickle=False)


tissue_fm = full_array[2]
tissue_fm =140
#np.save('fm_array.npy', full_array)
print(tissue_fm)