import numpy as np
import os

experiment_directory = r'E:\2nd_full_auto_test'
numpy_path = experiment_directory + '/' + 'np_arrays'
os.chdir(numpy_path)

fm_array = np.load('fm_array.npy', allow_pickle=False)


print(fm_array[4][0][0])


