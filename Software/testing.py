import numpy as np
import os
from skimage import io, morphology
from matplotlib import pyplot as plt
import os

experiment_directory = r'E:\tissue_identify'
numpy_path = experiment_directory + '/' + 'np_arrays'
os.chdir(numpy_path)
file_name = 'fm_array.npy'
fm_array = np.load(file_name, allow_pickle=False)


io.imshow(fm_array[10])
io.show()
