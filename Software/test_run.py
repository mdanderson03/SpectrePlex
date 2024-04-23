import time
import os
from subprocess import call
from fluidics_V3 import fluidics
import numpy as np


experiment_directory = r'E:\12-4-24 celiac'

numpy_path = experiment_directory + '/' + 'np_arrays'
os.chdir(numpy_path)
exp_filename = 'exp_array.npy'
file_name = 'fm_array.npy'
exp_array = np.load(exp_filename, allow_pickle=False)

exp_array[1] = 50
exp_array[2] = 50
exp_array[3] = 50

np.save(exp_filename, exp_array)


