from autocyplex import *
microscope = cycif() # initialize cycif object
arduino = arduino()
import numpy as np

experiment_directory = r'E:\test_folder'
numpy_path = experiment_directory + '/' + 'np_arrays'
os.chdir(numpy_path)
file_name = 'exp_calc_array.npy'
calc_array = np.load(file_name, allow_pickle=False)

print(core.get_exposure())