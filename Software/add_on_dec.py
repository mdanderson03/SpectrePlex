from pycromanager import Core, Acquisition, multi_d_acquisition_events, Dataset, MagellanAcquisition, Magellan, start_headless
import numpy as np
import os
import numpy as np
from skimage import io

experiment_directory = r'E:\auto'
numpy_path = experiment_directory + '/' + 'np_arrays'
os.chdir(numpy_path)
sp_array = np.load('DAPI_sp_array.npy', allow_pickle=False)
io.imshow(sp_array[3,:,:,0])
io.show()



