import numpy as np
import os
from skimage import io
from matplotlib import pyplot as plt

import os

from autocyplex import *
from pycromanager import Core, Studio, Magellan
microscope = cycif() # initialize cycif object
pump = fluidics(6, 3)




experiment_directory = r'E:\auto_focus testing'
numpy_path = experiment_directory + '/' + 'np_arrays'
os.chdir(numpy_path)

fm_array = np.load('fm_array.npy', allow_pickle=False)
dapi_sp_array = np.load('dapi_sp_array.npy', allow_pickle=False)
#images = np.load('images.npy', allow_pickle=False)

#io.imshow(images[2])
#io.show()

#scores = [microscope.focus_score(images[0], 17), microscope.focus_score(images[1], 17), microscope.focus_score(images[2], 17)]
#print(scores)


#focus_map = dapi_sp_array[3, ::, ::, 0] * dapi_sp_array[4, ::, ::, 0]
#print(np.shape(dapi_sp_array))

y = dapi_sp_array[0:3, 1, 5, 0]
x = dapi_sp_array[0:3, 1, 5, 1]
print(fm_array[2][1][1])
#io.imshow(focus_map)
#io.show()

#plt.scatter(x,y)
#plt.show()


