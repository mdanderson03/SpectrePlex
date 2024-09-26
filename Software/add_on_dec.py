import numpy as np
from skimage import io, util
import os



os.chdir(r'E:\test_olympus\np_arrays')

fm_array = np.load('fm_array.npy', allow_pickle=False)
print(fm_array[3][0][0])
fm_array[5] = 1
fm_array[7] = 1
fm_array[9] = 1
np.save('fm_array.npy', fm_array)

#x_tile_count = np.shape(full_array[0])[1]
#y_tile_count = np.shape(full_array[0])[0]
#z_planes = full_array[2]
#for x in range(0, x_tile_count):
#    for y in range(0, y_tile_count):
#        z_planes[y][x] = -366 - y*3.75
#np.save('fm_array.npy', full_array)
#io.imshow(full_array[2])
#io.show()
