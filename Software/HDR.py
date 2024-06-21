import numpy as np
import os
from skimage import io, filters, util
import time
from copy import deepcopy



times = np.array([10, 20, 30, 50, 100, 150, 250, 400, 800, 2000])
hdr_indicies = [0,3,9]

os.chdir(r'C:\Users\CyCIF PC\Desktop\linearity')
array = io.imread('linearity.tif')

hdr_array = array[hdr_indicies]
times = times[hdr_indicies]

start_time = time.time()

max_hdr_time = np.max(times)
hdr_array = hdr_array.astype('float32')
weight_array = deepcopy(hdr_array)

#subtract linear offset
hdr_array = hdr_array - 300

#populate weight array
for index in range(0,3):

    im = hdr_array[index]
    im[im >65234] = 0
    weight_array[index] = np.sqrt(im)
    hdr_array[index] = ((max_hdr_time + 28) / (times[index] + 28)) * hdr_array[index]

total_weight_array = np.sum(weight_array, axis=0)
scaled_weight_array = np.divide(weight_array,total_weight_array)

hdr_im = hdr_array[0]*scaled_weight_array[0] + hdr_array[1]*scaled_weight_array[1] + hdr_array[2]*scaled_weight_array[2]

end_time = time.time()
print('time to run', end_time - start_time)

hdr_im[hdr_im<0] = 0
os.chdir(r'C:\Users\CyCIF PC\Desktop\linearity\hdr')
io.imsave('bit32_hdr.tif', hdr_im)
hdr_im = hdr_im/np.max(hdr_im)
hdr_im = util.img_as_uint(hdr_im)
io.imsave('bit16_hdr.tif', hdr_im)


