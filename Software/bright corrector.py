import numpy as np
from skimage import io
import os

import numpy as np
from skimage import io
import os

experiment_directory = r'D:\23_4_26_casey_TMA_2'


numpy_path = experiment_directory + '/' + 'np_arrays'
os.chdir(numpy_path)
full_array = np.load('fm_array.npy', allow_pickle=False)

full_array[2] = full_array[2] - 18
np.save('fm_array.npy', full_array)
tissue_fm = full_array[2]
print(tissue_fm)



'''
experiment_directory = r'D:/1_4_26_casey_TMA_4'


numpy_path = experiment_directory + '/' + 'np_arrays'
os.chdir(numpy_path)
full_array = np.load('fm_array.npy', allow_pickle=False)
tissue_fm = full_array[12]
numpy_x = full_array[0]
numpy_y = full_array[1]

x_tile_count = np.unique(numpy_x).size
y_tile_count = np.unique(numpy_y).size

tile_array = np.ones([y_tile_count, x_tile_count])

for cycle in range(4,5):

    os.chdir(r'D:\1_4_26_casey_TMA_4\A647\Stain\cy_'+str(cycle) + '\Tiles')

    for x in range(0, x_tile_count):
        for y in range(0, y_tile_count):
            if tissue_fm[y][x] > 1:

                file_name = 'x'+str(x) + '_y_' + str(y) + '_c_A647.tif'
                image = io.imread(file_name)
                tile_array[y][x] = np.median(image)

    med_im = np.where(tile_array>1)
    system_median = np.median(tile_array[med_im])
    io.imshow(tile_array/system_median)
    io.show()
    
'''
'''
    tile_array = tile_array/system_median

    for x in range(0, x_tile_count):
        for y in range(0, y_tile_count):
            if tissue_fm[y][x] > 1.7:

                if tile_array[y][x] >1:

                    file_name = 'x' + str(x) + '_y_' + str(y) + '_c_A647.tif'
                    image = io.imread(file_name)
                    image = image/tile_array[y][x]
                    io.imsave(file_name, image)

'''