#from autocyplex import *
#microscope = cycif() # initialize cycif object
#arduino = arduino()
import numpy as np
import os
from skimage import io, filters
import matplotlib.pyplot as plt

os.chdir('D:\Images\AutoCyPlex/10-3-23 flat field correction')

dark = io.imread('dark_image.tif')

reflect_a647 = io.imread('blank_a647_close_up_time_lapse_1/blank_a647_close_up_time_lapse_1_MMStack_Pos0.ome.tif').astype('float32')
fluor_a647 = io.imread('fl_a647_time_lapse_1/fl_a647_time_lapse_1_MMStack_Pos0.ome.tif').astype('float32')

reflect_a488 = io.imread('a488_time_lapse_2/a488_time_lapse_2_MMStack_Pos0.ome.tif').astype('float32')
fluor_a488 = io.imread('fl_a488_time_lapse_1/fl_a488_time_lapse_1_MMStack_Pos0.ome.tif').astype('float32')

#reflect_a555 = io.imread('blank_a555_close_up_time_lapse_1/blank_a555_close_up_time_lapse_1_MMStack_Pos0.ome.tif').astype('float32')
#fluor_a555 = io.imread('fl_a555_time_lapse_1/fl_a555_time_lapse_1_MMStack_Pos0.ome.tif').astype('float32')


#a488 = io.imread('a488_ff.tif')
#a555 = io.imread('a555_ff.tif')
#a647 = io.imread('a647_ff.tif')

a647_raw = io.imread('nak_atpase_lower_right.tif').astype('float32')
a488_raw = io.imread('488_raw.tif').astype('float32')


def correction_map_generator(image, reflectance_image, channel, h_overlap, w_overlap):

    reflect_count = reflectance_image.shape[0]
    fluor_count = image.shape[0]


    avg_reflect = np.sum(reflect_a647, axis =0)/reflect_count
    avg_reflect = avg_reflect
    avg_fluor = np.sum(fluor_a647, axis =0)/fluor_count
    avg_fluor = avg_fluor

    filt_fluor = filters.gaussian(avg_fluor, 1)
    filt_reflect = filters.gaussian(avg_reflect, 1)

    sub_image = filt_fluor-filt_reflect
    max_pixel = np.max(sub_image)
    corrected_map = max_pixel/sub_image
    corrected_map = corrected_map/np.max(corrected_map)
    os.chdir('D:\Images\AutoCyPlex/10-3-23 flat field correction')

    file_save_name = 'ff_correction_map_' + str(channel) + '.npy'

    np.save(file_save_name, corrected_map)

    return corrected_map

def ff_correction(image, channel, h_overlap, w_overlap):

    file_name = 'ff_correction_map_' + str(channel) + '.npy'

    os.chdir('D:\Images\AutoCyPlex/10-3-23 flat field correction')
    correction_map = np.load(file_name)

    height = image.shape[0]
    width = image.shape[0]

    start_h = int(height * h_overlap / 200)
    end_h = int(height * (1 - h_overlap / 200))
    start_w = int(width * w_overlap / 200)
    end_w = int(width * (1 - w_overlap / 200))

    correction_map = correction_map[start_h:end_h, start_w:end_w]
    correction_map = correction_map/np.max(correction_map)
    image = image[start_h:end_h, start_w:end_w]

    corrected_image = image*correction_map

    return corrected_image


corrected_648 = ff_correction(a647_raw, 'A647', 20, 20)

io.imsave('a647_corrected.tif', corrected_648)
corrected_648 = filters.difference_of_gaussians(corrected_648, 2, high_sigma=25)
corrected_648[corrected_648 < 0] = 0
map =io.imshow(corrected_648)
plt.show()