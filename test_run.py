from autocyplex import *
microscope = cycif() # initialize cycif object
#arduino = arduino()
import numpy as np
from skimage import io, filters


dark = io.imread('E:/flat field images/dark_image.tif')
dapi = io.imread('E:/flat field images/dapi_ff.tif')
a488 = io.imread('E:/flat field images/a488_ff.tif')
a555 = io.imread('E:/flat field images/a555_ff.tif')
a647 = io.imread('E:/flat field images/a647_ff.tif')

a647_raw = io.imread('E:/flat field images/nak_atpase_center.tif')

def correction_map_generator(image, dark_image, channel):

    sub_image = image-dark_image
    sub_image = filters.gaussian(sub_image, 5)
    max_pixel = np.max(sub_image)
    corrected_map = max_pixel/sub_image
    corrected_map = corrected_map/np.max(corrected_map)
    os.chdir('E:/flat field images')

    file_save_name = 'ff_correction_map_' + str(channel) + '.npy'

    np.save(file_save_name, corrected_map)

    return corrected_map

correction_image = correction_map_generator(a647, dark, 'A647')
a647_corrected = a647_raw - dark
a647_corrected = correction_image * a647_corrected
io.imsave('a647_corrected.tif', a647_corrected)
map =io.imshow(correction_image)
plt.show()