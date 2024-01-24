import numpy as np
import skimage.util
from skimage import io, morphology, filters, measure
from matplotlib import pyplot as plt
import tifffile as tf
import os
import cv2 as cv
import time


def image_percentile_level(image, cut_off_threshold=0.999):
    '''
    Takes in image and cut off threshold and finds pixel value that exists at that threshold point.

    :param numpy array image: numpy array image
    :param float cut_off_threshold: percentile for cut off. For example a 0.99 would disregaurd the top 1% of pixels from calculations
    :return: intensity og pixel that resides at the cut off fraction that was entered in the image
    :rtype: int
    '''

    pixel_values = np.sort(image, axis=None)
    indicies = np.nonzero(pixel_values)[0]
    thresh = filters.threshold_otsu(pixel_values[indicies])
    indicies = np.where(pixel_values > thresh)[0]
    pixel_values = pixel_values[indicies]

    pixel_count = int(np.size(pixel_values))
    cut_off_index = int(pixel_count * cut_off_threshold)
    tail_intensity = pixel_values[cut_off_index]

    return tail_intensity


#Import images
junk_folder_path = r'C:\Users\mike\Desktop\junk image folder'
folder_path = r'C:\Users\mike\Desktop\junk image folder\cy7'
os.chdir(junk_folder_path)
a488_im = io.imread(r'x1_y_2_c_A488.tif')
a555_im = io.imread(r'x1_y_2_c_A555.tif')
a647_im = io.imread(r'x1_y_2_c_A647.tif')

os.chdir(junk_folder_path)
dapi_im = io.imread(r'x1_y_2_c_DAPI.tif')
os.chdir(folder_path)

#make binary of tissue area
foot_print = morphology.disk(70, decomposition='sequence')
#foot_print_2 = morphology.disk(25, decomposition='sequence')
dapi_dil = morphology.binary_dilation(dapi_im, foot_print)
#dapi_er = morphology.binary_erosion(dapi_dil, foot_print_2)
tissue_binary = dapi_dil



image_chosen = a555_im

start_time = time.time()

a647_filt = filters.butterworth(image_chosen, cutoff_frequency_ratio=0.25, high_pass = False)
thresh = filters.threshold_multiotsu(a647_filt, classes = 4)[2]
a647_binary = a647_filt > thresh

foot_print = morphology.disk(30, decomposition='sequence')
a647_er = morphology.binary_erosion(a647_binary, foot_print)
foot_print = morphology.disk(80, decomposition='sequence')
a647_dil = morphology.binary_dilation(a647_er, foot_print)
a647_dil = skimage.util.img_as_float(a647_dil)


try:
    regions = measure.regionprops(a647_dil)
    for props in regions:
        y0, x0 = props.centroid
        print('y', y0,'x', x0)
except:
    pass

end_time = time.time()
print('time', end_time - start_time)
overall_mask = tissue_binary - a647_dil
overall_mask[overall_mask < 0] = 0
masked_image = overall_mask * image_chosen





highest_int = image_percentile_level(masked_image, 0.99)
print('highest intensity', highest_int)
print('increased exp from original', 65500*0.3/highest_int)


#io.imsave('a647_binary.tif', a647_dil)
io.imshow(masked_image)
#io.imshow(tissue_binary)
io.show()