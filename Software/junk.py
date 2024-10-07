import numpy as np
import os
from skimage import io, filters, morphology, transform
from skimage.filters import rank
from matplotlib import pyplot as plt
import cv2
import math


from skimage.morphology import disk
#from stardist.models import StarDist2D
#from csbdeep.utils import normalize


def block_proc_min(array, block_y_pixels, block_x_pixels):
    '''
    breaks image up into blocks of size (block_y_pixels x block_x_pixels
    and then returns array that is the minimum of each block and is
    effectively down sampled by the block size

    :param array:
    :param block_y_pixels:
    :param block_x_pixles:
    :return:
    '''

    y_pixels = np.shape(array)[0]
    x_pixels = np.shape(array)[1]

    y_num_blocks = int(y_pixels / block_y_pixels)
    x_num_blocks = int(x_pixels / block_x_pixels)

    im = array

    blocked_array = np.lib.stride_tricks.as_strided(im,
                                                    shape=(x_num_blocks, y_num_blocks, block_x_pixels, block_y_pixels),
                                                    strides=(
                                                    im.strides[0] * block_x_pixels, im.strides[1] * block_y_pixels,
                                                    im.strides[0], im.strides[1]))
    min = np.min(blocked_array, axis=2)
    min = np.min(min, axis=2)

    return min


def block_proc_reshaper(array, block_y_pixels, block_x_pixels):
    '''
    takes in array and evenly clips off rows and columns to to their counts be integers
    when divided by block size pixel counts. Does no padding, only shrinks.
    :param array:
    :param block_y_pixels:
    :param block_x_pixels:
    :return:
    '''

    y_pixels = np.shape(array)[0]
    x_pixels = np.shape(array)[1]

    y_num_blocks = math.floor(y_pixels / block_y_pixels)
    x_num_blocks = math.floor(x_pixels / block_x_pixels)

    adjusted_y_pixels = block_y_pixels * y_num_blocks
    adjusted_x_pixels = block_x_pixels * x_num_blocks

    clipped_y_pixel_num = y_pixels - adjusted_y_pixels
    clipped_x_pixel_num = x_pixels - adjusted_x_pixels

    # deal with even and odd cases for y axis
    if clipped_y_pixel_num % 2 == 0:
        top_rows_2_clip = int(clipped_y_pixel_num / 2)
        bottom_rows_2_clip = int(clipped_y_pixel_num / 2)
        bottom_adjusted_row_num = y_pixels - bottom_rows_2_clip
        array = array[top_rows_2_clip:bottom_adjusted_row_num, ::]
    elif clipped_y_pixel_num % 2 != 0:
        top_rows_2_clip = int(clipped_y_pixel_num / 2)
        bottom_rows_2_clip = int((clipped_y_pixel_num / 2) + 1)
        bottom_adjusted_row_num = y_pixels - bottom_rows_2_clip
        array = array[top_rows_2_clip:bottom_adjusted_row_num, ::]

    # deal with even and odd cases for x axis
    if clipped_x_pixel_num % 2 == 0:
        left_col_2_clip = int(clipped_x_pixel_num / 2)
        right_col_2_clip = int(clipped_x_pixel_num / 2)
        right_adjusted_col_num = int(x_pixels - right_col_2_clip)
        array = array[::, left_col_2_clip:right_adjusted_col_num]
    elif clipped_x_pixel_num % 2 != 0:
        left_col_2_clip = int(clipped_x_pixel_num / 2)
        right_col_2_clip = int((clipped_x_pixel_num / 2) + 1)
        right_adjusted_col_num = int(x_pixels - right_col_2_clip)
        array = array[::, left_col_2_clip:right_adjusted_col_num]

    return array


def dark_frame_generate(array, block_y_pixels, block_x_pixels):
    '''

    :param array:
    :param block_y_pixels:
    :param block_x_pixels:
    :return:
    '''

    original_y_pixels = np.shape(array)[0]
    original_x_pixels = np.shape(array)[1]

    adjusted_array = block_proc_reshaper(array, block_y_pixels, block_x_pixels)
    min_array = block_proc_min(adjusted_array, block_y_pixels, block_x_pixels)
    resized_min_array = transform.resize(min_array, (original_y_pixels, original_x_pixels), preserve_range=True,anti_aliasing=True)
    resized_min_array = filters.butterworth(resized_min_array, cutoff_frequency_ratio=0.001, high_pass=False, order=2,npad=1000)

    return resized_min_array



os.chdir(r'C:\Users\mike\Desktop\dark')


#filename = 'x6_y_13_c_A647.tif'
filename = 'x6_y_13_c_A647_temp.tif'
filename_mean = 'x6_y_13_c_A647_mean.tif'
im = io.imread(filename)

im_mean = im/np.max(im)

footprint = disk(5)
im_mean = rank.mean(im_mean, footprint=footprint)
im_mean = im_mean.astype('int32')
im_mean = im_mean * (np.max(im)/np.max(im_mean))
io.imshow(im_mean)
io.show()
print(np.max(im_mean))
#im_mean = io.imread(filename_mean)

dark_im = dark_frame_generate(im_mean, 50, 50)
dark_subbed = im - dark_im
dark_subbed[dark_subbed< 0] = 0
