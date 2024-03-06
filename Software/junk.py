import numpy as np
import skimage.util
from skimage import io, morphology, filters, measure
from skimage.restoration import rolling_ball
from matplotlib import pyplot as plt
import tifffile as tf
import os
import cv2 as cv
import time
import scipy.optimize as opt
import math

tissue_folder = r'D:\Images\AutoCyPlex\6-1-24 multiplex\Tissue_Binary'
numpy_folder = r'D:\Images\AutoCyPlex\6-1-24 multiplex\np_arrays'
os.chdir(numpy_folder)
file_name = 'fm_array.npy'
fm_array = np.load(file_name, allow_pickle=False)

# find tile counts
x_tiles = np.shape(fm_array[0])[1]
y_tiles = np.shape(fm_array[0])[0]

def col_row_nonzero(image):

    #determine row and column counts
    row_count = np.shape(image)[0]
    column_count = np.shape(image)[1]

    # define row and column arrays
    row_nonzero = np.zeros((row_count))
    column_nonzero = np.zeros((column_count))

    #find column indicies with non zero values
    for x in range(0, column_count):
        column = image[::, x]
        sum = np.sum(column)
        if sum > 0:
            column_nonzero[x] = 1
        else:
            pass

    # find row indicies with non zero values
    for y in range(0, row_count):
        row = image[y, ::]
        sum = np.sum(row)
        if sum > 0:
            row_nonzero[y] = 1
        else:
            pass

    return row_nonzero, column_nonzero


def fm_grid_readjuster(experiment_directory, x_frame_size):
    numpy_path = experiment_directory + r'\np_arrays'
    tissue_path = experiment_directory + r'\Tissue_Binary'

    os.chdir(numpy_folder)
    file_name = 'fm_array.npy'
    fm_array = np.load(file_name, allow_pickle=False)

    # find tile counts
    x_tiles = np.shape(fm_array[0])[1]
    y_tiles = np.shape(fm_array[0])[0]

    #make row and column arrays that can contain all tissue images in row or col respectively
    row_image = np.random.rand(2960, x_frame_size * x_tiles).astype('float16')
    col_image = np.random.rand(2960 * y_tiles, x_frame_size ).astype('float16')

    os.chdir(tissue_path)

    found_upper_y = 0
    found_lower_y = 0
    found_upper_x = 0
    found_lower_x = 0


    # find upper and lower tissue containing tiles and what rows within them where the tissue starts showing up

    #upper row
    for y in range(0, y_tiles):
        if found_upper_y == 0:
            for x in range(0, x_tiles):
                # populate row image
                filename = 'x' + str(x) + '_y_' + str(y) + '_tissue.tif'
                individual_image = io.imread(filename)
                start_column = x * x_frame_size
                end_column = start_column + x_frame_size
                row_image[::, start_column:end_column] = individual_image
            row_array, col_array = col_row_nonzero(row_image)
            try:
                row_indicies = np.nonzero(row_array)[0]
                upper_y_index = row_indicies[0]
                upper_y_tile = y
                found_upper_y = 1
            except:
                pass

        else:
            pass

    #lower row
    for y in range(y_tiles - 1, -1, -1):
        if found_lower_y == 0:
            for x in range(0, x_tiles):
                # populate row image
                filename = 'x' + str(x) + '_y_' + str(y) + '_tissue.tif'
                individual_image = io.imread(filename)
                start_column = x * x_frame_size
                end_column = start_column + x_frame_size
                row_image[::, start_column:end_column] = individual_image

            row_image = np.flipud(row_image)
            row_array, col_array = col_row_nonzero(row_image)
            try:
                row_indicies = np.nonzero(row_array)[0]
                lower_y_index = 2960 - row_indicies[0]
                lower_y_tile = y
                found_lower_y = 1
            except:
                pass

        else:
            pass


    #upper column
    for x in range(x_tiles - 1, -1, -1):
        if found_upper_x == 0:
            for y in range(0, y_tiles):
                # populate row image
                filename = 'x' + str(x) + '_y_' + str(y) + '_tissue.tif'
                individual_image = io.imread(filename)
                start_row = y * x_frame_size
                end_row = start_row + 2960
                col_image[start_row:end_row, ::] = individual_image

            col_image = np.fliplr(col_image)
            row_array, col_array = col_row_nonzero(col_image)
            try:
                col_indicies = np.nonzero(col_array)[0]
                upper_x_index = x_frame_size - col_indicies[0]
                upper_x_tile = x
                found_upper_x = 1
            except:
                pass

        else:
            pass

    #lower column
    for x in range(0, x_tiles):
        if found_lower_x == 0:
            for y in range(0, y_tiles):
                # populate row image
                filename = 'x' + str(x) + '_y_' + str(y) + '_tissue.tif'
                individual_image = io.imread(filename)
                start_row = y * x_frame_size
                end_row = start_row + 2960
                col_image[start_row:end_row, ::] = individual_image

            row_array, col_array = col_row_nonzero(col_image)
            try:
                col_indicies = np.nonzero(col_array)[0]
                lower_x_index = col_indicies[0]
                lower_x_tile = x
                found_lower_x = 1
            except:
                pass

        else:
            pass


    # determine new X and Y grid size

    x_tile_range = fm_array[0][0][upper_x_tile] - fm_array[0][0][lower_x_tile] + (x_frame_size/2 - lower_x_index) * 0.204 + (upper_x_index - x_frame_size/2) * 0.204
    y_tile_range = fm_array[1][lower_y_tile][0] - fm_array[1][upper_y_tile][0] + (2960 / 2 - upper_y_index) * 0.204 + (lower_y_index - 2960 / 2) * 0.204

    # determine min number tiles to encompass tissue

    x_new_tiles = math.ceil(x_tile_range/(x_frame_size * 0.9 * 0.204))
    y_new_tiles = math.ceil(y_tile_range / (2960 * 0.9 * 0.204))

    # determine displacement vector for xy grid
    margin_frame_x_2_tissue = (((x_new_tiles - 1) * 0.9 + 1) * x_frame_size * 0.204 - x_tile_range)/2
    displacement_x = (lower_x_index * 0.204 - margin_frame_x_2_tissue)

    margin_frame_y_2_tissue = (((y_new_tiles - 1) * 0.9 + 1) * 2960 * 0.204 - y_tile_range) / 2
    displacement_y = (upper_y_index * 0.204 - margin_frame_y_2_tissue)
    print(displacement_x, displacement_y)
    print(fm_array[0][upper_y_tile:y_new_tiles, lower_x_tile:x_new_tiles])








experiment_directory = r'D:\Images\AutoCyPlex\6-1-24 multiplex'

fm_grid_readjuster(experiment_directory, 2960)


