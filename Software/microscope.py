import ome_types
from pycromanager import Core,Magellan
import numpy as np
import time
from skimage import io, filters, morphology
import skimage
import os
import math
from tifffile import imwrite
import tifffile as tf
from openpyxl import load_workbook, Workbook
from skspatial.objects import Plane, Points
from sklearn.linear_model import HuberRegressor
from ome_types import from_xml, OME, to_xml
from copy import deepcopy
from pystackreg import StackReg
from pybasic import shading_correction
from path import Path
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from matplotlib import pyplot as plt
import cv2

#magellan = Magellan()
#core = Core()

class cycif:

    def __init__(self):

        return

    ####################################################################
    ############ focus scoring
    ####################################################################

    def focus_score(self, image, derivative_jump, labels):
        '''
        Calculates focus score on image with Brenners algorithm on downsampled image.


        :param numpy image: single image from hooked from acquistion

        :return: focus score for image
        :rtype: float
        '''
        # Note: Uniform background is a bit mandatory

        # do Brenner score


        a = image[derivative_jump:, :]
        a = a.astype('float64')
        a = np.nan_to_num(a, posinf=65000)
        b = image[:-derivative_jump, :]
        b = b.astype('float64')
        b = np.nan_to_num(b, posinf=65000)
        c = (a - b)
        c = c / 1000 * c / 1000
        labels = labels[derivative_jump:, :]
        c = c * labels
        f_score_shadow = c.sum(dtype=np.float64)

        return f_score_shadow

    def focus_score_post_processing(self, image, derivative_jump):
        '''
        Calculates focus score on image with Brenners algorithm on downsampled image.


        :param numpy image: single image from hooked from acquistion

        :return: focus score for image
        :rtype: float
        '''
        # Note: Uniform background is a bit mandatory

        # do Brenner score

        a = image[derivative_jump:, :]
        a = a.astype('float64')
        a = np.nan_to_num(a, posinf=65000)
        b = image[:-derivative_jump, :]
        b = b.astype('float64')
        b = np.nan_to_num(b, posinf=65000)
        c = (a - b)
        c = c / 1000 * c / 1000
        f_score_shadow = c.sum(dtype=np.float64) + 0.00001

        return f_score_shadow

    #########################################################
    # Setup fm_array and exp_array
    #########################################################

    def establish_fm_array(self, experiment_directory, desired_cycle_count, z_slices, off_array, initialize=0,
                           x_frame_size=5056, autofocus=0, auto_expose=0):

        self.file_structure(experiment_directory, desired_cycle_count)

        if initialize == 1:
            xy_points = self.tile_xy_pos('New Surface 1')
            xyz_points = self.nonfocus_tile_DAPI(xy_points, experiment_directory)
            self.tile_pattern(xyz_points, experiment_directory)
            self.fm_channel_initial(experiment_directory, off_array, z_slices)
            self.establish_exp_arrays(experiment_directory)

            if x_frame_size != 5056:
                self.x_overlap_adjuster(x_frame_size, experiment_directory)
                self.establish_exp_arrays(experiment_directory)
            else:
                pass

        else:
            pass

        if autofocus == 1 and auto_expose == 1:
            self.recursive_stardist_autofocus(experiment_directory, desired_cycle_count)
            self.establish_exp_arrays(experiment_directory)
            self.auto_exposure(experiment_directory, x_frame_size, percentage_cut_off = 0.997, target_percentage = 0.1)
        if autofocus == 1 and auto_expose == 0:
            #self.DAPI_surface_autofocus(experiment_directory, 20, 2, x_frame_size)
            self.recursive_stardist_autofocus(experiment_directory, desired_cycle_count)
            #self.fm_channel_initial(experiment_directory, off_array, z_slices, 2)
        if autofocus == 0 and auto_expose == 1:
            self.establish_exp_arrays(experiment_directory)
            self.auto_exposure(experiment_directory, x_frame_size, percentage_cut_off = 0.997, target_percentage = 0.1)
        else:
            pass

        self.fm_channel_update(experiment_directory, off_array, z_slices)

    def establish_exp_arrays(self, experiment_directory):
        '''
        Make exp_array with default exp times and exp_calc_array to use with auto exposure

        :param experiment_directory:
        :return:
        '''
        numpy_path = experiment_directory + '/' + 'np_arrays'
        os.chdir(numpy_path)
        fm_array = np.load('fm_array.npy', allow_pickle=False)

        y_tiles = np.shape(fm_array[0])[0]
        x_tiles = np.shape(fm_array[0])[1]

        exp_calc_array = np.random.rand(4, 3, y_tiles, x_tiles)
        exp_array = [100, 100, 100, 100]
        exp_calc_array[::, 0, ::, ::] = 100

        file_name = 'exp_calc_array.npy'
        np.save(file_name, exp_calc_array)
        np.save('exp_array.npy', exp_array)


    ###########################################################
    # This section is the for the exposure functions.
    ###########################################################
    def auto_exposure(self, experiment_directory, x_frame_size, percentage_cut_off = 0.99, target_percentage = 0.08 ):

        # load in data structures
        numpy_path = experiment_directory + '/' + 'np_arrays'
        os.chdir(numpy_path)
        file_name = 'fm_array.npy'
        exp_filename = 'exp_array.npy'
        fm_array = np.load(file_name, allow_pickle=False)
        exp_array = np.load(exp_filename, allow_pickle=False)
        channels = ['DAPI', 'A488', 'A555', 'A647']
        number_channels = len(channels[0])
        slice_gap = 2

        #find tile counts
        x_tiles = np.shape(fm_array[0])[1]
        y_tiles = np.shape(fm_array[0])[0]
        tissue_fm = fm_array[10]

        #get XY arrays
        numpy_x = fm_array[0]
        numpy_y = fm_array[1]

        #make psuedo stack to hold images in
        #exp_image_stack = np.random.rand(number_channels, y_tiles, x_tiles, 2960, x_frame_size)
        exp_image_stack = np.zeros((number_channels, y_tiles, x_tiles, 2960, x_frame_size))

        #define pixel range in X dimension
        side_pixel_count = int((5056 - x_frame_size)/2)
        start_frame = side_pixel_count
        end_frame = side_pixel_count + x_frame_size

        start = time.time()

        for x in range(0, x_tiles):
            for y in range(0, y_tiles):
                print('x', x, 'y', y)

                #import tissue binary image
                tissue_path = experiment_directory + '/Tissue_Binary'
                os.chdir(tissue_path)
                tissue_binary_name = 'x' + str(x) + '_y_' + str(y) + '_tissue.tif'
                tissue_bin_im = io.imread(tissue_binary_name)

                if tissue_fm[y][x] == 1:

                    for channel in channels:
                        if channel == 'DAPI':
                            channel_index = 0
                        if channel == 'A488':
                            channel_index = 1
                        if channel == 'A555':
                            channel_index = 2
                        if channel == 'A647':
                            channel_index = 3

                        #exp time to use for snap shot image
                        exp_time = exp_array[channel_index]

                        #Go to XY location (mid stack)
                        core.set_xy_position(numpy_x[y][x], numpy_y[y][x])
                        time.sleep(0.3)

                        #Go to Z location
                        numpy_z = fm_array[channel_index*2 + 2]
                        slice_count = fm_array[channel_index*2 + 3][y][x]
                        z_range = slice_count * slice_gap
                        center_z = int(numpy_z[y][x] - z_range/2)
                        core.set_position(center_z)
                        time.sleep(0.3)

                        #set channel and exp time
                        exp_time = int(exp_array[channel_index])
                        core.set_config("Color", channel)
                        core.set_exposure(exp_time)

                        #take reference image
                        core.snap_image()
                        tagged_image = core.get_tagged_image()
                        pixels = np.reshape(tagged_image.pix,
                                            newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"]])
                        pixels = pixels[::, side_pixel_count:side_pixel_count + x_frame_size]

                        #Determine if intensity is in bounds and take diff image if not. Record new exp time
                        pixels, exp_time = self.exp_bound_solver(pixels, exp_time, 0.99)
                        exp_array[channel_index] = exp_time

                        #load in auto fluorescence image
                        auto_fl_path = experiment_directory + '/' + str(channel) + '/Bleach/' + 'cy_0/' + 'Tiles'
                        os.chdir(auto_fl_path)
                        filename = 'z_' + str(int(slice_count/2)) + '_x' + str(x) + '_y_' + str(y) + '_c_' + str(channel) + '.tif'
                        auto_fl_im = io.imread(filename)

                        #find ideal constant and sub off auto_fl_im from snapped image

                        if channel_index == 0:
                            subbed_image = pixels
                        else:
                            factor = self.autof_factor_estimator(pixels, auto_fl_im)
                            subbed_image = pixels - factor*auto_fl_im
                            subbed_image[subbed_image < 0] = 0
                            subbed_image = np.nan_to_num(subbed_image, posinf= 65500)

                        #multiply by tissue binary
                        masked_subbed_image = subbed_image * tissue_bin_im

                        #place subbed image into stack
                        exp_image_stack[channel_index][y][x] = masked_subbed_image

                if tissue_fm[y][x] == 0:
                    pass

        #save image array for analysis
        numpy_path = experiment_directory + '/' + 'np_arrays'
        os.chdir(numpy_path)
        np.save('auto_exp_images.npy', exp_image_stack)


        for channel in channels:
            if channel == 'DAPI':
                channel_index = 0
            if channel == 'A488':
                channel_index = 1
            if channel == 'A555':
                channel_index = 2
            if channel == 'A647':
                channel_index = 3

            # use Otsu's (or any other threshold method) to find well stained pixels
            channel_pixels = exp_image_stack[channel_index]
            non_zero_pixels = channel_pixels.ravel()[np.flatnonzero(channel_pixels)]
            thresh = filters.threshold_otsu(non_zero_pixels)

            #apply threshold and find brightest and dimmest pixels
            non_zero_thresh_pixels = channel_pixels.ravel()[np.flatnonzero(channel_pixels > thresh)]
            high_pixel, low_pixel = self.image_percentile_level(non_zero_thresh_pixels, cut_off_threshold= percentage_cut_off)

            #find new exp factor
            new_exp_factor = target_percentage * 65500 / low_pixel * exp_array[channel_index]

            if new_exp_factor > 200:
                new_exp_factor = 200
            if new_exp_factor < 50:
                new_exp_factor = 50

            #add to exp_array
            exp_array[channel_index] = new_exp_factor

        #save new exp_array
        finish = time.time()

        print('auto_exp time', finish - start)


        numpy_path = experiment_directory + '/' + 'np_arrays'
        os.chdir(numpy_path)
        print(exp_array)
        avg_exp_time = (exp_array[0] + exp_array[1] + exp_array[2] + exp_array[3])/4
        total_exp= 7*4*avg_exp_time * 8
        print('time exposing', total_exp)
        np.save('exp_array.npy', exp_array)

    def autof_factor_estimator(self, image, autof_image, num_images=2):
        top_range = 20
        x_factor = top_range / num_images

        image = np.nan_to_num(image, posinf= 65500)
        autof_image = np.nan_to_num(autof_image, posinf= 65500)

        image = image.astype('float32')
        autof_image = autof_image.astype('float32')

        x_axis = np.linspace(0, top_range, num_images).astype('float32')
        y_axis = np.linspace(0, top_range, num_images).astype('float32')
        for x in range(0, num_images):
            input_image = (image - x_factor * x * autof_image)
            mean = self.absolute_mean(input_image)
            y_axis[x] = mean
            x_axis[x] = x * x_factor

        projected_min_point = self.min_factor(x_axis, y_axis)

        return projected_min_point

    def absolute_mean(self, image):
        abs = np.absolute(image)
        mean = np.mean(abs)

        return mean

    def min_factor(self, x_axis, y_axis):
        sorted_list = np.sort(y_axis)
        index_1 = np.where(y_axis == sorted_list[0])[0][0]
        index_1 = x_axis[index_1]
        index_2 = np.where(y_axis == sorted_list[1])[0][0]
        index_2 = x_axis[index_2]
        min_value_1 = sorted_list[0]
        min_value_2 = sorted_list[1]

        total_value = min_value_1 + min_value_2
        rel_value_1 = total_value / min_value_1
        rel_value_2 = total_value / min_value_2
        total_rel_value = rel_value_1 + rel_value_2
        normalized_rel_value_1 = rel_value_1 / total_rel_value
        normalized_rel_value_2 = rel_value_2 / total_rel_value
        projected_min_point = normalized_rel_value_1 * index_1 + normalized_rel_value_2 * index_2

        return projected_min_point

    def image_percentile_level(self, image, cut_off_threshold=0.99):
        '''
        Takes in image and cut off threshold and finds pixel value that exists at that threshold point.

        :param numpy array image: numpy array image
        :param float cut_off_threshold: percentile for cut off. For example a 0.99 would disregaurd the top 1% of pixels from calculations
        :return: intensity og pixel that resides at the cut off fraction that was entered in the image
        :rtype: int
        '''

        pixel_values = np.sort(image, axis=None)
        indicies = np.nonzero(pixel_values)[0]
        # thresh = filters.threshold_otsu(pixel_values[indicies])
        # indicies = np.where(pixel_values > thresh)[0]
        pixel_values = pixel_values[indicies]

        pixel_count = int(np.size(pixel_values))
        cut_off_index = int(pixel_count * cut_off_threshold)
        min_cut_off_index = int(pixel_count - cut_off_index)
        tail_intensity = pixel_values[cut_off_index]
        min_intensity = pixel_values[min_cut_off_index]

        return tail_intensity, min_intensity

    def exp_bound_solver(self, image, exp_time, percentage_cutoff):
        '''
        Takes in image and determines if its in bounds. If not, it adjusts exp time to compensate and gives indicator
        if exp time alterations were used, ie was it triggered.

        :param image:
        :param exp_time:
        :return:
        '''

        target_intensity = 65535 * 0.05
        max_time = 100
        min_time = 30
        trigger_state = 0

        x_frame_size = np.shape(image)[1]

        intensity, low_intensity = self.image_percentile_level(image, percentage_cutoff)
        #print(intensity)

        while intensity > 50000:
            exp_time = exp_time / 3
            core.set_exposure(exp_time)
            core.snap_image()
            tagged_image = core.get_tagged_image()
            new_image = np.reshape(tagged_image.pix, newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"]])
            intensity, low_intensity = self.image_percentile_level(new_image, cut_off_threshold=percentage_cutoff)
            trigger_state = 1

        if intensity < 1000 and exp_time < max_time:
            trigger_state = 1

        if trigger_state == 1:
            scale_factor = target_intensity / (intensity - 300)
            if scale_factor * exp_time > max_time:
                exp_time = max_time
            if scale_factor < min_time:
                scale_factor = min_time
            else:
                exp_time = int(exp_time * scale_factor)

        if trigger_state == 1:

            #gather new image
            core.set_exposure(exp_time)
            core.snap_image()
            tagged_image = core.get_tagged_image()
            image = np.reshape(tagged_image.pix, newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"]])
            #crop image to equal first image
            side_pixel_count = int((5056 - x_frame_size)/2)
            image = image[::, side_pixel_count:side_pixel_count + x_frame_size]
        else:
            pass


        return image, exp_time

    def exp_logbook(self, experiment_directory, cycle):
        '''
        Opens up numpy object for exp times and records in excel sheet
        :param experiment_directory:
        :param cycle:
        :return:
        '''

        # import exp_calc_array
        numpy_path = experiment_directory + '/' + 'np_arrays'
        os.chdir(numpy_path)
        exp_array = np.load('exp_array.npy', allow_pickle=False)

        exp_path = experiment_directory + '/' + 'exposure_times'
        os.chdir(exp_path)

        # create or open workbook

        if os.path.isfile('Exp.xlsx') == False:
            wb = Workbook()
            ws = wb.active

            # populate headers
            ws.cell(row = 1, column = 1).value = 'Cycle #'
            ws.cell(row=1, column=2).value = 'Exp Time DAPI'
            #ws.cell(row=1, column=3).value = 'Average # DAPI'
            ws.cell(row=1, column=4).value = 'Exp Time A488'
            #ws.cell(row=1, column=5).value = 'Average # A488'
            ws.cell(row=1, column=6).value = 'Exp Time A555'
            #ws.cell(row=1, column=7).value = 'Average # A555'
            ws.cell(row=1, column=8).value = 'Exp Time A647'
            #ws.cell(row=1, column=9).value = 'Average # A647'

        if os.path.isfile('Exp.xlsx') == True:
            wb = load_workbook('Exp.xlsx')
            ws = wb.active

        # populate columns with times and cycle count
        ws.cell(row=int(cycle + 2), column=1).value = cycle

        ws.cell(row=int(cycle + 2), column=2).value = exp_array[0]
        ws.cell(row=int(cycle + 2), column=4).value = exp_array[1]
        ws.cell(row=int(cycle + 2), column=6).value = exp_array[2]
        ws.cell(row=int(cycle + 2), column=8).value = exp_array[3]

        wb.save('Exp.xlsx')
    ##########################################################
    #Initial population of fm_array and some mods to it
    ##########################################################

    def tile_xy_pos(self, surface_name):
        '''
        imports previously generated micro-magellan surface with name surface_name and outputs
        the coordinates of the center of each tile from it.

        :param str surface_name: name of micro-magellan surface
        :param object magellan: object created via = bridge.get_magellan()

        :return: XY coordinates of the center of each tile from micro-magellan surface
        :rtype: dictionary {{x:(float)}, {y:(float)}}
        '''
        surface = magellan.get_surface(surface_name)
        num = surface.get_num_positions()
        xy = surface.get_xy_positions()
        tile_points_xy = {}
        x_temp = []
        y_temp = []

        for q in range(0, num):
            pos = xy.get(q)
            pos = pos.get_center()
            x_temp.append(pos.x)
            y_temp.append(pos.y)

        tile_points_xy['x'] = x_temp  ## put all points in dictionary to ease use
        tile_points_xy['y'] = y_temp
        x_temp = np.array(tile_points_xy['x'])
        y_temp = np.array(tile_points_xy['y'])

        xy = np.append(x_temp, y_temp)
        xy = np.reshape(xy, (2, int(xy.size / 2)))

        return xy

    def tile_pattern(self, numpy_array, experiment_directory):
        '''
        Takes numpy array with N rows and known tile pattern and casts into new array that follows
        south-north, west-east snake pattern.


        :param numpy_array: dimensions [N, x_tiles*y_tiles]
        :param x_tiles: number x tiles in pattern
        :param y_tiles: number y tiles in pattern
        :return: numpy array with dimensions [N,x_tiles,y_tiles] with above snake pattern
        '''
        numpy_path = experiment_directory + '/' + 'np_arrays'
        os.chdir(numpy_path)
        filename = 'fm_array.npy'

        y_tiles = np.unique(numpy_array[1]).size
        x_tiles = np.unique(numpy_array[0]).size
        layers = np.shape(numpy_array)[0]
        numpy_array = numpy_array.reshape(layers, x_tiles, y_tiles)
        dummy = numpy_array.reshape(layers, y_tiles, x_tiles)
        new_numpy = np.empty_like(dummy)
        for x in range(0, layers):
            new_numpy[x] = numpy_array[x].transpose()
            new_numpy[x, ::, 1:y_tiles:2] = np.flipud(new_numpy[x, ::, 1:y_tiles:2])

        np.save(filename, new_numpy)

        return new_numpy

    def fm_channel_update(self, experiment_directory, off_array, z_slices, slice_gap = 2):

        numpy_path = experiment_directory + '/' + 'np_arrays'
        os.chdir(numpy_path)
        file_name = 'fm_array.npy'
        fm_array = np.load(file_name, allow_pickle=False)

        a488_channel_offset = off_array[1]  # determine if each of these are good and repeatable offsets
        a555_channel_offset = off_array[2]
        a647_channel_offset = off_array[3]

        fm_array[4] = fm_array[2] + a488_channel_offset  # index for a488 = 3
        fm_array[6] = fm_array[2] + a555_channel_offset
        fm_array[8] = fm_array[2] + a647_channel_offset

        np.save(file_name, fm_array)

    def fm_channel_initial(self, experiment_directory, off_array, z_slices, slice_gap=2):

        numpy_path = experiment_directory + '/' + 'np_arrays'
        os.chdir(numpy_path)
        file_name = 'fm_array.npy'
        fm_array = np.load(file_name, allow_pickle=False)

        a488_channel_offset = off_array[1]  # determine if each of these are good and repeatable offsets
        a555_channel_offset = off_array[2]
        a647_channel_offset = off_array[3]

        y_tiles = np.shape(fm_array[0])[0]
        x_tiles = np.shape(fm_array[0])[1]

        dummy_channel = np.empty_like(fm_array[0])
        dummy_channel = np.expand_dims(dummy_channel, axis=0)
        channel_count = np.shape(fm_array)[0]

        while channel_count < 11:
            fm_array = np.append(fm_array, dummy_channel, axis=0)
            channel_count = np.shape(fm_array)[0]

        fm_array[4] = fm_array[2] + a488_channel_offset  # index for a488 = 3
        fm_array[6] = fm_array[2] + a555_channel_offset
        fm_array[8] = fm_array[2] + a647_channel_offset
        y_tiles = int(np.shape(fm_array[0])[0])
        x_tiles = int(np.shape(fm_array[0])[1])
        z_slice_array = np.full((y_tiles, x_tiles), z_slices)
        tissue_all_ones_array = np.full((y_tiles, x_tiles), 1)

        fm_array[3] = z_slice_array
        fm_array[5] = z_slice_array
        fm_array[7] = z_slice_array
        fm_array[9] = z_slice_array

        fm_array[10] = tissue_all_ones_array

        fm_array[2] = fm_array[2] + int(((z_slice_array[0][0] - 1) * slice_gap) / 2)
        fm_array[4] = fm_array[4] + int(((z_slice_array[0][0] - 1) * slice_gap) / 2)
        fm_array[6] = fm_array[6] + int(((z_slice_array[0][0] - 1) * slice_gap) / 2)
        fm_array[8] = fm_array[8] + int(((z_slice_array[0][0] - 1) * slice_gap) / 2)

        np.save(file_name, fm_array)

        return fm_array

    def x_overlap_adjuster(self, new_x_pixel_count, experiment_directory):
        '''
        Increases overlap in focus map while cropping in the x dimension to preserve effective 10% overlap of cropped images
        :param new_x_pixel_count:
        :param experiment_directory:
        :return:
        '''

        # load in fm array
        numpy_path = experiment_directory + '/' + 'np_arrays'
        os.chdir(numpy_path)
        file_name = 'fm_array.npy'
        fm_array = np.load(file_name, allow_pickle=False)

        # Find pixels in each dimensions
        x_tiles = np.shape(fm_array[0])[1]
        y_tiles = np.shape(fm_array[0])[0]

        # find um/pixel in focus map

        try:
            x1 = fm_array[0][0][0]
            x2 = fm_array[0][0][1]
            diff = x2 - x1
        except:
            y1 = fm_array[0][0][0]
            y2 = fm_array[0][1][0]
            diff = y2 - y1

        um_per_pixel = diff / 4550  # 4550 = 0.9 * 5056

        # Find number tiles in adjusted grid
        x_range_pixels = (x_tiles - 0.2) * 5056
        number_new_x_dim_tiles = x_range_pixels / new_x_pixel_count
        new_x_tiles = math.ceil(number_new_x_dim_tiles)

        # generate new blank fm_array numpy array

        new_fm_array = np.random.rand(11, y_tiles, new_x_tiles).astype('float64')

        # Find border where x starts on the left (not center point, but x value for left most edge of left most tile

        left_x_center = fm_array[0][0][0]
        left_most_x = left_x_center - 2528 * um_per_pixel

        # Find center point in new image that makes edge of image align with left_most_x
        # Also find x to x + i spacing and populate rest of x values in new_fm_array

        x_col_0 = left_most_x + new_x_pixel_count / 2 * um_per_pixel
        x_spacing = (0.9) * new_x_pixel_count * um_per_pixel

        # Populate new_fm_array with row 0 x values

        for x in range(0, new_x_tiles):
            new_fm_array[0][0:y_tiles, x] = x_col_0 + x * x_spacing

        # populate new_fm_array with y values

        for y in range(0, y_tiles):
            new_fm_array[1][y, 0:new_x_tiles] = fm_array[1][y][0]

        # populate new_fm_array with dapi z values and everything else in planes 2-10
        for slice in range(2, 11):
            new_fm_array[slice, 0:y_tiles, 0:new_x_tiles] = fm_array[slice][0][0]

        np.save('fm_array.npy', new_fm_array)

    def nonfocus_tile_DAPI(self, full_array_no_pattern, experiment_directory):
        '''
        Makes micromagellen z the focus position at each XY point for DAPI
        auto_focus and outputs the paired in focus z coordinate

        :param dictionary tile_points_xy: dictionary containing all XY coordinates. In the form: {{x:(int)}, {y:(int)}}

        :return: XYZ points where XY are stage coords and Z is in focus coordinate. {{x:(int)}, {y:(int)}, {z:(float)}}
        :rtype: dictionary
        '''

        numpy_path = experiment_directory + '/' + 'np_arrays'
        os.chdir(numpy_path)
        file_name = 'fm_array.npy'

        z_pos = magellan.get_surface('New Surface 1').get_points().get(0).z
        num = np.shape(full_array_no_pattern)[1]
        z_temp = []
        for q in range(0, num):
            z_temp.append(z_pos)
        z_temp = np.expand_dims(z_temp, axis=0)
        xyz = np.append(full_array_no_pattern, z_temp, axis=0)

        np.save(file_name, xyz)

        return xyz

    #functions for itdentifying is tissue is present or not###################

    def tissue_region_identifier(self, experiment_directory):
        '''
        Looks at tissue binary images and updates fm object to encode information
        :return:
        '''

        # load in fm array
        numpy_path = experiment_directory + '/' + 'np_arrays'
        os.chdir(numpy_path)
        file_name = 'fm_array.npy'
        fm_array = np.load(file_name, allow_pickle=False)

        # Find pixels in each dimensions
        x_tiles = np.shape(fm_array[0])[1]
        y_tiles = np.shape(fm_array[0])[0]

        #make tissue binary images
        self.tissue_binary_generate(experiment_directory)
        tissue_path = experiment_directory + '/Tissue_Binary'
        os.chdir(tissue_path)

        tissue_status = 1

        for x in range(0, x_tiles):
            for y in range(0, y_tiles):

                tissue_binary_name = 'x' + str(x) + '_y_' + str(y) + '_tissue.tif'
                im = io.imread(tissue_binary_name)
                sum = np.sum(im)

                if sum > 0:
                    tissue_status = 1
                if sum == 0:
                    tissue_status = 0

                fm_array[10][y][x] = tissue_status

        os.chdir(numpy_path)
        np.save(file_name, fm_array)

    def tissue_filter(self, image):

        image = image.astype('bool')
        image_2 = morphology.remove_small_objects(image, min_size=80000, connectivity=1)
        image_2 = image_2.astype('int8')

        return image_2

    def tissue_binary_generate(self, experiment_directory):
        '''
        Generates tissue binary maps from star dist binary maps

        :param experiment_directory:
        :return:
        '''

        # load numpy arrays in
        numpy_path = experiment_directory + '/' + 'np_arrays'
        os.chdir(numpy_path)
        full_array = np.load('fm_array.npy', allow_pickle=False)

        numpy_x = full_array[0]
        numpy_y = full_array[1]

        y_tile_count = numpy_x.shape[0]
        x_tile_count = numpy_y.shape[1]

        star_dist_path = experiment_directory + '/Labelled_Nuc'
        tissue_path = experiment_directory + '/Tissue_Binary'

        try:
            os.chdir(tissue_path)
        except:
            os.chdir(experiment_directory)
            os.mkdir('Tissue_Binary')

        foot_print = morphology.disk(70, decomposition='sequence')

        for x in range(0, x_tile_count):
            for y in range(0, y_tile_count):

                os.chdir(star_dist_path)
                star_dist_filename =  'x' + str(x) + '_y_' + str(y) + '_c_DAPI.tif'
                star_dist_im = io.imread(star_dist_filename)

                tissue_binary_im = morphology.binary_dilation(star_dist_im, foot_print)
                tissue_binary_im = tissue_binary_im.astype(np.uint8)
                filtered_image = self.tissue_filter(tissue_binary_im)
                os.chdir(tissue_path)
                tissue_binary_name = 'x' + str(x) + '_y_' + str(y) + '_tissue.tif'
                io.imsave(tissue_binary_name, filtered_image)

    #recursize autofocusfunctions#####################################

    def highest_index(self, score_array):


        middle_index = (np.shape(score_array)[0] - 1)/2
        middle_index = int(middle_index)

        max_score = np.max(score_array)
        index = np.where(score_array == max_score)
        try:
            index = index[0][0]
        except:
            index = (np.shape(score_array)[0] - 1) / 2
            index = int(index)

        return index

    def generate_nuc_mask(self, experiment_directory):

        model = StarDist2D.from_pretrained('2D_versatile_fluo')

        # load numpy arrays in
        numpy_path = experiment_directory + '/' + 'np_arrays'
        os.chdir(numpy_path)
        full_array = np.load('fm_array.npy', allow_pickle=False)

        numpy_x = full_array[0]
        numpy_y = full_array[1]

        y_tile_count = numpy_x.shape[0]
        x_tile_count = numpy_y.shape[1]

        dapi_im_path = experiment_directory + '/' + 'DAPI' '\Bleach\cy_' + str(0) + '\Tiles'
        os.chdir(experiment_directory)
        try:
            os.mkdir('Labelled_Nuc')
        except:
            pass

        labelled_path = experiment_directory + '/Labelled_Nuc'

        for x in range(0, x_tile_count):
            for y in range(0, y_tile_count):
                os.chdir(dapi_im_path)
                file_name = 'z_' + str(4) + '_x' + str(x) + '_y_' + str(y) + '_c_DAPI.tif'
                labelled_file_name = 'x' + str(x) + '_y_' + str(y) + '_c_DAPI.tif'
                img = io.imread(file_name)
                labels, _ = model.predict_instances(normalize(img))
                labels[labels > 0] = 1

                os.chdir(labelled_path)
                io.imsave(labelled_file_name, labels)

    def recursive_stardist_autofocus(self, experiment_directory, cycle, slice_gap=2):
        '''
        Finds center of dapi z stack for each tile and updates focus map to center it.
        Cycle 1 uses cycle 0 dapi z stacks. So cycle n-1 informs focus map for cycle n.
        A pretrained network for stardist is used to screen out junk in images and increase
        reliability. Note, must have pregenerated binary nuclei images available to use.
        See generate_nuc_mask function to make them. This automakes if cycle = 0.

        :param experimental_directory:
        :param cycle:
        :return:
        '''

        labelled_path = experiment_directory + '/Labelled_Nuc'
        if cycle == 1:
            dapi_im_path = experiment_directory + '/' + 'DAPI' '\Bleach\cy_' + str(cycle - 1) + '\Tiles'
        else:
            dapi_im_path = experiment_directory + '/' + 'DAPI' '\Stain\cy_' + str(cycle - 1) + '\Tiles'

        # make nuclear masks if cycle 0

        if cycle == 1:
            self.generate_nuc_mask(experiment_directory)
            self.tissue_region_identifier(experiment_directory)
        else:
            pass


        # load numpy arrays in (focus map)
        numpy_path = experiment_directory + '/' + 'np_arrays'
        os.chdir(numpy_path)
        fm_array = np.load('fm_array.npy', allow_pickle=False)

        numpy_x = fm_array[0]
        numpy_y = fm_array[1]
        tissue_fm = fm_array[10]

        y_tile_count = numpy_x.shape[0]
        x_tile_count = numpy_y.shape[1]
        #y_tile_count = 1
        #x_tile_count = 1
        z_count = int(fm_array[3][0][0])
        z_middle = int(z_count / 2)  # if odd, z_count will round down. Since index counts from 0, it is the middle

        step_size = 17  # brenner score step size
        x_axis = np.array([0,1,2,3,4,5,6,7,8])


        # make numpy array to hold scores in for each tile
        score_array = np.random.rand(z_count)

        # iterate through tiles and find index of slice most in focus
        for x in range(0, x_tile_count):
            for y in range(0, y_tile_count):

                if tissue_fm[y][x] == 1:

                    for z in range(0, z_count):
                        # load in binary image mask
                        os.chdir(labelled_path)
                        file_name = 'x' + str(x) + '_y_' + str(y) + '_c_DAPI.tif'
                        labels = io.imread(file_name)
                        # load in z slice
                        os.chdir(dapi_im_path)
                        file_name = 'z_' + str(z) + '_x' + str(x) + '_y_' + str(y) + '_c_DAPI.tif'
                        img = io.imread(file_name)
                        # apply mask to image and find brenner score
                        score = self.focus_score(img, step_size, labels)
                        score_array[z] = score
                    # find highest score slice index and find shift amount
                    focus_index = self.highest_index(score_array)
                    print('cycle',cycle,'x', x, 'y', y)
                    #plt.scatter(x_axis, score_array)
                    #plt.show()
                    center_focus_index_difference = int(focus_index - z_middle)
                    print(center_focus_index_difference)
                    new_fm_z_position = center_focus_index_difference * slice_gap
                    # update focus map for all channels
                    fm_array[2][y][x] = fm_array[2][y][x] + new_fm_z_position

                if tissue_fm[y][x] == 0:
                    pass

        # save updated focus array
        numpy_path = experiment_directory + '/' + 'np_arrays'
        os.chdir(numpy_path)
        np.save('fm_array.npy', fm_array)

    ############################################
    # Using core snap and not pycromanager acquire
    ############################################

    def image_capture(self, experiment_directory, channel, exp_time, x, y, z):

        numpy_path = experiment_directory + '/' + 'np_arrays'
        os.chdir(numpy_path)
        fm_array = np.load('fm_array.npy', allow_pickle=False)

        numpy_x = fm_array[0]
        numpy_y = fm_array[1]
        numpy_z = fm_array[2]
        if z < 20:
            z_pos = numpy_z[y][x] - z*2
        else:
            z_pos = numpy_z[y][x]


        core.set_config("Color", channel)
        core.set_xy_position(numpy_x[y][x], numpy_y[y][x])
        core.set_position(z_pos)
        time.sleep(0.5)
        core.set_exposure(int(exp_time))

        core.snap_image()
        tagged_image = core.get_tagged_image()
        pixels = np.reshape(tagged_image.pix, newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"]])
        # time.sleep(1)

        return pixels

    def multi_channel_z_stack_capture(self, experiment_directory, cycle_number, Stain_or_Bleach,
                                      x_pixels=5056, slice_gap=2, channels=['DAPI', 'A488', 'A555', 'A647']):
        '''
        Captures and saves all images in XY and Z dimensions. Order of operation is ZC XY(snake). Entire z stack with all
        channels is made into a numpy data structure and saved before going to next tile and being reused. This is done
        to reduce overall memory usage.


        :param experiment_directory: directory that main experiment data is stored in
        :param cycle_number: integer of what cycle number it is. Counts from 0
        :param Stain_or_Bleach: string of Stain or Bleach depending on what action it is
        :param slice_gap: micron spacing in z slices
        :param channels: string list of what channels to acquire
        :return:
        '''

        core = Core()

        # load in focus map and exp array
        numpy_path = experiment_directory + '/' + 'np_arrays'
        os.chdir(numpy_path)
        full_array = np.load('fm_array.npy', allow_pickle=False)
        exp_time_array = np.load('exp_array.npy', allow_pickle=False)

        height_pixels = 2960
        width_pixels = x_pixels
        # determine attributes like tile counts,z slices and channel counts
        numpy_x = full_array[0]
        numpy_y = full_array[1]
        tissue_fm = full_array[10]

        side_pixel_count = int(5056 - x_pixels)

        x_tile_count = np.unique(numpy_x).size
        y_tile_count = np.unique(numpy_y).size
        #x_tile_count = 1
        #y_tile_count = 1
        #x_tile_count = 1
        #y_tile_count = 1
        z_slices = full_array[5][0][0]
        # z_slices = 11
        # go to upper left corner to start pattern
        core.set_xy_position(numpy_x[0][0], numpy_y[0][0])
        time.sleep(1)
        # generate numpy data structure
        zc_tif_stack = np.random.rand(4, int(z_slices), height_pixels, width_pixels).astype('float16')

        image_number_counter = 0

        for y in range(0, y_tile_count):
            if y % 2 != 0:
                for x in range(x_tile_count - 1, -1, -1):

                    if tissue_fm[y][x] == 1:

                        core.set_xy_position(numpy_x[y][x], numpy_y[y][x])
                        time.sleep(.3)

                        for channel in channels:

                            # determine the proper indecies to use for focus map z positions and exp array
                            if channel == 'DAPI':
                                channel_index = 2
                                tif_stack_c_index = 0
                                zc_index = 0
                            if channel == 'A488':
                                channel_index = 4
                                tif_stack_c_index = 1
                                zc_index = 1
                            if channel == 'A555':
                                channel_index = 6
                                tif_stack_c_index = 2
                                zc_index = 2
                            if channel == 'A647':
                                channel_index = 8
                                tif_stack_c_index = 3
                                zc_index = 3

                            numpy_z = full_array[channel_index]
                            exp_time = int(exp_time_array[tif_stack_c_index])
                            core.set_config("Color", channel)
                            core.set_exposure(exp_time)

                            z_end = int(numpy_z[y][x])
                            z_start = int(z_end - z_slices * slice_gap)

                            z_counter = 0

                            for z in range(z_start, z_end, slice_gap):
                                core.set_position(z)
                                #time.sleep(0.1)
                                core.snap_image()
                                tagged_image = core.get_tagged_image()
                                pixels = np.reshape(tagged_image.pix,
                                                    newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"]])
                                pixels = np.nan_to_num(pixels, posinf= 65500)
                                zc_tif_stack[zc_index][z_counter] = pixels[::, side_pixel_count:side_pixel_count + x_pixels]

                                # core.pop_next_tagged_image()
                                image_number_counter += 1
                                z_counter += 1

                        # save zc stack
                        self.zc_save(zc_tif_stack, channels, x, y, cycle_number, x_pixels, experiment_directory,
                                     Stain_or_Bleach)

                    if tissue_fm[y][x] == 0:
                        pass


            elif y % 2 == 0:
                for x in range(0, x_tile_count):

                    if tissue_fm[y][x] == 1:

                        #print('x', numpy_x[y][x], 'y', numpy_y[y][x])

                        core.set_xy_position(numpy_x[y][x], numpy_y[y][x])
                        time.sleep(.3)

                        for channel in channels:

                            # determine the proper indecies to use for focus map z positions and exp array
                            if channel == 'DAPI':
                                channel_index = 2
                                tif_stack_c_index = 0
                                zc_index = 0
                            if channel == 'A488':
                                channel_index = 4
                                tif_stack_c_index = 1
                                zc_index = 1
                            if channel == 'A555':
                                channel_index = 6
                                tif_stack_c_index = 2
                                zc_index = 2
                            if channel == 'A647':
                                channel_index = 8
                                tif_stack_c_index = 3
                                zc_index = 3

                            numpy_z = full_array[channel_index]
                            exp_time = int(exp_time_array[tif_stack_c_index])
                            core.set_config("Color", channel)
                            core.set_exposure(exp_time)

                            z_end = int(numpy_z[y][x])
                            z_start = int(z_end - z_slices * slice_gap)
                            z_counter = 0
                            #print('channel', channel, 'z_range', z_start, z_end)

                            for z in range(z_start, z_end, slice_gap):
                                core.set_position(z)
                                #time.sleep(0.1)

                                core.snap_image()
                                tagged_image = core.get_tagged_image()
                                pixels = np.reshape(tagged_image.pix,
                                                    newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"]])
                                pixels = np.nan_to_num(pixels, posinf=65500)
                                zc_tif_stack[zc_index][z_counter] = pixels[::, side_pixel_count:side_pixel_count + x_pixels]

                                # core.pop_next_tagged_image()
                                image_number_counter += 1
                                z_counter += 1

                        self.zc_save(zc_tif_stack, channels, x, y, cycle_number, x_pixels, experiment_directory,
                                     Stain_or_Bleach)

                    if tissue_fm[y][x] == 0:
                        pass

        return

    def image_cycle_acquire(self, cycle_number, experiment_directory, z_slices, stain_bleach, offset_array, x_frame_size=5056, establish_fm_array=0, auto_focus_run=0, auto_expose_run=0,
                            channels=['DAPI', 'A488', 'A555', 'A647']):

        self.establish_fm_array(experiment_directory, cycle_number, z_slices, offset_array,
                                initialize=establish_fm_array, x_frame_size=x_frame_size, autofocus=auto_focus_run,
                                auto_expose=auto_expose_run)


        self.image_capture(experiment_directory, 'DAPI', 50, 0, 0, 0)  # wake up lumencor light engine


        '''
        std_dev = np.std(start_image)
        if std_dev > threshold:
            pass
        else: 
            time.sleep(10)
        '''
        time.sleep(10)  # wait for it to wake up
        ''''
        exp_time = exp_time_array
        np.save('exp_array.npy', exp_time)

        for channel in channels:
            z_tile_stack = self.core_tile_acquire(experiment_directory, channel)
            self.save_files(z_tile_stack, channel, cycle_number, experiment_directory, stain_bleach)

        '''
        self.exp_logbook(experiment_directory, cycle_number)
        self.multi_channel_z_stack_capture(experiment_directory, cycle_number, stain_bleach,x_pixels=x_frame_size, slice_gap=2, channels=channels)
        # self.marker_excel_file_generation(experiment_directory, cycle_number)


    def full_cycle(self, experiment_directory, cycle_number, offset_array, stain_valve, fluidics_object, z_slices, incub_val=45, x_frame_size=2960):

        pump = fluidics_object

        if cycle_number == 0:
            self.image_cycle_acquire(cycle_number, experiment_directory, z_slices, 'Bleach', offset_array, x_frame_size=x_frame_size, establish_fm_array=1, auto_focus_run=0,
                                     auto_expose_run=0)
        else:

            # print(status_str)
            pump.liquid_action('Stain', incub_val=incub_val, stain_valve=stain_valve)  # nuc is valve=7, pbs valve=8, bleach valve=1 (action, stain_valve, heater state (off = 0, on = 1))
            # print(status_str)
            self.image_cycle_acquire(cycle_number, experiment_directory, z_slices, 'Stain', offset_array,x_frame_size=x_frame_size, establish_fm_array=0, auto_focus_run=1,
                                     auto_expose_run=1)
            time.sleep(5)

            # print(status_str)
            pump.liquid_action('Bleach', stain_valve=stain_valve)  # nuc is valve=7, pbs valve=8, bleach valve=1 (action, stain_valve, heater state (off = 0, on = 1))
            time.sleep(5)
            # print(status_str)
            self.image_cycle_acquire(cycle_number, experiment_directory, z_slices, 'Bleach', offset_array, x_frame_size=x_frame_size, establish_fm_array=0, auto_focus_run=0,
                                     auto_expose_run=0)
            time.sleep(3)

        # self.post_acquisition_processor(experiment_directory, x_frame_size)



    ######Folder System Generation########################################################

    def marker_excel_file_generation(self, experiment_directory, cycle_number):

        folder_path = experiment_directory + '/mcmicro'
        numpy_path = experiment_directory + '/' + 'np_arrays'
        os.chdir(numpy_path)
        exp_array = np.load('exp_array.npy', allow_pickle=False)
        exp_array = np.flip(exp_array)

        filter_sets = ['A647', 'A555', 'A488', 'DAPI']
        emission_wavelengths = ['675', '565', '525', '450']
        exciation_wavlengths = ['647', '555', '488', '405']

        try:
            os.chdir(folder_path)
            wb = load_workbook('markers.xlsx')
        except:
            wb = Workbook()
        ws = wb.active
        ws.cell(row=1, column=1).value = 'channel_number'
        ws.cell(row=1, column=2).value = 'cycle_number'
        ws.cell(row=1, column=3).value = 'marker_name'
        ws.cell(row=1, column=4).value = 'Filter'
        ws.cell(row=1, column=5).value = 'excitation_wavlength'
        ws.cell(row=1, column=6).value = 'emission_wavlength'
        ws.cell(row=1, column=7).value = 'background'
        ws.cell(row=1, column=8).value = 'exposure'
        ws.cell(row=1, column=9).value = 'remove'

        for row_number in range(2, (cycle_number) * 4 + 2):
            cycle_number = 4 // (row_number - 2) + 1
            intercycle_channel_number = cycle_number * 4 + 1 - row_number

            ws.cell(row=row_number, column=1).value = row_number
            ws.cell(row=row_number, column=2).value = cycle_number
            ws.cell(row=row_number, column=3).value = 'Marker_' + str(row_number)
            # ws.cell(row=row_number, column=4).value = filter_sets[intercycle_channel_number]
            # ws.cell(row=row_number, column=5).value = exciation_wavlengths[intercycle_channel_number]
            # ws.cell(row=row_number, column=6).value = emission_wavelengths[intercycle_channel_number]

        row_start = (cycle_number - 1) * 4 + 2
        row_end = row_start + 4

        for row_number in range(row_start, row_end):
            cycle_number = 4 // (row_number - 2) + 1
            intercycle_channel_number = cycle_number * 4 + 1 - row_number

            ws.cell(row=row_number, column=8).value = exp_array[row_number - 2]

        os.chdir(folder_path)
        wb.save(filename='markers.xlsx')

    def folder_addon(self, parent_directory_path, new_folder_names):

        os.chdir(parent_directory_path)

        for name in new_folder_names:

            add_on_folder = str(name)
            full_directory_path = parent_directory_path + '/' + add_on_folder

            try:
                os.mkdir(full_directory_path)
            except:
                pass

    def file_structure(self, experiment_directory, highest_cycle_count):

        channels = ['DAPI', 'A488', 'A555', 'A647']
        cycles = np.linspace(0, highest_cycle_count).astype(int)

        # folder layer one
        os.chdir(experiment_directory)
        self.folder_addon(experiment_directory, ['Quick_Tile'])
        self.folder_addon(experiment_directory, ['np_arrays'])
        self.folder_addon(experiment_directory, ['mcmicro'])
        self.folder_addon(experiment_directory, ['exposure_times'])
        self.folder_addon(experiment_directory, channels)

        # folder layer two

        mc_micro_directory = experiment_directory + '/mcmicro'
        self.folder_addon(mc_micro_directory, ['raw'])
        quick_tile_directory = experiment_directory + '/Quick_Tile'
        self.folder_addon(quick_tile_directory, ['DAPI'])
        self.folder_addon(quick_tile_directory, ['A488'])
        self.folder_addon(quick_tile_directory, ['A555'])
        self.folder_addon(quick_tile_directory, ['A647'])

        for channel in channels:

            experiment_channel_directory = experiment_directory + '/' + channel

            self.folder_addon(experiment_channel_directory, ['Stain'])
            self.folder_addon(experiment_channel_directory, ['Bleach'])

            # folder layers 3 and 4

            for cycle in cycles:
                experiment_channel_stain_directory = experiment_channel_directory + '/' + 'Stain'
                experiment_channel_bleach_directory = experiment_channel_directory + '/' + 'Bleach'

                self.folder_addon(experiment_channel_stain_directory, ['cy_' + str(cycle)])
                experiment_channel_stain_cycle_directory = experiment_channel_stain_directory + '/' + 'cy_' + str(cycle)

                self.folder_addon(experiment_channel_stain_cycle_directory, ['Tiles'])

                self.folder_addon(experiment_channel_bleach_directory, ['cy_' + str(cycle)])
                experiment_channel_bleach_cycle_directory = experiment_channel_bleach_directory + '/' + 'cy_' + str(
                    cycle)

                self.folder_addon(experiment_channel_bleach_cycle_directory, ['Tiles'])

        os.chdir(experiment_directory + '/' + 'np_arrays')

    #####################################################################################################
    ##########Saving/File Generation Methods#############################################################################

    def post_acquisition_processor(self, experiment_directory, x_pixels):

        mcmicro_path = experiment_directory + r'\mcmicro\raw'
        cycle_start = 1
        cycle_start_search = 0
        '''
        os.chdir(mcmicro_path)
        while cycle_start_search == 0:
            file_name = str(experiment_directory.split("\\")[-1]) + '-cycle-0' + str(cycle_start) + '.ome.tif'
            if os.path.isfile(file_name) == 1:
                cycle_start += 1
            else:
                cycle_start_search = 1
        '''
        cycle_end = 8
        cycle_start = 1

        self.tissue_binary_generate(experiment_directory)
        self.tissue_exist_array_generate(experiment_directory)

        for cycle_number in range(cycle_start, cycle_end):
            self.infocus(experiment_directory, cycle_number, x_pixels, 2, 2)
            #self.background_sub(experiment_directory, cycle_number)
            #self.illumination_flattening(experiment_directory, cycle_number)
            #self.mcmicro_image_stack_generator(cycle_number, experiment_directory, x_pixels)
            #self.stage_placement(experiment_directory, cycle_number, x_pixels)

    def mcmicro_image_stack_generator(self, cycle_number, experiment_directory, x_frame_size):

        numpy_path = experiment_directory + '/' + 'np_arrays'
        os.chdir(numpy_path)
        full_array = np.load('fm_array.npy', allow_pickle=False)

        xml_metadata = self.metadata_generator(experiment_directory, x_frame_size)

        numpy_x = full_array[0]
        numpy_y = full_array[1]
        y_tile_count = numpy_x.shape[0]
        x_tile_count = numpy_y.shape[1]
        tile_count = int(x_tile_count * y_tile_count)

        dapi_im_path = experiment_directory + '\DAPI\Stain\cy_' + str(
            cycle_number) + '\Tiles' + '/focused_basic_corrected'
        a488_im_path = experiment_directory + '\A488\Stain\cy_' + str(
            cycle_number) + '\Tiles' + '/focused_basic_corrected'
        a555_im_path = experiment_directory + '\A555\Stain\cy_' + str(
            cycle_number) + '\Tiles' + '/focused_basic_corrected'
        a647_im_path = experiment_directory + '\A647\Stain\cy_' + str(
            cycle_number) + '\Tiles' + '/focused_basic_corrected'

        mcmicro_path = experiment_directory + r'\mcmicro\raw'

        mcmicro_stack = np.random.rand(tile_count * 4, 2960, x_frame_size).astype('uint16')

        tile = 0
        for x in range(0, x_tile_count):
            for y in range(0, y_tile_count):
                dapi_file_name = 'x' + str(x) + '_y_' + str(y) + '_c_DAPI.tif'
                a488_file_name = 'x' + str(x) + '_y_' + str(y) + '_c_A488.tif'
                a555_file_name = 'x' + str(x) + '_y_' + str(y) + '_c_A555.tif'
                a647_file_name = 'x' + str(x) + '_y_' + str(y) + '_c_A647.tif'

                base_count_number_stack = tile * 4

                os.chdir(dapi_im_path)
                try:
                    image = io.imread(dapi_file_name)
                except:
                    image = cv2.imread(dapi_file_name)[::, ::, 0]
                mcmicro_stack[base_count_number_stack + 0] = image

                os.chdir(a488_im_path)
                try:
                    image = io.imread(a488_file_name)
                except:
                    image = cv2.imread(a488_file_name)[::, ::, 0]
                mcmicro_stack[base_count_number_stack + 1] = image

                os.chdir(a555_im_path)
                try:
                    image = io.imread(a555_file_name)
                except:
                    image = cv2.imread(a555_file_name)[::, ::, 0]
                mcmicro_stack[base_count_number_stack + 2] = image

                os.chdir(a647_im_path)
                try:
                    image = io.imread(a647_file_name)
                except:
                    image = cv2.imread(a647_file_name)[::, ::, 0]
                mcmicro_stack[base_count_number_stack + 3] = image

                tile += 1

        os.chdir(mcmicro_path)
        mcmicro_file_name = str(experiment_directory.split("\\")[-1]) + '-cycle-0' + str(cycle_number) + '.ome.tif'
        tf.imwrite(mcmicro_file_name, mcmicro_stack, photometric='minisblack', description=xml_metadata)

    def metadata_generator(self, experiment_directory, x_frame_size):

        new_ome = OME()
        #ome = from_xml(r'C:\Users\mike\Documents\GitHub\AutoCIF/image.xml', parser='lxml')
        ome = from_xml(r'C:\Users\CyCIF PC\Documents\GitHub\AutoCIF/image.xml', parser='lxml')
        ome = ome.images[0]

        numpy_path = experiment_directory + '/' + 'np_arrays'
        os.chdir(numpy_path)
        full_array = np.load('fm_array.npy', allow_pickle=False)

        numpy_x = full_array[0]
        numpy_y = full_array[1]
        y_tile_count = numpy_x.shape[0]
        x_tile_count = numpy_y.shape[1]
        total_tile_count = x_tile_count * y_tile_count

        y_gap = 532
        col_col_gap = 10
        # for r in range(3, -1, -1):
        #    numpy_y[r][0] = numpy_y[r + 1][0] - y_gap
        #    numpy_y[r][1] = numpy_y[r + 1][1] - y_gap - col_col_gap

        # sub in needed pixel size and pixel grid changes
        ome.pixels.physical_size_x = 0.2
        ome.pixels.physical_size_y = 0.2
        ome.pixels.size_x = x_frame_size
        ome.pixels.size_y = 2960
        # sub in other optional numbers to make metadata more accurate

        for x in range(0, total_tile_count):
            tile_metadata = deepcopy(ome)
            new_ome.images.append(tile_metadata)

        # sub in stage positional information into each tile. numpy[y][x]
        tile_counter = 0
        for x in range(0, x_tile_count):
            for y in range(0, y_tile_count):

                for p in range(0, 4):
                    new_x = numpy_x[y][x] - 11000
                    new_y = numpy_y[y][x] + 2300
                    new_ome.images[tile_counter].pixels.planes[p].position_y = deepcopy(new_y)
                    new_ome.images[tile_counter].pixels.planes[p].position_x = deepcopy(new_x)
                    new_ome.images[tile_counter].pixels.tiff_data_blocks[p].ifd = (4 * tile_counter) + p
                tile_counter += 1

        xml = to_xml(new_ome)

        return xml

    def star_dist_stage_placement(self, experiment_directory, x_pixels):
        '''
        Goal to place images via rough stage coords in a larger image. WIll have nonsense borders
        '''

        star_dist_path = experiment_directory + r'\Labelled_Nuc'

        # load in numpy matricies

        numpy_path = experiment_directory + '/' + 'np_arrays'
        os.chdir(numpy_path)
        full_array = np.load('fm_array.npy', allow_pickle=False)

        numpy_x = full_array[0]
        numpy_y = full_array[1]

        y_tile_count = numpy_x.shape[0]
        x_tile_count = numpy_y.shape[1]

        fov_x_pixels = x_pixels
        fov_y_pixels = 2960
        um_pixel = 0.20

        # generate large numpy image with rand numbers. Big enough to hold all images + 10%

        super_y = int(1.02 * (y_tile_count * fov_y_pixels))
        super_x = int(1.02 * (x_tile_count * fov_x_pixels))
        placed_image = np.random.rand(super_y, super_x).astype('uint16')

        # transform numpy x and y coords into new shifted space that starts at zero and is in units of pixels and not um
        numpy_x_pixels = numpy_x / um_pixel
        numpy_y_pixels = numpy_y / um_pixel

        y_displacement_vector = (2100 / um_pixel + fov_y_pixels / 2) * 1.02
        x_displacement_vector = (-11385 / um_pixel + fov_x_pixels / 2) * 1.02

        numpy_x_pixels = numpy_x_pixels + x_displacement_vector
        numpy_x_pixels = np.ceil(numpy_x_pixels)
        numpy_x_pixels = numpy_x_pixels.astype(int)

        numpy_y_pixels = numpy_y_pixels + y_displacement_vector
        numpy_y_pixels = np.ceil(numpy_y_pixels)
        numpy_y_pixels = numpy_y_pixels.astype(int)

        # load images into python

        os.chdir(star_dist_path)

        # place images into large array

        for x in range(0, x_tile_count):
            for y in range(0, y_tile_count):
                filename = 'x' + str(x) + '_y_' + str(y) + '_c_DAPI.tif'
                try:
                    image = io.imread(filename)
                except:
                    image = cv2.imread(filename)[::,::,0]

                # define subsection of large array that fits dimensions of single FOV
                # x_center = numpy_x_pixels[y][x]
                # y_center = numpy_y_pixels[y][x]
                # x_start = int(x_center - fov_x_pixels / 2)
                # x_end = int(x_center + fov_x_pixels / 2)
                # y_start = int(y_center - fov_y_pixels / 2)
                # y_end = int(y_center + fov_y_pixels / 2)

                if x == 0 and y == 0:
                    x_start = 0
                    x_end = x_pixels
                    y_start = 0
                    y_end = 2960
                else:

                    x_start = int(x * fov_x_pixels * 0.9)
                    x_end = x_start + x_pixels
                    y_start = int(y * fov_y_pixels * 0.9)
                    y_end = y_start + 2960

                # placed_image[y_start:y_end, x_start:x_end] = placed_image[y_start:y_end, x_start:x_end] + image
                placed_image[y_start:y_end, x_start:x_end] = image

        # save output image
        os.chdir(star_dist_path)
        tf.imwrite('star_dist_placed.tif', placed_image)

    def stage_placement(self, experiment_directory, cycle_number, x_pixels):
        '''
        Goal to place images via rough stage coords in a larger image. WIll have nonsense borders
        '''

        quick_tile_path = experiment_directory + r'\Quick_Tile'

        # load in numpy matricies

        numpy_path = experiment_directory + '/' + 'np_arrays'
        os.chdir(numpy_path)
        full_array = np.load('fm_array.npy', allow_pickle=False)

        numpy_x = full_array[0]
        numpy_y = full_array[1]

        y_tile_count = numpy_x.shape[0]
        x_tile_count = numpy_y.shape[1]

        fov_x_pixels = x_pixels
        fov_y_pixels = 2960
        um_pixel = 0.20

        # generate large numpy image with rand numbers. Big enough to hold all images + 10%

        super_y = int(1.02 * (y_tile_count * fov_y_pixels))
        super_x = int(1.02 * (x_tile_count * fov_x_pixels))
        placed_image = np.random.rand(super_y, super_x).astype('uint16')

        # transform numpy x and y coords into new shifted space that starts at zero and is in units of pixels and not um
        numpy_x_pixels = numpy_x / um_pixel
        numpy_y_pixels = numpy_y / um_pixel

        y_displacement_vector = (2100 / um_pixel + fov_y_pixels / 2) * 1.02
        x_displacement_vector = (-11385 / um_pixel + fov_x_pixels / 2) * 1.02

        numpy_x_pixels = numpy_x_pixels + x_displacement_vector
        numpy_x_pixels = np.ceil(numpy_x_pixels)
        numpy_x_pixels = numpy_x_pixels.astype(int)

        numpy_y_pixels = numpy_y_pixels + y_displacement_vector
        numpy_y_pixels = np.ceil(numpy_y_pixels)
        numpy_y_pixels = numpy_y_pixels.astype(int)

        # load images into python

        channels = ['DAPI', 'A488', 'A555', 'A647']
        types = ['Stain', 'Bleach']

        for type in types:
            for channel in channels:

                if type == 'Stain':
                    im_path = experiment_directory + '/' + channel + "/" + type + '\cy_' + str(
                        cycle_number) + '\Tiles' + '/focused_basic_corrected'
                elif type == 'Bleach':
                    im_path = experiment_directory + '/' + channel + "/" + type + '\cy_' + str(
                        cycle_number) + '\Tiles' + '/focused'
                os.chdir(im_path)

                # place images into large array

                for x in range(0, x_tile_count):
                    for y in range(0, y_tile_count):
                        filename = 'x' + str(x) + '_y_' + str(y) + '_c_' + channel + '.tif'
                        try:
                            image = io.imread(filename)
                        except:
                            image = cv2.imread(filename)[::,::,0]

                        # define subsection of large array that fits dimensions of single FOV
                        # x_center = numpy_x_pixels[y][x]
                        # y_center = numpy_y_pixels[y][x]
                        # x_start = int(x_center - fov_x_pixels / 2)
                        # x_end = int(x_center + fov_x_pixels / 2)
                        # y_start = int(y_center - fov_y_pixels / 2)
                        # y_end = int(y_center + fov_y_pixels / 2)

                        if x == 0 and y == 0:
                            x_start = 0
                            x_end = x_pixels
                            y_start = 0
                            y_end = 2960
                        else:

                            x_start = int(x * fov_x_pixels * 0.9)
                            x_end = x_start + x_pixels
                            y_start = int(y * fov_y_pixels * 0.9)
                            y_end = y_start + 2960

                        # placed_image[y_start:y_end, x_start:x_end] = placed_image[y_start:y_end, x_start:x_end] + image
                        placed_image[y_start:y_end, x_start:x_end] = image

                # save output image
                os.chdir(quick_tile_path + '/' + channel)
                tf.imwrite(channel + '_cy_' + str(cycle_number) + '_' + type + '_placed.tif', placed_image)

    def illumination_flattening(self, experiment_directory, cycle_number):

        # load numpy arrays in
        numpy_path = experiment_directory + '/' + 'np_arrays'
        os.chdir(numpy_path)
        full_array = np.load('fm_array.npy', allow_pickle=False)

        directory_start = experiment_directory + '//'

        for channel in range(0, 4):

            if channel == 0:
                channel_name = 'DAPI'
            if channel == 1:
                channel_name = 'A488'
            if channel == 2:
                channel_name = 'A555'
            if channel == 3:
                channel_name = 'A647'

            print('channel', channel_name, 'cycle', cycle_number)

            if channel == 0:
                directory = directory_start + channel_name + '\Stain\cy_' + str(cycle_number) + r'\Tiles\focused'
            else:
                directory = directory_start + channel_name + '\Stain\cy_' + str(cycle_number) + r'\Tiles\focused\background_subbed'
            output_directory = directory_start + channel_name + '\Stain\cy_' + str(
                cycle_number) + r'\Tiles\focused_basic_corrected'

            try:
                os.mkdir(output_directory)
            except:
                pass

            epsilon = 1e-06
            optimizer = shading_correction.BaSiC(directory)
            optimizer.prepare()
            optimizer.run()
            # Save the estimated fields (only if the profiles were estimated)
            directory = Path(output_directory)
            flatfield_name = directory / "flatfield.tif"
            darkfield_name = directory / "darkfield.tif"
            cv2.imwrite(str(flatfield_name), optimizer.flatfield_fullsize.astype(np.float32))
            cv2.imwrite(str(darkfield_name), optimizer.darkfield_fullsize.astype(np.float32))
            optimizer.write_images(output_directory, epsilon=epsilon)

    def infocus(self, experiment_directory, cycle_number, x_frame_size, x_sub_section_count = 1, y_sub_section_count = 1):

        print('cycle', cycle_number)
        bin_values = [10]
        channels = ['DAPI', 'A488', 'A555', 'A647']
        #channels = ['A488']

        dapi_im_path = experiment_directory + '/' + 'DAPI' '\Stain\cy_' + str(cycle_number) + '\Tiles'
        tissue_path = experiment_directory + '/Tissue_Binary'

        # load numpy arrays in
        numpy_path = experiment_directory + '/' + 'np_arrays'
        os.chdir(numpy_path)
        full_array = np.load('fm_array.npy', allow_pickle=False)
        tissue_exist = np.load('tissue_exist.npy', allow_pickle=False)

        numpy_x = full_array[0]
        numpy_y = full_array[1]

        y_tile_count = numpy_x.shape[0]
        x_tile_count = numpy_y.shape[1]

        # determine number z slices
        z_checker = 0
        z_slice_count = 0
        os.chdir(dapi_im_path)
        while z_checker == 0:
            file_name = 'z_' + str(z_slice_count) + '_x' + str(0) + '_y_' + str(0) + '_c_DAPI.tif'
            if os.path.isfile(file_name) == 1:
                z_slice_count += 1
            else:
                z_checker = 1

        # make object to hold all tissue binary maps
        tissue_binary_stack = np.random.rand(y_tile_count, x_tile_count, 2960, x_frame_size).astype('uint16')
        for x in range(0, x_tile_count):
            for y in range(0, y_tile_count):
                if tissue_exist[y][x] == 1:
                    os.chdir(tissue_path)
                    file_name = 'x' + str(x) + '_y_' + str(y) + '_tissue.tif'
                    image = tf.imread(file_name)
                    tissue_binary_stack[y][x] = image
                else:
                    pass

        for channel in channels:
            # generate imstack of z slices for tile
            im_path = experiment_directory + '/' + channel + '\Stain\cy_' + str(cycle_number) + '\Tiles'
            os.chdir(im_path)

            z_stack = np.random.rand(z_slice_count, 2960, x_frame_size).astype('float32')
            for x in range(0, x_tile_count):
                for y in range(0, y_tile_count):
                    if tissue_exist[y][x] == 1:
                        for z in range(0, z_slice_count):
                            im_path = experiment_directory + '/' + channel + '\Stain\cy_' + str(cycle_number) + '\Tiles'
                            os.chdir(im_path)
                            file_name = 'z_' + str(z) + '_x' + str(x) + '_y_' + str(y) + '_c_' + str(channel) + '.tif'
                            image = tf.imread(file_name)
                            z_stack[z] = image

                            # break into sub sections (2x2)

                        number_bins = len(bin_values)
                        brenner_sub_selector = np.random.rand(z_slice_count, number_bins, y_sub_section_count,
                                                              x_sub_section_count)
                        for z in range(0, z_slice_count):
                            for y_sub in range(0, y_sub_section_count):
                                for x_sub in range(0, x_sub_section_count):

                                    y_end = int((y_sub + 1) * (2960 / y_sub_section_count))
                                    y_start = int(y_sub * (2960 / y_sub_section_count))
                                    x_end = int((x_sub + 1) * (x_frame_size / x_sub_section_count))
                                    x_start = int(x_sub * (x_frame_size / x_sub_section_count))
                                    sub_image = z_stack[z][y_start:y_end, x_start:x_end]

                                    sub_tissue_bin = tissue_binary_stack[y][x][y_start:y_end, x_start:x_end]

                                    for b in range(0, number_bins):
                                        bin_value = int(bin_values[b])
                                        score =  self.focus_score(sub_image, bin_value, sub_tissue_bin)
                                        #score = self.focus_score_post_processing(sub_image, bin_value)
                                        #score = 500
                                        brenner_sub_selector[z][b][y_sub][x_sub] = score

                        reconstruct_array = self.brenner_reconstruct_array(brenner_sub_selector, z_slice_count, number_bins)
                        #reconstruct_array = skimage.filters.median(reconstruct_array)
                        self.image_reconstructor(experiment_directory, reconstruct_array, channel, cycle_number,
                                             x_frame_size, y, x)

                    else:
                        pass

    def brenner_reconstruct_array(self, brenner_sub_selector, z_slice_count, number_bins):
        '''
        take in sub selector array and slice to find max values for brenner scores and then find mode for various bin levels
        output array that shows what slice to grab each sub section from
        :param brenner_sub_selector:
        :return:
        '''
        # array to dictate the z slice to grab each sub section from
        y_sections = np.shape(brenner_sub_selector)[2]
        x_sections = np.shape(brenner_sub_selector)[3]

        reconstruct_array = np.random.rand(y_sections, x_sections).astype('uint16')

        temp_bin_max_indicies = np.random.rand(number_bins).astype('uint16')

        for y in range(0, y_sections):
            for x in range(0, x_sections):
                for b in range(0, number_bins):
                    sub_scores = brenner_sub_selector[::, b, y, x]
                    max_score = np.max(sub_scores)
                    max_index = np.where(sub_scores == max_score)[0][0]
                    print(sub_scores, max_index)
                    #max_index = 3
                    # temp_bin_max_indicies[b] = max_index
                # sub_section_index_mode = stats.mode(temp_bin_max_indicies)[0][0]
                # reconstruct_array[y][x] = sub_section_index_mode
                reconstruct_array[y][x] = max_index

        return reconstruct_array

    def image_reconstructor(self, experiment_directory, reconstruct_array, channel, cycle_number, x_frame_size,
                            y_tile_number, x_tile_number):

        y_sections = np.shape(reconstruct_array)[0]
        x_sections = np.shape(reconstruct_array)[1]

        cycle_types = ['Stain', 'Bleach']

        for cycle_type in cycle_types:

            im_path = experiment_directory + '/' + channel + '/' + cycle_type + '\cy_' + str(cycle_number) + '\Tiles'
            os.chdir(im_path)
            try:
                os.mkdir('focused')
            except:
                pass

            # rebuilt image container
            rebuilt_image = np.random.rand(2960, x_frame_size).astype('float32')

            for y in range(0, y_sections):
                for x in range(0, x_sections):
                    # define y and x start and ends subsection of rebuilt image
                    y_end = int((y + 1) * (2960 / y_sections))
                    y_start = int(y * (2960 / y_sections))
                    x_end = int((x + 1) * (x_frame_size / x_sections))
                    x_start = int(x * ((x_frame_size) / x_sections))

                    # find z for specific subsection
                    z_slice = reconstruct_array[y][x]
                    #z_slice = 5
                    # load in image to extract for subsection
                    file_name = 'z_' + str(z_slice) + '_x' + str(x_tile_number) + '_y_' + str(
                        y_tile_number) + '_c_' + str(channel) + '.tif'
                    image = tf.imread(file_name)
                    rebuilt_image[y_start:y_end, x_start:x_end] = image[y_start:y_end, x_start:x_end]

            reconstruct_path = experiment_directory + '/' + channel + '/' + cycle_type + '\cy_' + str(
                cycle_number) + '\Tiles' + '/focused'
            os.chdir(reconstruct_path)
            filename = 'x' + str(x_tile_number) + '_y_' + str(y_tile_number) + '_c_' + str(channel) + '.tif'
            tf.imwrite(filename, rebuilt_image)

    def background_sub(self, experiment_directory, cycle):

        experiment_directory = experiment_directory + '/'

        numpy_path = experiment_directory + '/' + 'np_arrays'
        os.chdir(numpy_path)
        file_name = 'fm_array.npy'
        fm_array = np.load(file_name, allow_pickle=False)
        tissue_exist = np.load('tissue_exist.npy', allow_pickle=False)

        y_tile_count = np.shape(fm_array[0])[0]
        x_tile_count = np.shape(fm_array[0])[1]

        channels = ['A488', 'A555', 'A647']

        for y in range(0, y_tile_count):
            for x in range(0, x_tile_count):

                if tissue_exist[y][x] == 1:

                    # load reference and "moved" image
                    ref_path = experiment_directory + 'DAPI\Bleach\cy_' + str(cycle) + '\Tiles/focused'
                    os.chdir(ref_path)
                    ref_name = 'x' + str(x) + '_y_' + str(y) + '_c_DAPI.tif'
                    ref = io.imread(ref_name)
                    ref = np.nan_to_num(ref, posinf=65500)
                    mov_path = experiment_directory + 'DAPI\Stain\cy_' + str(cycle) + '\Tiles/focused'
                    os.chdir(mov_path)
                    mov_name = 'x' + str(x) + '_y_' + str(y) + '_c_DAPI.tif'
                    mov = io.imread(mov_name)
                    mov = np.nan_to_num(mov, posinf=65500)

                    # Translational transformation
                    sr = StackReg(StackReg.TRANSLATION)
                    out_tra = sr.register_transform(ref, mov)

                    # apply translation to other color channels

                    for channel in channels:
                        stain_color_path = experiment_directory + channel + r'/Stain/cy_' + str(cycle) + '\Tiles/focused'
                        os.chdir(stain_color_path)
                        filename = 'x' + str(x) + '_y_' + str(y) + '_c_' + channel + '.tif'
                        color_im = io.imread(filename)
                        color_im = np.nan_to_num(color_im, posinf= 65500)
                        color_reg = sr.transform(color_im)

                        # sub background color channels
                        bleach_color_path = experiment_directory + channel + r'/Bleach/cy_' + str(cycle) + '\Tiles/focused'
                        os.chdir(bleach_color_path)
                        color_bleach = io.imread(filename)
                        color_bleach = np.nan_to_num(color_bleach, posinf=65500)
                        #coefficent = self.autof_factor_estimator(color_reg, color_bleach)
                        #color_subbed = color_reg - coefficent * color_bleach
                        color_subbed = color_reg - color_bleach
                        color_subbed[color_subbed < 0] = 0
                        color_subbed = color_subbed.astype('float32')

                        # save

                        save_path = experiment_directory + channel + r'/Stain/cy_' + str(cycle) + '\Tiles/focused/background_subbed'
                        try:
                            os.chdir(save_path)
                        except:
                            os.mkdir(save_path)
                            os.chdir(save_path)

                        subbed_filename = 'x' + str(x) + '_y_' + str(y) + '_c_' + channel + '.tif'
                        # bleach_filename ='x' + str(x) + '_y_' + str(y) + '_c_' + channel + '_bleach.tif'
                        # reg_filename = 'x' + str(x) + '_y_' + str(y) + '_c_' + channel + '_registered.tif'
                        tf.imwrite(subbed_filename, color_subbed)
                        # io.imsave(bleach_filename, color_bleach)
                        # io.imsave(reg_filename, color_reg)

                else:

                    for channel in channels:
                        stain_color_path = experiment_directory + channel + r'/Stain/cy_' + str(cycle) + '\Tiles/focused'
                        os.chdir(stain_color_path)
                        filename = 'x' + str(x) + '_y_' + str(y) + '_c_' + channel + '.tif'
                        color_im = io.imread(filename)

                        # save

                        save_path = experiment_directory + channel + r'/Stain/cy_' + str(
                            cycle) + '\Tiles/focused/background_subbed'
                        try:
                            os.chdir(save_path)
                        except:
                            os.mkdir(save_path)
                            os.chdir(save_path)

                        subbed_filename = 'x' + str(x) + '_y_' + str(y) + '_c_' + channel + '.tif'
                        # bleach_filename ='x' + str(x) + '_y_' + str(y) + '_c_' + channel + '_bleach.tif'
                        # reg_filename = 'x' + str(x) + '_y_' + str(y) + '_c_' + channel + '_registered.tif'
                        tf.imwrite(subbed_filename, color_im)
                        # io.imsave(bleach_filename, color_bleach)
                        # io.imsave(reg_filename, color_reg)

    def zc_save(self, zc_tif_stack, channels, x_tile, y_tile, cycle, x_pixels, experiment_directory, Stain_or_Bleach):

        z_tile_count = np.shape(zc_tif_stack)[1]

        for channel in channels:
            if channel == 'DAPI':
                zc_index = 0
            if channel == 'A488':
                zc_index = 1
            if channel == 'A555':
                zc_index = 2
            if channel == 'A647':
                zc_index = 3

            # establish x range to collect in image

            #side_pixel_count = int(5056 - x_pixels)

            save_directory = experiment_directory + '/' + str(channel) + '/' + Stain_or_Bleach + '/' + 'cy_' + str(
                cycle) + '/' + 'Tiles'
            os.chdir(save_directory)

            for z in range(0, z_tile_count):
                file_name = 'z_' + str(z) + '_x' + str(x_tile) + '_y_' + str(y_tile) + '_c_' + str(channel) + '.tif'
                image = zc_tif_stack[zc_index][z]
                imwrite(file_name, image, photometric='minisblack')

    def tissue_exist_array_generate(self, experiment_directory):

        tissue_path = experiment_directory + '/Tissue_Binary'

        # load numpy arrays in
        numpy_path = experiment_directory + '/' + 'np_arrays'
        os.chdir(numpy_path)
        full_array = np.load('fm_array.npy', allow_pickle=False)

        numpy_x = full_array[0]
        numpy_y = full_array[1]

        y_tile_count = numpy_x.shape[0]
        x_tile_count = numpy_y.shape[1]

        # make object to hold all tissue binary maps
        tissue_binary_stack = np.random.rand(y_tile_count, x_tile_count, 2960, 2960).astype('uint16')
        for x in range(0, x_tile_count):
            for y in range(0, y_tile_count):
                os.chdir(tissue_path)
                file_name = 'x' + str(x) + '_y_' + str(y) + '_tissue.tif'
                image = io.imread(file_name)
                tissue_binary_stack[y][x] = image

        # make object that hold info as to if the image has tissue in it or not
        tissue_or_not = np.random.rand(y_tile_count, x_tile_count).astype('uint16')
        for x in range(0, x_tile_count):
            for y in range(0, y_tile_count):
                if tissue_binary_stack[y][x].sum() > 0:
                    tissue_or_not[y][x] = 1
                if tissue_binary_stack[y][x].sum() == 0:
                    tissue_or_not[y][x] = 0

        os.chdir(numpy_path)
        np.save('tissue_exist.npy', tissue_or_not)

    ######Kinetics and its functions#####################################################

    def kinetic_autofocus(self, experiment_directory, focus_z_position, planes):
        '''
        Takes planes number of images, split in half above and below focus position. Gives back plane position
        that had highest brenner score with skip 17.
        :param experiment_directory: where the experiment files are contained
        :param focus_z_position: expected focus position in the z axis
        :param planes: number planes to acquire for focusing purposes
        :return:
        '''


        os.chdir(experiment_directory)
        try:
            os.mkdir('nuclei_focus_images')
        except:
            pass

        focus_image_path = experiment_directory + '/nuclei_focus_images'
        os.chdir(focus_image_path)

        # set channel and exposure times in ms
        a488_2_dapi_offset = 8
        core.set_config("Color", 'DAPI')
        core.set_exposure(75)
        core.snap_image()
        tagged_image = core.get_tagged_image()
        time.sleep(10)

        # make image stack object
        dapi_tif_stack = np.random.rand(planes, 2960, 5056).astype('float16')

        # find z range to be scanned
        dapi_focus_z_position = focus_z_position + a488_2_dapi_offset
        slice_step_size = 1 # in microns
        bottom_z = dapi_focus_z_position - int(planes/2) * slice_step_size #remember int() rounds down
        top_z = dapi_focus_z_position + int(planes/2) * slice_step_size #remember int() rounds down
        image_slice_counter = 0

        #capture images for stack
        for z in range(bottom_z, top_z + slice_step_size, slice_step_size):

            core.set_position(z)
            time.sleep(0.3)
            core.snap_image()
            tagged_image = core.get_tagged_image()
            pixels = np.reshape(tagged_image.pix,newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"]])
            dapi_tif_stack[image_slice_counter] = pixels
            image_slice_counter += 1

        #save images in folder
        filename = 'dapi_autofocus.tif'
        io.imsave(filename, dapi_tif_stack)

        scores = np.random.rand(planes).astype('float64')

        for plane in range(0, planes):

            score = self.focus_score_post_processing(dapi_tif_stack[plane], 17)
            scores[plane] = score

        #find highest brenner score index
        max_score = np.max(scores)
        focus_index = np.where(scores == max_score)
        #make range of z values scanned
        dapi_scan_range = np.linspace(bottom_z, top_z, num=planes)
        # find focus value
        dapi_in_focus_z_position = dapi_scan_range[focus_index]
        a488_in_focus_z_position = int(dapi_in_focus_z_position - a488_2_dapi_offset)

        return a488_in_focus_z_position


    def antibody_kinetics(self, experiment_directory, capture_rate_staining, duration_staining, stain_valve, fluidic_object, channels=['DAPI', 'A488', 'A555', 'A647']):

        '''
        Dispenses stain avia fluidic system and images until certain time point. After that, bleach dispenses and capturing continues at a different
        rate until set time point


        :param experiment_directory:
        :param capture_rate_staining: points/min
        :param capture_rate_bleaching: points/min
        :param duration_staining: minutes
        :param duration_bleaching: minutes
        :param channels:
        :return:
        '''

        exp_array = [50, 200, 75, 75]
        A488_to_channels_offset = [8, 0, 0, 3]

        # create folders

        os.chdir(experiment_directory)
        try:
            os.mkdir('DAPI')
        except:
            pass
        try:
            os.mkdir('A488')
        except:
            pass
        try:
            os.mkdir('A555')
        except:
            pass
        try:
            os.mkdir('A647')
        except:
            pass
        try:
            os.mkdir('auto_fluorescence')
        except:
            pass
        try:
            os.mkdir('post_wash')
        except:
            pass

        dapi_path = experiment_directory + r'\DAPI'
        a488_path = experiment_directory + r'\A488'
        a555_path = experiment_directory + r'\A555'
        a647_path = experiment_directory + r'\A647'

        # dimensional parameters
        y_pixel_count = 2960
        x_pixel_count = 5056
        channel_count = len(channels)
        time_point_stain_count = int(duration_staining * capture_rate_staining)
        time_gap_staining = 1 / capture_rate_staining * 60
        print('time gap stain', time_gap_staining)

        # find current focus position
        focus_position = core.get_position()
        print('focus position acquired')
        focus_position = int(focus_position)

        # create data structure for staining images
        data_points_stain = np.full((time_point_stain_count, channel_count, y_pixel_count, x_pixel_count), 0)
        fluidic_object.valve_select(stain_valve)
        print('valve selected')

        #capture pre stain images (so autofluorescence)

        os.chdir(experiment_directory + r'\auto_fluorescence')
        for channel in channels:

            if channel == 'DAPI':
                channel_index = 0
            if channel == 'A488':
                channel_index = 1
            if channel == 'A555':
                channel_index = 2
            if channel == 'A647':
                channel_index = 3

            exp_time = exp_array[channel_index]
            channel_offset = A488_to_channels_offset[channel_index]
            core.set_position(focus_position + channel_offset)
            time.sleep(0.5)

            core.set_config("Color", channel)
            core.set_exposure(exp_time)

            print('capturing pre image', channel)

            core.snap_image()
            tagged_image = core.get_tagged_image()
            pixels = np.reshape(tagged_image.pix,newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"]])
            io.imsave(channel + '.tif', pixels)

        os.chdir(experiment_directory)

        #start flow
        fluidic_object.flow(500)
        print('flowing stain')
        time.sleep(45)
        fluidic_object.flow(-3)
        print('flow stain ended')
        fluidic_object.valve_select(12)

        # acquire kinetic time points
        print('total time points', time_point_stain_count)
        for time_point in range(0, time_point_stain_count):
            print('time point', time_point)
            start_time = time.time()

            # run auto focus and set new focus position
            focus_position = self.kinetic_autofocus(experiment_directory, focus_z_position= focus_position, planes = 11)
            print('focus position', focus_position)

            for channel in channels:

                if channel == 'DAPI':
                    channel_index = 0
                if channel == 'A488':
                    channel_index = 1
                if channel == 'A555':
                    channel_index = 2
                if channel == 'A647':
                    channel_index = 3

                exp_time = exp_array[channel_index]

                channel_offset = A488_to_channels_offset[channel_index]
                core.set_position(focus_position + channel_offset)
                time.sleep(0.3)

                core.set_config("Color", channel)
                core.set_exposure(exp_time)

                core.snap_image()
                tagged_image = core.get_tagged_image()
                pixels = np.reshape(tagged_image.pix,
                                    newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"]])
                data_points_stain[time_point][channel_index] = pixels

            #adjust for time to acquire to make time between frames = time_gap_staining
            end_time = time.time()
            time_elapsed = end_time - start_time
            time.sleep(time_gap_staining - time_elapsed)

        # save images
        print('saving')
        os.chdir(dapi_path)
        tf.imwrite('dapi_stain_stack', data_points_stain[::, 0, ::, ::])
        os.chdir(a488_path)
        tf.imwrite('a488_stain_stack', data_points_stain[::, 1, ::, ::])
        os.chdir(a555_path)
        tf.imwrite('a555_stain_stack', data_points_stain[::, 2, ::, ::])
        os.chdir(a647_path)
        tf.imwrite('a647_stain_stack', data_points_stain[::, 3, ::, ::])
        print('done saving')

        # start flow of PBS
        print('starting wash')
        fluidic_object.valve_select(12)
        fluidic_object.flow(500)
        time.sleep(45)
        fluidic_object.flow(-3)
        print('wash completed')

        # acquire post wash images

        os.chdir(experiment_directory + r'/post_wash')
        for channel in channels:

            if channel == 'DAPI':
                channel_index = 0
            if channel == 'A488':
                channel_index = 1
            if channel == 'A555':
                channel_index = 2
            if channel == 'A647':
                channel_index = 3

            exp_time = exp_array[channel_index]
            channel_offset = A488_to_channels_offset[channel_index]
            core.set_position(focus_position  + channel_offset)
            time.sleep(0.3)

            core.set_config("Color", channel)
            core.set_exposure(exp_time)

            core.snap_image()
            tagged_image = core.get_tagged_image()
            pixels = np.reshape(tagged_image.pix,newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"]])
            io.imsave(channel + '.tif', pixels)

