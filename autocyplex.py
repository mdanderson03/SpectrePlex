import ome_types
from pycromanager import Core, Acquisition, multi_d_acquisition_events, Dataset, MagellanAcquisition, Magellan, start_headless, XYTiledAcquisition
import numpy as np
import time
from scipy.optimize import curve_fit
import paho.mqtt.client as mqtt
import matplotlib.pyplot as plt
from skimage import io, measure
from skimage.filters import median
import os
import math
from datetime import datetime
from tifffile import imsave, imwrite
from openpyxl import load_workbook, Workbook
from ome_types.model import Instrument, Microscope, Objective, InstrumentRef, Image, Pixels, Plane, Channel
from ome_types.model.simple_types import UnitsLength, PixelType, PixelsID, ImageID, ChannelID
from ome_types import from_xml, OME, from_tiff
import sys
from ctypes import *
sys.path.append(r'C:\Users\CyCIF PC\Documents\GitHub\AutoCIF\Python_64_elveflow\DLL64')#add the path of the library here
sys.path.append(r'C:\Users\CyCIF PC\Documents\GitHub\AutoCIF\Python_64_elveflow')#add the path of the LoadElveflow.py

from array import array
from Elveflow64 import *



client = mqtt.Client('autocyplex_server')
client.connect('10.3.141.1', 1883)

core = Core()
magellan = Magellan()

global level
level = []


class brenner:
    def __init__(self):
        brenner.value = []
        return

class exp_level:
    def __init__(self):
        exp_level.value = []
        return

class cycif:

    def __init__(self):

        return

    def tissue_center(self, mag_surface):
        '''
        take magellan surface and find the xy coordinates of the center of the surface
        :param mag_surface:
        :param magellan:
        :return: x tissue center position and y tissue center position
        :rtype: list[float, float]
        '''
        xy_pos = self.tile_xy_pos(mag_surface)
        x_center = (max(xy_pos[0]) + min(xy_pos[0])) / 2
        y_center = (max(xy_pos[1]) + min(xy_pos[1])) / 2
        return x_center, y_center

    def num_surfaces_count(self):
        '''
        Looks at magellan surfaces that start with New Surface in its name, ie. 'New Surface 1' as that is the default generated prefix.

        :param object magellan: magellan object from magellan = Magellan() in pycromanager
        :return: surface_count
        :rtype: int
        '''
        x = 1
        while magellan.get_surface("New Surface " + str(x)) != None:
            x += 1
        surface_count = x - 1
        time.sleep(1)

        return surface_count

    def surface_exist_check(self, surface_name):
        '''
        Checks name of surface to see if exists. If it does, returns 1, if not returns 0

        :param object magellan: magellan object from magellan = Magellan() in pycromanager
        :param str surface_name: name of surface to check if exists

        :return: status
        :rtype: int
        '''

        status = 0
        if magellan.get_surface(surface_name) != None:
            status += 1

        return status

    ####################################################################
    ############ All in section are functions for the autofocus function
    ####################################################################
    def image_process_hook( image, metadata):
        '''
        Method that hooks from autofocus image acquistion calls. It takes image, calculates a focus score for it
        via focus_score method and exports a list that contains both the focus score and the z position it was taken at

        :param numpy image: single image from hooked from acquistion
        :param list[float] metadata: metadata for image

        :return: Nothing
        '''

        z = metadata.pop('ZPosition_um_Intended')  # moves up while taking z stack
        image_focus_score = cycif.focus_bin_generator(image)
        brenner_scores.value.append([image_focus_score, z])

        return
    def score_array_generator(self):

        optimal_score_array = [[brenner_scores.value[0][0], brenner_scores.value[0][1]], [brenner_scores.value[1][0],
                        brenner_scores.value[1][1]], [brenner_scores.value[2][0], brenner_scores.value[2][1]]]

        return optimal_score_array


    def bin_selector(self):
        '''
        takes array of brenner scores and determines optimal bin level and outputs corresponding focus scores with z positions
        :return: list [optimal bin focus score1, z position1], [], [], ...
        '''

        bin_levels = [4, 8, 16, 32, 64, 128]
        range_array = []

        for x in range(0, len(brenner_scores.value[0][0]) ):
            score_array = [brenner_scores.value[0][0][x], brenner_scores.value[1][0][x], brenner_scores.value[2][0][x]]
            max_value = max(score_array)
            min_value = min(score_array)
            range_array.append(max_value - min_value)

        max_range = max(range_array)
        max_index = range_array.index(max_range)
        print(bin_levels[max_index])

        optimal_score_array = [[brenner_scores.value[0][0][max_index], brenner_scores.value[0][1]], [brenner_scores.value[1][0][max_index],
                        brenner_scores.value[1][1]], [brenner_scores.value[2][0][max_index], brenner_scores.value[2][1]]]

        return optimal_score_array

    def focus_bin_generator( image):
        '''
        takes image and calculates brenner scores for various bin levels of 64, 32, 16, 8 and 4 and outputs them.
        :param image: numpy array 2D
        :return: list of brenner scores of binned images
        '''

        #bin_levels = [4,8,16,32,64]
        #focus_scores = [0]*len(bin_levels)



        focus_score = cycif.focus_score(image, 1)

        return focus_score

    def focus_score(image, bin_level):
        '''
        Calculates focus score on image with Brenners algorithm on downsampled image.

        Image is downsampled by [binning_size x binning_size], where binning_size is currently hardcoded in.

        :param numpy image: single image from hooked from acquistion
        :return: focus score for image
        :rtype: float
        '''
        # Note: Uniform background is a bit mandatory

        binning_size = bin_level
        image_binned = measure.block_reduce(image, binning_size)

        # do Brenner score
        a = image_binned[2:, :]
        a = a.astype('int64')
        b = image_binned[:-2, :]
        b = b.astype('int64')
        c = a - b
        c = c * c
        c = c/100000
        f_score_shadow = c.sum()

        return 1/f_score_shadow

    def gauss_jordan_solver(self, three_point_array):
        '''
        Takes 3 points and solves quadratic equation in a generic fashion and returns constants and
        solves for x in its derivative=0 equation

        :param list[float, float] three_point_array: list that contains pairs of [focus_score, z]
        :results: z coordinate for in focus plane
        :rtype: float
        '''

        x1 = three_point_array[0][1]
        x2 = three_point_array[1][1]
        x3 = three_point_array[2][1]
        score_0 = three_point_array[0][0]
        score_1 = three_point_array[1][0]
        score_2 = three_point_array[2][0]

        aug_matrix = np.array([[x1 * x1, x1, 1, score_0], [x2 * x2, x2, 1, score_1], [x3 * x3, x3, 1, score_2]])

        aug_matrix[0] = aug_matrix[0] / (aug_matrix[0, 0] + 0.000001)
        aug_matrix[1] = -(aug_matrix[1, 0]) * aug_matrix[0] + aug_matrix[1]
        aug_matrix[2] = -(aug_matrix[2, 0]) * aug_matrix[0] + aug_matrix[2]

        aug_matrix[1] = -(aug_matrix[1, 1] - 1) / (aug_matrix[2, 1] + 0.0000001) * aug_matrix[2] + aug_matrix[1]
        aug_matrix[0] = -aug_matrix[0, 1] * aug_matrix[1] + aug_matrix[0]
        aug_matrix[2] = -aug_matrix[2, 1] * aug_matrix[1] + aug_matrix[2]

        aug_matrix[2] = aug_matrix[2] / (aug_matrix[2, 2] + 0.000001)
        aug_matrix[0] = -aug_matrix[0, 2] * aug_matrix[2] + aug_matrix[0]
        aug_matrix[1] = -aug_matrix[1, 2] * aug_matrix[2] + aug_matrix[1]

        a = aug_matrix[0, 3]
        b = aug_matrix[1, 3]
        c = aug_matrix[2, 3]

        derivative = -b / (2 * a)

        return a,b,c,derivative

    def auto_focus(self, z_range, exposure_time, channel='DAPI'):
        '''
        Runs entire auto focus algorithm in current XY position. Gives back predicted
        in focus z position via focus_score method which is the Brenner score.

        :param list[int, int, int] z_range: defines range and stepsize with [z start, z end, z step]
        :param str channel: channel to autofocus with
        :return: z coordinate for in focus plane
        :rtype: float
        '''
        global brenner_scores
        brenner_scores = brenner()  # found using class for brenner was far superior to using it as a global variable.
        # I had issues with the image process hook function not updating brenner as a global variable

        with Acquisition(directory='C:/Users/CyCIF PC/Desktop/test_images', name='trash',
                         show_display=False,
                         image_process_fn=cycif.image_process_hook) as acq:
            events = multi_d_acquisition_events(channel_group='Color',
                                                channels=[channel],
                                                z_start=z_range[0],
                                                z_end=z_range[1],
                                                z_step=z_range[2],
                                                order='zc', channel_exposures_ms=[exposure_time])
            acq.acquire(events)

        optimal_score_array = self.score_array_generator()
        [a,b,c, derivative] = self.gauss_jordan_solver(optimal_score_array)
        z_ideal = derivative

        return z_ideal

    ###########################################################
    #This section is the for the exposure functions.
    ###########################################################

    def image_percentile_level( image, cut_off_threshold):
        '''
        Takes in image and cut off threshold and finds pixel value that exists at that threshold point.

        :param numpy array image: numpy array image
        :param float cut_off_threshold: percentile for cut off. For example a 0.99 would disregaurd the top 1% of pixels from calculations
        :return: intensity og pixel that resides at the cut off fraction that was entered in the image
        :rtype: int
        '''
        pixel_values = np.sort(image, axis=None)
        pixel_count = int(np.size(pixel_values))
        cut_off_index = int(pixel_count * cut_off_threshold)
        tail_intensity = pixel_values[cut_off_index]

        return tail_intensity

    def exposure_hook(image, metadata):
        '''
        Hook for expose method. Returns metric based on image_percentile_level method.

        :param numpy array image: image from expose
        :param metadata: metadata from exposure
        :return: nothing
        '''

        global level
        level = cycif.image_percentile_level(image, 0.85)

        return

    def expose( seed_exposure, channel='DAPI'):
        '''
        Runs entire auto exposure algorithm in current XY position. Gives back predicted
        in exposure level via image_percentile_level method with the exposure_hook

        :param list[int, int, int] z_range: defines range and stepsize with [z start, z end, z step]
        :param str channel: channel to autofocus with

        :return: z coordinate for in focus plane
        :rtype: float
        '''

        with Acquisition(directory='C:/Users/CyCIF PC/Desktop/test_images/', name='trash', show_display=False,
                         image_process_fn=cycif.exposure_hook) as acq:
            # Create some acquisition events here:

            event = {'channel': {'group': 'Color', 'config': channel}, 'exposure': seed_exposure}
            acq.acquire(event)

        return level

    def auto_expose(self, directory, seed_expose, benchmark_threshold, channels=['DAPI', 'A488', 'A555', 'A647']):
        '''

        :param object core: core object from Core() in pycromananger
        :param object magellan: magellan object from Magellan() in pycromanager
        :param int seed_expose: initial exposure time
        :param dict tile_points_xy: dictionary that contains keys of X and Y with associated coordinates
        :param int benchmark_threshold: integer of threshold that top 99% pixel will hit.
        :param float z_focused_pos: z position where image is in focus
        :param [str] channels: list of strings of channels that are wanted to be used
        :param str surface_name: name of surface to be used. If blank, just executes in current XY position
        :return: list of exposures
        '''

        numpy_path = experiment_directory +'/' + 'np_arrays'
        os.chdir(numpy_path)
        full_array = np.load('fm_array.npy', allow_pickle=False)

        new_x = full_array[0][0][0]
        new_y = full_array[1][0][0]
        z_position_channel_list = [full_array[2][0][0],full_array[3][0][0], full_array[4][0][0], full_array[5][0][0]]
        core.set_xy_position(new_x, new_y)

        bandwidth = 0.1
        sat_max = 65000
        exp_time_limit = 2000
        exposure_array = [10, 10, 10, 10]  # dapi, a488, a555, a647

        for fluor_channel in channels:

            if fluor_channel == 'DAPI':
                exp_index = 0
            if fluor_channel == 'A488':
                exp_index = 1
            if fluor_channel == 'A555':
                exp_index = 2
            if fluor_channel == 'A647':
                exp_index = 3

            z_pos = z_position_channel_list[exp_index]
            core.set_position(z_pos)

            intensity = cycif.expose(seed_expose, fluor_channel)
            new_exp = seed_expose
            while intensity < (1 - bandwidth) * benchmark_threshold or intensity > (
                    1 + bandwidth) * benchmark_threshold:
                if intensity < benchmark_threshold:
                    new_exp = benchmark_threshold / intensity * new_exp
                    if new_exp >= exp_time_limit:
                        new_exp = exp_time_limit
                        break
                    else:
                        intensity = cycif.expose(new_exp, fluor_channel)
                elif intensity > benchmark_threshold and intensity < sat_max:
                    new_exp = benchmark_threshold / intensity * new_exp
                    intensity = cycif.expose(new_exp, fluor_channel)
                elif intensity > sat_max:
                    new_exp = new_exp / 10
                    intensity = cycif.expose(new_exp, fluor_channel)
                elif new_exp >= sat_max:
                    new_exp = sat_max
                    break

            exposure_array[exp_index] = new_exp

        return exposure_array

    def z_scan_exposure_hook( image, metadata):
        '''
        Method that hooks from autofocus image acquistion calls. It takes image, calculates a focus score for it
        via focus_score method and exports a list that contains both the focus score and the z position it was taken at

        :param numpy image: single image from hooked from acquistion
        :param list[float] metadata: metadata for image

        :return: Nothing
        '''
        z = metadata.pop('ZPosition_um_Intended')  # moves up while taking z stack
        z_intensity_level = cycif.image_percentile_level(image, 0.99)
        intensity.value.append([z_intensity_level, z])

        return

    def auto_initial_expose(self, seed_expose, benchmark_threshold, channel, z_range, surface_name):
        '''
        Scans z levels around surface z center and finds brightest z position via z_scan_exposure method.
        Moves machine to that z plane and executes auto_expose method to determine proper exposure. This is meant for
        an initial exposure value for autofocus pruposes.

        :param: str surface_name: string of name of magellan surface to use
        :param list z_centers: list of z points associated with xy points where the slide tilt was compensated for
        :param: str channels: list that contains strings with channel names, for example 'DAPI'

        :return: exposure time: time for inputted channels exposure to be used for autofocus
        :rtype: int
        '''

        [x_pos, y_pos] = self.tissue_center(surface_name)
        core.set_xy_position(x_pos, y_pos)

        z_brightest = z_range[0] + z_range[2]
        core.set_position(z_brightest)

        new_exp = cycif.auto_expose(seed_expose, benchmark_threshold, z_brightest, [channel])

        return new_exp

    ##############################################

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
        xy = np.reshape(xy, (2, int(xy.size/2)))

        return xy


############################################################################################
#####focus_tile is depreciated. Need to convert this to not go to every tile, but an even sampling of them
    #######################################################################################

    def tile_pattern(self, numpy_array):
        '''
        Takes numpy array with N rows and known tile pattern and casts into new array that follows
        south-north, west-east snake pattern.


        :param numpy_array: dimensions [N, x_tiles*y_tiles]
        :param x_tiles: number x tiles in pattern
        :param y_tiles: number y tiles in pattern
        :return: numpy array with dimensions [N,x_tiles,y_tiles] with above snake pattern
        '''
        y_tiles = np.unique(numpy_array[1]).size
        x_tiles = np.unique(numpy_array[0]).size
        layers = np.shape(numpy_array)[0]
        numpy_array = numpy_array.reshape(layers, x_tiles, y_tiles)
        dummy = numpy_array.reshape(layers, y_tiles, x_tiles)
        new_numpy = np.empty_like(dummy)
        for x in range(0, layers):
            new_numpy[x] = numpy_array[x].transpose()
            new_numpy[x, ::, 1:y_tiles:2] = np.flipud(new_numpy[x, ::, 1:y_tiles:2])

        return new_numpy

    def fm_channel_initial(self, full_array, off_array):

        a488_channel_offset = off_array[0] #determine if each of these are good and repeatable offsets
        a555_channel_offset = off_array[1]
        a647_channel_offset = off_array[2]

        dummy_channel = np.empty_like(full_array[0])
        dummy_channel = np.expand_dims(dummy_channel, axis=0)
        channel_count = np.shape(full_array)[0]

        while channel_count < 6:
            full_array = np.append(full_array, dummy_channel, axis = 0)
            channel_count = np.shape(full_array)[0]

        full_array[3] = full_array[2] + a488_channel_offset #index for a488 = 3
        full_array[4] = full_array[2] + a555_channel_offset
        full_array[5] = full_array[2] + a647_channel_offset

        return full_array


    def median_fm_filter(self, full_array, channel):

        if channel == 'DAPI':
            channel_index = 2
        if channel == 'A488':
            channel_index = 3
        if channel == 'A555':
            channel_index = 4
        if channel == 'A647':
            channel_index = 5

        full_array[channel_index] = median(full_array[channel_index])

        return full_array

    def focus_tile_DAPI(self, full_array_no_pattern, z_range, exposure_time):
         '''
         Takes dictionary of XY coordinates, moves to each of them, executes autofocus algorithm from method
         auto_focus and outputs the paired in focus z coordinate

         :param dictionary tile_points_xy: dictionary containing all XY coordinates. In the form: {{x:(int)}, {y:(int)}}
         :param MMCore_Object core: Object made from Bridge.core()
         :param list z_centers: list of z points associated with xy points where the slide tilt was compensated for

         :return: XYZ points where XY are stage coords and Z is in focus coordinate. {{x:(int)}, {y:(int)}, {z:(float)}}
         :rtype: dictionary
         '''

         z_temp = []
         num = np.shape(full_array_no_pattern)[1]
         for q in range(0, num):
             z_range = [z_range[0], z_range[1], z_range[2]]

             new_x = full_array_no_pattern[0][q]
             new_y = full_array_no_pattern[1][q]
             core.set_xy_position(new_x, new_y)
             z_focused = self.auto_focus(z_range, exposure_time,'DAPI')  # here is where autofocus results go. = auto_focus
             z_temp.append(z_focused)
         z_temp = np.expand_dims(z_temp, axis = 0)
         xyz = np.append(full_array_no_pattern, z_temp, axis =0)


         return xyz

    def nonfocus_tile_DAPI(self, full_array_no_pattern):
         '''
         Makes micromagellen z the focus position at each XY point for DAPI
         auto_focus and outputs the paired in focus z coordinate

         :param dictionary tile_points_xy: dictionary containing all XY coordinates. In the form: {{x:(int)}, {y:(int)}}

         :return: XYZ points where XY are stage coords and Z is in focus coordinate. {{x:(int)}, {y:(int)}, {z:(float)}}
         :rtype: dictionary
         '''

         z_pos = magellan.get_surface('New Surface 1').get_points().get(0).z
         num = np.shape(full_array_no_pattern)[1]
         z_temp = []
         for q in range(0, num):
             z_temp.append(z_pos)
         z_temp = np.expand_dims(z_temp, axis = 0)
         xyz = np.append(full_array_no_pattern, z_temp, axis =0)

         return xyz


    def focus_tile_stain(self, full_array_with_pattern, search_range, channel, exposure_time):
         '''
         Takes dictionary of XY coordinates, moves to each of them, executes autofocus algorithm from method
         auto_focus and outputs the paired in focus z coordinate

         :param dictionary tile_points_xy: dictionary containing all XY coordinates. In the form: {{x:(int)}, {y:(int)}}
         :param list z_centers: list of z points associated with xy points where the slide tilt was compensated for

         :return: XYZ points where XY are stage coords and Z is in focus coordinate. {{x:(int)}, {y:(int)}, {z:(float)}}
         :rtype: dictionary
         '''

         if channel == 'A488':
             channel_index = 3
         if channel == 'A555':
             channel_index = 4
         if channel == 'A647':
             channel_index = 5

         #exposure_time = cycif.auto_expose(50, 2500, channel)

         shape = np.shape(full_array_with_pattern)
         x_tiles = shape[2]
         y_tiles = shape[1]
         for x in range(0, x_tiles):
             for y in range(0, y_tiles):
                 center = full_array_with_pattern[channel_index][y][x]
                 z_range = [center - search_range/2, center + search_range/2, search_range/2]
                 new_x = full_array_with_pattern[0][y][x]
                 new_y = full_array_with_pattern[1][y][x]
                 core.set_xy_position(new_x, new_y)
                 z_focused = self.auto_focus(z_range, exposure_time,channel)  # here is where autofocus results go. = auto_focus
                 full_array_with_pattern[channel_index][y][x] = z_focused

         return full_array_with_pattern

 ########################################################################################################################



    def tiled_acquire(self, full_array, channel, exposure_time, cycle_number, directory_name='E://test_control_staining'):

        add_on_folder = 'cycle_' + str(cycle_number)
        full_directory_path = directory_name + add_on_folder
        if os.path.exists(full_directory_path) == 'False':
            os.mkdir(full_directory_path)
        time.sleep(0.5)

        if channel == 'DAPI':
            channel_index = 2
        if channel == 'A488':
            channel_index = 2
        if channel == 'A555':
            channel_index = 2
        if channel == 'A647':
            channel_index = 2


        numpy_x = full_array[0]
        numpy_y = full_array[1]
        numpy_z = full_array[channel_index]
        x_tile_count = np.unique(numpy_x).size
        y_tile_count = np.unique(numpy_y).size

        core.set_xy_position(numpy_x[0][0], numpy_y[0][0])
        core.set_position(numpy_z[0][0])

        with XYTiledAcquisition(directory=full_directory_path, name=channel, show_display=False, tile_overlap=10) as acq:
            for y in range(0, y_tile_count):
                if y % 2 != 0:
                    for x in range(x_tile_count -1, -1, -1):
                        print(x,y)
                        event = {'channel': {'group': 'Color', 'config': channel}, 'exposure': exposure_time, 'row': y,
                                 'col': x}

                        core.set_position(numpy_z[y][x])
                        time.sleep(0.5)

                        acq.acquire(event)

                elif y % 2 == 0:
                    for x in range(0, x_tile_count):
                        print(x, y)
                        event = {'channel': {'group': 'Color', 'config': channel}, 'exposure': exposure_time, 'row': y,
                                 'col': x}

                        core.set_position(numpy_z[y][x])
                        time.sleep(0.5)

                        acq.acquire(event)



    def fm_map_generate(self, channels=['DAPI', 'A488', 'A555', 'A647']):
        '''
        Takes generated micro-magellan surface with name: surface_name and extracts all points from it.
        uses multi_d_acquistion to acquire all images in defined surface via xyz_acquire method,  auto exposes DAPI, A488, A555 and A647 channels.
        Takes autofocus from center tile in surface and applies value to every other tile in surface

        ::param str directory_name: highest level folder name to store all images in
        :param: list[str] channels: list that contains strings with channel names
        :return: Nothing
        '''

        surface_name = 'New Surface 1'
        tile_surface_xy = self.tile_xy_pos(surface_name)  # pull center tile coords from manually made surface
        z_center = magellan.get_surface(surface_name).get_points().get(0).z
        dapi_z_range = [z_center - 20, z_center + 20, 10]

        exp_time_array = []

        for channel in channels:

            if channel == 'DAPI':

                auto_focus_exposure_time = self.auto_initial_expose(50, 2500, 'DAPI', dapi_z_range, surface_name)
                tile_surface_xyz = self.focus_tile_DAPI(tile_surface_xy, dapi_z_range, auto_focus_exposure_time)
                tile_surface_xyz = self.tile_pattern(tile_surface_xyz)
                tile_surface_xyz = self.median_fm_filter(tile_surface_xyz, 'DAPI')
                full_array = tile_surface_xyz
                z_focused = tile_surface_xyz[2][0][0]
                exp_time = self.auto_expose(auto_focus_exposure_time, 2500, z_focused, [channel], surface_name)
                exp_time_array.append(exp_time)

                full_array = self.fm_channel_initial(full_array)

            if channel != 'DAPI':

                if channel == 'A488':
                    channel_index = 3
                if channel == 'A555':
                    channel_index = 4
                if channel == 'A647':
                    channel_index = 5

                full_array = self.focus_tile_stain(full_array, 15,  channel)
                z_focused = full_array[channel_index][0][0]
                exp_time = self.auto_expose(60, 2500, z_focused, [channel], surface_name)
                exp_time_array.append(exp_time)

        np.save('exp_array.npy', exp_time_array)
        np.save('fm_array.npy', full_array)

        return exp_time_array, full_array


    def acquire_all_tiled_surfaces(self, cycle_number, channels=['DAPI', 'A488', 'A555', 'A647'], directory_name='E://test_control_staining/'):

        full_array = np.load('fm_array.npy', allow_pickle=False)
        exp_time_array = np.load('exp_array.npy', allow_pickle=False)

        for channel in channels:

            if channel == 'DAPI':
                channel_index = 0
            if channel == 'A488':
                channel_index = 1
            if channel == 'A555':
                channel_index = 2
            if channel == 'A647':
                channel_index = 3

            exp_time = exp_time_array[channel_index]
            self.tiled_acquire(full_array, channel, exp_time, cycle_number, directory_name)

############################################
#experimental using core snap and not pycromanager acquire
############################################

    def core_tile_acquire(self, experiment_directory, channel = 'DAPI', z_slices = 3):
        '''
        Makes numpy files that contain all tiles and z slices. Order is z, tiles.

        :param self:
        :param channels:
        :param z_slices:
        :return:
        '''

        numpy_path = experiment_directory +'/' + 'np_arrays'
        os.chdir(numpy_path)
        full_array = np.load('fm_array.npy', allow_pickle=False)
        exp_time_array = np.load('exp_array.npy', allow_pickle=False)

        height_pixels = 2960
        width_pixels = 5056

        z_end = int((z_slices - 1)/2)
        z_start = (-1*z_end)

        numpy_x = full_array[0]
        numpy_y = full_array[1]
        x_tile_count = np.unique(numpy_x).size
        y_tile_count = np.unique(numpy_y).size
        total_tile_count = x_tile_count * y_tile_count

        core.set_xy_position(numpy_x[0][0], numpy_y[0][0])
        time.sleep(1)

        tif_stack = np.random.rand(z_slices, total_tile_count, height_pixels, width_pixels).astype('float16')


        if channel == 'DAPI':
            channel_index = 2
            tif_stack_c_index = 0
        if channel == 'A488':
            channel_index = 3
            tif_stack_c_index = 1
        if channel == 'A555':
            channel_index = 4
            tif_stack_c_index = 2
        if channel == 'A647':
            channel_index = 5
            tif_stack_c_index = 3

        numpy_z = full_array[channel_index]
        exp_time = int(exp_time_array[tif_stack_c_index])
        core.set_config("Color", channel)
        core.set_exposure(exp_time)
        tile_counter = 0

        for y in range(0, y_tile_count):
            if y % 2 != 0:
                for x in range(x_tile_count - 1, -1, -1):

                    core.set_xy_position(numpy_x[y][x], numpy_y[y][x])
                    time.sleep(.5)

                    z_counter = 0

                    for z in range(z_start, z_end + 1, 1):
                        core.set_position(numpy_z[y][x] + 2*z)
                        core.snap_image()
                        tagged_image = core.get_tagged_image()
                        pixels = np.reshape(tagged_image.pix,newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"]])
                        tif_stack[z_counter][tile_counter] = pixels

                        z_counter += 1

                    #print(tile_counter)
                    tile_counter += 1


            elif y % 2 == 0:
                for x in range(0, x_tile_count):

                    core.set_xy_position(numpy_x[y][x], numpy_y[y][x])
                    time.sleep(.5)

                    z_counter = 0

                    for z in range(z_start, z_end + 1, 1):
                        core.set_position(numpy_z[y][x] + 2*z)
                        core.snap_image()
                        tagged_image = core.get_tagged_image()
                        pixels = np.reshape(tagged_image.pix,
                                            newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"]])

                        tif_stack[z_counter][tile_counter] = pixels


                        z_counter += 1

                    #print(tile_counter)
                    tile_counter += 1

        return tif_stack
    
    def quick_tile_placement(self, z_tile_stack, overlap = 10):

        numpy_path = 'E:/folder_structure' +'/' + 'np_arrays'
        os.chdir(numpy_path)
        full_array = np.load('fm_array.npy', allow_pickle=False)

        numpy_x = full_array[0]
        numpy_y = full_array[1]

        x_tile_count = np.unique(numpy_x).size
        y_tile_count = np.unique(numpy_y).size

        height = z_tile_stack[0].shape[0]
        width = z_tile_stack[0].shape[1]
        overlapped_height = int(height * (1 - overlap / 100))
        overlapped_width = int(width * (1 - overlap / 100))

        pna_height = int(y_tile_count * height - int((y_tile_count) * overlap / 100 * height))
        pna_width = int(x_tile_count * width - int((x_tile_count) * overlap / 100 * width))

        pna = np.random.rand(pna_height, pna_width).astype('float16')
        tile_counter = 0

        for y in range(0, y_tile_count):
            if y % 2 != 0:
                for x in range(x_tile_count - 1, -1, -1):
                    pna[y * overlapped_height:(y + 1) * overlapped_height,
                    x * overlapped_width:(x + 1) * overlapped_width] = z_tile_stack[tile_counter][0:overlapped_height,
                                                                       0:overlapped_width]

                    tile_counter += 1


            elif y % 2 == 0:
                for x in range(0, x_tile_count):
                    pna[y * overlapped_height:(y + 1) * overlapped_height,
                    x * overlapped_width:(x + 1) * overlapped_width] = z_tile_stack[tile_counter][0:overlapped_height,
                                                                       0:overlapped_width]

                    tile_counter += 1

        return pna

    def quick_tile_optimal_z(self, z_tile_stack):

        z_slice_count = z_tile_stack.shape[0]
        tile_count = z_tile_stack[0].shape[0]

        height = z_tile_stack[0].shape[1]
        width = z_tile_stack[0].shape[2]

        optimal_stack = np.random.rand(tile_count, height, width).astype('float16')
        score_array = np.random.rand(z_slice_count, 1).astype('float32')


        for tile in range(0, tile_count):

            for z in range(0, z_slice_count):
                score_array[z] = cycif.focus_bin_generator(z_tile_stack[z][tile])

            min_score = np.min(score_array)
            optimal_index = np.where(score_array == min_score)[0][0]
            optimal_stack[tile] = z_tile_stack[optimal_index][tile]

        return optimal_stack

    def optimal_quick_preview_qt(self, z_tile_stack, channel, cycle, experiment_directory,  overlap = 10):

        optimal_stack = self.quick_tile_optimal_z(z_tile_stack)
        optimal_qt = self.quick_tile_placement(optimal_stack, overlap)
        optimal_qt_binned = optimal_qt[0:-1:4, 0:-1:4]
        self.save_optimal_quick_tile(optimal_qt_binned, channel, cycle, experiment_directory)



    def quick_tile_all_z_save(self, z_tile_stack, channel, cycle, experiment_directory, stain_bleach,  overlap = 0):


        z_slice_count = z_tile_stack.shape[0]
        numpy_path = experiment_directory +'/' + 'np_arrays'
        os.chdir(numpy_path)
        full_array = np.load('fm_array.npy', allow_pickle=False)

        numpy_x = full_array[0]
        numpy_y = full_array[1]

        x_tile_count = int(np.unique(numpy_x).size)
        y_tile_count = int(np.unique(numpy_y).size)

        height = int(z_tile_stack.shape[2])
        width = int(z_tile_stack.shape[3])

        pna_height = int(y_tile_count * height - int((y_tile_count) * overlap / 100 * height))
        pna_width = int(x_tile_count * width - int((x_tile_count) * overlap / 100 * width))

        pna_stack = np.random.rand(z_slice_count, pna_height, pna_width).astype('float16')

        for z in range(0, z_slice_count):
            pna = self.quick_tile_placement(z_tile_stack[z], overlap)
            pna_stack[z] = pna

        self.save_quick_tile(pna_stack, channel, cycle, experiment_directory, stain_bleach)


    def cycle_acquire(self, cycle_number, experiment_directory, z_slices, stain_bleach, offset_array, exp_time_array = 0, channels = ['DAPI', 'A488', 'A555', 'A647']):

        self.file_structure(experiment_directory, cycle_number)
        xy_pos = self.tile_xy_pos('New Surface 1')
        xyz_pos = self.nonfocus_tile_DAPI(xy_pos)
        xyz_tile_pattern = self.tile_pattern(xyz_pos)
        fm_array = self.fm_channel_initial(xyz_tile_pattern, offset_array)

        np.save('fm_array.npy', fm_array)

        if exp_time_array == 0:
            exp_time = self.auto_expose(experiment_directory, 300, 5000)
        else:
            exp_time = exp_time_array

        np.save('exp_array.npy', exp_time)

        for channel in channels:
            z_tile_stack = self.core_tile_acquire(experiment_directory, channel, z_slices)
            self.save_files(z_tile_stack, channel, cycle_number, experiment_directory, stain_bleach)

        self.marker_excel_file_generation(experiment_directory, cycle_number)

    def full_cycle(self, experiment_directory, cycle_number, offset_array, stain_valve, heater_state = 0):

        z_slices = 7

        pump.liquid_action('Stain', stain_valve, heater_state)  # nuc is valve=7, pbs valve=8, bleach valve=1 (action, stain_valve, heater state (off = 0, on = 1))
        microscope.cycle_acquire(cycle_number, experiment_directory, z_slices, 'Stain', offset_array)
        pump.liquid_action('Bleach')  # nuc is valve=7, pbs valve=8, bleach valve=1 (action, stain_valve, heater state (off = 0, on = 1))
        microscope.cycle_acquire(cycle_number, experiment_directory, z_slices, 'Bleach', offset_array)

######Folder System Generation########################################################


    def marker_excel_file_generation(self, experiment_directory, cycle_number):

        folder_path = experiment_directory + '/mcmicro'
        numpy_path = experiment_directory +'/' + 'np_arrays'
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
        
        for row_number in range(2, (cycle_number)*4 + 2):

            cycle_number = 4//(row_number - 2) + 1
            intercycle_channel_number = cycle_number * 4 + 1 - row_number

            ws.cell(row=row_number, column=1).value = row_number
            ws.cell(row=row_number, column=2).value = cycle_number
            ws.cell(row=row_number, column=3).value = 'Marker_' + str(row_number)
            #ws.cell(row=row_number, column=4).value = filter_sets[intercycle_channel_number]
            #ws.cell(row=row_number, column=5).value = exciation_wavlengths[intercycle_channel_number]
            #ws.cell(row=row_number, column=6).value = emission_wavelengths[intercycle_channel_number]


        row_start = (cycle_number - 1)*4 + 2
        row_end = row_start + 4

        for row_number in range(row_start, row_end):

            cycle_number = 4//(row_number - 2) + 1
            intercycle_channel_number = cycle_number * 4 + 1 - row_number

            ws.cell(row=row_number, column=8).value = exp_array[row_number-2]

        os.chdir(folder_path)
        wb.save(filename = 'markers.xlsx')


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
        cycles = np.linspace(0,highest_cycle_count,highest_cycle_count).astype(int)

        #folder layer one
        os.chdir(experiment_directory)
        self.folder_addon(experiment_directory, ['Quick_Tile'])
        self.folder_addon(experiment_directory, ['np_arrays'])
        self.folder_addon(experiment_directory, ['mcmicro'])
        self.folder_addon(experiment_directory, channels)

        #folder layer two

        for channel in channels:

            experiment_channel_directory = experiment_directory + '/' + channel

            self.folder_addon(experiment_channel_directory, ['Stain'])
            self.folder_addon(experiment_channel_directory, ['Bleach'])

        #folder layers 3 and 4

            for cycle in cycles:

                experiment_channel_stain_directory = experiment_channel_directory + '/' + 'Stain'
                experiment_channel_bleach_directory = experiment_channel_directory + '/' + 'Bleach'


                self.folder_addon(experiment_channel_stain_directory, ['cy_' + str(cycle)])
                experiment_channel_stain_cycle_directory = experiment_channel_stain_directory + '/' + 'cy_' + str(cycle)

                self.folder_addon(experiment_channel_stain_cycle_directory, ['Tiles'])


                self.folder_addon(experiment_channel_bleach_directory, ['cy_' + str(cycle)])
                experiment_channel_bleach_cycle_directory = experiment_channel_bleach_directory + '/' + 'cy_' + str(cycle)

                self.folder_addon(experiment_channel_bleach_cycle_directory, ['Tiles'])

        os.chdir(experiment_directory + '/' + 'np_arrays')

#####################################################################################################
##########Saving/File Generation Methods#############################################################################

    def post_acquisition_processor(self, experiment_directory):

        mcmicro_path = experiment_directory + r'\mcmicro\raw'
        dapi_im_path = experiment_directory + '\DAPI\Stain'
        cycle_start = 1
        cycle_start_search = 0

        os.chdir(mcmicro_path)
        while cycle_start_search == 0:
            file_name = str(experiment_directory.split("\\")[-1]) + '-cycle-0' + str(cycle_start) + '.ome.tif'
            if os.path.isfile(file_name) == 1:
                cycle_start += 1
            else:
                cycle_start_search = 1

        cycle_end = len(os.listdir(dapi_im_path)) + 1

        for cycle_number in range(cycle_start, cycle_end):
            self.infocus(experiment_directory, cycle_number, 10 ,10)
            self.mcmicro_image_stack_generator(cycle_number, experiment_directory)
            self.stage_placement(experiment_directory, cycle_number)

    def mcmicro_image_stack_generator(self, cycle_number, experiment_directory):

        numpy_path = experiment_directory + '/' + 'np_arrays'
        os.chdir(numpy_path)
        full_array = np.load('fm_array.npy', allow_pickle=False)

        xml_metadata = cycif.metadata_generator(experiment_directory)

        numpy_x = full_array[0]
        numpy_y = full_array[1]
        y_tile_count = numpy_x.shape[0]
        x_tile_count = numpy_y.shape[1]
        tile_count = int(x_tile_count * y_tile_count)

        dapi_im_path = experiment_directory + '\DAPI\Stain\cy_' + str(cycle_number) + '\Tiles' + '/focused'
        a488_im_path = experiment_directory + '\A488\Stain\cy_' + str(cycle_number) + '\Tiles' + '/focused'
        a555_im_path = experiment_directory + '\A555\Stain\cy_' + str(cycle_number) + '\Tiles' + '/focused'
        a647_im_path = experiment_directory + '\A647\Stain\cy_' + str(cycle_number) + '\Tiles' + '/focused'

        mcmicro_path = experiment_directory + r'\mcmicro\raw'

        mcmicro_stack = np.random.rand(tile_count * 4, 2960, 5056).astype('uint16')

        tile = 0
        for x in range(0, x_tile_count):
            for y in range(0, y_tile_count):
                dapi_file_name = 'x' + str(x) + '_y_' + str(y) + '_c_DAPI.tif'
                a488_file_name = 'x' + str(x) + '_y_' + str(y) + '_c_A488.tif'
                a555_file_name = 'x' + str(x) + '_y_' + str(y) + '_c_A555.tif'
                a647_file_name = 'x' + str(x) + '_y_' + str(y) + '_c_A647.tif'

                base_count_number_stack = tile * 4

                os.chdir(dapi_im_path)
                image = io.imread(dapi_file_name).astype('uint16')
                mcmicro_stack[base_count_number_stack + 0] = image

                os.chdir(a488_im_path)
                image = io.imread(a488_file_name).astype('uint16')
                mcmicro_stack[base_count_number_stack + 1] = image

                os.chdir(a555_im_path)
                image = io.imread(a555_file_name).astype('uint16')
                mcmicro_stack[base_count_number_stack + 2] = image

                os.chdir(a647_im_path)
                image = io.imread(a647_file_name).astype('uint16')
                mcmicro_stack[base_count_number_stack + 3] = image

                tile += 1

        os.chdir(mcmicro_path)
        mcmicro_file_name = str(experiment_directory.split("\\")[-1]) + '-cycle-0' + str(cycle_number) + '.ome.tif'
        tf.imwrite(mcmicro_file_name, mcmicro_stack, photometric='minisblack', description=xml_metadata)

    def metadata_generator(self, experiment_directory):

        new_ome = OME()
        ome = from_xml(r'C:\Users\mike\Documents\GitHub\AutoCIF/image.xml')
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
        for r in range(3, -1, -1):
            numpy_y[r][0] = numpy_y[r + 1][0] - y_gap
            numpy_y[r][1] = numpy_y[r + 1][1] - y_gap - col_col_gap

        # sub in needed pixel size and pixel grid changes
        ome.pixels.physical_size_x = 0.2
        ome.pixels.physical_size_y = 0.2
        ome.pixels.size_x = 5056
        ome.pixels.size_y = 2960
        # sub in other optional numbers to make metadata more accurate

        for x in range(0, total_tile_count):
            tile_metadata = copy.deepcopy(ome)
            new_ome.images.append(tile_metadata)

        # sub in stage positional information into each tile. numpy[y][x]
        tile_counter = 0
        for x in range(0, x_tile_count):
            for y in range(0, y_tile_count):

                for p in range(0, 4):
                    new_x = numpy_x[y][x] - 11000
                    new_y = numpy_y[y][x] + 2300
                    new_ome.images[tile_counter].pixels.planes[p].position_y = copy.deepcopy(new_y)
                    new_ome.images[tile_counter].pixels.planes[p].position_x = copy.deepcopy(new_x)
                    new_ome.images[tile_counter].pixels.tiff_data_blocks[p].ifd = (4 * tile_counter) + p
                tile_counter += 1

        xml = to_xml(new_ome)

        return xml

    def stage_placement(self, experiment_directory, cycle_number):
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

        fov_x_pixels = 5056
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

        for channel in channels:

            im_path = experiment_directory + '/' + channel + '\Stain\cy_' + str(cycle_number) + '\Tiles' + '/focused'
            os.chdir(im_path)

            # place images into large array

            for x in range(0, x_tile_count):
                for y in range(0, y_tile_count):
                    filename = 'x' + str(x) + '_y_' + str(y) + '_c_' + channel + '.tif'
                    image = io.imread(filename)
                    # define subsection of large array that fits dimensions of single FOV
                    x_center = numpy_x_pixels[y][x]
                    y_center = numpy_y_pixels[y][x]
                    x_start = int(x_center - fov_x_pixels / 2)
                    x_end = int(x_center + fov_x_pixels / 2)
                    y_start = int(y_center - fov_y_pixels / 2)
                    y_end = int(y_center + fov_y_pixels / 2)

                    # placed_image[y_start:y_end, x_start:x_end] = placed_image[y_start:y_end, x_start:x_end] + image
                    placed_image[y_start:y_end, x_start:x_end] = image

            # save output image
            os.chdir(quick_tile_path)
            tf.imwrite(channel + '_cy' + str(cycle_number) + '_placed.tif', placed_image)

    def focus_score(self, image, bin_level):
        '''
        Calculates focus score on image with Brenners algorithm on downsampled image.

        Image is downsampled by [binning_size x binning_size], where binning_size is currently hardcoded in.

        :param numpy image: single image from hooked from acquistion
        :return: focus score for image
        :rtype: float
        '''
        # Note: Uniform background is a bit mandatory

        # binning_size = bin_level
        binning_size = 1
        image_binned = measure.block_reduce(image, binning_size)

        # do Brenner score
        a = image_binned[1::bin_level, ::bin_level]
        # a = image_binned[2:, :]
        a = a
        b = image_binned[:-1:bin_level, ::bin_level]
        # b = image_binned[:-2, :]
        b = b
        c = (a - b)
        c = (c * c)
        c = c / 100000
        c = c.astype('int64')
        # tf.imwrite('c.tif', c)
        f_score = c.sum().astype('longlong')

        return f_score

    def infocus(self, experiment_directory, cycle_number, x_sub_section_count, y_sub_section_count):

        bin_values = [4]
        channels = ['DAPI', 'A488', 'A555', 'A647']

        dapi_im_path = experiment_directory + '/' + 'DAPI' '\Stain\cy_' + str(cycle_number) + '\Tiles'

        # load numpy arrays in
        numpy_path = experiment_directory + '/' + 'np_arrays'
        os.chdir(numpy_path)
        full_array = np.load('fm_array.npy', allow_pickle=False)

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

        for channel in channels:
            # generate imstack of z slices for tile
            # channel = 'DAPI'
            im_path = experiment_directory + '/' + channel + '\Stain\cy_' + str(cycle_number) + '\Tiles'
            os.chdir(im_path)

            z_stack = np.random.rand(z_slice_count, 2960, 5056).astype('uint16')
            for x in range(0, x_tile_count):
                for y in range(0, y_tile_count):
                    for z in range(0, z_slice_count):
                        im_path = experiment_directory + '/' + channel + '\Stain\cy_' + str(cycle_number) + '\Tiles'
                        os.chdir(im_path)
                        file_name = 'z_' + str(z) + '_x' + str(x) + '_y_' + str(y) + '_c_' + str(channel) + '.tif'
                        image = tf.imread(file_name)
                        z_stack[z] = image

                        # break into sub sections (2x3)

                    number_bins = len(bin_values)
                    brenner_sub_selector = np.random.rand(z_slice_count, number_bins, y_sub_section_count,
                                                          x_sub_section_count).astype('longlong')
                    for z in range(0, z_slice_count):
                        for y_sub in range(0, y_sub_section_count):
                            for x_sub in range(0, x_sub_section_count):

                                y_end = int((y_sub + 1) * (2960 / y_sub_section_count))
                                y_start = int(y_sub * (2960 / y_sub_section_count))
                                x_end = int((x_sub + 1) * (5056 / x_sub_section_count))
                                x_start = int(x_sub * (5056 / x_sub_section_count))
                                sub_image = z_stack[z][y_start:y_end, x_start:x_end]

                                for b in range(0, number_bins):
                                    bin_value = int(bin_values[b])
                                    score = cycif.focus_score(sub_image, bin_value)
                                    brenner_sub_selector[z][b][y_sub][x_sub] = score

                    reconstruct_array = brenner_reconstruct_array(brenner_sub_selector, z_slice_count, number_bins)
                    reconstruct_array = skimage.filters.median(reconstruct_array)
                    image_reconstructor(reconstruct_array, channel, cycle_number, y, x)

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
                    sub_scores = brenner_sub_selector[0:z_slice_count, b, y, x]
                    max_score = np.max(sub_scores)
                    max_index = np.where(sub_scores == max_score)[0][0]
                    temp_bin_max_indicies[b] = max_index
                sub_section_index_mode = stats.mode(temp_bin_max_indicies)[0][0]
                reconstruct_array[y][x] = sub_section_index_mode

        return reconstruct_array

    def image_reconstructor(self, reconstruct_array, channel, cycle_number, y_tile_number, x_tile_number):

        y_sections = np.shape(reconstruct_array)[0]
        x_sections = np.shape(reconstruct_array)[1]

        cycle_types = ['Stain', 'Bleach']

        for cycle_type in cycle_types:

            im_path = experiment_directory + '/' + channel + '/' + cycle_type + '\cy_' + str(cycle_number) + '\Tiles'
            os.chdir(im_path)
            try:
                os.mkdir('focused')
            except:
                t = 5

            # rebuilt image container
            rebuilt_image = np.random.rand(2960, 5056).astype('uint16')

            for y in range(0, y_sections):
                for x in range(0, x_sections):
                    # define y and x start and ends subsection of rebuilt image
                    y_end = int((y + 1) * (2960 / y_sections))
                    y_start = int(y * (2960 / y_sections))
                    x_end = int((x + 1) * (5056 / x_sections))
                    x_start = int(x * (5056 / x_sections))

                    # find z for specific subsection
                    z_slice = reconstruct_array[y][x]
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

    def save_files(self, z_tile_stack, channel, cycle, experiment_directory, Stain_or_Bleach = 'Stain'):

        save_directory = experiment_directory + '/' + str(channel) + '/' + Stain_or_Bleach + '/' + 'cy_' + str(cycle) + '/' + 'Tiles'

        numpy_path = experiment_directory +'/' + 'np_arrays'
        os.chdir(numpy_path)
        full_array = np.load('fm_array.npy', allow_pickle=False)

        numpy_x = full_array[0]
        numpy_y = full_array[1]

        x_tile_count = np.unique(numpy_x).size
        y_tile_count = np.unique(numpy_y).size

        z_tile_count = z_tile_stack.shape[0]


        for z in range(0, z_tile_count):

            tile_counter = 0

            for y in range(0, y_tile_count):
                if y % 2 != 0:
                    for x in range(x_tile_count - 1, -1, -1):

                        #meta = self.image_metadata_generation(x, y, channel, experiment_directory)
                        file_name = 'z_' + str(z) + '_x' + str(x) + '_y_' + str(y) + '_c_' + str(channel)+ '.tif'
                        image = z_tile_stack[z][tile_counter]
                        os.chdir(save_directory)
                        imwrite(file_name, image, photometric='minisblack')
                        tile_counter += 1

                if y % 2 == 0:
                    for x in range(0, x_tile_count):

                        #meta = self.image_metadata_generation(x, y, channel, experiment_directory)
                        file_name = 'z_' + str(z) + '_x' + str(x) + '_y_' + str(y) + '_c_' + str(channel)+ '.tif'
                        image = z_tile_stack[z][tile_counter]
                        os.chdir(save_directory)
                        imwrite(file_name, image, photometric='minisblack')
                        tile_counter += 1

    def save_tif_stack(self, tif_stack, cycle_number,  directory_name):

        add_on_folder = 'cycle_' + str(cycle_number)
        full_directory_path = directory_name + add_on_folder
        try:
            os.mkdir(full_directory_path)
        except:
           pass
        os.chdir(full_directory_path)
        file_name = 'image_array.tif'

        imwrite(file_name, tif_stack,
                bigtiff=True,
                photometric='minisblack',
                compression = 'zlib',
                compressionargs = {'level': 8} )


############################################
#depreciated or unused
############################################

    def save_optimal_quick_tile(self, image, channel, cycle, experiment_directory):

        file_name = 'quick_tile_' + str(channel) + '_' + 'cy' + str(cycle) + '.tif'

        top_path = experiment_directory + '/' + 'Quick_Tile'

        os.chdir(top_path)
        imwrite(file_name, image, photometric='minisblack' )

    def save_quick_tile(self, image, channel, cycle, experiment_directory, Stain_or_Bleach = 'Stain'):

        file_name = 'quick_tile_' + str(channel)  + '_' + 'cy' + str(cycle) + '.tif'

        top_path = experiment_directory + '/' + 'Quick_Tile'
        bottom_path = experiment_directory + '/' + str(channel) + '/' + Stain_or_Bleach + '/' + 'cy_' + str(cycle) + '/' + 'Quick_Tile'

        os.chdir(top_path)
        imwrite(file_name, image, photometric='minisblack' )

        os.chdir(bottom_path)
        imwrite(file_name, image, photometric='minisblack' )

    def image_metadata_generation(self, tile_x_number, tile_y_number, channel, experiment_directory):

        ome = OME()

        numpy_path = experiment_directory + '/' + 'np_arrays'
        os.chdir(numpy_path)
        full_array = np.load('fm_array.npy', allow_pickle=False)
        exp_time_array = np.load('exp_array.npy', allow_pickle=False)

        numpy_x = full_array[0]
        numpy_y = full_array[1]

        stage_x = numpy_x[tile_y_number][tile_x_number]
        stage_y = numpy_y[tile_y_number][tile_x_number]

        if channel == 'DAPI':
            channel_array_index = 2
            ex_wavelength = 405
            em_wavelength = 455
        if channel == 'A488':
            channel_array_index = 3
            ex_wavelength = 488
            em_wavelength = 525
        if channel == 'A555':
            channel_array_index = 4
            ex_wavelength = 540
            em_wavelength = 590
        if channel == 'A647':
            channel_array_index = 5
            ex_wavelength = 640
            em_wavelength = 690

        microscope_mk4 = Microscope(
            manufacturer='ASI',
            model='AutoCyPlex',
            serial_number='CFIC-1',
        )

        objective_16x = Objective(
            manufacturer='Nikon',
            model='16x water dipping',
            nominal_magnification=21.0,
        )

        instrument = Instrument(
            microscope=microscope_mk4,
            objectives=[objective_16x],
        )

        p_type = PixelType('uint16')
        p_id = PixelsID('Pixels:0')
        i_id = ImageID('Image:0')
        c_id = ChannelID('Channel:1:' + str(channel_array_index - 2))

        channel = Channel(
            id=c_id,
            emission_wavelength=em_wavelength,
            emission_wavelength_unit='nm',
            excitation_wavelength=ex_wavelength,
            excitation_wavelength_unit='nm',
            samples_per_pixel=1
        )

        plane = Plane(
            the_c=0,
            the_t=0,
            the_z=0,
            exposure_time=exp_time_array[channel_array_index - 2],
            exposure_time_unit='ms',
            position_x=stage_x,
            position_y=stage_y,
            position_z=1
        )

        image_pixels = Pixels(dimension_order='XYZCT', id=p_id, size_c=1, size_t=1, size_x=5056, size_y=2960, size_z=1,
                              type=p_type,
                              physical_size_x=202,
                              physical_size_x_unit='nm',
                              physical_size_y=202,
                              physical_size_y_unit='nm',
                              physical_size_z=1000,
                              physical_size_z_unit='nm',
                              planes=[plane],
                              channels=[channel], metadata_only = 'True')

        image1 = Image(id=i_id, pixels=image_pixels)
        ome.images.append(image1)
        ome.instruments.append(instrument)
        ome.images[0].instrument_ref = InstrumentRef(id=instrument.id)

        metadata = ome_types.to_xml(ome)

        return metadata

    def z_scan_exposure(self, z_range, seed_exposure, channel='DAPI'):
        '''
        Goes through z scan to over z range to determine maximum intensity value in said range and step size.

        :param list[int, int, int] z_range: defines range and stepsize with [z start, z end, z step]
        :param str channel: channel to autofocus with

        :return: z coordinate for in focus plane
        :rtype: float
        '''
        global intensity
        intensity = exp_level()  # found using class for exp_level was far superior to using it as a global variable.
        # I had issues with the image process hook function not updating brenner as a global variable

        with Acquisition(directory='C:/Users/CyCIF PC/Desktop/test_images', name='trash',
                         show_display=False,
                         image_process_fn=self.z_scan_exposure_hook) as acq:
            events = multi_d_acquisition_events(channel_group='Color',
                                                channels=[channel],
                                                z_start=z_range[0],
                                                z_end=z_range[1],
                                                z_step=z_range[2],
                                                order='zc', channel_exposures_ms=[seed_exposure])
            acq.acquire(events)
        intensity_list = [x[0] for x in intensity.value]
        brightest = max(intensity_list)
        z_level_brightest_index = intensity_list.index(brightest)
        z_level_brightest = intensity.value[z_level_brightest_index][1]

        return z_level_brightest


    def tile_xyz_gen(self, tile_points_xy, z_focused):
        '''
        Takes dictionary of XY coordinates applies inputted same focused z postion to all of them to make a xyz array

        :param dictionary tile_points_xy: dictionary containing all XY coordinates. In the form: {{x:(int)}, {y:(int)}}
        :param float z_focused: z position where surface is in focus

        :return: XYZ points where XY are stage coords and Z is in focus coordinate. {{x:(int)}, {y:(int)}, {z:(float)}}
        :rtype: dictionary
        '''

        z_temp = []
        num = len(tile_points_xy['x'])
        for q in range(0, num):
            z_temp.append(z_focused)
        tile_points_xy['z'] = z_temp
        surface_points_xyz = tile_points_xy

        return surface_points_xyz

    def numpy_xyz_gen(self, tile_points_xy, z_focused):
        '''
        Takes dictionary of XY coordinates applies inputted same focused z postion to all of them to make a xyz array

        :param dictionary tile_points_xy: dictionary containing all XY coordinates. In the form: {{x:(int)}, {y:(int)}}
        :param float z_focused: z position where surface is in focus

        :return: XYZ points where XY are stage coords and Z is in focus coordinate. {{x:(int)}, {y:(int)}, {z:(float)}}
        :rtype: numpy array
        '''

        z_temp = []
        x_temp = np.array(tile_points_xy['x'])
        y_temp = np.array(tile_points_xy['y'])
        num = len(tile_points_xy['x'])
        for q in range(0, num):
            z_temp.append(z_focused)
        z_temp_numpy = np.array(z_temp)

        xyz = np.hstack([x_temp[:,None], y_temp[:,None], z_temp_numpy[:,None]])

        return xyz

    def focused_surface_generate_xyz(self, new_surface_name, surface_points_xyz):
        '''
        Generates new micro-magellan surface with name new_surface_name and uses all points in surface_points_xyz dictionary
        as interpolation points. If surface already exists, then it checks for it and updates its xyz points.

        :param dictionary surface_points_xyz: all paired-wise points of XYZ where XY are stage coords and Z is in focus. {{x:(int)}, {y:(int)}, {z:(float)}}
        :param str new_surface_name: name that generated surface will have

        :return: Nothing
        '''
        creation_status = self.surface_exist_check(magellan, new_surface_name)  # 0 if surface with that name doesnt exist, 1 if it does
        if creation_status == 0:
            magellan.create_surface(new_surface_name)  # make surface if it doesnt already exist

        focused_surface = magellan.get_surface(new_surface_name)
        num = len(surface_points_xyz['x'])
        for q in range(0, num):
            focused_surface.add_point(surface_points_xyz['x'][q], surface_points_xyz['y'][q],surface_points_xyz['z'][q])

        return

    def focused_surface_acq_settings(self, exposure, original_surface_name, surface_name, acq_surface_num, channel):
        '''
        Takes already generated micro-magellan surface with name surface_name, sets it as a 2D surface, what channel group
        to use, sets exposure levels for all 4 channels and where to make the savings directory.

        :param numpy array exposure: a numpy array of [dapi_exposure, a488_exposure, a555_exposure, a647_exposure] with exposure times in milliseconds
        :param str surface_name: name of micro-magellan surface that is to be used as the space coordinates for this acquistion event
        :param object magellan: magellan object from magellan = Magellan() in pycromanager
        :param list[int] channel_offsets: list of offsets with respect to nuclei. Order is [DAPI, A488, A555, A647]
        :param int acq_surface_num: number that corresponds to the surface number ie. 1 in 'New Surface 1' and so on

        :return: Nothing
        '''
        acquisition_name = channel + ' surface ' + str(acq_surface_num)  # make name always be channel + surface number

        i = 0
        error = 0
        acquisition_name_array = []
        while error == 0:  # create an array of all names contained in acquistion events in MM
            try:
                acquisition_name_array.append(magellan.get_acquisition_settings(i).name_)
                i += 1
            except:
                error = 1

        try:  # see if acquistion name is within that array, if not create new event
            name_index = acquisition_name_array.index(acquisition_name)
            acq_settings = magellan.get_acquisition_settings(name_index)
        except:
            magellan.create_acquisition_settings()
            acq_settings = magellan.get_acquisition_settings(i)

        acq_settings.set_acquisition_name(
            acquisition_name)  # make same name as in focused_surface_generate function (all below as well too)
        acq_settings.set_acquisition_space_type('2d_surface')
        acq_settings.set_xy_position_source(original_surface_name)
        acq_settings.set_surface(surface_name)
        acq_settings.set_channel_group('Color')
        acq_settings.set_use_channel('DAPI', False)  # channel_name, use
        acq_settings.set_use_channel('A488', False)  # channel_name, use
        acq_settings.set_use_channel('A555', False)  # channel_name, use
        acq_settings.set_use_channel('A647', False)  # channel_name, use
        acq_settings.set_use_channel('A750', False)  # channel_name, use
        acq_settings.set_use_channel(channel, True)  # channel_name, use
        acq_settings.set_channel_exposure('DAPI', int(exposure[
                                                          0]))  # channel_name, exposure in ms can auto detect channel names and iterate names with exposure times
        acq_settings.set_channel_exposure('A488', int(exposure[1]))  # channel_name, exposure in ms
        acq_settings.set_channel_exposure('A555', int(exposure[2]))  # channel_name, exposure in ms
        acq_settings.set_channel_exposure('A647', int(exposure[3]))  # channel_name, exposure in ms
        acq_settings.set_channel_exposure('A750', 10)  # channel_name, exposure in ms

        return

    def surface_acquire(self, channels=['DAPI', 'A488', 'A555', 'A647']):
        '''
        Takes generated micro-magellan surface with name: surface_name and generates new micro-magellan surface with name:
        new_focus_surface_name and makes an acquistion event after latter surface and auto exposes DAPI, A488, A555 and A647 channels.
        Takes autofocus from center tile in surface and applies value to every other tile in surface

        :param MMCore_Object core: Object made from Bridge.core()
        :param object magellan: object created via = bridge.get_magellan()
        :param: list[str] channels: list that contains strings with channel names

        :return: Nothing
        '''

        num_surfaces = self.num_surfaces_count(magellan)  # checks how many 'New Surface #' surfaces exist. Not actual total

        for channel in channels:

            for x in range(1, num_surfaces + 1):
                surface_name = 'New Surface ' + str(x)
                new_focus_surface_name = 'Focused Surface ' + str(channel)

                tile_surface_xy = self.tile_xy_pos(surface_name, magellan)  # pull center tile coords from manually made surface
                auto_focus_exposure_time = self.auto_initial_expose(core, magellan, 50, 6500, channel, surface_name)

                z_center = magellan.get_surface(surface_name).get_points().get(0).z
                z_range = [z_center - 10, z_center + 10, 1]

                z_focused = self.auto_focus(z_range, auto_focus_exposure_time, channel)  # here is where autofocus results go. = auto_focus
                surface_points_xyz = self.tile_xyz_gen(tile_surface_xy, z_focused)  # go to each tile coord and autofocus and populate associated z with result
                self.focused_surface_generate_xyz(magellan, new_focus_surface_name, surface_points_xyz)  # will generate surface if not exist, update z points if exists

                exposure_array = self.auto_expose(core, magellan, auto_focus_exposure_time, 6500, z_focused, [channel], surface_name)
                self.focused_surface_acq_settings(exposure_array, surface_name, new_focus_surface_name, magellan, x, channel)

        return



    def xyz_acquire(self, xyz_array, channel, exposure_time, cycle_number, directory_name='E:/images/'):
        '''
        :param numpy xyz_array: numpy array 3xN where is N number of points that contain all xyz coords of positions
        :param str channel: channel name ie. DAPI, A488, A555, A647, etc.
        :param str exposure_time: exposure time required for channel in ms
        :param str directory_name: highest level folder name to store all images in
        :param int cycle_number: cycle number
        :return:  nothing
        '''

        add_on_folder = 'cycle_/' + str(cycle_number)
        full_directory_path = os.path.join(directory_name, add_on_folder)
        if os.path.exists(full_directory_path) == 'False':
            os.mkdir(full_directory_path)

        with Acquisition(directory=full_directory_path, name=channel) as acq:
            events = multi_d_acquisition_events(channel_group='Color', channels=[channel], xyz_positions=xyz_array,
                                                channel_exposures_ms=[exposure_time])
            acq.acquire(events)
            acq.await_completion()

    def mm_focus_hook(self, event):
        z_center = core.get_position()
        core.snap_image()
        tagged_image = core.get_tagged_image()
        # z_range = [z_center - 50, z_center + 50, 20]
        # exposure_time = core.get_exposure()
        # z_focused_position = self.auto_focus(z_range, exposure_time, 'DAPI')
        # core.set_position(z_focused_position)
        time.sleep(0.5)
        print(z_center)

        return event

    def micro_magellan_acq_auto_focus(self):
        '''
        going through micromagellan list and acquire each one while autofocusing at each tile
        '''

        for x in range(0, 100):
            try:
                acq = MagellanAcquisition(magellan_acq_index=x, post_hardware_hook_fn=self.core_snap_auto_focus)
                acq.await_completion()
                print('acq ' + str(x) + ' finished')
            except:
                continue

            print('acq ' + str(x) + ' finished')

    def micro_magellan_acq(self):
        '''
        go through micromagellan list and acquire each one
        '''

        for x in range(0, 100):
            try:
                acq = MagellanAcquisition(magellan_acq_index=x)
                acq.await_completion()
                print('acq ' + str(x) + ' finished')
            except:
                continue

            print('acqs ' + ' finished')

############################################
##Control arduino based microfluidic system
############################################

class arduino:

    def __init__(self):
        return

    def mqtt_publish(self, message, subtopic, topic="control", client=client):
        '''
        takes message and publishes message to server defined by client and under topic of topic/subtopic

        :param str subtopic: second tier of topic heirarchy
        :param str topic: first tier of topic heirarchy
        :param object client: client that MQTT server is on. Established in top of module
        :return:
        '''

        full_topic = topic + "/" + subtopic

        client.loop_start()
        client.publish(full_topic, message)
        client.loop_stop()

    def heater_state(self, state):

        if state == 'on':
            self.mqtt_publish(210, 'dc_pump')
        elif state == 'off':
            self.mqtt_publish(200, 'dc_pump')

    def chamber(self, fill_drain, run_time=27):
        '''
        Aquarium pumps to fill or drain outer chamber with water. Uses dispense function as backbone.

        :param str fill_drain: fill = fills chamber, drain = drains chamber
        :param int time: time in secs to fill chamber and conversely drain it
        :return: nothing
        '''

        if fill_drain == 'drain':
            self.mqtt_publish(110, 'dc_pump')
            time.sleep(run_time + 3)
            self.mqtt_publish(100, 'dc_pump')

        elif fill_drain == 'fill':
            self.mqtt_publish(111, 'dc_pump')
            time.sleep(run_time)
            self.mqtt_publish(101, 'dc_pump')


class fluidics:

    def __init__(self, mux_com_port, ob1_com_port):

        # OB1 initialize
        ob1_path = 'ASRL' + str(ob1_com_port) + '::INSTR'
        Instr_ID = c_int32()
        pump = OB1_Initialization(ob1_path.encode('ascii'), 0, 0, 0, 0, byref(Instr_ID))
        pump = OB1_Add_Sens(Instr_ID, 1, 5, 1, 0, 7, 0) #16bit working range between 0-1000uL/min, also what are CustomSens_Voltage_5_to_25 and can I really choose any digital range?

        Calib = (c_double * 1000)()
        Elveflow_Calibration_Default(byref(Calib), 1000)
        OB1_Start_Remote_Measurement(Instr_ID.value, byref(Calib), 1000)
        self.calibration_array = byref(Calib)

        set_channel_regulator = int(1)  # convert to int
        set_channel_regulator = c_int32(set_channel_regulator)  # convert to c_int32
        set_channel_sensor = int(1)
        set_channel_sensor = c_int32(set_channel_sensor)  # convert to c_int32
        PID_Add_Remote(Instr_ID.value, set_channel_regulator, Instr_ID.value, set_channel_sensor, .9, 0.002, 1)


        # MUX intiialize
        path = 'ASRL' + str(mux_com_port) + '::INSTR'
        mux_Instr_ID = c_int32()
        MUX_DRI_Initialization(path.encode('ascii'), byref(mux_Instr_ID))  # choose the COM port, it can be ASRLXXX::INSTR (where XXX=port number)

        #home
        #answer = (c_char * 40)()
        self.mux_ID = mux_Instr_ID.value
        #MUX_DRI_Send_Command(self.mux_ID, 0, answer, 40)

        self.pump_ID = Instr_ID.value

        return

    def mux_end(self):

        MUX_DRI_Destructor(self.mux_ID)

    def valve_select(self, valve_number):
        '''
        Selects valve in mux unit with associated mux_id to the valve_number declared.
        :param c_int32 mux_id: mux_id given from mux_initialization method
        :param int valve_number: number of desired valve to be selected
        :return: Nothing
        '''

        valve_number = c_int32(valve_number)
        MUX_DRI_Set_Valve(self.mux_ID, valve_number, 0) #0 is shortest path. clockwise and cc are also options
        time.sleep(1)

    def flow(self, flow_rate):

        set_channel=int(1)#convert to int
        set_channel=c_int32(set_channel)#convert to c_int32

        set_target=float(flow_rate) # in uL/min for flow
        set_target=c_double(set_target)#convert to c_double

        OB1_Set_Remote_Target(self.pump_ID, set_channel, set_target)
        #OB1_Stop_Remote_Measurement(self.pump_ID)

    def ob1_end(self):

        set_channel = int(1)  # convert to int
        set_channel = c_int32(set_channel)  # convert to c_int32

        data_sens=c_double()
        data_reg=c_double()

        x = 0
        self.flow(0)

        while x == 0:
            OB1_Get_Remote_Data(self.pump_ID, set_channel, byref(data_reg), byref(data_sens))
            flow_rate = data_sens.value

            if flow_rate < 10:
                x = 1
            if flow_rate > 10:
                x = 0
            time.sleep(1)

        OB1_Stop_Remote_Measurement(self.pump_ID)
        OB1_Destructor(self.pump_ID)

    def measure(self):

        set_channel = int(1)  # convert to int
        set_channel = c_int32(set_channel)  # convert to c_int32

        data_sens=c_double()
        data_reg=c_double()

        OB1_Get_Remote_Data(self.pump_ID, set_channel, byref(data_reg),byref(data_sens) )

        pressure = data_reg.value
        flow_rate = data_sens.value
        time_stamp = time.time()

        return pressure, flow_rate, time_stamp

    def flow_recorder(self, time_step, total_time, file_name = 'none', plot = 1):

        wb = Workbook()
        ws = wb.active
        ws.cell(row=1, column=1).value = 'Time'
        ws.cell(row=1, column=2).value = 'Flow Rate'
        ws.cell(row=1, column=3).value = 'Pressure'

        total_steps = int(total_time/time_step)
        pressure_points = np.random.rand(total_steps).astype('float16')
        time_points = np.random.rand(total_steps).astype('float16')
        flow_points = np.random.rand(total_steps).astype('float16')

        for t in range(0, total_steps):

            pressure_point, flow_point, time_point = self.measure()
            pressure_points[t] = pressure_point
            time_points[t] = t * time_step
            flow_points[t] = flow_point

            ws.cell(row= t + 2, column=1).value = t * time_step
            ws.cell(row= t + 2, column=2).value = flow_point
            ws.cell(row= t + 2, column=3).value = pressure_point

            time.sleep(time_step)

        #wb.save(filename = file_name)

        if plot == 1:
            plt.plot(time_points, flow_points, 'o', color='black')
            plt.show()


    def liquid_action(self, action_type, stain_valve = 0, heater_state = 0):

        bleach_valve = 1
        pbs_valve = 8
        bleach_time = 3 #minutes
        stain_flow_time = 45 #seconds
        if heater_state == 0:
            stain_inc_time = 45 #minutes
        if heater_state == 1:
            stain_inc_time = 45  # minutes
        nuc_valve = 7
        nuc_flow_time = 60 #seconds
        nuc_inc_time = 3 #minutes

        if action_type == 'Bleach':

            self.valve_select(bleach_valve)
            self.flow(500)
            time.sleep(60)
            self.flow(0)
            time.sleep(bleach_time*60)

            self.valve_select(pbs_valve)
            self.flow(500)
            time.sleep(60)
            self.flow(0)

        elif action_type == 'Stain':

            if heater_state == 1:
                arduino.heater_state(1)
                arduino.chamber('drain')
            else:
                pass

            self.valve_select(stain_valve)
            self.flow(500)
            time.sleep(stain_flow_time)
            self.flow(0)
            time.sleep(stain_inc_time*60)

            if heater_state == 1:
                arduino.heater_state(0)
                arduino.chamber('fill')
            else:
                pass

            self.valve_select(pbs_valve)
            self.flow(500)
            time.sleep(60)
            self.flow(0)

        elif action_type == "Wash":

            self.valve_select(pbs_valve)
            self.flow(500)
            time.sleep(60)
            self.flow(0)

        elif action_type == 'Nuc_Touchup':

            self.valve_select(nuc_valve)
            self.flow(500)
            time.sleep(nuc_flow_time)
            self.flow(0)
            time.sleep(nuc_inc_time*60)

            self.valve_select(pbs_valve)
            self.flow(500)
            time.sleep(60)
            self.flow(0)











    
    




