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
import openpyxl
from ome_types.model import Instrument, Microscope, Objective, InstrumentRef, Image, Pixels, Plane, Channel
from ome_types.model.simple_types import UnitsLength, PixelType, PixelsID, ImageID, ChannelID
from ome_types import from_xml, OME, from_tiff


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

    def auto_expose(self, seed_expose, benchmark_threshold, channels=['DAPI', 'A488', 'A555', 'A647']):
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

        #if surface_name != 'none':
            #new_x, new_y = cycif.tissue_center(self, surface_name)  # uncomment if want center of tissue to expose
            #core.set_xy_position(new_x, new_y)
            #z_pos = z_focused_pos
            # z_pos = magellan.get_surface(surface_name).get_points().get(0).z
            #core.set_position(z_pos)

        bandwidth = 0.1
        sat_max = 65000
        exp_time_limit = 1000
        exposure_array = [10, 10, 10, 10]  # dapi, a488, a555, a647

        for fluor_channel in channels:

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

        return new_exp

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

    def fm_channel_initial(self, full_array):

        a488_channel_offset = -7.5 #determine if each of these are good and repeatable offsets
        a555_channel_offset = -13
        a647_channel_offset = -7

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



    def quick_tile_all_z_save(self, z_tile_stack, channel, cycle, experiment_directory,  overlap = 0):


        z_slice_count = z_tile_stack.shape[0]
        numpy_path = experiment_directory +'/' + 'np_arrays'
        os.chdir(numpy_path)
        full_array = np.load('fm_array.npy', allow_pickle=False)

        numpy_x = full_array[0]
        numpy_y = full_array[1]

        x_tile_count = np.unique(numpy_x).size
        y_tile_count = np.unique(numpy_y).size

        height = z_tile_stack.shape[2]
        width = z_tile_stack.shape[3]

        pna_height = int(y_tile_count * height - int((y_tile_count) * overlap / 100 * height))
        pna_width = int(x_tile_count * width - int((x_tile_count) * overlap / 100 * width))

        pna_stack = np.random.rand(z_slice_count, pna_height, pna_width).astype('float16')

        for z in range(0, z_slice_count):
            pna = self.quick_tile_placement(z_tile_stack[z], overlap)
            pna_stack[z] = pna

        self.save_quick_tile(pna_stack, channel, cycle, experiment_directory)

######Folder System Generation########################################################

    def image_metadata_generation(self, tile_x_number, tile_y_number, channel, experiment_directory):

        ome = OME()

        numpy_path = experiment_directory + '/' + 'np_arrays'
        os.chdir(numpy_path)
        full_array = np.load('fm_array.npy', allow_pickle=False)
        exp_time_array = np.load('exp_array.npy', allow_pickle=False)

        numpy_x = full_array[0]
        numpy_y = full_array[1]

        stage_x = numpy_x[tile_x_number][tile_y_number]
        stage_y = numpy_y[tile_x_number][tile_y_number]

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
            exposure_time=exp_time_array[channel_array_index],
            exposure_time_unit='ms',
            position_x=stage_x,
            position_y=stage_y,
            position_z=0
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
                              channels=[channel])

        image1 = Image(id=i_id, pixels=image_pixels)
        ome.images.append(image1)
        ome.instruments.append(instrument)
        ome.images[0].instrument_ref = InstrumentRef(id=instrument.id)

        metadata = to_xml(ome)

        return metadata


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
            wb = load_workbook(markers.xlsx)
        except:
            wb = openpyxl.Workbook()
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

            cycle_number = 4//row_number + 1
            intercycle_channel_number = cycle_number * 4 + 1 - row_number

            ws.cell(row=row_number, column=1).value = row_number
            ws.cell(row=row_number, column=2).value = cycle_number
            ws.cell(row=row_number, column=3).value = 'Marker_' + str(row_number)
            ws.cell(row=row_number, column=4).value = filter_sets[intercycle_channel_number]
            ws.cell(row=row_number, column=5).value = exciation_wavlengths[intercycle_channel_number]
            ws.cell(row=row_number, column=6).value = emission_wavelengths[intercycle_channel_number]


        row_start = (cycle_number - 1)*4 + 2
        row_end = row_start + 4

        for row_number in range(row_start, row_end):

            cycle_number = 4 // row_number + 1
            intercycle_channel_number = cycle_number * 4 + 1 - row_number

            ws.cell(row=row_number, column=8).value = exp_array[intercycle_channel_number]

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
        cycles = np.linspace(1,highest_cycle_count,highest_cycle_count).astype(int)

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
                self.folder_addon(experiment_channel_stain_cycle_directory, ['Quick_Tile'])


                self.folder_addon(experiment_channel_bleach_directory, ['cy_' + str(cycle)])
                experiment_channel_bleach_cycle_directory = experiment_channel_bleach_directory + '/' + 'cy_' + str(cycle)

                self.folder_addon(experiment_channel_bleach_cycle_directory, ['Tiles'])
                self.folder_addon(experiment_channel_bleach_cycle_directory, ['Quick_Tile'])

        os.chdir(experiment_directory + '/' + 'np_arrays')

#####################################################################################################
##########Saving Methods#############################################################################

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

    def save_files(self, z_tile_stack, channel, cycle, experiment_directory, Stain_or_Bleach = 'Stain'):

        save_directory = experiment_directory + '/' + str(channel) + '/' + Stain_or_Bleach + '/' + 'cy_' + str(cycle) + '/' + 'Tiles'

        z_tile_count = z_tile_stack.shape[0]
        tile_count = z_tile_stack.shape[1]

        for z in range(0, z_tile_count):
            for t in range(0, tile_count):

                file_name = 'z_' + str(z) + '_' + str(t) + '_' + str(channel)+ '.tif'
                image = z_tile_stack[z][t]
                os.chdir(save_directory)
                imwrite(file_name, image, photometric='minisblack')

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

    def auto_load(self, time_small=26, time_large=40):
        '''
        Load in all 8 liquids into multiplexer. Numbers 2-7 just reach multiplexer while 1 and 8 flow through more.

        :param float time_small: time in secs to pump liquid from Eppendorf to multiplexer given a speed of 7ms per step.
        :param float time_large: time in secs to load larger volume liquids of 1 and 8 into multiplexer. Will make default for both.
        :return: nothing
        '''

        self.mqtt_publish(170, 'peristaltic')  # turn pump on

        for x in range(3, 8):
            on_command = (x * 100) + 10  # multiplication by 100 forces x value into the 1st spot on a 3 digit code.
            off_command = (x * 100) + 00  # multiplication by 100 forces x value into the 1st spot on a 3 digit code.
            self.mqtt_publish(on_command, 'valve')
            time.sleep(time_small)
            self.mqtt_publish(off_command, 'valve')

        for x in range(1, 9, 7):
            on_command = (x * 100) + 10  # multiplication by 100 forces x value into the 1st spot on a 3 digit code.
            off_command = (x * 100) + 00  # multiplication by 100 forces x value into the 1st spot on a 3 digit code.
            self.mqtt_publish(on_command, 'valve')
            time.sleep(time_large)
            self.mqtt_publish(off_command, 'valve')

        self.mqtt_publish(810, 'valve')
        time.sleep(60)
        self.mqtt_publish(800, 'valve')

        self.mqtt_publish(0o70, 'peristaltic')  # turn pump off

    def dispense(self, liquid_selection, volume, plex_chamber_time=24):
        '''
        Moves volume defined of liquid selected into chamber.
        Acts different if volume requested is larger than volume from multiplexer through chamber.
        Difference being PBS flow is not activated if volume is larger, but is if not.
        Make sure to have volume be greater than chamber volume of 60uL

        :param int liquid_selection: liquid number to be dispensed
        :param int volume: volume of chosen liquid to be dispensed in uL
        :param float plex_chamber_time: time in secs to flow from multiplexer to chambers end.
        :return: nothing
        '''

        on_command = (
                                 liquid_selection * 100) + 10  # multiplication by 100 forces x value into the 1st spot on a 3 digit code.
        off_command = (
                                  liquid_selection * 100) + 00  # multiplication by 100 forces x value into the 1st spot on a 3 digit code.
        speed = 11  # in uL per second
        time_dispense_volume = volume / speed
        transition_zone_time = 6  # time taken at 7ms steps to move past transition zone in liquid front

        if time_dispense_volume >= (plex_chamber_time + transition_zone_time):
            self.mqtt_publish(on_command, 'valve')
            self.mqtt_publish(170, 'peristaltic')  # turn pump on
            time.sleep(time_dispense_volume)
            self.mqtt_publish(0o70, 'peristaltic')  # turn pump off
            self.mqtt_publish(off_command, 'valve')

        elif time_dispense_volume < (plex_chamber_time + transition_zone_time):
            self.mqtt_publish(on_command, 'valve')
            self.mqtt_publish(170, 'peristaltic')  # turn pump on
            time.sleep(time_dispense_volume)
            self.mqtt_publish(off_command, 'valve')

            self.mqtt_publish(810, 'valve')  # start PBS flow
            time.sleep(
                plex_chamber_time + transition_zone_time - time_dispense_volume * 3 / 4)  # adjust time to get front 1/4 into chamber drain line
            self.mqtt_publish(0o70, 'peristaltic')  # turn pump off
            self.mqtt_publish(800, 'valve')  # end PBS flow

    def flow(self, liquid_selection, run_time=-1, chamber_volume=60, plex_chamber_time=24):
        '''
        Flow liquid selected through chamber. If defaults are used, flows through chamber volume 4x plus volume to reach chamber from multiplexer.
        If time is used, it overrides the last two parameters of chamber_volume and plex_chamber_time.

        :param int liquid_selection: liquid number to be dispensed
        :param int time: time in secs to flow fluid in absolute total
        :param float chamber_volume: volume in uL that chamber holds
        :param float plex_chamber_time: time in secs for liquid to go from multiplexer to end of chamber
        :return: nothing
        '''

        on_command = (
                                 liquid_selection * 100) + 10  # multiplication by 100 forces x value into the 1st spot on a 3 digit code.
        off_command = (
                                  liquid_selection * 100) + 00  # multiplication by 100 forces x value into the 1st spot on a 3 digit code.
        speed = 11  # in uL per second
        time_chamber_volume = chamber_volume / speed
        transition_zone_time = 6  # time taken at 7ms steps to move past transition zone in liquid front

        if run_time == -1:
            self.mqtt_publish(on_command, 'valve')
            self.mqtt_publish(170, 'peristaltic')  # turn pump on
            time.sleep(plex_chamber_time + transition_zone_time + 4 * time_chamber_volume)
            self.mqtt_publish(0o70, 'peristaltic')  # turn pump off
            self.mqtt_publish(off_command, 'valve')

        else:
            self.mqtt_publish(on_command, 'valve')
            self.mqtt_publish(170, 'peristaltic')  # turn pump on
            time.sleep(run_time)
            self.mqtt_publish(0o70, 'peristaltic')  # turn pump off
            self.mqtt_publish(off_command, 'valve')

    def bleach(self, run_time):
        '''
        Flows bleach solution into chamber and keeps it on for time amount of time. Uses flow function as backbone.

        :param int time: time in secs for bleach solution to rest on sample
        :return: nothing
        '''

        self.flow(1)  # flow bleach solution through chamber. Should be at number 8 slot.
        time.sleep(run_time)
        self.flow(8)  # Flow wash with PBS

    def stain(self, liquid_selection, run_time=2700):
        '''
        Flows stain solution into chamber and keeps it on for time amount of time. Uses dispense function as backbone.

        :param int liquid_selection: liquid number to be dispensed
        :param int time: time in secs for bleach solution to rest on sample
        :return: nothing
        '''

        self.dispense(liquid_selection, 300)
        time.sleep(run_time)
        self.flow(8)  # flow wash with PBS

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

    def nuc_touch_up(self, liquid_selection, run_time=60):
        '''
        Flows hoescht solution into chamber and keeps it on for time amount of time. Uses dispense function as backbone.

        :param int liquid_selection: liquid number to be dispensed
        :param int time: time hoescht solution rests on sample
        :return: nothing
        '''

        self.dispense(liquid_selection, 200)
        time.sleep(run_time)
        self.flow(8)  # flow wash with PBS

    def primary_secondary_cycle(self, primary_liq_selection, secondary_liquid_selection):
        '''
        Puts primary stain on and then secondary stain.

        :param int primary_liq_selection: slot that contains primary antibody solution
        :param int secondary_liquid_selection: slot that contains secondary antibody solution
        :return: nothing
        '''
        self.stain(primary_liq_selection)
        self.stain(secondary_liquid_selection)





