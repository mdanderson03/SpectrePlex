from pycromanager import Core, Acquisition, multi_d_acquisition_events, Dataset, MagellanAcquisition, Magellan, start_headless, XYTiledAcquisition
import numpy as np
import time
from scipy.optimize import curve_fit
import paho.mqtt.client as mqtt
import matplotlib.pyplot as plt
from skimage import io, measure
import os
import math
from datetime import datetime


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
    def image_process_hook(self, image, metadata):
        '''
        Method that hooks from autofocus image acquistion calls. It takes image, calculates a focus score for it
        via focus_score method and exports a list that contains both the focus score and the z position it was taken at

        :param numpy image: single image from hooked from acquistion
        :param list[float] metadata: metadata for image

        :return: Nothing
        '''

        z = metadata.pop('ZPosition_um_Intended')  # moves up while taking z stack
        image_focus_score = self.focus_bin_generator(image)
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

    def focus_bin_generator(self, image):
        '''
        takes image and calculates brenner scores for various bin levels of 64, 32, 16, 8 and 4 and outputs them.
        :param image: numpy array 2D
        :return: list of brenner scores of binned images
        '''

        #bin_levels = [4,8,16,32,64]
        #focus_scores = [0]*len(bin_levels)



        focus_score = self.focus_score(image, 4)

        return focus_score

    def focus_score(self, image, bin_level):
        '''
        Calculates focus score on image with Brenners algorithm on downsampled image.

        Image is downsampled by [binning_size x binning_size], where binning_size is currently hardcoded in.

        :param numpy image: single image from hooked from acquistion
        :return: focus score for image
        :rtype: float
        '''
        # Note: Uniform background is a bit mandatory

        binning_size = bin_level
        image_binned = io.measure.block_reduce(image, binning_size)

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
                         image_process_fn=self.image_process_hook) as acq:
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

    def image_percentile_level(self, image, cut_off_threshold):
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

    def exposure_hook(self, image, metadata):
        '''
        Hook for expose method. Returns metric based on image_percentile_level method.

        :param numpy array image: image from expose
        :param metadata: metadata from exposure
        :return: nothing
        '''

        global level
        level = self.image_percentile_level(image, 0.85)

        return

    def expose(self, seed_exposure, channel='DAPI'):
        '''
        Runs entire auto exposure algorithm in current XY position. Gives back predicted
        in exposure level via image_percentile_level method with the exposure_hook

        :param list[int, int, int] z_range: defines range and stepsize with [z start, z end, z step]
        :param str channel: channel to autofocus with

        :return: z coordinate for in focus plane
        :rtype: float
        '''

        with Acquisition(directory='C:/Users/CyCIF PC/Desktop/test_images/', name='trash', show_display=False,
                         image_process_fn=self.exposure_hook) as acq:
            # Create some acquisition events here:

            event = {'channel': {'group': 'Color', 'config': channel}, 'exposure': seed_exposure}
            acq.acquire(event)

        return level

    def auto_expose(self, seed_expose, benchmark_threshold, z_focused_pos, channels=['DAPI', 'A488', 'A555', 'A647'], surface_name='none'):
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

        if surface_name != 'none':
            new_x, new_y = self.tissue_center(surface_name)  # uncomment if want center of tissue to expose
            core.set_xy_position(new_x, new_y)
            z_pos = z_focused_pos
            # z_pos = magellan.get_surface(surface_name).get_points().get(0).z
            core.set_position(z_pos)

        bandwidth = 0.1
        sat_max = 65000
        exp_time_limit = 1000
        exposure_array = [10, 10, 10, 10]  # dapi, a488, a555, a647

        for fluor_channel in channels:

            intensity = self.expose(seed_expose, fluor_channel)
            new_exp = seed_expose
            while intensity < (1 - bandwidth) * benchmark_threshold or intensity > (
                    1 + bandwidth) * benchmark_threshold:
                if intensity < benchmark_threshold:
                    new_exp = benchmark_threshold / intensity * new_exp
                    if new_exp >= exp_time_limit:
                        new_exp = exp_time_limit
                        break
                    else:
                        intensity = self.expose(new_exp, fluor_channel)
                elif intensity > benchmark_threshold and intensity < sat_max:
                    new_exp = benchmark_threshold / intensity * new_exp
                    intensity = self.expose(new_exp, fluor_channel)
                elif intensity > sat_max:
                    new_exp = new_exp / 10
                    intensity = self.expose(new_exp, fluor_channel)
                elif new_exp >= sat_max:
                    new_exp = sat_max
                    break

        return new_exp

    def z_scan_exposure_hook(self, image, metadata):
        '''
        Method that hooks from autofocus image acquistion calls. It takes image, calculates a focus score for it
        via focus_score method and exports a list that contains both the focus score and the z position it was taken at

        :param numpy image: single image from hooked from acquistion
        :param list[float] metadata: metadata for image

        :return: Nothing
        '''
        z = metadata.pop('ZPosition_um_Intended')  # moves up while taking z stack
        z_intensity_level = self.image_percentile_level(image, 0.99)
        intensity.value.append([z_intensity_level, z])

        return

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

        z_center_initial = magellan.get_surface(surface_name).get_points().get(0).z

        z_brightest = self.z_scan_exposure(z_range, seed_expose, channel)
        core.set_position(z_brightest)

        new_exp = self.auto_expose(seed_expose, benchmark_threshold, z_brightest, [channel])

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

        xy = np.hstack([x_temp[:, None], y_temp[:, None]])

        return xy

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

############################################################################################
#####focus_tile is depreciated. Need to convert this to not go to every tile, but an even sampling of them
    #######################################################################################

    def tile_pattern(self, numpy_array, x_tiles, y_tiles):
        '''
        Takes numpy array with N rows and known tile pattern and casts into new array that follows
        south-north, west-east snake pattern.


        :param numpy_array: dimensions [N, x_tiles*y_tiles]
        :param x_tiles: number x tiles in pattern
        :param y_tiles: number y tiles in pattern
        :return: numpy array with dimensions [N,x_tiles,y_tiles] with above snake pattern
        '''

        layers = np.shape(numpy_array)[0]
        numpy_array = numpy_array.reshape(layers, x_tiles, y_tiles)
        dummy = numpy_array.reshape(layers, y_tiles, x_tiles)
        new_numpy = np.empty_like(dummy)
        for x in range(0, layers):
            new_numpy[x] = numpy_array[x].transpose()
            new_numpy[x, ::, 1:y_tiles:2] = np.flipud(new_numpy[x, ::, 1:y_tiles:2])

        return new_numpy

    def median_fm_filter(self, full_array):
        full_array[2] = median(full_array[2])

        return full_array

    def focus_tile(self, full_array_no_pattern, z_range, offset, exposure_time, channel):
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
         num = np.shape(full_array_no_pattern)[0]
         print(num)
         for q in range(0, num):
             z_range = [z_range[0] + offset, z_range[1] + offset, z_range[2]]

             new_x = full_array_no_pattern[0][q]
             new_y = full_array_no_pattern[1][q]
             core.set_xy_position(new_x, new_y)
             z_focused = self.auto_focus(z_range, exposure_time,channel)  # here is where autofocus results go. = auto_focus
             z_temp.append(z_focused)
         xyz = np.append(full_array_no_pattern, z_temp)


         return xyz

 ########################################################################################################################

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

    def tiled_acquire(self, full_array, channel, exposure_time, cycle_number, directory_name='E://test_control_staining'):

        add_on_folder = 'cycle_' + str(cycle_number)
        full_directory_path = directory_name + add_on_folder
        if os.path.exists(full_directory_path) == 'False':
            os.mkdir(full_directory_path)
        time.sleep(0.5)

        if channel == 'DAPI':
            channel_index = 3
        if channel == 'A488':
            channel_index = 3
        if channel == 'A555':
            channel_index = 3
        if channel == 'A647':
            channel_index = 3


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
                        event = {'channel': {'group': 'Color', 'config': channel}, 'exposure': exposure_time, 'row': y,
                                 'col': x}

                        core.set_position(numpy_z[y][x])
                        time.sleep(0.5)

                        #need to experiment here and see how y and x are packed together to make sure we are getting the right z
                        #associated with the right xy
                        acq.acquire(event)

                elif y % 2 == 0:
                    for x in range(0, x_tile_count):
                        event = {'channel': {'group': 'Color', 'config': channel}, 'exposure': exposure_time, 'row': y,
                                 'col': x}

                        core.set_position(numpy_z[y][x])
                        time.sleep(0.5)

                        acq.acquire(event)



    def mmsurface_2_acquire(self, cycle_number, channel=['DAPI', 'A488', 'A555', 'A647'], directory_name='E:/test_control_staining/'):
        '''
        Takes generated micro-magellan surface with name: surface_name and extracts all points from it.
        uses multi_d_acquistion to acquire all images in defined surface via xyz_acquire method,  auto exposes DAPI, A488, A555 and A647 channels.
        Takes autofocus from center tile in surface and applies value to every other tile in surface

        :param int cycle_number: cycle number
        ::param str directory_name: highest level folder name to store all images in
        :param: list[str] channels: list that contains strings with channel names
        :return: Nothing
        '''

        surface_name = 'New Surface 1'
        #num_channels = len(channels)  # checks how many 'New Surface #' surfaces exist. Not actual total
        tile_surface_xy = self.tile_xy_pos(surface_name)  # pull center tile coords from manually made surface
        z_center = magellan.get_surface(surface_name).get_points().get(0).z
        z_range = [z_center - 20, z_center + 20, 10]

        z_focused_array = []
        exp_time_array = []


        auto_focus_exposure_time = self.auto_initial_expose(50, 2500, channel, z_range, surface_name)
        tile_surface_xyz = self.focus_tile( tile_surface_xy, z_range, 0, auto_focus_exposure_time, channel)
        z_focused = tile_surface_xyz[2][0][0]
        exp_time = self.auto_expose(auto_focus_exposure_time, 2500, z_focused, [channel], surface_name)
        tile_surface_xyz = self.median_fm_filter(tile_surface_xyz)

        self.tiled_acquire(tile_surface_xyz, channel, exp_time, cycle_number, directory_name)

        return

    def acquire_all_tiled_surfaces(self, cycle_number, channels=['DAPI', 'A488', 'A555', 'A647'], directory_name='E://test_control_staining/'):
        for channel in channels:
            self.mmsurface_2_acquire(cycle_number, channel, directory_name)

    ##################################################################
######Acquire all MM acquistion surfaces and autofocus each one right before taking image
##################################################################

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

#######################################################################
####Acquire MM acquistion surfaces with no additional autofocus
#######################################################################

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
            time.sleep(time)
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

        self.dispense(liquid_selection, 275)
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





