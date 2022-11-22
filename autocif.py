from pycromanager import Core, Acquisition, multi_d_acquisition_events, Dataset, MagellanAcquisition, Magellan, start_headless
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import time
from scipy import stats
from skimage import io
from scipy.io import loadmat
from scipy.optimize import curve_fit
import pandas as pd
import serial
import paho.mqtt.client as mqtt

client = mqtt.Client('autocyplex_server')
client.connect('10.3.141.1', 1883)
#core = Core()
#magellan = Magellan()

import autocif


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



    def tissue_center(self, mag_surface, magellan_object):
        '''
        take magellan surface and find the xy coordinates of the center of the surface
        :param mag_surface:
        :param magellan_object:
        :return: x tissue center position and y tissue center position
        :rtype: list[float, float]
        '''
        xy_pos = self.tile_xy_pos(mag_surface, magellan_object)
        x_center = (max(xy_pos['x']) + min(xy_pos['x']))/2
        y_center = (max(xy_pos['y']) + min(xy_pos['y']))/2
        return x_center, y_center

    def syr_obj_switch(self, state):
        '''
        Switch stage from objective to syringe,
        0 = objective
        1 = syringe

        :param int state: positional state of stage.

        :return: Nothing
        '''
        core = Core()
        diff_vec_x = -92000
        diff_vec_y = 7000
        y = core.get_y_position()
        x = core.get_x_position()
        z = core.get_position()
        if state == 0:
            new_y = y + diff_vec_y
            new_x = x + diff_vec_x
            core.set_xy_position(new_x, new_y)
            time.sleep(15)
            core.set_position(z - 15000)
            time.sleep(12)

        if state == 1:
            core.set_position(z + 15000)
            time.sleep(12)
            new_y = y - diff_vec_y
            new_x = x - diff_vec_x
            core.set_xy_position(new_x, new_y)
            time.sleep(15)

        return

    def num_surfaces_count(self, magellan):
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
        return surface_count

    def surface_exist_check(self, magellan, surface_name):
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

############ All in section are functions for the autofocus function
    def image_process_hook(self, image, metadata):
        '''
        Method that hooks from autofocus image acquistion calls. It takes image, calculates a focus score for it
        via focus_score method and exports a list that contains both the focus score and the z position it was taken at

        :param numpy image: single image from hooked from acquistion
        :param list[float] metadata: metadata for image

        :return: Nothing
        '''

        binning_size = 1
        z = metadata.pop('ZPosition_um_Intended') # moves up while taking z stack
        image = image[::binning_size, ::binning_size]
        image_focus_score = self.focus_score(image)
        brenner.value.append([image_focus_score, z])
        image[:, :] = []





        return

    def focus_score(self, image):
        '''
        Calculates focus score on image with Brenners algorithm on downsampled image.

        Image is downsampled by [binning_size x binning_size], where binning_size is currently hardcoded in.

        :param numpy image: single image from hooked from acquistion

        :return: focus score for image
        :rtype: float
        '''
        # Note: Uniform background is a bit mandatory

        # do binning. maybe set this as an input?
        binning_size = 4
        image_binned = image[::binning_size, ::binning_size]

        # do Brenner score
        a = image_binned[2:, :]
        b = image_binned[:-2, :]
        c = a - b
        c = c * c
        f_score_shadow = c.sum()

        return f_score_shadow



    def autofocus_fit(self):
        '''
        Takes focus scores and its associated z and fits data with gaussian. Gives back position of the fitted gaussian's middle
        (x0 parameter) which is the ideal/ in focus z plane

        :param list[float, float] brenner: list that contains pairs of [focus_score, z]

        :results: z coordinate for in focus plane
        :rtype: float
        '''

        def gauss(x, A, x0, sig, y0):

            y = y0 + (A * np.exp(-((x - x0) / sig) ** 2))
            return y


        f_score_temp = [l[0] for l in brenner.value[0:-1]]
        z = [l[1] for l in brenner.value[0:-1]]


        # curve fitted with bounds relating to the inputs
        # let's force it such that z_ideal is within our range.
        parameters, covariance = curve_fit(gauss, z, f_score_temp,
                                           bounds=[(min(f_score_temp) / 4, min(z), 0, 0),
                                                   (max(f_score_temp) * 4, max(z), (max(z) - min(z)),
                                                    max(f_score_temp))])

        # a previous iteration of bounds that are more general
        # bounds = [(min(f_score_temp) / 4, min(z) / 2, f_start / 2, 0),
        #           (max(f_score_temp) * 4, max(z) * 2, f_start * 2, max(f_score_temp))])

        #print('Z focus is located at: (microns)')
        #print(parameters[1])

        # for a sanity check, let's plot this
        #fit_f_score_gauss = cycif.gauss(z, *parameters)
        #plt.plot(z, f_score_temp, 'o', label='data')
        #plt.plot(z, fit_f_score_gauss, '-', label='fit')
        #plt.title(['fstart,fend,fstep: ', str(f_start), ' ', str(f_end), ' ', str(f_step)])
        #plt.legend()
        #plt.grid()
        #plt.show()
        return parameters[1]
#################################################
    ###This section is the hook for the initial exposure function

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


##############################################

    def core_snap_auto_focus(self, event):

        brenner = autocif.brenner()
        z_center = event['z']
        channel = event['channel']['config']
        core.set_config("Color", channel)
        exposure = event['exposure']
        core.set_exposure(exposure)

        #print(event)
        z_range = [z_center - 30, z_center + 30, 10]
        for z in range(int(z_range[0]), int(z_range[1]), int(z_range[2])):
            core.set_position(z)
            time.sleep(0.2)
            core.snap_image()
            pixels = core.get_tagged_image()
            image = np.reshape(pixels.pix, newshape=[pixels.tags['Height'], pixels.tags['Width']])
            image_focus_score = self.focus_score(image)
            brenner.value.append([image_focus_score, z])
        z_focused = self.autofocus_fit()
        core.set_position(z_focused)
        time.sleep(0.3)
        #print(z_focused)

        return event




    def auto_focus(self, z_range, exposure_time, channel = 'DAPI'):
        '''
        Runs entire auto focus algorithm in current XY position. Gives back predicted
        in focus z position via focus_score method which is the Brenner score.

        :param list[int, int, int] z_range: defines range and stepsize with [z start, z end, z step]
        :param str channel: channel to autofocus with

        :return: z coordinate for in focus plane
        :rtype: float
        '''

        brenner = autocif.brenner() # found using class for brenner was far superior to using it as a global variable.
        # I had issues with the image process hook function not updating brenner as a global variable



        with Acquisition(directory = 'C:/Users/CyCIF PC/Desktop/test_images', name='trash',
                         show_display=False,
                         image_process_fn=self.image_process_hook) as acq:
            events = multi_d_acquisition_events(channel_group='Color',
                                                channels=[channel],
                                                z_start=z_range[0],
                                                z_end=z_range[1],
                                                z_step=z_range[2],
                                                order='zc',channel_exposures_ms = [exposure_time])
            acq.acquire(events)

        z_ideal = self.autofocus_fit()

        return z_ideal

    def z_scan_exposure(self, z_range, seed_exposure, channel = 'DAPI'):
        '''
        Goes through z scan to over z range to determine maximum intensity value in said range and step size.

        :param list[int, int, int] z_range: defines range and stepsize with [z start, z end, z step]
        :param str channel: channel to autofocus with

        :return: z coordinate for in focus plane
        :rtype: float
        '''
        global intensity
        intensity = autocif.exp_level() # found using class for exp_level was far superior to using it as a global variable.
        # I had issues with the image process hook function not updating brenner as a global variable




        with Acquisition(directory = 'C:/Users/CyCIF PC/Desktop/test_images', name='trash',
                         show_display=False,
                         image_process_fn=self.z_scan_exposure_hook) as acq:
            events = multi_d_acquisition_events(channel_group='Color',
                                                channels=[channel],
                                                z_start=z_range[0],
                                                z_end=z_range[1],
                                                z_step=z_range[2],
                                                order='zc',channel_exposures_ms = [seed_exposure])
            acq.acquire(events)
        intensity_list = [x[0] for x in intensity.value]
        brightest = max(intensity_list)
        z_level_brightest_index = intensity_list.index(brightest)
        z_level_brightest = intensity.value[z_level_brightest_index][1]

        return z_level_brightest

    def auto_initial_expose(self, core, magellan_object, seed_expose, benchmark_threshold, tile_points_xy, channel, surface_name):
        '''
        Autoexposure algorithm to be executed before auto focus is used. This function scans over the entire z range that is to be used with auto focus
        and determines what the brightest intensity values are up to 99%. It then takes that and determines what is the correct
        exposure time to use via the benchmark_threshold. It automatically executes this in the upper left hand corner of the inputted MM surface.

        :param: str surface_name: string of name of magellan surface to use
        :param list z_centers: list of z points associated with xy points where the slide tilt was compensated for
        :param: str channels: list that contains strings with channel names, for example 'DAPI'

        :return: exposure time: time for inputted channels exposure to be used for autofocus
        :rtype: int
        '''
        #[x_pos, y_pos] = self.tissue_center(surface_name, magellan_object)
        #core.set_xy_position(x_pos, y_pos)
        x_initial = tile_points_xy['x'][0]
        y_initial = tile_points_xy['y'][0]
        core.set_xy_position(x_initial, y_initial)
        z_center_initial = magellan_object.get_surface(surface_name).get_points().get(0).z
        z_range = [z_center_initial - 50, z_center_initial + 50, 5]

        bandwidth = 0.1
        sat_max = 65000
        exp_time_limit = 1000


        z_brightest = self.z_scan_exposure(z_range, seed_expose, channel)
        core.set_position(z_brightest)
        intensity = self.expose(seed_expose, channel)
        new_exp = seed_expose
        while intensity < (1 - bandwidth)*benchmark_threshold or intensity > (1 + bandwidth)*benchmark_threshold:
            if intensity < benchmark_threshold:
                new_exp = benchmark_threshold/intensity * new_exp
                if new_exp >= exp_time_limit:
                    new_exp = exp_time_limit
                    break
                else:
                    intensity = self.expose(new_exp, channel)
            elif intensity > benchmark_threshold and intensity < sat_max:
                new_exp = benchmark_threshold/intensity * new_exp
                intensity = self.expose(new_exp, channel)
            elif intensity > sat_max:
                new_exp = new_exp/10
                intensity = self.expose(new_exp, channel)
            elif new_exp >= sat_max:
                new_exp = sat_max
                break

        return new_exp



    def tile_xy_pos(self, surface_name, magellan_object):
        '''
        imports previously generated micro-magellan surface with name surface_name and outputs
        the coordinates of the center of each tile from it.

        :param str surface_name: name of micro-magellan surface
        :param object magellan_object: object created via = bridge.get_magellan()

        :return: XY coordinates of the center of each tile from micro-magellan surface
        :rtype: dictionary {{x:(float)}, {y:(float)}}
        '''
        surface = magellan_object.get_surface(surface_name)
        num = surface.get_num_positions()
        xy = surface.get_xy_positions()
        tile_points_xy = {}
        x_temp = []
        y_temp = []

        for q in range(0,num ):
            pos = xy.get(q)
            pos = pos.get_center()
            x_temp.append(pos.x)
            y_temp.append(pos.y)

        tile_points_xy['x'] = x_temp  ## put all points in dictionary to ease use
        tile_points_xy['y'] = y_temp

        return tile_points_xy

    def z_range(self, tile_points_xy, surface_name, magellan_object, core, cycle_number, channel, auto_focus_exposure_time):

        '''
        takes all tile points and starts with the first position (upper left corner of surface) and adds on an amount to shift the center of the z range
        which compensates for the tilt of the slide.

        :param tile_points_xy:
        :param str surface_name: name of micro-magellan surface
        :param object magellan_object: object created via = Magellan()
        :return: list of z centers for points as compensated for slide tilt
        :rtype: list[float]
        '''

        x_slide_slope = 0.00 #rise over run of z focus change over x change in microns
        y_slide_slope = 0.00 #rise over run of z focus change over y change in microns

        z_centers = []
        if cycle_number == 1:
            z_center_initial = magellan_object.get_surface(surface_name).get_points().get(0).z
            #z_center_initial = seed_plane
        if cycle_number != 1:
            first_cycle_z = magellan_object.get_surface(surface_name).get_points().get(0).z
            z_center_initial = self.long_range_z(tile_points_xy, first_cycle_z, core, channel, auto_focus_exposure_time)

        z_centers.append(z_center_initial)

        num_points = len(tile_points_xy['x'])
        x_initial = tile_points_xy['x'][0]
        y_initial = tile_points_xy['y'][0]


        for x in range(1, num_points):
            x_point = tile_points_xy['x'][x]
            y_point = tile_points_xy['y'][x]

            x_diff = x_point - x_initial
            y_diff = y_point - y_initial

            z_offset_x = x_diff * x_slide_slope
            z_offset_y = y_diff * y_slide_slope
            z_offset = z_offset_y + z_offset_x
            adjusted_z = z_center_initial + z_offset
            z_centers.append(adjusted_z)


        return z_centers


    def long_range_z(self, tile_points_xy, first_cycle_z, core, channel, auto_focus_exposure_time):

        x_point = tile_points_xy['x'][0]
        y_point = tile_points_xy['y'][0]

        core.set_xy_position(x_point, y_point)
        z = first_cycle_z
        z_range = [z - 50, z + 50, 2]
        z_focused = self.auto_focus(z_range, channel, auto_focus_exposure_time)

        return z_focused



    def focus_tile(self, tile_points_xy, z_centers, core, exposure_time, channel):
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
        num = len(tile_points_xy['x'])
        for q in range(0, num):

            z_center = z_centers[q]
            z_range = [z_center - 50, z_center + 50, 20]

            new_x = tile_points_xy['x'][q]
            new_y = tile_points_xy['y'][q]
            core.set_xy_position(new_x, new_y)
            z_focused = self.auto_focus(z_range, exposure_time, channel)  # here is where autofocus results go. = auto_focus
            z_temp.append(z_focused)
        tile_points_xy['z'] = z_temp
        surface_points_xyz = tile_points_xy

        return surface_points_xyz

    def tile_xyz_gen(self, tile_points_xy, z_focused):
        '''
        Takes dictionary of XY coordinates applies inputted same focused z postion to all of them to make an xyz array

        :param dictionary tile_points_xy: dictionary containing all XY coordinates. In the form: {{x:(int)}, {y:(int)}}

        :return: XYZ points where XY are stage coords and Z is in focus coordinate. {{x:(int)}, {y:(int)}, {z:(float)}}
        :rtype: dictionary
        '''

        z_temp = []
        num = len(tile_points_xy['x'])
        for q in range(0, num):

            z_center = z_focused[q]

            new_x = tile_points_xy['x'][q]
            new_y = tile_points_xy['y'][q]
            z_temp.append(z_focused)
        tile_points_xy['z'] = z_temp
        surface_points_xyz = tile_points_xy

        return surface_points_xyz

    def focused_surface_generate(self, magellan_object, new_surface_name):
        '''
        Generates new micro-magellan surface with name new_surface_name and uses all points in surface_points_xyz dictionary
        as interpolation points. If surface already exists, then it checks for it and updates its xyz points.

        :param dictionary surface_points_xyz: all paired-wise points of XYZ where XY are stage coords and Z is in focus. {{x:(int)}, {y:(int)}, {z:(float)}}
        :param str new_surface_name: name that generated surface will have

        :return: Nothing
        '''
        creation_status = self.surface_exist_check(magellan_object, new_surface_name) # 0 if surface with that name doesnt exist, 1 if it does
        if creation_status == 0:
            magellan_object.create_surface(new_surface_name)  # make surface if it doesnt alreay exist
        '''
        focused_surface = magellan_object.get_surface(new_surface_name)
        num = len(surface_points_xyz['x'])
        for q in range(0, num):
            focused_surface.add_point(surface_points_xyz['x'][q], surface_points_xyz['y'][q], surface_points_xyz['z'][q])
            # access point_list and add on relevent points to surface
        '''

    def focused_surface_generate_xyz(self, magellan_object, new_surface_name, surface_points_xyz):
        '''
        Generates new micro-magellan surface with name new_surface_name and uses all points in surface_points_xyz dictionary
        as interpolation points. If surface already exists, then it checks for it and updates its xyz points.

        :param dictionary surface_points_xyz: all paired-wise points of XYZ where XY are stage coords and Z is in focus. {{x:(int)}, {y:(int)}, {z:(float)}}
        :param str new_surface_name: name that generated surface will have

        :return: Nothing
        '''
        creation_status = self.surface_exist_check(magellan_object, new_surface_name) # 0 if surface with that name doesnt exist, 1 if it does
        if creation_status == 0:
            magellan_object.create_surface(new_surface_name)  # make surface if it doesnt alreay exist
        
        focused_surface = magellan_object.get_surface(new_surface_name)
        num = len(surface_points_xyz['x'])
        for q in range(0, num):
            focused_surface.add_point(surface_points_xyz['x'][q], surface_points_xyz['y'][q], surface_points_xyz['z'][q])
            # access point_list and add on relevent points to surface
        

    def auto_expose(self, core, magellan_object, seed_expose, tile_points_xy, benchmark_threshold, z_focused_pos, channels = ['DAPI', 'A488', 'A555', 'A647'], surface_name = 'none'):
        '''
        Autoexposure algorithm. Currently, just sets each exposure to 100ms to determine exposure times.
        It also goes to center of micromagellan surface and finds channel offsets with respect to the nuclei/DAPI channel.

        :param: str surface_name: string of name of magellan surface to use
        :param list z_centers: list of z points associated with xy points where the slide tilt was compensated for
        :param: list of str channels: list that contains strings with channel names

        :return: exposure times: [dapi_exposure, a488_exposure, a555_exposure, a647_exposure]
        :rtype: numpy array
        '''
        if surface_name != 'none':
            x_initial = tile_points_xy['x'][0]
            y_initial = tile_points_xy['y'][0]
            core.set_xy_position(x_initial, y_initial)
            #new_x, new_y = self.tissue_center(surface_name, magellan_object)
            #core.set_xy_position(new_x, new_y)
            z_pos = z_focused_pos
            #z_pos = magellan_object.get_surface(surface_name).get_points().get(0).z
            core.set_position(z_pos)

        bandwidth = 0.1
        sat_max = 65000
        exp_time_limit = 1000
        exposure_array = [10,10,10,10] #dapi, a488, a555, a647

        for fluor_channel in channels:

            intensity = self.expose(seed_expose, fluor_channel)
            new_exp = seed_expose
            while intensity < (1 - bandwidth)*benchmark_threshold or intensity > (1 + bandwidth)*benchmark_threshold:
                if intensity < benchmark_threshold:
                    new_exp = benchmark_threshold/intensity * new_exp
                    if new_exp >= exp_time_limit:
                        new_exp = exp_time_limit
                        break
                    else:
                        intensity = self.expose(new_exp, fluor_channel)
                elif intensity > benchmark_threshold and intensity < sat_max:
                    new_exp = benchmark_threshold/intensity * new_exp
                    intensity = self.expose(new_exp, fluor_channel)
                elif intensity > sat_max:
                    new_exp = new_exp/10
                    intensity = self.expose(new_exp, fluor_channel)
                elif new_exp >= sat_max:
                    new_exp = sat_max
                    break

            if fluor_channel == 'DAPI':
                exposure_array[0] = new_exp
            elif fluor_channel == 'A488':
                exposure_array[1] = new_exp
            elif fluor_channel == 'A555':
                exposure_array[2] = new_exp
            elif fluor_channel == 'A647':
                exposure_array[3] = new_exp

        return exposure_array

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
        global level
        level = self.image_percentile_level(image, 0.85)

        return


    def expose(self, seed_exposure, channel = 'DAPI'):
        '''
        Runs entire auto focus algorithm in current XY position. Gives back predicted
        in focus z position via focus_score method which is the Brenner score.

        :param list[int, int, int] z_range: defines range and stepsize with [z start, z end, z step]
        :param str channel: channel to autofocus with

        :return: z coordinate for in focus plane
        :rtype: float
        '''


        with Acquisition(directory = 'C:/Users/CyCIF PC/Desktop/test_images', name='trash', show_display=False ,image_process_fn=self.exposure_hook) as acq:
            # Create some acquisition events here:

            event =  {'channel': {'group': 'Color', 'config': channel},'exposure': seed_exposure}
            acq.acquire(event)


        return level


    def channel_offsets(self, surface_name, z_centers, core, channels):
        '''
        Offset algorithm It goes to center of micromagellan surface and finds channel offsets with respect to the nuclei/DAPI channel.

        :param: str surface_name: string of name of magellan surface to use
        :param list[int] z_centers: list of z points associated with xy points where the slide tilt was compensated for
        :param: list[str] channels: list that contains strings with channel names

        :return: channel offsets: [dapi_offset a488_offset, a555_offset, a647_offset]
        :rtype: numpy array
        '''

        z_center = z_centers[0]
        z_range = [z_center - 15, z_center + 15, 2]

        center_xy = self.tissue_center(surface_name)
        core.set_xy_position(center_xy[0], center_xy[1])

        num_channels = len(channels)
        channel_offsets = np.empty(num_channels)

        for x in range(0, num_channels):
            z_focused = self.auto_focus(z_range, channels[x])
            channel_offsets[x] = z_focused

        return channel_offsets

    def focused_surface_acq_settings(self, exposure, original_surface_name, surface_name, magellan_object, acq_surface_num, channel):
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
        acquisition_name = channel + ' surface ' + str(acq_surface_num)  #make name always be channel + surface number

        i = 0
        error = 0
        acquisition_name_array = []
        while error == 0:   #create an array of all names contained in acquistion events in MM
            try:
                acquisition_name_array.append(magellan_object.get_acquisition_settings(i).name_)
                i += 1
            except:
                error = 1

        try:  # see if acquistion name is within that array, if not create new event
            name_index = acquisition_name_array.index(acquisition_name)
            acq_settings = magellan_object.get_acquisition_settings(name_index)
        except:
            magellan_object.create_acquisition_settings()
            acq_settings = magellan_object.get_acquisition_settings(i)

        acq_settings.set_acquisition_name(acquisition_name)  # make same name as in focused_surface_generate function (all below as well too)
        acq_settings.set_acquisition_space_type('2d_surface')
        acq_settings.set_xy_position_source(original_surface_name)
        acq_settings.set_surface(original_surface_name)
        #acq_settings.set_saving_dir(r'C:\Users\CyCIF PC\Desktop\test_images\tiled_images')  # standard saving directory
        acq_settings.set_channel_group('Color')
        acq_settings.set_use_channel('DAPI', False)  # channel_name, use
        acq_settings.set_use_channel('A488', False)  # channel_name, use
        acq_settings.set_use_channel('A555', False)  # channel_name, use
        acq_settings.set_use_channel('A647', False)  # channel_name, use
        acq_settings.set_use_channel(channel, True)  # channel_name, use
        acq_settings.set_channel_exposure('DAPI', int(exposure[0]))  # channel_name, exposure in ms can auto detect channel names and iterate names with exposure times
        acq_settings.set_channel_exposure('A488', int(exposure[1]))  # channel_name, exposure in ms
        acq_settings.set_channel_exposure('A555', int(exposure[2]))  # channel_name, exposure in ms
        acq_settings.set_channel_exposure('A647', int(exposure[3]))  # channel_name, exposure in ms


    def surf2focused_surf(self, core, magellan_object, channels = ['DAPI', 'A488', 'A555', 'A647']):
        '''
        Takes generated micro-magellan surface with name: surface_name and generates new micro-magellan surface with name:
        new_focus_surface_name and makes an acquistion event after latter surface and auto exposes DAPI, A488, A555 and A647 channels.
        It also compensates for slope of slide issues using the z_range function.

        :param MMCore_Object core: Object made from Bridge.core()
        :param object magellan_object: object created via = bridge.get_magellan()
        :param int cycle_number: object created via = bridge.get_magellan()
        :param: list[str] channels: list that contains strings with channel names

        :return: Nothing
        '''

        num_surfaces = self.num_surfaces_count(magellan_object) # checks how many 'New Surface #' surfaces exist. Not actual total

        for channel in channels:

            for x in range(1,num_surfaces + 1):

                surface_name = 'New Surface ' + str(x)
                new_focus_surface_name = 'Focused Surface ' + str(channel)

                tile_surface_xy = self.tile_xy_pos(surface_name,magellan_object)  # pull center tile coords from manually made surface
                auto_focus_exposure_time = self.auto_initial_expose(core, magellan_object, 50, 6500, tile_surface_xy, channel, surface_name)
                #z_centers = self.z_range(tile_surface_xy, surface_name, magellan_object, core, cycle_number, channel, auto_focus_exposure_time)


                #[x_pos, y_pos] = self.tissue_center(surface_name,magellan_object)
                #core.set_xy_position(x_pos, y_pos)
                z_center = magellan_object.get_surface(surface_name).get_points().get(0).z
                z_range = [z_center - 50, z_center + 50, 20]
                z_focused = self.auto_focus(z_range, auto_focus_exposure_time,channel)  # here is where autofocus results go. = auto_focus





                #surface_points_xyz = self.focus_tile(tile_surface_xy, z_centers, core, auto_focus_exposure_time, channel)  # go to each tile coord and autofocus and populate associated z with result
                #self.focused_surface_generate(magellan_object, new_focus_surface_name) # will generate surface if not exist, update z points if exists
                time.sleep(5)
                exposure_array = self.auto_expose(core, magellan_object, auto_focus_exposure_time, tile_surface_xy, 6500, z_focused, [channel], surface_name)
                self.focused_surface_acq_settings(exposure_array, surface_name, new_focus_surface_name, magellan_object, x, channel)

        return

    def surface_acquire(self, core, magellan_object, channels = ['DAPI', 'A488', 'A555', 'A647']):
        '''
        Takes generated micro-magellan surface with name: surface_name and generates new micro-magellan surface with name:
        new_focus_surface_name and makes an acquistion event after latter surface and auto exposes DAPI, A488, A555 and A647 channels.
        Takes auto focus from center tile in surface and applies value to every other tile in surface

        :param MMCore_Object core: Object made from Bridge.core()
        :param object magellan_object: object created via = bridge.get_magellan()
        :param int cycle_number: object created via = bridge.get_magellan()
        :param: list[str] channels: list that contains strings with channel names

        :return: Nothing
        '''

        num_surfaces = self.num_surfaces_count(magellan_object) # checks how many 'New Surface #' surfaces exist. Not actual total

        for channel in channels:

            for x in range(1,num_surfaces + 1):

                surface_name = 'New Surface ' + str(x)
                new_focus_surface_name = 'Focused Surface ' + str(channel)

                tile_surface_xy = self.tile_xy_pos(surface_name,magellan_object)  # pull center tile coords from manually made surface
                auto_focus_exposure_time = self.auto_initial_expose(core, magellan_object, 50, 6500, tile_surface_xy, channel, surface_name)
                z_center = magellan_object.get_surface(surface_name).get_points().get(0).z
                z_range = [z_center - 10, z_center + 10, 1]
                z_focused = self.auto_focus(z_range, auto_focus_exposure_time,channel)  # here is where autofocus results go. = auto_focus
                surface_points_xyz = self.focus_tile(tile_surface_xy, z_focused)  # go to each tile coord and autofocus and populate associated z with result
                self.focused_surface_generate_xyz(magellan_object, new_focus_surface_name, surface_points_xyz) # will generate surface if not exist, update z points if exists


    def mm_focus_hook(self, event):
        z_center = core.get_position()
        core.snap_image()
        tagged_image = core.get_tagged_image()
        #z_range = [z_center - 50, z_center + 50, 20]
        #exposure_time = core.get_exposure()
        #z_focused_position = self.auto_focus(z_range, exposure_time, 'DAPI')
        #core.set_position(z_focused_position)
        time.sleep(0.5)
        print(z_center)


        return event



    def micro_magellan_acq_auto_focus(self):
        '''
        going through micromagellan list and acquire each one while autofocusing at each tile
        '''

        for x in range(0,100):
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

        for x in range(0,100):
            try:
                acq = MagellanAcquisition(magellan_acq_index=x)
                acq.await_completion()
                print('acq ' + str(x) + ' finished')
            except:
                continue


            print('acqs ' + ' finished')






class arduino:

    def mqtt_publish(message, subtopic, topic = "control", client = client):
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

    def auto_load(self, time_small = 22.7, time_large = 35):
        '''
        Load in all 8 liquids into multiplexer. Numbers 2-7 just reach multiplexer while 1 and 8 flow through more.

        :param float time_small: time in secs to pump liquid from Eppendorf to multiplexer given a speed of 7ms per step.
        :param float time_large: time in secs to load larger volume liquids of 1 and 8 into multiplexer. Will make default for both.
        :return: nothing
        '''

        self.mqtt_publish(170, 'peristaltic') # turn pump on

        for x in range(2,8):
            on_command = (x * 100) + 10   # multiplication by 100 forces x value into the 1st spot on a 3 digit code.
            off_command = (x * 100) + 00  # multiplication by 100 forces x value into the 1st spot on a 3 digit code.
            self.mqtt_publish(on_command, 'valve')
            time.sleep(time_small)
            self.mqtt_publish(off_command, 'valve')

        for x in range(1,9,7):
            on_command = (x * 100) + 10   # multiplication by 100 forces x value into the 1st spot on a 3 digit code.
            off_command = (x * 100) + 00  # multiplication by 100 forces x value into the 1st spot on a 3 digit code.
            self.mqtt_publish(on_command, 'valve')
            time.sleep(time_large)
            self.mqtt_publish(off_command, 'valve')

        self.mqtt_publish(0o70, 'peristaltic') # turn pump off

    def dispense(self, liquid_selection, volume, plex_chamber_time = 24):
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

        on_command = (liquid_selection * 100) + 10  # multiplication by 100 forces x value into the 1st spot on a 3 digit code.
        off_command = (liquid_selection * 100) + 00  # multiplication by 100 forces x value into the 1st spot on a 3 digit code.
        speed = 11 # in uL per second
        time_dispense_volume = volume/speed
        transition_zone_time = 6  #time taken at 7ms steps to move past transition zone in liquid front

        if time_dispense_volume >= (plex_chamber_time + transition_zone_time):
            self.mqtt_publish(on_command, 'valve')
            self.mqtt_publish(170, 'peristaltic')  # turn pump on
            time.sleep(time_dispense_volume)
            self.mqtt_publish(0o70, 'peristaltic') # turn pump off
            self.mqtt_publish(off_command, 'valve')

        elif time_dispense_volume < (plex_chamber_time + transition_zone_time):
            self.mqtt_publish(on_command, 'valve')
            self.mqtt_publish(170, 'peristaltic')  # turn pump on
            time.sleep(time_dispense_volume)
            self.mqtt_publish(off_command, 'valve')

            self.mqtt_publish(810, 'valve') # start PBS flow
            time.sleep(plex_chamber_time + transition_zone_time - time_dispense_volume*3/4) # adjust time to get front 1/4 into chamber drain line
            self.mqtt_publish(0o70, 'peristaltic') # turn pump off
            self.mqtt_publish(800, 'valve') # end PBS flow

    def flow(self, liquid_selection, time = -1, chamber_volume = 60, plex_chamber_time = 24):
        '''
        Flow liquid selected through chamber. If defaults are used, flows through chamber volume 4x plus volume to reach chamber from multiplexer.
        If time is used, it overrides the last two parameters of chamber_volume and plex_chamber_time.

        :param int liquid_selection: liquid number to be dispensed
        :param int time: time in secs to flow fluid in absolute total
        :param float chamber_volume: volume in uL that chamber holds
        :param float plex_chamber_time: time in secs for liquid to go from multiplexer to end of chamber
        :return: nothing
        '''

        on_command = (liquid_selection * 100) + 10  # multiplication by 100 forces x value into the 1st spot on a 3 digit code.
        off_command = (liquid_selection * 100) + 00  # multiplication by 100 forces x value into the 1st spot on a 3 digit code.
        speed = 11 # in uL per second
        time_chamber_volume = chamber_volume/speed
        transition_zone_time = 6  # time taken at 7ms steps to move past transition zone in liquid front

        if time == -1:
            self.mqtt_publish(on_command, 'valve')
            self.mqtt_publish(170, 'peristaltic')  # turn pump on
            time.sleep(plex_chamber_time + transition_zone_time + 4*time_chamber_volume)
            self.mqtt_publish(0o70, 'peristaltic') # turn pump off
            self.mqtt_publish(off_command, 'valve')

        else:
            self.mqtt_publish(on_command, 'valve')
            self.mqtt_publish(170, 'peristaltic')  # turn pump on
            time.sleep(time)
            self.mqtt_publish(0o70, 'peristaltic') # turn pump off
            self.mqtt_publish(off_command, 'valve')

    def bleach(self, time):
        '''
        Flows bleach solution into chamber and keeps it on for time amount of time. Uses flow function as backbone.

        :param int time: time in secs for bleach solution to rest on sample
        :return: nothing
        '''

        self.flow(8) # flow bleach solution through chamber. Should be at number 8 slot.
        time.sleep(time)
        self.flow(1) # Flow wash with PBS

    def stain(self, liquid_selection, time = 300):
        '''
        Flows stain solution into chamber and keeps it on for time amount of time. Uses dispense function as backbone.

        :param int liquid_selection: liquid number to be dispensed
        :param int time: time in secs for bleach solution to rest on sample
        :return: nothing
        '''

        self.dispense(liquid_selection, 200)
        time.sleep(time)
        self.flow(1) # flow wash with PBS

    def chamber(self, fill_drain, time = 30):
        '''
        Aquarium pumps to fill or drain outer chamber with water. Uses dispense function as backbone.

        :param str fill_drain: fill = fills chamber, drain = drains chamber
        :param int time: time in secs to fill chamber and conversely drain it
        :return: nothing
        '''

        if fill_drain == 'drain':
            self.mqtt_publish(110, 'dc_pump')
            time.sleep(time)
            self.mqtt_publish(010, 'dc_pump')

        elif fill_drain == 'fill':
            self.mqtt_publish(111, 'dc_pump')
            time.sleep(time)
            self.mqtt_publish(011, 'dc_pump')

    def nuc_touch_up(self, liquid_selection, time = 60):
        '''
        Flows hoescht solution into chamber and keeps it on for time amount of time. Uses dispense function as backbone.

        :param int liquid_selection: liquid number to be dispensed
        :param int time: time hoescht solution rests on sample
        :return: nothing
        '''

        self.dispense(liquid_selection, 200)
        time.sleep(time)
        self.flow(1) # flow wash with PBS

    def primary_secondary_cycle(self, primary_liq_selection, secondary_liquid_selection):
        '''
        Puts primary stain on and then secondary stain.

        :param int primary_liq_selection: slot that contains primary antibody solution
        :param int secondary_liquid_selection: slot that contains secondary antibody solution
        :return: nothing
        '''
        self.stain(primary_liq_selection)
        self.stain(secondary_liquid_selection)





