import openpyxl
from pycromanager import Core, Acquisition, multi_d_acquisition_events, Dataset
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
from openpyxl import Workbook

import autocif


class brenner:
    def __init__(self):
        brenner.value = []
        return

class cycif:

    def __init__(self):

        return



    def tissue_center(mag_surface):
        surface_points = {}
        interp_points = (mag_surface.get_points())  # pull all points that are contained in micromagellan surface
        x_temp = []
        y_temp = []
        z_temp = []

        for i in range(interp_points.size()):
            point = interp_points.get(i)
            x_temp.append(point.x)
            y_temp.append(point.y)
            z_temp.append(point.z)
        surface_points["x"] = x_temp  ## put all points in dictionary to ease use
        surface_points["y"] = y_temp
        surface_points["z"] = z_temp
        x_middle = (max(surface_points["x"]) - min(surface_points["x"])) / 2 + min(
            surface_points["x"]
        )  # finds center of tissue in X and below in Y
        y_middle = (max(surface_points["y"]) - min(surface_points["y"])) / 2 + min(
            surface_points["y"]
        )
        xy_pos = list((x_middle, y_middle))
        return xy_pos

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
        x = 1
        while magellan.get_surface("New Surface " + str(x)) != None:
            x += 1
        return x - 1

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
    def auto_focus(self, z_range):
        '''
        Runs entire auto focus algorithm in current XY position. Gives back predicted
        in focus z position via focus_score method which is the Brenner score.

        :param list[int, int, int] z_range: defines range and stepsize with [z start, z end, z step]

        :return: z coordinate for in focus plane
        :rtype: float
        '''

        brenner = autocif.brenner() # found using class for brenner was far superior to using it as a global variable.
        # I had issues with the image process hook function not updating brenner as a global variable



        with Acquisition(directory='C:/Users/CyCIF PC/Desktop/test_images',
                         name='z_stack_DAPI',
                         show_display=False,
                         image_process_fn=self.image_process_hook) as acq:
            events = multi_d_acquisition_events(channel_group='Color',
                                                channels=['DAPI'],
                                                z_start=z_range[0],
                                                z_end=z_range[1],
                                                z_step=z_range[2],
                                                order='zc')
            acq.acquire(events)

        z_ideal = self.autofocus_fit()
        print(z_ideal)
        return z_ideal

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

    def z_range(self, tile_points_xy, surface_name, magellan_object):

        '''
        takes all tile points and starts with the first position (upper left corner of surface) and adds on an amount to shift the center of the z range
        which compensates for the tilt of the slide.

        :param tile_points_xy:
        :param str surface_name: name of micro-magellan surface
        :param object magellan_object: object created via = Magellan()
        :return: list of z centers for points as compensated for slide tilt
        :rtype: list[float]
        '''

        x_slide_slope = -0.0032 #rise over run of z focus change over x change in microns
        y_slide_slope = -0.0098 #rise over run of z focus change over y change in microns

        z_centers = []
        z_center_initial = magellan_object.get_surface(surface_name).get_points().get(0).z
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

            z_centers.append(z_center_initial + z_offset)

        return z_centers
        
    def focus_tile(self, tile_points_xy, z_centers, core):
        '''
        Takes dictionary of XY coordinates, moves to each of them, executes autofocus algorithm from method
        auto_focus and outputs the paired in focus z coordinate

        :param dictionary tile_points_xy: dictionary containing all XY coordinates. In the form: {{x:(int)}, {y:(int)}}
        :param MMCore_Object core: Object made from Bridge.core()
        :param z_centers: list of z points associated with xy points where the slide tilt was compensated for

        :return: XZY points where XY are stage coords and Z is in focus coordinate. {{x:(int)}, {y:(int)}, {z:(float)}}
        :rtype: dictionary
        '''
        z_temp = []
        num = len(tile_points_xy['x'])
        for q in range(0, num):

            z_center = z_centers[q]
            z_range = [z_center - 10, z_center + 10, 2]

            new_x = tile_points_xy['x'][q]
            new_y = tile_points_xy['y'][q]
            core.set_xy_position(new_x, new_y)
            #time.sleep(2)  # wait long enough for stage to translate to new location
            z_focused = self.auto_focus(z_range)  # here is where autofocus results go. = auto_focus()
            z_temp.append(z_focused)
        tile_points_xy['z'] = z_temp
        surface_points_xyz = tile_points_xy

        return surface_points_xyz

    def focused_surface_generate(self, surface_points_xyz, magellan_object, new_surface_name):  # only get 1/2 points anticipated, dont know why
        '''
        Generates new micro-magellan surface with name new_surface_name and uses all points in surface_points_xyz dictionary
        as interpolation points.

        :param dictionary surface_points_xyz: all paired-wise points of XYZ where XY are stage coords and Z is in focus. {{x:(int)}, {y:(int)}, {z:(float)}}
        :param str new_surface_name: name that generated surface will have

        :return: Nothing
        '''
        magellan_object.create_surface(new_surface_name)  # need to make naming convention
        focused_surface = magellan_object.get_surface(new_surface_name)
        num = len(surface_points_xyz['x'])
        for q in range(0, num):
            focused_surface.add_point(surface_points_xyz['x'][q], surface_points_xyz['y'][q], surface_points_xyz['z'][q])
            # access point_list and add on relevent points to surface

    def auto_expose(self):
        '''
        Autoexposure algorithm. Currently, just sets each exposure to 100ms

        :param: Nothing

        :return: exposure times: [dapi_exposure, a488_exposure, a555_exposure, a647_exposure]
        :rtype: numpy array
        '''
        exposure = np.array([10, 100, 100, 100])  # exposure time in milliseconds
        return exposure

    def focused_surface_acq_settings(self, exposure, original_surface_name, surface_name, magellan_object, acq_surface_num):
        '''
        Takes already generated micro-magellan surface with name surface_name, sets it as a 2D surface, what channel group
        to use, sets exposure levels for all 4 channels and where to make the savings directory.

        :param numpy array exposure: a numpy array of [dapi_exposure, a488_exposure, a555_exposure, a647_exposure] with exposure times in milliseconds
        :param str surface_name: name of micro-magellan surface that is to be used as the space coordinates for this acquistion event

        :return: Nothing
        '''
        magellan_object.create_acquisition_settings()

        acq_settings = magellan_object.get_acquisition_settings(acq_surface_num)
        acq_settings.set_acquisition_name(surface_name)  # make same name as in focused_surface_generate function (all below as well too)
        acq_settings.set_acquisition_space_type('2d_surface')
        acq_settings.set_xy_position_source(original_surface_name)
        acq_settings.set_surface(surface_name)
        acq_settings.set_bottom_surface(surface_name)
        acq_settings.set_top_surface(surface_name)
        acq_settings.set_saving_dir(r'C:\Users\CyCIF PC\Desktop\test_images\tiled_images')  # standard saving directory
        acq_settings.set_channel_group('Color')
        acq_settings.set_use_channel('DAPI', True)  # channel_name, use
        acq_settings.set_channel_exposure('DAPI', int(exposure[0]))  # channel_name, exposure in ms can auto detect channel names and iterate names with exposure times
        acq_settings.set_channel_exposure('A488', int(exposure[1]))  # channel_name, exposure in ms
        acq_settings.set_channel_exposure('A555', int(exposure[2]))  # channel_name, exposure in ms
        acq_settings.set_channel_exposure('A647', int(exposure[3]))  # channel_name, exposure in ms
        acq_settings.set_channel_z_offset('DAPI', 0)  # channel_name, offset in um

    def surf2focused_surf(self, core, magellan_object):
        '''
        Takes generated micro-magellan surface with name: surface_name and generates new micro-magellan surface with name:
        new_focus_surface_name and makes an acquistion event after latter surface and auto exposes DAPI, A488, A555 and A647 channels.
        It also compensates for slope of slide issues using the z_range function.

        :param MMCore_Object core: Object made from Bridge.core()
        :param object magellan_object: object created via = bridge.get_magellan()


        :return: Nothing
        '''

        num_surfaces = self.num_surfaces_count(magellan_object)
        elapsed_time_array = []






        for x in range(1,num_surfaces + 1):

            surface_name = 'New Surface ' + str(x)
            new_focus_surface_name = 'Focused Surface ' + str(x)

            tile_surface_xy = self.tile_xy_pos(surface_name,magellan_object)  # pull center tile coords from manually made surface


            z_centers = self.z_range(tile_surface_xy, surface_name, magellan_object)

            start = time.perf_counter()
            surface_points_xyz = self.focus_tile(tile_surface_xy, z_centers, core)  # go to each tile coord and autofocus and populate associated z with result
            end = time.perf_counter()
            elapsed_time = end - start

            self.focused_surface_generate(surface_points_xyz, magellan_object, new_focus_surface_name)
            exposure_array = self.auto_expose()
            self.focused_surface_acq_settings(exposure_array, surface_name, new_focus_surface_name, magellan_object, x)

            elapsed_time_array.append(elapsed_time)

        return elapsed_time_array


    def tilt_angle_chip(self, core):
        x_0 = core.get_x_position()
        y_0 = core.get_y_position()
        z_0 = core.get_position(core.get_focus_device())
        z_range= [z_0 - 10, z_0 + 10, 2]
        corners_ideal_z = []
        for x in range(0,2):
            for y in range(0,2):
                core.set_roi(x * 842, y * 493, 421, 246)
                core.set_xy_position(x_0 - x * 636, y_0 - y * 372)
                z_focused = self.auto_focus(z_range)  # here is where autofocus results go. = auto_focus()
                corners_ideal_z.append(z_focused)
                print(z_focused)
        core.set_xy_position(x_0, y_0)
        core.clear_roi()
        angle_x = np.arctan([(corners_ideal_z[3] - corners_ideal_z[1]) + (corners_ideal_z[2] - corners_ideal_z[0])][0]/1272) * 1000 * 57.32 #angle in millidegrees
        angle_y = np.arctan([(corners_ideal_z[1] - corners_ideal_z[0]) + (corners_ideal_z[3] - corners_ideal_z[2])][0]/ 1272) * 1000 * 57.32  # angle in millidegrees
        print('angle X: ' + str(angle_x) + ' millidegrees', 'angle Y: ' + str(angle_y) + ' millidegrees')
        print(corners_ideal_z)




class arduino:

    def __init__(self, com_address, baudrate =9600, timeout =5):
        '''
        Establishes connection to arduino

        :param str com_address: address of com port that arduino is connected to. Example: 'COM5'
        :param int baudrate: baudrate that connection to arduino is. Defaults to 9600.
        :param int timeout: how long to wait to check for information in line in ms. Defaults to 5

        :return: Nothing
        '''

        self.connection = serial.Serial(port=com_address, baudrate=baudrate, timeout=timeout)

        return

    def order_execute(self, orders):
        '''
        input list of commands for arduino and executes them one at a time, left to right. See serial command decoder file for info.

        :param list[int, int, ...] orders: List of all commands to be sent to arduino. This list has no limit in length.

        :return: Nothing
        '''
        # input list of serial codes for arduino and executes them one at a time, left to right. See serial command decoder for info.
        for order in orders:
            exit = 1
            command = str(order)
            command = command.encode()
            self.connection.write(command)
            while exit != 0:  # this part reads the finished command from arduino to know that the entered command was fully executed
                exit = self.connection.readline()
                exit = exit.decode()
                try:
                    exit = int(exit)
                except:
                    exit = 1
        return

    def prim_secondary_cycle(self, prim_syringe_num, secondary_syringe_num, microscope_object):
        '''
        First cycle of CyCIF consists of primary and secondaries.

        :param int prim_syringe_num: Number on autopipettor revolver that the primary antibody is in
        :param int secondary_syringe_num: Number on autopipettor revolver that the secdonary antibody is in
        :param object microscope_object: pycromanager Class object that controls micromanager core API

        :return: Nothing
        '''
        prim_syringe_command = str(str(1) + str(prim_syringe_num))
        second_syringe_command = str(str(1) + str(secondary_syringe_num))
        #microscope_object.syr_obj_switch(1)
        list_of_orders = [70, prim_syringe_command, 21, 34, 20, 61]
        self.order_execute(list_of_orders)
        time.sleep(3600)
        list_of_orders = [49, 71, 85, 70, 49, 60, second_syringe_command, 21, 34, 20, 61]
        self.order_execute(list_of_orders)
        time.sleep(2700)
        list_of_orders = [49, 71, 85, 70, 49, 60, 71]
        self.order_execute(list_of_orders)
        microscope_object.syr_obj_switch(0)

        return

    def post_acquistion_cycle(self, syringe_num):
        '''
        Normal directly conjugated cycle. It bleaches sample, restains and washes sample.

        :param int syringe_num: Number on autopipettor revolver that the direct conjugated stain is in

        :return: Nothing
        '''
        syringe_command = int(str(1) + str(syringe_num))
        cycif.syr_obj_switch(1)
        list_of_orders = [70, 53, 49, 49, 49, syringe_command, 21, 32, 20, 61]
        self.order_execute(list_of_orders)
        time.sleep(2700)
        list_of_orders = [49, 49, 49, 60, 71]
        self.order_execute(list_of_orders)
        cycif.syr_obj_switch(0)

        return


