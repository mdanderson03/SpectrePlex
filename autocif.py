from pycromanager import Bridge, Acquisition, multi_d_acquisition_events, Dataset
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


class cycif:

    #bridge = Bridge()
    #core = bridge.core()
    z_range = [5800, 5900, 10] #[z start, z end, z step]

    def __init__(self):

        return



    def tissue_center(mag_surface):
        surface_points = {}
        interp_points = (
            mag_surface.get_points()
        )  # pull all points that are contained in micromagellan surface
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

        Args:
            state (int)

        Returns:
            nothing
        '''
        core = cycif.core
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
            core.set_position(z - 12000)
            time.sleep(12)

        if state == 1:
            core.set_position(z + 12000)
            time.sleep(12)
            new_y = y - diff_vec_y
            new_x = x - diff_vec_x
            core.set_xy_position(new_x, new_y)
            time.sleep(15)

        return




############ All in section are functions for the autofocus function
    def focus(self, image, metadata):
        '''
        Method that hook from autofocus image acquistion calls. It takes image, calculates a focus score for it
        via focus_score method and exports a list that contains both the focus score and the z position it was taken at

        Args:
            image (numpy array),
            metadata (unknown)

        Returns:
            Nothing
        '''
        # append z
        z = f_start + metadata['Axes']['z'] * f_step  ### is z change in z here?

        # focus_score(image,z) outputs f_score and z
        # focus_score(image, z)

        # image in numpy array and z in integer with micron units
        # output should something like this:
        # [[fscore1,z1],[fscore2,z2],...]
        brenner.append(cycif.focus_score(image, z))

        # null the image to save memory
        # plt.imshow(image)
        # plt.show()
        # image[:100, :100] = 0
        image[:, :] = []

        return

    def focus_score(self, image):
        '''
        Calculates focus score with Brenners algorithm.

        Args:
            image (numpy array)

        Returns:
            focus score (float)
        '''
        # focus score using Brenner's score function
        # Note: Uniform background is a bit mandatory
        a = image[2:, :]
        b = image[:-2, :]
        c = a - b
        c = c * c
        f_score_shadow = c.sum()

        # check to see if this works. and it does.
        # print(f_score_shadow)
        # print(z)
        return f_score_shadow

    def gauss(self, x, A, x0, sig, y0):
        '''
        gaussian function that gives y value back when given all parameters including x.

        Args:
            x (float),
            A (float),
            X0 (float),
            sig (float),
            y0 (float)

        Returns:
            y (float)
        '''
        # fit to a gaussian
        y = y0 + (A * np.exp(-((x - x0) / sig) ** 2))
        return y

    def autofocus_fit(self, brenner):
        '''
        Takes focus scores and z and fits data with gaussian. Gives back z position of the fitted gaussian's middle
        which is the ideal/ in focus z plane

        Args:
            brenner (float), list that contains pairs of [focus_score, z]

        Results:
            ideal_z (float)
        '''
        brenner_temp = np.array(brenner)  # brenner's a global variable. there's no reason to re-call it
        f_score_temp = brenner_temp[:, 0]
        z = brenner_temp[:, 1]

        # print(brenner_temp)
        # print(f_score_temp)
        # print(z)

        # curve fitted with bounds relating to the inputs
        # let's force it such that z_ideal is within our range.
        parameters, covariance = curve_fit(gauss, z, f_score_temp,
                                           bounds=[(min(f_score_temp) / 4, min(z), 0, 0),
                                                   (max(f_score_temp) * 4, max(z), (max(z) - min(z)),
                                                    max(f_score_temp))])

        # a previous iteration of bounds that are more general
        # bounds = [(min(f_score_temp) / 4, min(z) / 2, f_start / 2, 0),
        #           (max(f_score_temp) * 4, max(z) * 2, f_start * 2, max(f_score_temp))])

        print('Z focus is located at: (microns)')
        print(parameters[1])

        # for a sanity check, let's plot this
        fit_f_score_gauss = cycif.gauss(z, *parameters)
        plt.plot(z, f_score_temp, 'o', label='data')
        plt.plot(z, fit_f_score_gauss, '-', label='fit')
        plt.title(['fstart,fend,fstep: ', str(f_start), ' ', str(f_end), ' ', str(f_step)])
        plt.legend()
        plt.grid()
        plt.show()
        return parameters[1]
#################################################
    def auto_focus(self):
        '''
        Runs entire auto focus algorithm in current XY position. Gives back predicted
        in focus z position via focus_score method which is the Brenner score.

        Args:
            None

        Returns:
            ideal_z (float)
        '''

        brenner = [] #need to test, may pose issue here outside of focus function

        with Acquisition(directory='C:/Users/CyCIF PC/Desktop/test_images',
                         name='z_stack_DAPI',
                         show_display=False,
                         image_process_fn=cycif.focus) as acq:
            events = multi_d_acquisition_events(channel_group='Color',
                                                channels=['DAPI'],
                                                z_start=cycif.z_range[0],
                                                z_end=cycif.z_range[0],
                                                z_step=cycif.z_range[0],
                                                order='zc')
            acq.acquire(events)

        z_ideal = cycif.autofocus_fit()

        return z_ideal




class arduino:

    def __init__(self, com_address, baudrate, timeout):
        '''
        Establishes connection to arduino

        Args:
            com_address(int),
            baudrate (int),
            timeout (int)

        Returns:
            Nothing
        '''

        connection = 1

        return connection

    def order_execute(self, orders):
        '''
        input list of serial codes for arduino and executes them one at a time, left to right. See serial command decoder for info. Also, for arduino object,
        it is something of the form: serial.Serial(port='COM5', baudrate=9600, timeout=5)

        Args:
            orders (int)

        Returns:
            Nothing
        '''
        # input list of serial codes for arduino and executes them one at a time, left to right. See serial command decoder for info.
        for order in orders:
            exit = 1
            command = str(order)
            command = command.encode()
            arduino.write(command)
            while exit != 0:  # this part reads the finished command from arduino to know that the entered command was fully executed
                exit = arduino.readline()
                exit = exit.decode()
                try:
                    exit = int(exit)
                except:
                    exit = 1
        return

    def prim_secondary_cycle(self, prim_syringe_num, secondary_syringe_num):
        '''
        First cycle of CyCIF consists of primary and secondaries.

        Args:
            prim_syringe_num (int)|
            secondary_syringe_num (int)

        Returns:
            Nothing

        '''
        prim_syringe_command = str(str(1) + str(prim_syringe_num))
        second_syringe_command = str(str(1) + str(secondary_syringe_num))
        cycif.syr_obj_switch(1)
        list_of_orders = [70, prim_syringe_command, 21, 34, 20, 61]
        cycif.order_execute(list_of_orders, arduino)
        time.sleep(2700)
        list_of_orders = [49, 71, 85, 70, 49, 60, second_syringe_command, 21, 34, 20, 61]
        cycif.order_execute(list_of_orders, arduino)
        time.sleep(2700)
        list_of_orders = [49, 71, 85, 70, 49, 60, 71]
        cycif.order_execute(list_of_orders, arduino)
        cycif.syr_obj_switch(0)

        return

    def post_acquistion_cycle(syringe_num):

        '''
                What post_acquistion_cycle is doing
                1. move stage underneath pipettor and raise objective out of pool|
                2. drain chamber|
                3. bleach cycle|
                4. waterfall wash 1 times|
                5. wait 5 mionutes|
                6. waterfall wash 1 times|
                7. stain (calibrate syringe, prime, dispense and unprime)|
                8. close lid|
                9. incubate stain for 45 minutes|
                10. open lid|
                11. fill chamber w/ PBS|
                12. place objective into pool and move over tissue|  

        Args:
            syringe_num (int), 0-7 of what syringe is to be used to stain in next cycle

        Returns:
             Nothing
        '''
        syringe_command = int(str(1) + str(syringe_num))
        cycif.syr_obj_switch(1)
        list_of_orders = [70, 53, 49, 49, 49, syringe_command, 21, 32, 20, 61]
        cycif.order_execute(list_of_orders)
        time.sleep(2700)
        list_of_orders = [49, 49, 49, 60, 71]
        cycif.order_execute(list_of_orders)
        cycif.syr_obj_switch(0)

        return


