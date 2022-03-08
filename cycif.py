from pycromanager import Bridge, Acquisition, multi_d_acquisition_events, Dataset
import numpy as np
import matplotlib.pyplot as plt
import math
from tifffile import imread, imwrite
import os
import time

class cycif:

    bridge = Bridge()
    core = bridge.get_core()


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

    def syr_obj_switch(state):
        core = cycif.core
        diff_vec_x = 75000
        diff_vec_y = 1000
        y = core.get_y_position()
        x= core.get_x_position()
        z = core.get_position()
        if state == 0:
            new_y = y + diff_vec_y
            new_x = x + diff_vec_x
            core.set_xy_position(new_x, new_y)
            time.sleep(15)
            core.set_position(z - 8000)

        if state == 1:
            core.set_position(z + 8000)
            time.sleep(6)
            new_y = y - diff_vec_y
            new_x = x - diff_vec_x
            core.set_xy_position(new_x, new_y)
