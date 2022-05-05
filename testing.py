from autocif import *
from pycromanager import Core, Magellan
core = Core()
magellan = Magellan()
microscope = cycif() # initialize cycif object
arduino = arduino('COM3')
import numpy as np
import time
import serial



time.sleep(5)

arduino.prim_secondary_cycle(0,1,microscope)

#time_array = microscope.surf2focused_surf(core, magellan)
#print(time_array)
'''
xy = microscope.tile_xy_pos('New Surface 1', magellan)
z_centers = microscope.z_range(xy, 'New Surface 1', magellan)

print(z_centers)
'''








#microscope.tilt_angle_chip(core)