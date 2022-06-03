from autocif import *
from pycromanager import Core, Magellan

core = Core()
magellan = Magellan() # magellan class object
microscope = cycif() # initialize cycif object
#arduino = arduino('COM3')
import numpy as np
import time
import serial

microscope.surf2focused_surf(core, magellan, 1)



#time.sleep(5)

#arduino.bleach_cycle()

#arduino.prim_secondary_cycle(0,1,microscope)
#arduino.post_acquistion_cycle(0, microscope)
#time_array = microscope.surf2focused_surf(core, magellan)
#print(time_array)
'''
xy = microscope.tile_xy_pos('New Surface 1', magellan)
z_centers = microscope.z_range(xy, 'New Surface 1', magellan)

print(z_centers)
'''








#microscope.tilt_angle_chip(core)