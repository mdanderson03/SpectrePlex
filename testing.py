from autocif import *
from pycromanager import Core, Magellan

core = Core()
magellan = Magellan() # magellan class object
microscope = cycif() # initialize cycif object
#arduino = arduino('COM3')
import numpy as np
import time
import serial




#auto_focus_exposure_times = [10,50,100,200]  #[DAPI, A488, A555, A647] in ms
#seed_plane = x   # z position when DAPI is in focus
#microscope.surf2focused_surf(core, magellan, 1, auto_focus_exposure_times, seed_plane)








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