from autocyplex import *
from pycromanager import Core, Magellan, MagellanAcquisition
import numpy as np


microscope = cycif() # initialize cycif object
arduino = arduino()

#arduino.auto_load() #uncomment to flush through all vials or load in fluid from new ones


#arduino.nuc_touch_up(6,300) #put hoescht into chamber for time (valve, time)

arduino.stain(3)
#arduino.stain(3)
#microscope.acquire_all_tiled_surfaces(1, directory_name='E://12-20-22 test slide s_16_00078620/')
#arduino.auto_load()
#arduino.nuc_touch_up(6,360)
directory = 'E://1_5_2022_control_duo/'    # 'E://12-20-22 test slide s_16_00078620/' example how directory address should be. Note the back slashes
'''
#arduino.primary_secondary_cycle(3,4)
microscope.acquire_all_tiled_surfaces(1, directory_name=directory)
arduino.bleach(300)
microscope.acquire_all_tiled_surfaces(2, directory_name=directory)
arduino.stain(5)
arduino.nuc_touch_up(6,300)
microscope.acquire_all_tiled_surfaces(3, directory_name='directory')
arduino.bleach(300)
microscope.acquire_all_tiled_surfaces(4, directory_name=directory)
arduino.stain(7)
microscope.acquire_all_tiled_surfaces(5, directory_name=directory)
'''