from autocif import *
from pycromanager import Core, Magellan
microscope = cycif() # initialize cycif object
import numpy as np

core = Core()
magellan = Magellan() # initialize magellan object
microscope = cycif() # initialize cycif object
#robotics = arduino('COM5') # initialize arduino controlled robotics object at defined COM terminal number

#time.sleep(3) # wait for ardiuno to reboot and connect with computer

#mm_app_path = 'C:/Program Files/Micro-Manager-2.0gamma'
#config_file = 'C:/Users/CyCIF PC/Desktop/backup config files/auto_cycif.cfg'
#start_headless(mm_app_path, config_file, timeout=15000)




print(microscope.tissue_center('Focused Surface DAPI',magellan))
#time.sleep(3)

microscope.surf2focused_surf(core, magellan, 1, ['DAPI'])











