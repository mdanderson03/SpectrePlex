from autocif import *
from pycromanager import Core, Magellan
import numpy as np

core = Core()
magellan = Magellan() # initialize magellan object
microscope = cycif() # initialize cycif object
robotics = arduino('COM5') # initialize arduino controlled robotics object at defined COM terminal number

time.sleep(3) # wait for ardiuno to reboot and connect with computer

#mm_app_path = 'C:/Program Files/Micro-Manager-2.0gamma'
#config_file = 'C:/Users/CyCIF PC/Desktop/backup config files/auto_cycif.cfg'
#start_headless(mm_app_path, config_file, timeout=15000)





#time.sleep(3)

#z_range = [5800, 5900, 10] #[z start, z end, z step]
#microscope.surf2focused_surf(z_range, 'New Surface 1', 'Focus Surface') #takes manual surface and autofocuses and generates new focused surface











