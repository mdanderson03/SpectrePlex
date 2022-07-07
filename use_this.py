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




A = [[1, 2], [1,3], [1,4]]
list = [x[1] for x in A]
result = max([x[1] for x in A])
index = list.index(result)
print(A[index][1])











