from autocyplex import *
from pycromanager import Core, Magellan, MagellanAcquisition
import numpy as np


microscope = cycif() # initialize cycif object
arduino = arduino()

#microscope.cycle_acquire(1, r'E:\24-3-23_control', 'Bleach') #(cycle_number, experiment_directory, 'Stain' or 'Bleach')

#arduino.auto_load()


#arduino.bleach(360)
#arduino.stain(3)
#arduino.nuc_touch_up(6,300)


#microscope.cycle_acquire(2, r'E:\24-3-23_control', 'Stain') #(cycle_number, experiment_directory, 'Stain' or 'Bleach')
#arduino.bleach(360)
#microscope.cycle_acquire(2, r'E:\24-3-23_control', 'Bleach') #(cycle_number, experiment_directory, 'Stain' or 'Bleach')
arduino.stain(4)
microscope.cycle_acquire(3, r'E:\24-3-23_control', 'Stain') #(cycle_number, experiment_directory, 'Stain' or 'Bleach')
arduino.bleach(360)
microscope.cycle_acquire(3, r'E:\24-3-23_control', 'Bleach') #(cycle_number, experiment_directory, 'Stain' or 'Bleach')

