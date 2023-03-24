from autocyplex import *
from pycromanager import Core, Magellan, MagellanAcquisition
import numpy as np


microscope = cycif() # initialize cycif object
arduino = arduino()

#microscope.cycle_acquire(1, 'E:/folder_structure', 'Stain', ['DAPI']) #(cycle_number, experiment_directory, 'Stain' or 'Bleach')

arduino.auto_load()


#arduino.bleach(1700)
#arduino.stain(3)
#arduino.nuc_touch_up(6,360)

