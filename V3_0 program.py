from autocif import *
from pycromanager import Core, Magellan, MagellanAcquisition
import numpy as np

core = Core() # initialize core object
magellan = Magellan() # initialize magellan object
microscope = cycif() # initialize cycif object
magellan_acq = MagellanAcquisition() # intialize mag acq object
robotics = arduino('COM5') # initialize arduino controlled robotics object at defined COM terminal number

time.sleep(3) # wait for ardiuno to reboot and connect with computer

total_cycles = 3 # type in total amount of cycles to be done
for x in range(1, total_cycles + 1):
    microscope.surf2focused_surf(core, magellan, cycle_number)
    magellan_acq()

