import os

from autocyplex import *
from optparse import OptionParser
from pycromanager import Core, Studio, Magellan
microscope = cycif() # initialize cycif object
pump = fluidics(6, 3)






experiment_directory = r'E:\14-11-23_test'
offset_array = [0, -8, -7, -11.5]
z_slices = 9





#for cycle in range(1,9):
#    microscope.full_cycle(experiment_directory, cycle, offset_array, cycle, pump)

#microscope.full_cycle(experiment_directory, 0, offset_array, 0, pump)
microscope.post_acquisition_processor(experiment_directory)








