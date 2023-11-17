import os

from autocyplex import *
from optparse import OptionParser
microscope = cycif() # initialize cycif object
pump = fluidics(6, 3)






experiment_directory = r'E:\16-11-23 square frame'
offset_array = [0, -8, -7, -11.5]
z_slices = 7
x_frame_size = 2960


microscope.establish_fm_array(experiment_directory, 2, z_slices, off_array, initialize=1,x_frame_size=x_frame_size, autofocus=1, auto_expose=0)

for cycle in range(2,4):
    microscope.full_cycle(experiment_directory, cycle, offset_array, cycle, pump)



microscope.post_acquisition_processor(experiment_directory, x_frame_size)

