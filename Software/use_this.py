import datetime

from autocyplex import *
from optparse import OptionParser
microscope = cycif() # initialize cycif object
experiment_directory = r'E:\fluid_test'
pump = fluidics(experiment_directory, 6, 3, flow_control=1)

z_slices = 7
x_frame_size = 2960
offset_array = [0, -8, -7, -7]


for cycle in range(0,8):
    microscope.full_cycle(experiment_directory, cycle, offset_array, cycle, pump, z_slices)



#microscope.post_acquisition_processor(experiment_directory, x_frame_size)



