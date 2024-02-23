import datetime

from autocyplex import *
from optparse import OptionParser
microscope = cycif() # initialize cycif object
experiment_directory = r'D:\Images\AutoCyPlex\12-12-23_real_test'
#pump = fluidics(experiment_directory, 6, 3, flow_control=1)

z_slices = 7
x_frame_size = 2960
offset_array = [0, -8, -7, -7]


microscope.post_acquisition_processor(experiment_directory, x_frame_size)



