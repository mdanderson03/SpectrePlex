import datetime

from autocyplex import *
from optparse import OptionParser
#microscope = cycif() # initialize cycif object
experiment_directory = r'E:\14-2-24 fluidics testing'
pump = fluidics(experiment_directory, 6, 3)

z_slices = 7
x_frame_size = 2960
offset_array = [0, -8, -7, -7]

