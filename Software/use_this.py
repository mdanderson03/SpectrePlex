from autocyplex import *
from optparse import OptionParser
microscope = cycif() # initialize cycif object
experiment_directory = r'E:\24-5-24 celiac'
pump = fluidics(experiment_directory, 6, 13, flow_control=1)

z_slices = 7
x_frame_size = 2960
offset_array = [0, -8, -7, -7]
focus_position = 152


#for cycle in range(2, 9):
#    microscope.full_cycle(experiment_directory, cycle, offset_array, cycle, pump, z_slices, focus_position=focus_position)

microscope.post_acquisition_processor(experiment_directory, x_frame_size, rolling_ball=0)

