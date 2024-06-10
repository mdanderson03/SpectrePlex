from autocyplex import *
from optparse import OptionParser
microscope = cycif() # initialize cycif object
experiment_directory = r'E:\6-6-24 marco'
pump = fluidics(experiment_directory, 6, 13, flow_control=1)

z_slices = 5
x_frame_size = 2960
offset_array = [0, -7, -7, -6]
focus_position = 155



#for cycle in range(6, 7):
#    microscope.full_cycle(experiment_directory, cycle, offset_array, cycle, pump, z_slices, focus_position=focus_position)

#microscope.post_acquisition_processor(experiment_directory, x_frame_size, rolling_ball=0)
microscope.brightness_uniformer(experiment_directory, cycle_number=1)
