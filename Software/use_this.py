import os

from autocyplex import *
from optparse import OptionParser
microscope = cycif() # initialize cycif object
pump = fluidics(6, 3)






experiment_directory = r'E:\16-11-23 square frame'
offset_array = [0, -8, -7, -11.5]
z_slices = 7
x_frame_size = 2960

#for cycle in range(0,9):
#    microscope.full_cycle(experiment_directory, cycle, offset_array, cycle, pump)


#microscope.post_acquisition_processor(experiment_directory, x_pixels)
#for cycle in range (1,9):
#    microscope.mcmicro_image_stack_generator(cycle, experiment_directory)

#microscope.establish_fm_array(experiment_directory, 0, z_slices, offset_array, initialize= 1, x_frame_size = x_frame_size, autofocus=1, auto_expose=0)

#microscope.multi_channel_z_stack_capture(experiment_directory, 0, 'Stain', x_pixels = x_frame_size,  slice_gap=2)

#microscope.post_acquisition_processor(experiment_directory, x_frame_size)

pump.liquid_action('PBS_flow_on')