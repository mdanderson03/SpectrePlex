from autocyplex import *
from optparse import OptionParser
microscope = cycif() # initialize cycif object
experiment_directory = r'E:\27-6-24 marco'
pump = fluidics(experiment_directory, 6, 13, flow_control=1)
core = Core()

z_slices = 3
x_frame_size = 2960
offset_array = [0, -7, -7, -6]
focus_position = 187



#for cycle in range(2, 8):
#    microscope.full_cycle(experiment_directory, cycle, offset_array, cycle, pump, z_slices, focus_position=focus_position)
#microscope.image_cycle_acquire(1, experiment_directory, 3, 'Stain', offset_array,x_frame_size=x_frame_size)
#microscope.post_acquisition_processor(experiment_directory, x_frame_size, rolling_ball=0)
#microscope.brightness_uniformer(experiment_directory, cycle_number=1)
#microscope.tissue_region_identifier(experiment_directory)
#microscope.recursive_stardist_autofocus(experiment_directory, 0,remake_nuc_binary=0)
#microscope.post_acquisition_processor(experiment_directory, x_pixels=x_frame_size)
microscope.hdr_compression(experiment_directory, cycle_number=1, apply_2_subbed=0, apply_2_bleached=1)

