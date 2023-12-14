import os

import cv2

from autocyplex import *
from optparse import OptionParser
microscope = cycif() # initialize cycif object
pump = fluidics(6, 3)






experiment_directory = r'E:\13-12-23 kinetics'
offset_array = [0, -8, -7, -7]

z_slices = 9
x_frame_size = 2960
cycle = 0



#microscope.full_cycle(experiment_directory, cycle, offset_array, cycle, pump, z_slices)



#pump.liquid_action('Stain', stain_valve = 1, incub_val=45)
#pump.liquid_action('Bleach')
#microscope.image_cycle_acquire(1, experiment_directory, z_slices, 'Bleach', offset_array, x_frame_size=x_frame_size, establish_fm_array=0, auto_focus_run=0, auto_expose_run=0)
#microscope.establish_fm_array(experiment_directory, 4, z_slices, offset_array, initialize=0,x_frame_size=x_frame_size, autofocus=0, auto_expose=1)

#print(core.get_position())

#microscope.image_cycle_acquire(0, experiment_directory, z_slices, 'Bleach', offset_array, x_frame_size=x_frame_size, establish_fm_array=1, auto_focus_run=0, auto_expose_run=0)
#microscope.image_cycle_acquire(4, experiment_directory, z_slices, 'Stain', offset_array, x_frame_size=x_frame_size, establish_fm_array=0, auto_focus_run=0, auto_expose_run=1)


#pump.liquid_action('Bleach')
#pump.liquid_action('Wash')
#microscope.image_cycle_acquire(4, experiment_directory, z_slices, 'Bleach', offset_array, x_frame_size=x_frame_size, establish_fm_array=0, auto_focus_run=0, auto_expose_run=0)


#microscope.recursive_stardist_autofocus(experiment_directory, 1)

microscope.antibody_kinetics(experiment_directory, 1, 1, 120, 3, 1, pump)




#for cycle in range(5,8):
#    microscope.full_cycle(experiment_directory, cycle, offset_array, cycle, pump, z_slices, incub_val=45)

#microscope.post_acquisition_processor(experiment_directory, x_frame_size)
#microscope.stage_placement(experiment_directory, 1, x_frame_size)

#microscope.mcmicro_image_stack_generator(1, experiment_directory, x_frame_size)

