from autocyplex import *
from optparse import OptionParser
microscope = cycif() # initialize cycif object
pump = fluidics(6, 3)

experiment_directory = r'E:\6-1-24 multiplex'
z_slices = 7
x_frame_size = 2960
offset_array = [0, -8, -7, -7]

#microscope.full_cycle(experiment_directory, cycle, offset_array, cycle, pump, z_slices)



#pump.liquid_action('Stain', stain_valve = 1, incub_val=45)
#pump.liquid_action('Bleach')
#microscope.image_cycle_acquire(1, experiment_directory, z_slices, 'Bleach', offset_array, x_frame_size=x_frame_size, establish_fm_array=0, auto_focus_run=0, auto_expose_run=0)
#microscope.establish_fm_array(experiment_directory, 4, z_slices, offset_array, initialize=0,x_frame_size=x_frame_size, autofocus=0, auto_expose=1)

#print(core.get_position())

#microscope.image_cycle_acquire(0, experiment_directory, z_slices, 'Bleach', offset_array, x_frame_size=x_frame_size, establish_fm_array=1, auto_focus_run=0, auto_expose_run=0)
#microscope.image_cycle_acquire(4, experiment_directory, z_slices, 'Stain', offset_array, x_frame_size=x_frame_size, establish_fm_array=0, auto_focus_run=0, auto_expose_run=1)


#pump.liquid_action('Bleach')
pump.liquid_action('Wash')
#microscope.establish_fm_array(experiment_directory, 1, z_slices, offset_array, initialize=1)
#microscope.fm_channel_initial(experiment_directory, offset_array, z_slices)
# microscope.image_cycle_acquire(0, experiment_directory, z_slices, 'Bleach', offset_array, x_frame_size=x_frame_size, establish_fm_array=1, auto_focus_run=0, auto_expose_run=0)


#microscope.recursive_stardist_autofocus(experiment_directory, 1)

#microscope.antibody_kinetics(experiment_directory, 0.5, 90, 1, pump)
#microscope.antibody_kinetics(experiment_directory, 1, 2, 1, pump)

#print(microscope.kinetic_autofocus(experiment_directory, -87, 11))

# for cycle in range(1,8):
#
#    microscope.full_cycle(experiment_directory, cycle, offset_array, cycle, pump, z_slices, incub_val=45)

#microscope.post_acquisition_processor(experiment_directory, x_frame_size)
#microscope.stage_placement(experiment_directory, 1, x_frame_size)

#microscope.mcmicro_image_stack_generator(1, experiment_directory, x_frame_size)

#experiment_directory = r'E:\11-1-24 test kinetics'
#offset_array = [0, -8, -7, -7]

#z_slices = 9
#x_frame_size = 2960
#cycle = 0


