from autocyplex import *
from optparse import OptionParser
microscope = cycif() # initialize cycif object
experiment_directory = r'E:\3-6-24 marco'
pump = fluidics(experiment_directory, 6, 13, flow_control=1)

z_slices = 5
x_frame_size = 2960
offset_array = [0, -7, -7, -6]
focus_position = 138


#pump.liquid_action('Wash')


#for cycle in range(5, 6):
#    microscope.full_cycle(experiment_directory, cycle, offset_array, cycle, pump, z_slices, focus_position=focus_position)
#time_start = time.time()
#microscope.reacquire_run_autofocus(experiment_directory, 1, z_slices, offset_array, x_frame_size)
#print(time.time() - time_start)
microscope.post_acquisition_processor(experiment_directory, x_frame_size, rolling_ball=0)
#pump.liquid_action('Wash')
