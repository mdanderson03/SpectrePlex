import datetime
import os

from autocyplex import *
from optparse import OptionParser
microscope = cycif() # initialize cycif object
experiment_directory = r'E:\15-5-24 healthy'
pump = fluidics(experiment_directory, 6, 13, flow_control=1)

z_slices = 7
x_frame_size = 2960
offset_array = [0, -8, -7, -7]
focus_position = 105



#pump.liquid_action('Stain', stain_valve=12, incub_val=1)

#microscope.establish_fm_array(experiment_directory, 1,z_slices, off_array= offset_array, initialize=1, x_frame_size=x_frame_size)


#pump.liquid_action('Stain', stain_valve=4)
#microscope.image_cycle_acquire(0, experiment_directory, z_slices, 'Bleach', offset_array, x_frame_size=x_frame_size, establish_fm_array=0, auto_focus_run=0, auto_expose_run=0)
#pump.liquid_action('Bleach')
#microscope.image_cycle_acquire(9, experiment_directory, z_slices, 'Stain', offset_array, x_frame_size=x_frame_size, establish_fm_array=0, auto_focus_run=0, auto_expose_run=2)
#pump.liquid_action('Bleach')
#microscope.image_cycle_acquire(9, experiment_directory, z_slices, 'Bleach', offset_array, x_frame_size=x_frame_size, establish_fm_array=1, auto_focus_run=0, auto_expose_run=0)

#numpy_path = experiment_directory + '/' + 'np_arrays'
#os.chdir(numpy_path)
#file_name = 'fm_array.npy'
#fm_array = np.load(file_name, allow_pickle=False)
#print(fm_array[2])

#microscope.full_cycle(experiment_directory, 0, offset_array, 0, pump, z_slices, focus_position=focus_position)
#pump.liquid_action('Wash')
#microscope.recursive_stardist_autofocus(experiment_directory, 2)
#microscope.image_cycle_acquire(2, experiment_directory, z_slices, 'Stain', offset_array, x_frame_size=x_frame_size, establish_fm_array=0, auto_focus_run=0, auto_expose_run=2)
#pump.liquid_action('Bleach')
#microscope.image_cycle_acquire(2, experiment_directory, z_slices, 'Bleach', offset_array, x_frame_size=x_frame_size, establish_fm_array=0, auto_focus_run=0, auto_expose_run=0)
#microscope.image_cycle_acquire(0, experiment_directory, z_slices, 'Bleach', offset_array, x_frame_size=x_frame_size, establish_fm_array=1, auto_focus_run=0,auto_expose_run=0, channels = ['DAPI'], focus_position = focus_position)
#microscope.image_cycle_acquire(0, experiment_directory, z_slices, 'Bleach', offset_array,x_frame_size=x_frame_size, fm_array_adjuster=0, establish_fm_array=0, auto_focus_run=1,auto_expose_run=0, focus_position=focus_position)


#microscope.image_cycle_acquire(1, experiment_directory, z_slices, 'Stain', offset_array, x_frame_size=x_frame_size,establish_fm_array=0, auto_focus_run=0,auto_expose_run=2)
#time.sleep(5)

# print(status_str)
#pump.liquid_action('Bleach')  # nuc is valve=7, pbs valve=8, bleach valve=
# 1 (action, stain_valve, heater state (off = 0, on = 1))
#time.sleep(5)
# print(status_str)
#microscope.image_cycle_acquire(1, experiment_directory, z_slices, 'Bleach', offset_array,x_frame_size=x_frame_size, establish_fm_array=0, auto_focus_run=0,auto_expose_run=0)
#time.sleep(3)



#for cycle in range(6, 9):
#    microscope.full_cycle(experiment_directory, cycle, offset_array, cycle, pump, z_slices, focus_position=focus_position)

#microscope.fm_grid_readjuster(experiment_directory, x_frame_size)
#microscope.post_acquisition_processor(experiment_directory, x_frame_size, rolling_ball=0)
#microscope.tissue_binary_generate(experiment_directory)

microscope.tissue_cluster_filter(experiment_directory,x_frame_size=x_frame_size, number_clusters_retained=1)

#pump.liquid_action('Wash')

#microscope.establish_fm_array(experiment_directory, 1, z_slices, offset_array, initialize=0, x_frame_size=x_frame_size, autofocus=1, auto_expose=1)
#microscope.image_cycle_acquire(1, experiment_directory, z_slices, 'Stain', offset_array, x_frame_size=x_frame_size, establish_fm_array=0, auto_focus_run=0,auto_expose_run=1)
#microscope.post_acquisition_processor(experiment_directory, x_frame_size, rolling_ball=0)
