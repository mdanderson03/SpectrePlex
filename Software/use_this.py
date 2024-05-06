import datetime
import os

from autocyplex import *
from optparse import OptionParser
microscope = cycif() # initialize cycif object
experiment_directory = r'E:\3-5-24 healthy'
pump = fluidics(experiment_directory, 6, 13, flow_control=1)

z_slices = 7
x_frame_size = 2960
offset_array = [0, -8, -7, -7]

'''
numpy_path = experiment_directory + '/' + 'np_arrays'
os.chdir(numpy_path)
exp_filename = 'exp_array.npy'
file_name = 'fm_array.npy'
fm_array = np.load(file_name, allow_pickle=False)
exp_array = np.load(exp_filename, allow_pickle=False)
fm_array[2][::,::] = 1

exp_array[1] = 10
exp_array[2] = 200
exp_array[3] = 200

fm_array[12][0][0] = 3

fm_array[13][0][0] = 1

fm_array[14][0][0] = 1

#print('dapi frames', fm_array[11][0][0])
#print('a488 frames', fm_array[12][0][0])
#print('a555 frames', fm_array[13][0][0])
#print('a647 frames', fm_array[14][0][0])
#np.save('fm_array.npy', fm_array)
np.save('exp_array.npy', exp_array)
'''




#pump.liquid_action('Wash')
#microscope.image_cycle_acquire(1, experiment_directory, z_slices, 'Stain', offset_array, x_frame_size=x_frame_size, establish_fm_array=0, auto_focus_run=0, auto_expose_run=0)
#pump.liquid_action('Bleach')
#microscope.image_cycle_acquire(1, experiment_directory, z_slices, 'Stain', offset_array, x_frame_size=x_frame_size, establish_fm_array=0, auto_focus_run=0, auto_expose_run=1)

#for cycle in range(1, 9):
#    microscope.full_cycle(experiment_directory, cycle, offset_array, cycle, pump, z_slices)
# end = time.time()
# print(end-start)

microscope.post_acquisition_processor(experiment_directory, x_frame_size, rolling_ball=0)




#pump.liquid_action('Wash')

#microscope.establish_fm_array(experiment_directory, 1, z_slices, offset_array, initialize=0, x_frame_size=x_frame_size, autofocus=1, auto_expose=1)
#microscope.image_cycle_acquire(1, experiment_directory, z_slices, 'Stain', offset_array, x_frame_size=x_frame_size, establish_fm_array=0, auto_focus_run=0,auto_expose_run=1)
#microscope.post_acquisition_processor(experiment_directory, x_frame_size, rolling_ball=0)
