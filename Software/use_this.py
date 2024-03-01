import datetime

from autocyplex import *
from optparse import OptionParser
microscope = cycif() # initialize cycif object
experiment_directory = r'E:\1-3-24 celiac multiplex'
pump = fluidics(experiment_directory, 6, 3, flow_control=1)

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
print(fm_array[0])

exp_array = [75, 50, 50, 100]
#fm_array[12][::, ::] = 1
fm_array[11][::, ::] = 1
print('dapi frames', fm_array[11][0][0])
print('a488 frames', fm_array[12][0][0])
print('a555 frames', fm_array[13][0][0])
print('a647 frames', fm_array[14][0][0])
np.save('fm_array.npy', fm_array)
np.save('exp_array.npy', exp_array)
'''





#cycle = 0
#microscope.full_cycle(experiment_directory, cycle, offset_array, cycle, pump, z_slices)

pump.liquid_action('Stain', stain_valve=1, incub_val=45)

#microscope.establish_fm_array(experiment_directory, 0, z_slices, offset_array, initialize=1, x_frame_size=x_frame_size, autofocus=0, auto_expose=0)
#microscope.image_cycle_acquire(1, experiment_directory, z_slices, 'Stain', offset_array, x_frame_size=x_frame_size, establish_fm_array=0, auto_focus_run=0,auto_expose_run=1)
#microscope.post_acquisition_processor(experiment_directory, x_frame_size, rolling_ball=0)
