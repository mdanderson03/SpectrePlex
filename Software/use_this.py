import os

from autocyplex import *
from optparse import OptionParser
from pycromanager import Core, Studio, Magellan
microscope = cycif() # initialize cycif object
pump = fluidics(6, 3)






#numpy_path = experiment_directory + '/' + 'np_arrays'
#os.chdir(numpy_path)
#np.save('exp_array.npy', exp_array)
#pump.liquid_action('Bleach')
#pump.liquid_action('PBS flow off')
#pump.liquid_action('Wash')
#pump.liquid_action('Stain', 1)
#pump.liquid_action('Stain', stain_valve = 7)

experiment_directory = r'E:\24-10-23 test'
#exp_array = np.array([100,35, 35, 35])
offset_array = [0, -7, -7, -8.5]
cycle_number = 0
stain_valve = 1
z_slices = 9


for cycle in range(0,9):
    microscope.full_cycle(experiment_directory, cycle, offset_array, cycle, pump)





#pump.liquid_action('Wash')
#microscope.full_cycle(experiment_directory, 0, offset_array, 0)





#microscope.establish_fm_array(experiment_directory, 1, z_slices, offset_array, initialize=1, autofocus=0, auto_expose=0)
#pump.liquid_action('Bleach')
#pump.liquid_action('Wash')
#microscope.image_cycle_acquire(cycle_number, experiment_directory, z_slices, 'Bleach', offset_array, establish_fm_array=1, auto_focus_run=0, auto_expose_run = 1)



'''
test_directory = r'D:\Images\AutoCyPlex\parallel_test_with_thunder\A647\Stain\cy_1\Tiles'
os.chdir(test_directory)


x = 0
y = 0
channel = 'A647'
x_range = np.array([0,1,2,3,4,5])
scores = np.array([0,0,0,0,0,0])

for z in range(0,6):

    file_name = 'z_' + str(z) + '_x' + str(x) + '_y_' + str(y) + '_c_' + str(channel) + '.tif'
    image = tf.imread(file_name)
    image = image[2100:2960, 4500:5056]
    score = microscope.focus_score(image, 4)
    scores[z] = score

scores = scores/min(scores)


plt.scatter(x_range, scores)
plt.show()
'''



microscope.post_acquisition_processor(experiment_directory)

#microscope.metadata_generator(experiment_directory)






