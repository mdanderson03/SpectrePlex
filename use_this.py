import os

from autocyplex import *
from pycromanager import Core, Studio, Magellan
microscope = cycif() # initialize cycif object
#pump = fluidics(6, 3)



experiment_directory = r'D:\Images\AutoCyPlex\parallel_test_with_thunder'
exp_array = [20,35, 35, 35]
offset_array = [0, -8, -8, -8]
cycle_number = 0
stain_valve = 4


#microscope.full_cycle(experiment_directory, cycle_number, exp_time_array, offset_array, stain_valve)


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






