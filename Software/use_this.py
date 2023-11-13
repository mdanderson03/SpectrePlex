import os

from autocyplex import *
from optparse import OptionParser
microscope = cycif() # initialize cycif object
#pump = fluidics(6, 3)






experiment_directory = r'D:\Images\AutoCyPlex\KRISH-UPLOAD\17-10-23 testing'
offset_array = [0, -8, -7, -11.5]
z_slices = 7

#for cycle in range(0,9):
#    microscope.full_cycle(experiment_directory, cycle, offset_array, cycle, pump)


#microscope.post_acquisition_processor(experiment_directory)
for cycle in range (1,9):
    microscope.mcmicro_image_stack_generator(cycle, experiment_directory)







