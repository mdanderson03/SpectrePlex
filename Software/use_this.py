from autocyplex import *
#from optparse import OptionParser
microscope = cycif() # initialize cycif object
#pump = fluidics(6, 3)


experiment_directory = r'D:\Images\AutoCyPlex\6-1-24 multiplex'
z_slices = 7
x_frame_size = 2960
offset_array = [0, -8, -7, -7]

microscope.post_acquisition_processor(experiment_directory, x_frame_size)






