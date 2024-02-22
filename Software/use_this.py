import datetime

from autocyplex import *
from optparse import OptionParser
microscope = cycif() # initialize cycif object
experiment_directory = r'E:\tissue_testing'

z_slices = 7
x_frame_size = 2960
offset_array = [0, -8, -7, -7]


start = time.time()

microscope.image_cycle_acquire(1, experiment_directory, z_slices, 'Stain', offset_array, x_frame_size=x_frame_size, establish_fm_array=0, auto_focus_run=0,auto_expose_run=1)

finish = time.time()

print('total time', (finish - start))

#io.imshow(pixels)
#io.show()



