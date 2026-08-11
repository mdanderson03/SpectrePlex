from autocyplex import *
import sys

microscope = cycif() # initialize cycif object
pump = fluidics(6,5,4)
z_slices = 15
x_frame_size = 2960
#offset_array = [0, -7, -6.7, -6, -7]
offset_array = [0, -9.6, -9, -8.6, -7]


stain_vial = int(sys.argv[1])
cycle = int(sys.argv[2])
experiment_directory = sys.argv[3]
print(sys.argv[0],sys.argv[1],sys.argv[2],sys.argv[3])

microscope.full_cycle(experiment_directory, cycle, offset_array, stain_vial, pump, z_slices, incub_val = 45, x_frame_size=x_frame_size)