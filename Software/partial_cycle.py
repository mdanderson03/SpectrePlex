from autocyplex import *
import sys

microscope = cycif() # initialize cycif object
pump = fluidics(6,5,4)
z_slices = 5
x_frame_size = 2960
offset_array = [0, -7, -6.7, -6, -7]


stain_vial = int(sys.argv[1])
cycle = int(sys.argv[2])
experiment_directory = sys.argv[3]
print(sys.argv[0],sys.argv[1],sys.argv[2],sys.argv[3])

microscope.prim_second_full_cycle(experiment_directory, cycle_number=1, offset_array=offset_array, fluidics_object=pump, z_slices=z_slices, prim_vial=1, second_vial=2)