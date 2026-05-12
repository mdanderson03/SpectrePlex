from autocyplex import *

microscope = cycif() # initialize cycif object
experiment_directory = r'D:\11_5_26_casey_TMA_1'
import matplotlib.pyplot as plt
pump = fluidics(6,5,7)
#pump = 'pump'
z_slices = 3
x_frame_size = 2960

offset_array = [0, -7, -6.7, -6, -7]

focus_position = 13

#pump.valve_prime()
#use first to set cluster surface
#microscope.wide_net_auto_focus(experiment_directory, x_frame_size=x_frame_size, offset_array=offset_array, z_slice_search_range=7, focus_position=focus_position, number_clusters_retained=5, manual_cluster_update=0)

#uncomment to get autofluorescence
#microscope.full_cycle(experiment_directory, 0, offset_array, 0, pump, z_slices, incub_val = 45, x_frame_size=x_frame_size,focus_position=focus_position)

#below are rest of loops and prim and secondary cycle

microscope.prim_second_full_cycle(experiment_directory, cycle_number=1, offset_array=offset_array, fluidics_object=pump, z_slices=z_slices, prim_vial=1, second_vial=2)
#
for cycle in range(2, 12):
     microscope.full_cycle(experiment_directory, cycle, offset_array, cycle + 1, pump, z_slices, incub_val = 45, x_frame_size=x_frame_size,focus_position=focus_position)
#
microscope.inter_cycle_processing(experiment_directory, 1, x_frame_size=x_frame_size)
cycles = [0,2,3,4,5,6,7, 8,9,10,11]

for cycle in cycles:
    microscope.inter_cycle_processing(experiment_directory, cycle, x_frame_size=x_frame_size)

microscope.archive(experiment_directory)