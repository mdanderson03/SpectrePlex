from autocyplex import *

microscope = cycif() # initialize cycif object
experiment_directory = r'D:\24_3_26_casey_TMA_3'
pump = fluidics(6,5,7)

z_slices = 9
x_frame_size = 2960

offset_array = [0, -7, -6.7, -6, -7]

focus_position = -12

#pump.valve_prime()
#use first to set cluster surface
#microscope.wide_net_auto_focus(experiment_directory, x_frame_size=x_frame_size, offset_array=offset_array, z_slice_search_range=15, focus_position=focus_position, number_clusters_retained=5, manual_cluster_update=0)
#pump.valve_prime()
#pump.liquid_action('Stain', stain_valve=1, incub_val=2)
#Use second to take initial autofluorescence cycle
#microscope.full_cycle(experiment_directory, 0, offset_array, 0,
# pump, z_slices, x_frame_size =x_frame_size, focus_position=focus_position)

#microscope.full_cycle(experiment_directory, 8, offset_array, 9, pump, z_slices, incub_val = 45, x_frame_size=x_frame_size,focus_position=focus_position)

#microscope.prim_second_full_cycle(experiment_directory, cycle_number=1, offset_array=offset_array, fluidics_object=pump, z_slices=z_slices, prim_vial=1, second_vial=2)

for cycle in range(11, 12):
    microscope.full_cycle(experiment_directory, cycle, offset_array, cycle + 1, pump, z_slices, incub_val = 45, x_frame_size=x_frame_size,focus_position=focus_position)

microscope.inter_cycle_processing(experiment_directory, 1, x_frame_size=x_frame_size)
cycles = [0,2,3,4,5,6,7, 8,9,10]
#
for cycle in cycles:
    microscope.inter_cycle_processing(experiment_directory, cycle, x_frame_size=x_frame_size)
microscope.archive(experiment_directory)