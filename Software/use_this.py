from autocyplex import *

microscope = cycif() # initialize cycif object
experiment_directory = r'D:\20-7-26_Kate_SP24_9928_A1'
import matplotlib.pyplot as plt
#pump = fluidics(6,5,4)
pump = 'pump'
z_slices = 5
x_frame_size = 2960

offset_array = [0, -7, -6.7, -6, -7]

focus_position = 155

#pump.valve_prime()
#use first to set cluster surface
#microscope.wide_net_auto_focus(experiment_directory, x_frame_size=x_frame_size, offset_array=offset_array, z_slice_search_range=9, focus_position=focus_position, number_clusters_retained=5, manual_cluster_update=1)

#uncomment to get autofluorescence
#microscope.full_cycle(experiment_directory, 0, offset_array, 0, pump, z_slices, incub_val = 45, x_frame_size=x_frame_size,focus_position=focus_position)



#result = subprocess.run(['python','partial_cycle.py', str(1), str(2), experiment_directory], capture_output=True, text=True)





for cycle in range(2,12):
     stain_vial_num = cycle + 1
     subprocess.run(['python','full_cycle.py', str(stain_vial_num), str(cycle), experiment_directory], capture_output=True, text=True)


#microscope.inter_cycle_processing(experiment_directory, 1, x_frame_size=x_frame_size)
#cycles = [0,2,3,4,5,6,7,8,9,10,11]
#for cycle in cycles:
#    microscope.inter_cycle_processing(experiment_directory, cycle, x_frame_size=x_frame_size)
#microscope.archive(experiment_directory)
