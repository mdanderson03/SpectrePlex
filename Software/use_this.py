from autocyplex import *

microscope = cycif() # initialize cycif object
experiment_directory = r'D:\24-6-26_casey_TMA_5'
import matplotlib.pyplot as plt
pump = fluidics(6,5,4)
#pump = 'pump'
z_slices = 3
x_frame_size = 2960

offset_array = [0, -7, -6.7, -6, -7]

focus_position = -67

pump.valve_prime()
#use first to set cluster surface
#microscope.wide_net_auto_focus(experiment_directory, x_frame_size=x_frame_size, offset_array=offset_array, z_slice_search_range=7, focus_position=focus_position, number_clusters_retained=5, manual_cluster_update=0)

#uncomment to get autofluorescence
#microscope.full_cycle(experiment_directory, 0, offset_array, 0, pump, z_slices, incub_val = 45, x_frame_size=x_frame_size,focus_position=focus_position)

#for x in range(4,12):

#    microscope.image_cycle_acquire(x, experiment_directory,z_slices, 'Stain', offset_array,x_frame_size=x_frame_size, establish_fm_array=0, auto_focus_run=0,auto_expose_run=3)
#    microscope.image_cycle_acquire(x, experiment_directory, z_slices, 'Bleach', offset_array,x_frame_size=x_frame_size, establish_fm_array=0, auto_focus_run=0, auto_expose_run=0)
#    x += 1
#    print('completed round')
#print('count_started')
#time.sleep(45*60)
#print('count ended')
#subprocess.run(['python','partial_cycle.py', str(1), str(2), experiment_directory], capture_output=True, text=True)
#microscope.image_cycle_acquire(2, experiment_directory,z_slices, 'Stain', offset_array,x_frame_size=x_frame_size, establish_fm_array=0, auto_focus_run=0,auto_expose_run=3)
#pump.liquid_action('Bleach')
#microscope.image_cycle_acquire(9, experiment_directory,z_slices, 'Bleach', offset_array,x_frame_size=x_frame_size, establish_fm_array=0, auto_focus_run=0,auto_expose_run=0)


#below are rest of loops and prim and secondary cycle

#microscope.prim_second_full_cycle(experiment_directory, cycle_number=1, offset_array=offset_array, fluidics_object=pump, z_slices=z_slices, prim_vial=1, second_vial=2)



# for cycle in range(10,12):
#     stain_vial_num = cycle + 1
#     subprocess.run(['python','full_cycle.py', str(stain_vial_num), str(cycle), experiment_directory], capture_output=True, text=True)
#    print(result)

#for cycle in range(4, 9):
#     microscope.full_cycle(experiment_directory, cycle, offset_array, cycle - 3, pump, z_slices, incub_val = 45, x_frame_size=x_frame_size,focus_position=focus_position)
#
#microscope.inter_cycle_processing(experiment_directory, 1, x_frame_size=x_frame_size)
#cycles = [0,2,3,4,5,6,7,8,9,10,11]

#for cycle in cycles:
#    microscope.inter_cycle_processing(experiment_directory, cycle, x_frame_size=x_frame_size)

# microscope.archive(experiment_directory)