from autocyplex import *

microscope = cycif() # initialize cycif object
experiment_directory = r'D:\19_8_26_CLEM'
import matplotlib.pyplot as plt
#pump = fluidics(6,5,4)
pump = 'pump'
z_slices = 3
x_frame_size = 2960

offset_array = [0, -7, -6.7, -6, -7]
#offset_array = [0, -7, -7.5, -7.3, -7]
focus_position=0

xyz_points = [(-6242,2522,-657),(297,2865,-671), (-5797,-3627,-552),(775,-3951,-560)]
microscope.initialize(experiment_directory)
fm = microscope.generate_fm_array_from_xyz(experiment_directory, xyz_points)
microscope.establish_exp_arrays(experiment_directory)
microscope.hdr_exp_generator(experiment_directory, threshold_level=10000, max_exp=700, min_exp=20)
microscope.establish_exp_arrays(experiment_directory)

print(fm[12])
#use first to set cluster surface
microscope.wide_net_auto_focus(experiment_directory, x_frame_size=x_frame_size, offset_array=offset_array, z_slice_search_range=5, focus_position=focus_position, number_clusters_retained=4, manual_cluster_update=0)
#microscope.image_cycle_acquire(0, experiment_directory,5, 'Bleach', offset_array, x_frame_size=x_frame_size,establish_fm_array=0, auto_focus_run=0, auto_expose_run=0, channels=['DAPI'],focus_position=focus_position, slice_gap = 1)


#uncomment to get autofluorescence
#microscope.full_cycle(experiment_directory, 0, offset_array, 0, pump, z_slices, incub_val = 45, x_frame_size=x_frame_size)



#result = subprocess.run(['python','partial_cycle.py', str(1), str(2), experiment_directory], capture_output=True, text=True)





# for cycle in range(11,12):
#      stain_vial_num = cycle + 1
#      subprocess.run(['python','full_cycle.py', str(stain_vial_num), str(cycle), experiment_directory], capture_output=True, text=True)


#microscope.inter_cycle_processing(experiment_directory, 1, x_frame_size=x_frame_size)
#cycles = [0,2,3,4,5,6,7,8,9,10,11]
#for cycle in cycles:
#   microscope.inter_cycle_processing(experiment_directory, cycle, x_frame_size=x_frame_size)
#microscope.archive(experiment_directory)
