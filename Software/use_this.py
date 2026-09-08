from autocyplex import *
from pathlib import Path as FilePath

microscope = cycif() # initialize cycif object
experiment_directory = r'D:\2_9_26_kate_sp24_9928'
import matplotlib.pyplot as plt
#pump = fluidics(6,5,4)
pump = 'pump'
z_slices = 3
x_frame_size = 2960

offset_array = [0, -7, -6.7, -6, -7]
#offset_array = [0, -7, -7.5, -7.3, -7]
focus_position=-71

#pump.valve_prime()

#xyz_points = [(-6144,2548,-695),(297,2865,-708), (-5707,-3597,-579),(813,-3924,-588)]
#microscope.initialize(experiment_directory)
#fm = microscope.generate_fm_array_from_xyz(experiment_directory, xyz_points)
#microscope.establish_exp_arrays(experiment_directory-)
#microscope.hdr_exp_generator(experiment_directory, threshold_level=10000, max_exp=700, min_exp=20)
#microscope.establish_exp_arrays(experiment_directory)


#use first to set cluster surface
#microscope.wide_net_auto_focus(experiment_directory, x_frame_size=x_frame_size, offset_array=offset_array, z_slice_search_range=5, focus_position=focus_position, number_clusters_retained=5, manual_cluster_update=0)



#uncomment to get autofluorescence
#microscope.full_cycle(experiment_directory, 0, offset_array, 0, pump, z_slices, incub_val = 45, x_frame_size=x_frame_size)



#script_dir = FilePath(__file__).resolve().parent
#partial_cycle_script = script_dir / "partial_cycle.py"
#full_cycle_script = script_dir / "full_cycle.py"


# result = subprocess.run(['python','partial_cycle.py', str(1), str(2), experiment_directory], capture_output=True, text=True)
#
#
# for cycle in range(2,11):
#       stain_vial_num = cycle + 1
#       subprocess.run(['python','full_cycle.py', str(stain_vial_num), str(cycle), experiment_directory], capture_output=True, text=True)


microscope.inter_cycle_processing(experiment_directory, 1, x_frame_size=x_frame_size)
cycles = [0,2,3,4,5,6,7,8,9,10]
for cycle in cycles:
   microscope.inter_cycle_processing(experiment_directory, cycle, x_frame_size=x_frame_size)
microscope.archive(experiment_directory)
