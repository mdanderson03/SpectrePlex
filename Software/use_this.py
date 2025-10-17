#import gc
#import math
#import os
#import io
#import sys
#import multiprocessing
#import numpy as np

from autocyplex import *
#from optparse import OptionParser
microscope = cycif() # initialize cycif object
# experiment_directory = r'D:\09_25_casey_S16_7223_2'
experiment_directory = r'D:\09_25_casey_S16_7223_2'
#pump = fluidics(6,5,7)



z_slices = 3
x_frame_size = 2960

offset_array = [0, -7, -7, -6]
focus_position = 4





# def parallel_processing(experiment_directory, cycles, x_frame_size=2960):
#     number_cores = int(len(cycles))
#     inputs = []
#     for cycle in cycles:
#         inputs.append((experiment_directory, cycle, x_frame_size))
#
#     if __name__ == '__main__':
#         with multiprocessing.Pool(processes=number_cores) as pool:
#             pool.starmap(microscope.inter_cycle_processing, inputs)

#pump.liquid_action('Wash')

#pump.valve_prime()

#make sure this is upper left hand corner focus z position


#microscope.repeated_image_acquistion('E:\poisson_noise_images', 25, 'DAPI', 200)

#use first to set cluster surface
#microscope.wide_net_auto_focus(experiment_directory, x_frame_size=x_frame_size, offset_array=offset_array, z_slice_search_range=5, focus_position=focus_position, number_clusters_retained=3, manual_cluster_update=1)

#Use second to take initial autofluorescence cycle
#microscope.full_cycle(experiment_directory, 0, offset_array, 0, pump, z_slices, x_frame_size =x_frame_size, focus_position=focus_position)


#microscope.prim_second_full_cycle(experiment_directory, cycle_number=1, offset_array=offset_array, fluidics_object=pump, z_slices=z_slices, prim_vial=1, second_vial=2)

#for cycle in range(2, 9):
#    microscope.full_cycle(experiment_directory, cycle, offset_array, cycle + 1, pump, z_slices, incub_val = 45, x_frame_size=x_frame_size,focus_position=focus_position)

#pump.clean_valve()

#microscope.inter_cycle_processing(experiment_directory, 1, x_frame_size=x_frame_size)
# cycles = [1,2,3,4,5,6,7,8]
cycles = [0]
for cycle in cycles:
    microscope.inter_cycle_processing(experiment_directory, cycle, x_frame_size=x_frame_size)
#parallel_processing(experiment_directory, cycles, x_frame_size=x_frame_size)