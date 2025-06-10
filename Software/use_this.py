import gc
import math
import os
import io
import sys
import multiprocessing

from KasaSmartPowerStrip import SmartPowerStrip
import ob1

import numpy as np

from autocyplex import *
from optparse import OptionParser
microscope = cycif() # initialize cycif object
experiment_directory = r'E:\03_6_25_test_double_hamilton_wirth_flow_meter'
pump = fluidics(experiment_directory, 12, 11, 7, flow_control=1)
#core = Core()


z_slices = 3
x_frame_size = 2960

offset_array = [0, -7, -7, -6]
focus_position = 594


def parallel_processing(experiment_directory, cycles, x_frame_size=2960):
    number_cores = int(len(cycles))
    inputs = []
    for cycle in cycles:
        inputs.append((experiment_directory, cycle, x_frame_size))

    if __name__ == '__main__':
        with multiprocessing.Pool(processes=number_cores) as pool:
            pool.starmap(microscope.inter_cycle_processing, inputs)

#pump.liquid_action('Wash')
#pump.valve_prime()

#make sure this is upper left hand corner focus z position


#microscope.repeated_image_acquistion('E:\poisson_noise_images', 25, 'DAPI', 200)

#use first to set cluster surface
#microscope.wide_net_auto_focus(experiment_directory, x_frame_size=x_frame_size, offset_array=offset_array, z_slice_search_range=5, focus_position=focus_position, number_clusters_retained=1, manual_cluster_update=0)

#Use second to take initial autofluorescence cycle
#microscope.full_cycle(experiment_directory, 0, offset_array, 0, pump, z_slices, x_frame_size =x_frame_size, focus_position=focus_position)

for cycle in range(1, 11):
    microscope.full_cycle(experiment_directory, cycle, offset_array, cycle, pump, z_slices, x_frame_size=x_frame_size,focus_position=focus_position)

#for cycle in range(3, 7):
#    microscope.full_cycle(experiment_directory, cycle, offset_array, cycle + 2, pump, z_slices, x_frame_size=x_frame_size,focus_position=focus_position)

#cycles = [4,5]
#parallel_processing(experiment_directory, cycles, x_frame_size=x_frame_size)