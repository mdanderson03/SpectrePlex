import math
import os
import io

import numpy as np

from autocyplex import *
from optparse import OptionParser
microscope = cycif() # initialize cycif object
experiment_directory = r'E:\21_1_24_MCPA_multiplex'
pump = fluidics(experiment_directory, 6, 10, flow_control=1)
#core = Core()


z_slices = 3
x_frame_size = 2960

offset_array = [0, -7, -7, -6]
focus_position = -120 #make sure this is upper left hand corner focus z position




#use first to set cluster surface
#microscope.wide_net_auto_focus(experiment_directory, x_frame_size=x_frame_size, offset_array=offset_array, z_slice_search_range=5, focus_position=focus_position, number_clusters_retained=1, manual_cluster_update=0)

#Use second to take initial autofluorescence cycle
#microscope.full_cycle(experiment_directory, 0, offset_array, 0, pump, z_slices, x_frame_size =x_frame_size, focus_position=focus_position)

#for cycle in range(1, 9):
#    microscope.full_cycle(experiment_directory, cycle, offset_array, cycle, pump, z_slices, x_frame_size=x_frame_size,focus_position=focus_position)




#for cycle in range(0, 1):
#    microscope.inter_cycle_processing(experiment_directory, cycle_number=cycle, x_frame_size=x_frame_size)
#microscope.image_cycle_acquire(9, experiment_directory, z_slices, 'Stain', offset_array=offset_array, x_frame_size=x_frame_size, fm_array_adjuster=0, establish_fm_array=0, auto_focus_run=0, auto_expose_run=3)

for cycle in range(1,2):
    microscope.delete_intermediate_folders(experiment_directory, cycle)