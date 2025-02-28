import math
import os
import io

import numpy as np

from autocyplex import *
from optparse import OptionParser
microscope = cycif() # initialize cycif object
experiment_directory = r'E:\26-2-25_sp24_07971_rest of cycles'
pump = fluidics(experiment_directory, 6, 10, flow_control=1)
#core = Core()


z_slices = 3
x_frame_size = 2960

offset_array = [0, -7, -7, -6]
focus_position = -126
#make sure this is upper left hand corner focus z position




#use first to set cluster surface
#microscope.wide_net_auto_focus(experiment_directory, x_frame_size=x_frame_size, offset_array=offset_array, z_slice_search_range=5, focus_position=focus_position, number_clusters_retained=3, manual_cluster_update=0)

#Use second to take initial autofluorescence cycle
#microscope.full_cycle(experiment_directory, 0, offset_array, 0, pump, z_slices, x_frame_size =x_frame_size, focus_position=focus_position)

#for c in range(7,8):

#    pump.liquid_action('Stain', incub_val=45, stain_valve=c)
#    pump.liquid_action('Bleach')

pump.liquid_action('low flow on')
microscope.image_cycle_acquire(2, experiment_directory, z_slices, 'Stain', offset_array, x_frame_size=x_frame_size,establish_fm_array=0, auto_focus_run=0, auto_expose_run=3)
pump.liquid_action('flow off')
time.sleep(5)

# print(status_str)
pump.liquid_action('Bleach',stain_valve=11)  # nuc is valve=7, pbs valve=8, bleach valve=1 (action, stain_valve, heater state (off = 0, on = 1))
time.sleep(5)
# print(status_str)
microscope.image_cycle_acquire(2, experiment_directory, z_slices, 'Bleach', offset_array,x_frame_size=x_frame_size, establish_fm_array=0, auto_focus_run=0, auto_expose_run=0)
time.sleep(3)



for cycle in range(3, 8):
    microscope.full_cycle(experiment_directory, cycle, offset_array, cycle, pump, z_slices, x_frame_size=x_frame_size,focus_position=focus_position)

for cycle in range(8, 10):
    microscope.full_cycle(experiment_directory, cycle, offset_array, cycle + 1, pump, z_slices,x_frame_size=x_frame_size, focus_position=focus_position)

#fm_array = np.ones((15,16,12))
#os.chdir(r'E:\7-2-25_Dou11_aku\np_arrays')
#np.save('fm_array.npy', fm_array)
#microscope.tissue_region_identifier(experiment_directory, x_frame_size=x_frame_size, clusters_retained=5)

#for cycle in range(2, 10):
#    microscope.inter_cycle_processing(experiment_directory, cycle_number=cycle, x_frame_size=x_frame_size)
#microscope.image_cycle_acquire(9, experiment_directory, z_slices, 'Stain', offset_array=offset_array, x_frame_size=x_frame_size, fm_array_adjuster=0, establish_fm_array=0, auto_focus_run=0, auto_expose_run=3)

#microscope.archive(experiment_directory)
#for cycle in range(2,9):
    #microscope.delete_intermediate_folders(experiment_directory, cycle)
#    microscope.zlib_compress_raw(experiment_directory, cycle)