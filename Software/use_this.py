import math

import numpy as np

from autocyplex import *
from optparse import OptionParser
microscope = cycif() # initialize cycif object
experiment_directory = r'E:\14-11-24_integrity_testing'
pump = fluidics(experiment_directory, 6, 10, flow_control=1)
#core = Core()


z_slices = 3
x_frame_size = 2960

offset_array = [0, -7, -7, -6]
focus_position = -100 #make sure this is upper left hand corner focus z position




#microscope.image_cycle_acquire(9, experiment_directory, z_slices, 'Bleach', offset_array,x_frame_size=x_frame_size, auto_focus_run=0, auto_expose_run=0)
#microscope.hdr_compression_2(experiment_directory, cycle_number=1)
#microscope.stage_placement(experiment_directory, 1, x_pixels=x_frame_size, down_sample_factor=4)
#microscope.fm_stage_tilt_compensation(experiment_directory, tilt=3.75)
#microscope.tissue_cluster_filter(experiment_directory, x_frame_size, number_clusters_retained=8, area_threshold=0.25)

#microscope.tilt_determination()


#microscope.tissue_integrity_cycles(experiment_directory, 0, offset_array, 12, pump, z_slices, x_frame_size =x_frame_size, focus_position=focus_position, number_clusters=1)
#for valve in range(1,13):
#    pump.liquid_action('Stain', stain_valve=valve, incub_val=0)
#pump.liquid_action('Wash')
#sma cycle
#microscope.tissue_integrity_cycles(experiment_directory, 1, offset_array, 10, pump, z_slices, x_frame_size =x_frame_size, focus_position=focus_position, number_clusters=1)


#pump.liquid_action('low flow on')
#microscope.image_cycle_acquire(41, experiment_directory, z_slices, 'Stain', offset_array, x_frame_size=x_frame_size,establish_fm_array=0, auto_focus_run=0, auto_expose_run=3)
#pump.liquid_action('flow off')
#time.sleep(5)

# print(status_str)
#pump.liquid_action('Bleach')  # nuc is valve=7, pbs valve=8, bleach valve=1 (action, stain_valve, heater state (off = 0, on = 1))
#time.sleep(5)


#blank cycles
#for cycle in range(35,40):
#    stain_valve = 1 + math.floor((cycle-35)/3)
#    microscope.tissue_integrity_cycles(experiment_directory, cycle, offset_array, stain_valve, pump, z_slices, x_frame_size=x_frame_size,focus_position=focus_position, number_clusters=1)

#sma cycle
#microscope.tissue_integrity_cycles(experiment_directory, 40, offset_array, 10, pump, z_slices, x_frame_size =x_frame_size, focus_position=focus_position, number_clusters=1)




# for cycle in range(8, 11):
    #microscope.inter_cycle_processing(experiment_directory, cycle_number=cycle, x_frame_size=x_frame_size)
    # microscope.full_cycle(experiment_directory, cycle, offset_array, cycle, pump, z_slices, x_frame_size =x_frame_size, focus_position=focus_position, number_clusters=6)
#microscope.image_cycle_acquire(1, experiment_directory, z_slices, 'Stain', offset_array,x_frame_size=x_frame_size, auto_focus_run=0, auto_expose_run=3)
#microscope.image_cycle_acquire(0, experiment_directory, z_slic
# es, 'Bleach', offset_array,x_frame_size=x_frame_size, auto_focus_run=0, auto_expose_run=3)
#microscope.post_acquisition_processor(experiment_directory, x_frame_size, rolling_ball=0)

#microscope.generate_nuc_mask(experiment_directory, 1)
for cycle in range(41, 42):
     microscope.inter_cycle_processing(experiment_directory, cycle_number=cycle, x_frame_size=x_frame_size)



#microscope.tissue_region_identifier(experiment_directory, x_frame_size=x_frame_size, clusters_retained=6)

#microscope.recursive_stardist_autofocus(experiment_directory, 0,remake_nuc_binary=0)
#microscope.hdr_compression(experiment_directory, cycle_number=1, apply_2_subbed=0, apply_2_bleached=0)
#microscope.post_acquisition_processor(experiment_directory, x_pixels=x_frame_size)
#microscope.hdr_compression(experiment_directory, cycle_number=2, apply_2_subbed=1, apply_2_bleached=1, apply_2_focused=1, apply_2_flattened=1)
#microscope.tissue_binary_generate(experiment_directory, x_frame_size=x_frame_size, clusters_retained=1, area_threshold=0.1)
#pump.liquid_action('Bleach')
#microscope.tissue_cluster_filter(experiment_directory, x_frame_size=x_frame_size, number_clusters_retained=6)
#microscope.mcmicro_image_stack_generator_separate_clusters(cycle_number=2, experiment_directory=experiment_directory, x_frame_size=x_frame_size)
#microscope.inter_cycle_processing(experiment_directory, cycle_number=3, x_frame_size=x_frame_size)
#microscope.stage_placement(experiment_directory, cycle_number=1, x_pixels=x_frame_size)
#microscope.hdr_compression(experiment_directory, cycle_number=2)
#os.chdir(r'E:\3-7-24 marco\Tissue_Binary')
#im = io.imread('x1_y_2label_tissue.tif')
#print(np.unique(im[np.nonzero(im)]))
#cluster_counts = microscope.number_tiles_each_cluster(experiment_directory)
#print(cluster_counts)