import numpy as np

from autocyplex import *
from optparse import OptionParser
microscope = cycif() # initialize cycif object
experiment_directory = r'E:\29-7-24 gutage'
pump = fluidics(experiment_directory, 6, 13, flow_control=1)
core = Core()


z_slices = 3
x_frame_size = 2960
offset_array = [0, -7, -7, -6]
focus_position = -104

#pump.liquid_action('Bleach')
#microscope.image_cycle_acquire(1, experiment_directory, z_slices, 'Bleach', offset_array,x_frame_size=x_frame_size, auto_focus_run=0, auto_expose_run=3)
#for cycle in range(2, 11):
#    microscope.full_cycle(experiment_directory, cycle, offset_array, cycle, pump, z_slices, focus_position=focus_position)
#microscope.image_cycle_acquire(1, experiment_directory, z_slices, 'Stain', offset_array,x_frame_size=x_frame_size, auto_focus_run=0, auto_expose_run=3)
#microscope.image_cycle_acquire(0, experiment_directory, z_slices, 'Bleach', offset_array,x_frame_size=x_frame_size, auto_focus_run=0, auto_expose_run=3)
#microscope.post_acquisition_processor(experiment_directory, x_frame_size, rolling_ball=0)
#microscope.brightness_uniformer(experiment_directory, cycle_number=1)

#microscope.tissue_region_identifier(experiment_directory, x_frame_size=x_frame_size, clusters_retained=3)

#microscope.recursive_stardist_autofocus(experiment_directory, 0,remake_nuc_binary=1)
#microscope.hdr_compression(experiment_directory, cycle_number=1, apply_2_subbed=0, apply_2_bleached=0)
#microscope.post_acquisition_processor(experiment_directory, x_pixels=x_frame_size)
#microscope.hdr_compression(experiment_directory, cycle_number=1, apply_2_subbed=0, apply_2_bleached=1)
#microscope.tissue_binary_generate(experiment_directory, x_frame_size=x_frame_size, clusters_retained=1, area_threshold=0.1)
#pump.liquid_action('Bleach')
microscope.mcmicro_image_stack_generator_separate_clusters(cycle_number=1, experiment_directory=experiment_directory, x_frame_size=x_frame_size)
#microscope.inter_cycle_processing(experiment_directory, cycle_number=3, x_frame_size=x_frame_size)
#microscope.stage_placement(experiment_directory, cycle_number=1, x_pixels=x_frame_size)
#microscope.hdr_compression(experiment_directory, cycle_number=2)
#os.chdir(r'E:\3-7-24 marco\Tissue_Binary')
#im = io.imread('x1_y_2label_tissue.tif')
#print(np.unique(im[np.nonzero(im)]))
#cluster_counts = microscope.number_tiles_each_cluster(experiment_directory)
#print(cluster_counts)