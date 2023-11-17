import os

from autocyplex import *
from optparse import OptionParser
microscope = cycif() # initialize cycif object
pump = fluidics(6, 3)






experiment_directory = r'E:\16-11-23 square frame'
offset_array = [0, -8, -7, -11.5]
z_slices = 7
x_frame_size = 2960


microscope.image_cycle_acquire(1, experiment_directory, 'Stain', offset_array, x_frame_size=2960, establish_fm_array=0, auto_focus_run=0, auto_expose_run=1)
time.sleep(5)
pump.liquid_action('Bleach', stain_valve=stain_valve)  # nuc is valve=7, pbs valve=8, bleach valve=1 (action, stain_valve, heater state (off = 0, on = 1))
pump.liquid_action('Wash', stain_valve=stain_valve)  # nuc is valve=7, pbs valve=8, bleach valve=1 (action, stain_valve, heater state (off = 0, on = 1))
time.sleep(5)
self.image_cycle_acquire(cycle_number, experiment_directory, z_slices, 'Bleach', offset_array, x_frame_size=x_frame_size, establish_fm_array=0, auto_focus_run=0,auto_expose_run=0)
time.sleep(10)


#for cycle in range(2,4):
#    microscope.full_cycle(experiment_directory, cycle, offset_array, cycle, pump)







#microscope.post_acquisition_processor(experiment_directory, x_frame_size)

