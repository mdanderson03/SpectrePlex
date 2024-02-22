import datetime

from autocyplex import *
from optparse import OptionParser
microscope = cycif() # initialize cycif object
experiment_directory = r'E:\20-2-2024 healthy multiplex'
pump = fluidics(experiment_directory, 6, 3, flow_control=1)

z_slices = 7
x_frame_size = 2960
offset_array = [0, -8, -7, -7]


microscope.image_cycle_acquire(1, experiment_directory, z_slices, 'Stain', offset_array,x_frame_size=x_frame_size, establish_fm_array=0, auto_focus_run=1,auto_expose_run=1)
time.sleep(5)
pump.liquid_action('Bleach', stain_valve=1)  # nuc is valve=7, pbs valve=8, bleach valve=1 (action, stain_valve, heater state (off = 0, on = 1))
time.sleep(5)
microscope.image_cycle_acquire(1, experiment_directory, z_slices, 'Bleach', offset_array, x_frame_size=x_frame_size, establish_fm_array=0, auto_focus_run=0,auto_expose_run=0)
time.sleep(3)

for cycle in range(2,8):
    microscope.full_cycle(experiment_directory, cycle, offset_array, cycle,  pump, z_slices)
