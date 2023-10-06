from autocyplex import *
from pycromanager import Core, Magellan, MagellanAcquisition
import numpy as np


microscope = cycif() # initialize cycif object
pump = fluidics(6, 3) #initialize fluidics object


folder = r'E:\elveflow_test_2'
z_slices = 7
exp_time = np.array([200,500,50,1500]) # make, exp_time = 0 if want to use auto expose
offset_array = [-7, -7, -7]  #micron deviation from nuc focus position [a488, a555, a647]



#initial autofluorescence image (only run once)
microscope.cycle_acquire(0, folder, z_slices, 'Stain', offset_array, exp_time) #(cycle_number, experiment_directory, folder_path, 'Stain' or 'Bleach', number z slices total for each tile)

# cycle 1

cycle = 1

pump.liquid_action('Stain', 2)
microscope.cycle_acquire(cycle, folder, z_slices, 'Stain', offset_array, exp_time) #(cycle_number, experiment_directory, folder_path, 'Stain' or 'Bleach', number z slices total for each tile)
pump.liquid_action('Bleach')
microscope.cycle_acquire(cycle, folder, z_slices, 'Bleach', offset_array, exp_time) #(cycle_number, experiment_directory, folder_path, 'Stain' or 'Bleach', number z slices total for each tile)


pump.ob1_end()
pump.mux_end()

