from autocyplex import *

microscope = cycif()
pump = fluidics(6, 3)

experiment_directory = r'D:\Images\AutoCyPlex\elveflow_test_2'
offset_array = [0, -7, -7, -7]
# intial auto fluorescence
pump.liquid_action('Nuc_Touchup') # nuc is valve=7, pbs valve=8, bleach valve=1 (action, stain_valve, heater state (off = 0, on = 1))
microscope.cycle_acquire(cycle_number = 1, experiment_directory, z_slices = 7, 'Bleach', offset_array)

# cycle 1
pump.liquid_action('Stain', 2, 1) # nuc is valve=7, pbs valve=8, bleach valve=1 (action, stain_valve, heater state (off = 0, on = 1))
microscope.cycle_acquire(cycle_number = 1, experiment_directory, z_slices = 7, 'Stain', offset_array)
pump.liquid_action('Bleach') # nuc is valve=7, pbs valve=8, bleach valve=1 (action, stain_valve, heater state (off = 0, on = 1))
microscope.cycle_acquire(cycle_number = 2, experiment_directory, z_slices = 7, 'Bleach', offset_array)

#microscope.full_cycle(experiment_directory, 1, offset_array, 2, 1) # effectively does block above



microscope.post_acquisition_processor(experiment_directory)

pump.ob1_end()
pump.mux_end()



