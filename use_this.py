from autocyplex import *
from pycromanager import Core, Studio, Magellan
microscope = cycif() # initialize cycif object
pump = fluidics(6, 3)



experiment_directory = r'E:\parallel_test_with_thunder'


#pump.liquid_action('Nuc_Touchup') # nuc is valve=7, pbs valve=8, bleach valve=1 (action, stain_valve, heater state (off = 0, on = 1))

exp_array = [20,35, 35, 35]
offset_array = [0, -8, -8, -8]
#pump.liquid_action('Bleach') # nuc is valve=7, pbs valve=8, bleach valve=1 (action, stain_valve, heater state (off = 0, on = 1))
#microscope.image_cycle_acquire(4, experiment_directory, 6, 'Bleach', exp_array, offset_array)
#time.sleep(5)


#pump.liquid_action('Wash') # nuc is valve=7, pbs valve=8, bleach valve=1 (action, stain_valve, heater state (off = 0, on = 1))


#pump.liquid_action('Stain', 2) # nuc is valve=7, pbs valve=8, bleach valve=1 (action, stain_valve, heater state (off = 0, on = 1))
#time.sleep(5)
#pump.liquid_action('Wash') # nuc is valve=7, pbs valve=8, bleach valve=1 (action, stain_valve, heater state (off = 0, on = 1))
#time.sleep(5)
#exp_array = [200,150, 300, 300]
#microscope.image_cycle_acquire(5, experiment_directory, 6, 'Stain', exp_array, offset_array)

'''
pump.liquid_action('Bleach') # nuc is valve=7, pbs valve=8, bleach valve=1 (action, stain_valve, heater state (off = 0, on = 1))
microscope.image_cycle_acquire(5, experiment_directory, 6, 'Bleach', exp_array, offset_array)
time.sleep(5)
pump.liquid_action('Stain', 4) # nuc is valve=7, pbs valve=8, bleach valve=1 (action, stain_valve, heater state (off = 0, on = 1))
time.sleep(5)
pump.liquid_action('Wash') # nuc is valve=7, pbs valve=8, bleach valve=1 (action, stain_valve, heater state (off = 0, on = 1))
microscope.image_cycle_acquire(6, experiment_directory, 6, 'Stain', exp_array, offset_array)

time.sleep(5)
'''
#microscope.image_cycle_acquire(8, experiment_directory, 1, 'Bleach', exp_array, offset_array)

microscope.post_acquisition_processor(experiment_directory)






