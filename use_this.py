from autocyplex import *
from pycromanager import Core, Studio, Magellan
microscope = cycif() # initialize cycif object
pump = fluidics(6, 3)



experiment_directory = r'E:\pure_image_system_testing'


#pump.liquid_action('Nuc_Touchup') # nuc is valve=7, pbs valve=8, bleach valve=1 (action, stain_valve, heater state (off = 0, on = 1))

exp_array = [50,50, 5, 5]
offset_array = [0, -8, -8, -8]
microscope.image_cycle_acquire(0, experiment_directory, 6, 'Bleach', exp_array, offset_array)
#pump.liquid_action('Bleach') # nuc is valve=7, pbs valve=8, bleach valve=1 (action, stain_valve, heater state (off = 0, on = 1))

#pump.liquid_action('Stain', 6) # nuc is valve=7, pbs valve=8, bleach valve=1 (action, stain_valve, heater state (off = 0, on = 1))



#microscope.image_cycle_acquire(1, experiment_directory, 5, 'Stain', exp_array, offset_array)






