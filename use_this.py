from autocyplex import *
from pycromanager import Core, Studio, Magellan
microscope = cycif() # initialize cycif object
pump = fluidics(6, 3)



experiment_directory = r'E:\8_3_23 test 2 cycle'


#pump.liquid_action('Nuc_Touchup') # nuc is valve=7, pbs valve=8, bleach valve=1 (action, stain_valve, heater state (off = 0, on = 1))

exp_array = [100,10000,100,100]
offset_array = [0,-7, -7,-7]
#microscope.image_cycle_acquire(0, experiment_directory, 5, 'Bleach', exp_array, offset_array, x_crop_percentage = 0.20)
pump.liquid_action('Bleach') # nuc is valve=7, pbs valve=8, bleach valve=1 (action, stain_valve, heater state (off = 0, on = 1))

#pump.liquid_action('Stain', 6) # nuc is valve=7, pbs valve=8, bleach valve=1 (action, stain_valve, heater state (off = 0, on = 1))



#microscope.image_cycle_acquire(1, experiment_directory, 5, 'Stain', exp_array, offset_array)






