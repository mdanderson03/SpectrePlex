import datetime

from autocyplex import *
from optparse import OptionParser
#microscope = cycif() # initialize cycif object
experiment_directory = r'E:\14-2-24 fluidics testing'
pump = fluidics(experiment_directory, 6, 3)

z_slices = 7
x_frame_size = 2960
offset_array = [0, -8, -7, -7]




pump.liquid_action('Stain', stain_valve=1, incub_val=10)
pump.liquid_action('Wash')
pump.liquid_action('Bleach')
pump.liquid_action('Wash')

pump.liquid_action('Stain', stain_valve=2, incub_val=10)
pump.liquid_action('Wash')
pump.liquid_action('Bleach')
pump.liquid_action('Wash')

pump.liquid_action('Stain', stain_valve=3, incub_val=10)
pump.liquid_action('Wash')
pump.liquid_action('Bleach')
pump.liquid_action('Wash')

pump.liquid_action('Stain', stain_valve=4, incub_val=10)
pump.liquid_action('Wash')
pump.liquid_action('Bleach')
pump.liquid_action('Wash')

pump.liquid_action('Stain', stain_valve=5, incub_val=10)
pump.liquid_action('Wash')
pump.liquid_action('Bleach')
pump.liquid_action('Wash')

pump.liquid_action('Stain', stain_valve=6, incub_val=10)
pump.liquid_action('Wash')
pump.liquid_action('Bleach')
pump.liquid_action('Wash')

pump.liquid_action('Stain', stain_valve=7, incub_val=10)
pump.liquid_action('Wash')
pump.liquid_action('Bleach')
pump.liquid_action('Wash')

pump.liquid_action('Stain', stain_valve=8, incub_val=10)
pump.liquid_action('Wash')
pump.liquid_action('Bleach')
pump.liquid_action('Wash')





