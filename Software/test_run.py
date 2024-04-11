import time
from subprocess import call
from fluidics_V3 import fluidics


experiment_directory = r'E:\8-4-24 celiac'
pump = fluidics(experiment_directory, 6, 13, flow_control=1)
pump.liquid_action('Stain', stain_valve=12, incub_val=2)
pump.liquid_action('Bleach')
pump.liquid_action('Wash')
