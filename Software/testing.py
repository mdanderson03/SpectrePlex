import numpy as np
import os
from skimage import io, morphology
from matplotlib import pyplot as plt
import os
from autocyplex import *

pump = fluidics(6, 3)


pump.liquid_action('Stain', stain_valve= 1, incub_val= 15)
pump.liquid_action('Bleach')