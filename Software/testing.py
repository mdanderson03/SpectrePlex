import numpy as np
import os
from skimage import io, morphology
from matplotlib import pyplot as plt
import os
from autocyplex import *

pump = fluidics(6, 3)

print('start')
pump.flow(500)
time.sleep(10)
print('stop')
pump.flow(0)
time.sleep(10)
print('no meter flow')
pump.flow(500)
time.sleep(10)
print('stop')
pump.flow(0)
time.sleep(5)