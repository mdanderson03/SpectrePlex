import math
'''
from autocif import *
microscope = autocif.cycif()


microscope.surf2focused_surf(core, magellan, ['DAPI'])
microscope.micro_magellan_acq()
'''
time = 3850


double_value = 43237*(1* math.exp(-0.0009504* time) + 0*math.exp(-0.008874 * time)) + 20568

threshold = 0.05 *(43237 - 20568)
print(double_value - 20568, threshold)

