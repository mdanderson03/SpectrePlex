from autocif import *
import time

bridge = Bridge()
core = bridge.get_core()
#magellan = bridge.get_magellan()
microscope = cycif() # initialize cycif object

z_range = [4845, 4865, 2]
sd= microscope.auto_focus(z_range)
print(sd)




