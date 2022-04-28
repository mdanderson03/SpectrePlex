from autocif import *
from pycromanager import Core, Magellan
core = Core()
magellan = Magellan()
microscope = cycif() # initialize cycif object
import numpy as np



z_range = [2665, 2700, 2]

#z_range = [4690, 4720, 4]

#sd= microscope.auto_focus(z_range)
#print(sd)
#z_ideal = microscope.auto_focus(z_range)
#print(z_ideal)


# microscope.surf2focused_surf(z_range, 'New Grid 1', 'Focused Surface', core, magellan)
microscope.surf2focused_surf(z_range, 'New Surface 1', 'Focused Surface', core, magellan)

#microscope.surf2focused_surf(z_range, 'New Surface 1', 'Focused Surface', core, magellan)


#surface = magellan.get_surface('Focused Surface')
#exposure = microscope.auto_expose()
#microscope.focused_surface_acq_settings(exposure, 'New Surface 1', 'Focused Surface', magellan)



#core.set_roi(385,160,420,494)

#microscope.tilt_angle_chip(core)