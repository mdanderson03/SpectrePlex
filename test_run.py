from autocif import *
from pycromanager import Core
core = Core()
#magellan = bridge.get_magellan()
microscope = cycif() # initialize cycif object
import numpy as np


<<<<<<< Updated upstream
#z_range = [4680, 4720, 2]
=======
z_range = [4690, 4720, 4]
>>>>>>> Stashed changes
#sd= microscope.auto_focus(z_range)
#print(sd)
#z_ideal = microscope.auto_focus(z_range)
#print(z_ideal)

<<<<<<< Updated upstream
# microscope.surf2focused_surf(z_range, 'New Grid 1', 'Focused Surface', core, magellan)
# microscope.surf2focused_surf(z_range, 'New Surface 1', 'Focused Surface', core, magellan)
=======
microscope.surf2focused_surf(z_range, 'New Surface 1', 'Focused Surface', core, magellan)

>>>>>>> Stashed changes
#surface = magellan.get_surface('Focused Surface')
#exposure = microscope.auto_expose()
#microscope.focused_surface_acq_settings(exposure, 'New Surface 1', 'Focused Surface', magellan)


<<<<<<< Updated upstream
#core.set_roi(385,160,420,494)

microscope.tilt_angle_chip(core)




=======
>>>>>>> Stashed changes
'''
xy = microscope.tile_xy_pos('New Surface 1', magellan)
print(xy)
xyz = microscope.focus_tile(xy, z_range, core)
microscope.focused_surface_generate(xyz, magellan, 'Focused Surface')
print(xyz)

exposure = microscope.auto_expose()
microscope.focused_surface_acq_settings(exposure, 'New Surface 1', 'Focused Surface', magellan)
'''

