from autocif import *
import time

bridge = Bridge()
core = bridge.get_core()
magellan = bridge.get_magellan()
microscope = cycif() # initialize cycif object

z_range = [4680, 4720, 2]
sd= microscope.auto_focus(z_range)
print(sd)


#microscope.surf2focused_surf(z_range, 'New Surface 1', 'Focused Surface', core, magellan)
#surface = magellan.get_surface('Focused Surface')
#exposure = microscope.auto_expose()
#microscope.focused_surface_acq_settings(exposure, 'New Surface 1', 'Focused Surface', magellan)




'''
xy = microscope.tile_xy_pos('New Surface 1', magellan)
print(xy)
xyz = microscope.focus_tile(xy, z_range, core)
microscope.focused_surface_generate(xyz, magellan, 'Focused Surface')
print(xyz)

exposure = microscope.auto_expose()
microscope.focused_surface_acq_settings(exposure, 'New Surface 1', 'Focused Surface', magellan)
'''

