from autocyplex import *
from pycromanager import Core, Magellan, MagellanAcquisition
import numpy as np


microscope = cycif() # initialize cycif object
arduino = arduino()
#arduino.auto_load()
#arduino.nuc_touch_up(5,360)
#arduino.dispense(7, 400)
#arduino.primary_secondary_cycle(3, 4)
#microscope.surface_acquire(core, magellan, ['DAPI'])
#acq = MagellanAcquisition(magellan_acq_index=0)
#acq.await_completion()
#print('acq ' + str(1) + ' finished')

#microscope.surface_acquire(core, magellan, ['A488'])
#microscope.micro_magellan_acq()
#arduino.bleach(300)
#microscope.micro_magellan_acq()
#arduino.stain(7)
#microscope.surface_acquire(core, magellan)
#microscope.micro_magellan_acq()
#arduino.bleach(300)
#microscope.micro_magellan_acq()

#level = microscope.expose(200)
#print(level)

'''
num_surfaces = microscope.num_surfaces_count(magellan)
channel = 'DAPI'

new_focus_surface_name = 'Focused Surface ' + str(channel)

z_center = magellan.get_surface('New Surface 1').get_points().get(0).z
z_range = [z_center - 10, z_center + 10, 1]
tile_surface_xy = microscope.tile_xy_pos('New Surface 1', magellan)  # pull center tile coords from manually made surface
auto_focus_exposure_time = microscope.auto_initial_expose(core, magellan, 50, 6500, tile_surface_xy, channel, 'New Surface 1')
z_focused = microscope.auto_focus(z_range, auto_focus_exposure_time,channel)  # here is where autofocus results go. = auto_focus
surface_points_xyz = microscope.focus_tile_center(tile_surface_xy, z_focused)
microscope.focused_surface_generate_xyz(magellan, new_focus_surface_name, surface_points_xyz) # will generate surface if not exist, update z points if exists

exposure_array = microscope.auto_expose(core, magellan, auto_focus_exposure_time, tile_surface_xy,6500, z_focused, [channel])
microscope.focused_surface_acq_settings(exposure_array, 'New Surface 1', new_focus_surface_name, magellan,1, channel)

#print(num_surfaces)
print(auto_focus_exposure_time)
print(z_focused)
print(tile_surface_xy)
print(surface_points_xyz)
print(exposure_array)
'''
#arduino.chamber('drain')

'''
arduino.stain(3)
microscope.acquire_all_tiled_surfaces(3)
arduino.bleach(1700)
arduino.bleach(1700)
microscope.acquire_all_tiled_surfaces(4)
arduino.stain(4)
microscope.acquire_all_tiled_surfaces(5)
arduino.bleach(300)
microscope.acquire_all_tiled_surfaces(6)
arduino.nuc_touch_up(6,360)
arduino.stain(5)
microscope.acquire_all_tiled_surfaces(7)
arduino.bleach(300)
microscope.acquire_all_tiled_surfaces(8)
'''
#arduino.nuc_touch_up(6,360)
arduino.bleach(1700)
arduino.bleach(1700)
microscope.acquire_all_tiled_surfaces(4)
arduino.stain(3)
arduino.nuc_touch_up(6,360)
microscope.acquire_all_tiled_surfaces(5)