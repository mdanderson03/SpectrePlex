from autocyplex import *


microscope = cycif()

max_z = 6000
min_z = 5950

z_center = magellan.get_surface('New Surface 1').get_points().get(0).z
#z_center = 5893
z_range = [z_center - 20, z_center + 20, 20]
print(z_range)

#num = 0

#ile_surface_xy = microscope.tile_xy_pos('New Surface 1')
#x_pos = tile_surface_xy['x'][num]
#y_pos = tile_surface_xy['y'][num]
#core.set_xy_position(x_pos, y_pos)

#z_range = [min_z, max_z, (max_z-min_z)/2]
#z_ideal = microscope.auto_focus(z_range, 100, 'DAPI')
#core.set_position(z_ideal)
#print(z_ideal)

channel = 'DAPI'

tile_surface_xy = microscope.tile_xy_pos('New Surface 1')

#z_focused = microscope.auto_focus(z_range, 100,channel)  # here is where autofocus results go. = auto_focus
#print(z_focused)


auto_focus_exposure_time = microscope.auto_initial_expose(50, 2500, channel, z_range, 'New Surface 1')
xyz = microscope.focus_tile( tile_surface_xy, z_range, 0, auto_focus_exposure_time, channel)
tile_surface_xyz = microscope.tile_pattern(xyz)
print(tile_surface_xyz[2])
tile_surface_xyz = microscope.median_fm_filter(tile_surface_xyz)
print(tile_surface_xyz[2])
tile_surface_xyz = microscope.median_fm_filter(tile_surface_xyz)
print(tile_surface_xyz[2])
z_focused = tile_surface_xyz[2][0][0]
exp_time = microscope.auto_expose(auto_focus_exposure_time, 2500, z_focused, [channel], 'New Surface 1')
microscope.tiled_acquire(tile_surface_xyz, channel, exp_time, 1, 'E:/garbage')
