from autocyplex import *


microscope = cycif()

max_z = 6000
min_z = 5950

z_center = magellan.get_surface('New Surface 1').get_points().get(0).z
z_range = [z_center - 20, z_center + 20, 20]
print(z_range)

#num = 0

tile_surface_xy = microscope.tile_xy_pos('New Surface 1')
#x_pos = tile_surface_xy['x'][num]
#y_pos = tile_surface_xy['y'][num]
#core.set_xy_position(x_pos, y_pos)

#z_range = [min_z, max_z, (max_z-min_z)/2]
#z_ideal = microscope.auto_focus(z_range, 100, 'DAPI')
#core.set_position(z_ideal)
#print(z_ideal)

channel = 'A488'

tile_surface_xy = microscope.tile_xy_pos('New Surface 1')
print(tile_surface_xy['x'])
print(tile_surface_xy['y'])

auto_focus_exposure_time = microscope.auto_initial_expose(50, 2500, channel, z_range, 'New Surface 1')
xyz = microscope.focus_tile( tile_surface_xy, z_range, 0, auto_focus_exposure_time, channel)


print(xyz['z'])
microscope.tiled_acquire(xyz, channel, auto_focus_exposure_time, 1, 'E:/garbage')
