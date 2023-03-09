from autocyplex import *


microscope = cycif()




microscope.file_structure('E:/folder_structure', 1)



#max_z = 6000
#min_z = 5950

z_center = magellan.get_surface('New Surface 1').get_points().get(0).z
#z_center = 6697
z_range = [z_center - 15, z_center + 15, 15]
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
#print(tile_surface_xy)
#z_focused = microscope.auto_focus(z_range, 200,channel)  # here is where autofocus results go. = auto_focus
#print(z_focused)

#exp = microscope.expose(200)

auto_focus_exposure_time = microscope.auto_initial_expose(50, 2500, 'DAPI', z_range, 'New Surface 1')
xyz = microscope.focus_tile_DAPI( tile_surface_xy, z_range, auto_focus_exposure_time)

tile_surface_xyz = microscope.tile_pattern(xyz)
tile_surface_xyz = microscope.median_fm_filter(tile_surface_xyz, channel)
tile_surface_xyz = microscope.fm_channel_initial(tile_surface_xyz)

'''
#exposure_time_a488 = microscope.auto_initial_expose(50, 2500, 'A488', z_range, 'New Surface 1')
#exposure_time_a555 = microscope.auto_initial_expose(50, 2500, 'A555', z_range, 'New Surface 1')
#exposure_time_a647 = microscope.auto_initial_expose(50, 2500, 'A647', z_range, 'New Surface 1')
'''

#tile_surface_xyz = microscope.focus_tile_stain(tile_surface_xyz,10, 'A488', 200)
#tile_surface_xyz = microscope.focus_tile_stain(tile_surface_xyz,10, 'A555', 200)
#tile_surface_xyz = microscope.focus_tile_stain(tile_surface_xyz,10, 'A647', 200)



#z_focused = tile_surface_xyz[2][0][0]
#exp_time = microscope.auto_expose(auto_focus_exposure_time, 2500, z_focused, [channel], 'New Surface 1')
exp_time = [100,200,200,200]

np.save('exp_array.npy', exp_time)
np.save('fm_array.npy', tile_surface_xyz)





dapi_tif = microscope.core_tile_acquire('DAPI', 11)
#a488_tif = microscope.core_tile_acquire('A488', 3)
#a555_tif = microscope.core_tile_acquire('A555', 3)
#a647_tif = microscope.core_tile_acquire('A647', 3)



microscope.optimal_quick_preview_qt(dapi_tif, 'DAPI', 1, 'E:/folder_structure')
#microscope.quick_tile_all_z_save(optimal_dapi_tif, 'DAPI', 1, 'E:/folder_structure')
#microscope.quick_tile_all_z_save(a488_tif, 'A488', 1, 'E:/folder_structure')
#microscope.quick_tile_all_z_save(a555_tif, 'A555', 1, 'E:/folder_structure')
#microscope.quick_tile_all_z_save(a647_tif, 'A647', 1, 'E:/folder_structure')

#microscope.save_files(dapi_tif, 'DAPI', 1, 'E:/folder_structure')
#microscope.save_files(a488_tif, 'A488', 1, 'E:/folder_structure')
#microscope.save_files(a555_tif, 'A555', 1, 'E:/folder_structure')
#microscope.save_files(a647_tif, 'A647', 1, 'E:/folder_structure')

#print(tile_surface_xyz[0])
#print(tile_surface_xyz[1])
#print(tile_surface_xyz[2])
#print(tile_surface_xyz[3])
#print(tile_surface_xyz[4])
#print(tile_surface_xyz[5])