from autocif import *
core = Core()
magellan = Magellan()
microscope = autocif.cycif()


magellan_object = magellan
channel = 'DAPI'
cycle_number = 1

surface_name = 'New Surface 1'
new_focus_surface_name = 'Focused Surface ' + 'DAPI'

#tile_surface_xy = microscope.tile_xy_pos(surface_name, magellan_object)  # pull center tile coords from manually made surface
#auto_focus_exposure_time = microscope.auto_initial_expose(core, magellan_object, 50, 6500, tile_surface_xy, channel,surface_name)
#z_centers = microscope.z_range(tile_surface_xy, surface_name, magellan_object, core, cycle_number, channel, auto_focus_exposure_time)

#surface_points_xyz = microscope.focus_tile(tile_surface_xy, z_centers, core, auto_focus_exposure_time,channel)  # go to each tile coord and autofocus and populate associated z with result

#microscope.focused_surface_generate(surface_points_xyz, magellan_object, new_focus_surface_name)  # will generate surface if not exist, update z points if exists
#exposure_array = microscope.auto_expose(core, magellan_object, auto_focus_exposure_time, 6500, [channel], new_focus_surface_name)
#microscope.focused_surface_acq_settings(exposure_array, surface_name, new_focus_surface_name, magellan_object, 1, channel)

z_center = 1100
z_focused1 = microscope.auto_focus([z_center - 50, z_center + 50,20 ], 6, 'DAPI')

z_center = 1100
z_focused2 = microscope.auto_focus([z_center - 50, z_center + 50,20 ], 6, 'DAPI')

z_center = 1100
z_focused3 = microscope.auto_focus([z_center - 50, z_center + 50,20 ], 6, 'DAPI')

print(z_focused1, z_focused2, z_focused3)






#microscope.surf2focused_surf(core, magellan, 1, ['DAPI'])
#microscope.micro_magellan_acq()
