from autocif import *
core = Core()
magellan = Magellan()
microscope = autocif.cycif()


magellan_object = magellan
channel = 'DAPI'
cycle_number = 1

surface_name = 'New Surface 1'
new_focus_surface_name = 'Focused Surface ' + str(channel)

tile_surface_xy = microscope.tile_xy_pos(surface_name, magellan_object)  # pull center tile coords from manually made surface
auto_focus_exposure_time = microscope.auto_initial_expose(core, magellan_object, 50, 6500, tile_surface_xy, channel,surface_name)
z_centers = microscope.z_range(tile_surface_xy, surface_name, magellan_object, core, cycle_number, channel, auto_focus_exposure_time)

surface_points_xyz = microscope.focus_tile(tile_surface_xy, z_centers, core, auto_focus_exposure_time,channel)  # go to each tile coord and autofocus and populate associated z with result

microscope.focused_surface_generate(surface_points_xyz, magellan_object, new_focus_surface_name)  # will generate surface if not exist, update z points if exists
exposure_array = microscope.auto_expose(core, magellan_object, auto_focus_exposure_time, 6500, [channel], new_focus_surface_name)
microscope.focused_surface_acq_settings(exposure_array, surface_name, new_focus_surface_name, magellan_object, 1, channel)


microscope.micro_magellan_acq()
