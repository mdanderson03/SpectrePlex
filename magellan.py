from autocif import *

bridge = Bridge()
core = bridge.get_core()
magellan = bridge.get_magellan()


surface = magellan.get_surface('New Surface 1')

def tile_xy_pos(surface):  #input is magellan surface, output is dictionary with x and y terms
    num = surface.get_num_positions()
    xy = surface.get_xy_positions()
    surface_points_xy = {}
    x_temp = []
    y_temp = []

    for q in range(0,num ):
        pos = xy.get(q)
        pos = pos.get_center()
        x_temp.append(pos.x)
        y_temp.append(pos.y)

    surface_points_xy['x'] = x_temp  ## put all points in dictionary to ease use
    surface_points_xy['y'] = y_temp

    return surface_points_xy


def focus_tile(surface_points_xy):
    z_temp = []
    num = len(surface_points_xy['x'])
    for q in range(0, num):
        new_x = surface_points_xy['x'][q]
        new_y = surface_points_xy['y'][q]
        core.set_xy_position(new_x, new_y)
        time.sleep(0.25) #wait long enough for stage to translate to new location
        z_focused = cycif.auto_focus()  #here is where autofocus results go. = auto_focus()
        z_temp.append(z_focused)
    surface_points_xy['z'] = z_temp
    surface_points_xyz = surface_points_xy

    return surface_points_xyz


def focused_surface_generate(surface_points_xyz):  #only get 1/2 points anticipated, dont know why
    magellan.create_surface('Focused Surface')  #need to make naming convention
    focused_surface = magellan.get_surface('Focused Surface')
    num = len(surface_points_xyz['x'])
    for q in range(0, num):
        focused_surface.add_point(surface_points_xyz['x'][q], surface_points_xyz['y'][q], surface_points_xyz['z'][q])  # access point_list and add on relavent points to surface


def auto_expose():
    exposure = np.array([100,100,100,100]) #exposure time in milliseconds
    return exposure

def focused_surface_acq_settings(exposure): #exposure is a numpy array (1x4)
    magellan.create_acquisition_settings()

    acq_settings = magellan.get_acquisition_settings(2)
    acq_settings.set_acquisition_name('Focused Surface') #make same name as in focused_surface_generate function (all below as well too)
    acq_settings.set_acquisition_space_type('2d_surface')
    acq_settings.set_xy_position_source('Focused Surface')
    acq_settings.set_surface('Focused Surface')
    acq_settings.set_bottom_surface('Focused Surface')
    acq_settings.set_top_surface('Focused Surface')
    acq_settings.set_saving_dir(r'C:\Users\CyCIF PC\Desktop\test_images\tiled_images') #standard saving directory
    acq_settings.set_channel_group('Color')
    acq_settings.set_use_channel('DAPI', True)  # channel_name, use
    acq_settings.set_channel_exposure('DAPI', int(exposure[0]))  # channel_name, exposure in ms can auto detect channel names and iterate names with exposure times
    acq_settings.set_channel_exposure('A488', int(exposure[1]))  # channel_name, exposure in ms
    acq_settings.set_channel_exposure('A555', int(exposure[2]))  # channel_name, exposure in ms
    acq_settings.set_channel_exposure('A647', int(exposure[3]))  # channel_name, exposure in ms
    acq_settings.set_channel_z_offset('DAPI', 0)  # channel_name, offset in um




surface_points_xy = tile_xy_pos(surface)
surface_points_xyz = focus_tile(surface_points_xy)
focused_surface_generate(surface_points_xyz)
exposure = auto_expose()
focused_surface_acq_settings(exposure)






