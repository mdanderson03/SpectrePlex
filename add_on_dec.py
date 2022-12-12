from pycromanager import Core, Acquisition, multi_d_acquisition_events, Dataset, MagellanAcquisition, Magellan, start_headless
import numpy as np
import os

#for autocyplex
#################
def auto_expose(self, core, magellan, seed_expose, benchmark_threshold, z_focused_pos,
                channels=['DAPI', 'A488', 'A555', 'A647'], surface_name='none'):
    '''

    :param object core: core object from Core() in pycromananger
    :param object magellan: magellan object from Magellan() in pycromanager
    :param int seed_expose: initial exposure time
    :param dict tile_points_xy: dictionary that contains keys of X and Y with associated coordinates
    :param int benchmark_threshold: integer of threshold that top 99% pixel will hit.
    :param float z_focused_pos: z position where image is in focus
    :param [str] channels: list of strings of channels that are wanted to be used
    :param str surface_name: name of surface to be used. If blank, just executes in current XY position
    :return: list of exposures
    '''

    if surface_name != 'none':
        new_x, new_y = self.tissue_center(surface_name, magellan)  # uncomment if want center of tissue to expose
        core.set_xy_position(new_x, new_y)
        z_pos = z_focused_pos
        # z_pos = magellan.get_surface(surface_name).get_points().get(0).z
        core.set_position(z_pos)

    bandwidth = 0.1
    sat_max = 65000
    exp_time_limit = 1000
    exposure_array = [10, 10, 10, 10]  # dapi, a488, a555, a647

    for fluor_channel in channels:

        intensity = self.expose(seed_expose, fluor_channel)
        new_exp = seed_expose
        while intensity < (1 - bandwidth) * benchmark_threshold or intensity > (
                1 + bandwidth) * benchmark_threshold:
            if intensity < benchmark_threshold:
                new_exp = benchmark_threshold / intensity * new_exp
                if new_exp >= exp_time_limit:
                    new_exp = exp_time_limit
                    break
                else:
                    intensity = self.expose(new_exp, fluor_channel)
            elif intensity > benchmark_threshold and intensity < sat_max:
                new_exp = benchmark_threshold / intensity * new_exp
                intensity = self.expose(new_exp, fluor_channel)
            elif intensity > sat_max:
                new_exp = new_exp / 10
                intensity = self.expose(new_exp, fluor_channel)
            elif new_exp >= sat_max:
                new_exp = sat_max
                break

        if fluor_channel == 'DAPI':
            exposure_array[0] = new_exp
        elif fluor_channel == 'A488':
            exposure_array[1] = new_exp
        elif fluor_channel == 'A555':
            exposure_array[2] = new_exp
        elif fluor_channel == 'A647':
            exposure_array[3] = new_exp

    return new_exp


def numpy_xyz_gen(self, tile_points_xy, z_focused):
    '''
    Takes dictionary of XY coordinates applies inputted same focused z postion to all of them to make a xyz array

    :param dictionary tile_points_xy: dictionary containing all XY coordinates. In the form: {{x:(int)}, {y:(int)}}
    :param float z_focused: z position where surface is in focus

    :return: XYZ points where XY are stage coords and Z is in focus coordinate. {{x:(int)}, {y:(int)}, {z:(float)}}
    :rtype: numpy array
    '''

    z_temp = []
    x_temp = tile_points_xy['x']
    y_temp = tile_points_xy['y']
    num = len(tile_points_xy['x'])
    for q in range(0, num):
        z_temp.append(z_focused)

    xyz = np.hstack([x_temp[:, None], y_temp[:, None], z_temp[:, None]])

    return xyz


def xyz_acquire(self, xyz_array, channel, exposure_time, cycle_number, directory_name = 'E:/images/'):
    '''
    :param numpy xyz_array: numpy array 3xN where is N number of points that contain all xyz coords of positions
    :param str channel: channel name ie. DAPI, A488, A555, A647, etc.
    :param str exposure_time: exposure time required for channel in ms
    :param str directory_name: highest level folder name to store all images in
    :param int cycle_number: cycle number
    :return:  nothing
    '''

    add_on_folder = 'cycle_/' + str(cycle_number)
    full_directory_path = os.path.join(directory_name, add_on_folder)
    if os.path.exists(full_directory_path) == 'False':
        os.mkdir(full_directory_path)

    with Acquisition(directory= full_directory_path, name=channel) as acq:
        events = multi_d_acquisition_events(channel_group='Color', channels=[channel], xyz_positions=xyz_array, channel_exposures_ms=[exposure_time])
        acq.acquire(events)
        acq.await_completion()


def surface_acquire(self, cycle_number, directory_name='E:/images/', channels=['DAPI', 'A488', 'A555', 'A647']):
    '''
    Takes generated micro-magellan surface with name: surface_name and extracts all points from it.
    uses multi_d_acquistion to acquire all images in defined surface via xyz_acquire method,  auto exposes DAPI, A488, A555 and A647 channels.
    Takes autofocus from center tile in surface and applies value to every other tile in surface

    :param int cycle_number: cycle number
    ::param str directory_name: highest level folder name to store all images in
    :param: list[str] channels: list that contains strings with channel names
    :return: Nothing
    '''

    surface_name = 'New Surface 1'
    num_channels = len(channels)  # checks how many 'New Surface #' surfaces exist. Not actual total
    tile_surface_xy = self.tile_xy_pos(surface_name, magellan)  # pull center tile coords from manually made surface
    z_center = magellan.get_surface(surface_name).get_points().get(0).z
    z_range = [z_center - 10, z_center + 10, 1]

    xyz_position_array = []
    exp_time_array = []

    for channel in channels:

        auto_focus_exposure_time = self.auto_initial_expose(core, magellan, 50, 6500, channel, surface_name)
        z_focused = self.auto_focus(z_range, auto_focus_exposure_time, channel)  # here is where autofocus results go. = auto_focus

        xyz_position_array.append(self.numpy_xyz_gen(tile_surface_xy, z_focused))  # go to each tile coord and autofocus and populate associated z with result
        exp_time_array.append(self.auto_expose(core, magellan, auto_focus_exposure_time, 6500, z_focused, [channel], surface_name))

    for x in range(0,num_channels):
        self.xyz_acquire(xyz_position_array[x], channel[x], exp_time_array[x], cycle_number, directory_name)

    return




