from pycromanager import Core, Acquisition, multi_d_acquisition_events, Dataset, MagellanAcquisition, Magellan, start_headless
import numpy as np
import os

#for autocyplex
#################



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




from pykuwahara import kuwahara
from skimage import io, measure
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from datetime import datetime

a = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

def  tile_pattern(numpy_array, x_tiles, y_tiles):
    numpy_array = numpy_array.reshape(x_tiles, y_tiles)
    numpy_array = numpy_array.transpose()
    numpy_array[::, 1:y_tiles:2] = np.flipud(numpy_array[::, 1:y_tiles:2])

    return numpy_array

a = tile_pattern(a, 4, 5)
print(a)


