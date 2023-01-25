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










from pykuwahara import kuwahara
from skimage import io, measure
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
from skimage.filters import threshold_otsu, butterworth, median
import math
import time
from datetime import datetime

a = np.array([[0,0,0,0,0,1000,1000,1000,1000,1000,2000,2000,2000,2000,2000,3000,3000,3000,3000,3000],
[0,600,1200,1800,2400,2400,1800,1200,600,0,0,600,1200,1800,2400,2400,1800,1200,600,0],
[50,51,52,80,53,52,57,52,62,50,53,50,40,53,60,52,51,55,52,52]])

def  tile_pattern(numpy_array, x_tiles, y_tiles):
    '''
    Takes numpy array with N rows and known tile pattern and casts into new array that follows
    south-north, west-east snake pattern.


    :param numpy_array: dimensions [N, x_tiles*y_tiles]
    :param x_tiles: number x tiles in pattern
    :param y_tiles: number y tiles in pattern
    :return: numpy array with dimensions [N,x_tiles,y_tiles] with above snake pattern
    '''

    layers = np.shape(numpy_array)[0]
    numpy_array = numpy_array.reshape(layers, x_tiles, y_tiles)
    dummy = numpy_array.reshape(layers, 5, 4)
    new_numpy = np.empty_like(dummy)
    for x in range(0,layers):

        new_numpy[x] = numpy_array[x].transpose()
        new_numpy[x,::, 1:y_tiles:2] = np.flipud(new_numpy[x,::, 1:y_tiles:2])

    return new_numpy

def fm_outlier_identifier(full_array):
    fm = full_array[2]
    em = np.empty_like(fm)
    max_derivative = 4
    y_dim = np.shape(fm)[0]
    x_dim = np.shape(fm)[1]


    for x in range(0,x_dim):
        for y in range(0,y_dim):


            score = np.array([0,0,0,0])

            h = np.array([0, 0, 0, 0])
            above = y + 1
            below = y - 1
            right = x + 1
            left = x - 1
            directional_array = np.array([above, below, right, left])
            h[0:2] = directional_array[0:2] < y_dim
            h[2:4] = directional_array[2:4] < x_dim
            directional_array = directional_array >= 0
            directional_array = directional_array * h

            score[0] = directional_array[0]*(fm[y][x] - fm[directional_array[0]*above][x])
            score[1] = directional_array[1] * (fm[y][x] - fm[directional_array[1]*below][x])
            score[2] = directional_array[2] * (fm[y][x] - fm[y][directional_array[2]*right])
            score[3] = directional_array[3] * (fm[y][x] - fm[y][directional_array[3]*left])
            score = score * score
            score = score > max_derivative

            em[y][x] = int(np.sum(score)/np.sum(directional_array))


    j, i = np.where(em == 1)
    #io.imshow(em)
    #plt.show()
    em = np.expand_dims(em, axis=0)
    full_array = np.append(full_array, em, axis = 0)

    return i,j,full_array

def interpolator_maxtrix_generator(full_array, i, j):

    x_array = full_array[0]
    y_array = full_array[1]
    value_array = full_array[2]

    excluded_pairs_x = i
    excluded_pairs_y = j
    excluded_pairs = np.stack((excluded_pairs_x, excluded_pairs_y), axis=-1)

    max_x = np.max(x_array)
    max_y = np.max(y_array)
    min_x = np.min(x_array)
    min_y = np.min(y_array)
    x_unique = np.unique(x_array)
    x_unique = x_unique.size
    y_unique = np.unique(y_array)
    y_unique = y_unique.size
    x_step = (max_x - min_x)/(x_unique - 1)
    y_step = (max_y - min_y) / (y_unique - 1)
    grid_x, grid_y = np.mgrid[min_y:max_y + y_step:y_step, min_x:max_x + x_step:x_step]  # already setup for this input

    x_points = np.array([])
    y_points = np.array([])
    values = np.array([])
    for i in range(0, x_unique):
        for j in range(0, y_unique):
            try:
                if exclusion_checker(i,j,excluded_pairs) != 1:
                    x_points = np.append(x_points, x_array[j][i])
                    y_points = np.append(y_points, y_array[j][i])
                    values = np.append(values, value_array[j][i])
            except:
                x_points = np.append(x_points, x_array[j][i])
                y_points = np.append(y_points, y_array[j][i])
                values = np.append(values, value_array[j][i])

    points = np.stack((x_points, y_points), axis = -1)
    #print(exclusion_checker(0, 3, excluded_pairs))

    return grid_x, grid_y, points, values

def exclusion_checker(i, j, excluded_pairs):
    length = np.shape(excluded_pairs)[0]

    xs = excluded_pairs - [i,j]
    #print(xs)
    checker = 0
    for x in range(0, length):
        xf = np.sum(np.isin(xs[x], [0, 0]))
        #print(xs[x], i, j, xf)
        if xf == 2:
            checker = 1
    #print(i,j, checker)
    return checker

def interpolate_missing_fm_points(x_grid, y_grid, points, values):
    #print(points)
    #print(values)
    grid_z1 = griddata(points, values, (x_grid, y_grid), method='nearest')

    return grid_z1

def median_fm_filter(full_array):
    full_array[2] = median(full_array[2])

    return full_array

#grid_x, grid_y = np.mgrid[0:3000:600, 0:4000:1000] #already setup for this input

#points = np.array([[0,0],[1200,2000], [600,1000], [0,3000]])
#values = np.array([50, 52, 51,52])
#grid_z1 = griddata(points, values, (grid_x, grid_y), method='nearest')


full_array = tile_pattern(a, 4, 5)
print(full_array[2])
#median_fm = median_fm_filter(full_array)[2]
#print(median_fm)
[i,j,full_array] = fm_outlier_identifier(full_array)
[x_grid, y_grid, points, values] = interpolator_maxtrix_generator(full_array, i, j)
#print(full_array[2])
new_fm = interpolate_missing_fm_points(x_grid, y_grid, points, values)




