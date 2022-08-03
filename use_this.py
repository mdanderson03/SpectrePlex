#from autocif import *
from pycromanager import Core, Studio, Magellan
#microscope = cycif() # initialize cycif object
core = Core()

#microscope = cycif() # initialize cycif object
#magellan = Magellan()

xyz_points = {'x':[1000,1000,2000, 2000], 'y':[2340,2680,2340,2680], 'z':[1100,1100,1103, 1105]}

def snake_order(xyz_points):
    '''
    Takes set of XYZ points and makes snake pattern order for points going left to right and top to bottom. Assumes points form recilinear pattern

    :param dictionary xyz_points: XY locations from MM surface and z from autofocus at each XY point
    :return: list that is the order for points to be acquired in
    '''
    unique_x_values = list(set(sorted(xyz_points['x'], reverse = True))) #all unique values and sorted from highest to lowest in list form
    unique_y_values = list(set(sorted(xyz_points['y'], reverse = True)))
    num_unique_x_positions = len(unique_x_values)
    num_unique_y_positions = len(unique_y_values)
    snake_order = []

    for n in range(0, num_unique_y_positions):
        y_value = unique_y_values[n]
        if n % 2 == 1:
            unique_x_values = list(set(sorted(xyz_points['x'], reverse=True)))
        elif n % 2 == 0:
            unique_x_values = list(set(sorted(xyz_points['x'])))
        for m in range(0, num_unique_x_positions):
            x_value = unique_x_values[m]
            print(x_value)
            position_index = xy_point_index(xyz_points, x_value, y_value)
            snake_order.append(position_index)

    return snake_order


def dict_index_array(dictionary, key_value, value):
    term_array = dictionary[key_value]
    index_array = []
    term_count = len(term_array)
    for x in range(0, term_count):
        term = term_array[x]
        if term == value:
            index_array.append(x)

    return index_array

def xy_point_index(xyz_points, x_point, y_point):
    indicies_x = dict_index_array(xyz_points, 'x', x_point)
    indicies_y = dict_index_array(xyz_points, 'y', y_point)
    index = list(set(indicies_y) & set(indicies_x))[0]
    return index

print(core.get_exposure() )












