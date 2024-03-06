import os
from skimage import io
import numpy as np
from ome_types.model import Instrument, Microscope, Objective, InstrumentRef, Image, Pixels, Plane, Channel
from ome_types.model.simple_types import UnitsLength, PixelType, PixelsID, ImageID, ChannelID
from ome_types import from_xml, OME, from_tiff, to_xml
import tifffile as tf
import openpyxl
import sys

os.chdir(r'D:\Images\AutoCyPlex\folder_structure\A647\Stain\cy_1\Tiles')


input_array = np.array([[1,1,1,1,1, 1], #each row is associated with channel.
                        [1,1,1,1,1,1], #each element is what z index to choose
                        [1,1,1,1,1,1],
                        [1,1,1,1,1,1]])


def mcmicro_meta_data_generation(input_array, experiment_directory):

    os.chdir(experiment_directory + '\DAPI\Stain\cy_1\Tiles')


    #Build structure of new metadata. Fill in details in next section
    ome = OME()

    tile_count = input_array.shape[1]
    p_type = PixelType('uint16')
    p_id = PixelsID('Pixels:0')
    i_id = ImageID('Image:0')
    em_wavelength_array = [455, 525, 590, 690]
    ex_wavelength_array = [405,488, 550, 640]

    channel1 = Channel(
        id = ChannelID('Channel:1:1'),
        emission_wavelength=em_wavelength_array[0],
        emission_wavelength_unit='nm',
        excitation_wavelength=ex_wavelength_array[0],
        excitation_wavelength_unit='nm',
        samples_per_pixel=1
    )
    channel2 = Channel(
        id = ChannelID('Channel:1:2'),
        emission_wavelength=em_wavelength_array[1],
        emission_wavelength_unit='nm',
        excitation_wavelength=ex_wavelength_array[1],
        excitation_wavelength_unit='nm',
        samples_per_pixel=1
    )
    channel3 = Channel(
        id = ChannelID('Channel:1:3'),
        emission_wavelength=em_wavelength_array[2],
        emission_wavelength_unit='nm',
        excitation_wavelength=ex_wavelength_array[2],
        excitation_wavelength_unit='nm',
        samples_per_pixel=1
    )
    channel4 = Channel(
        id = ChannelID('Channel:1:4'),
        emission_wavelength=em_wavelength_array[3],
        emission_wavelength_unit='nm',
        excitation_wavelength=ex_wavelength_array[3],
        excitation_wavelength_unit='nm',
        samples_per_pixel=1
    )

    plane1 = Plane(
        the_c=0,
        the_t=0,
        the_z=0,
        exposure_time= 100,
        exposure_time_unit='ms',
        position_x=1000,
        position_y=1000,
        position_z=0
    )
    plane2 = Plane(
        the_c=1,
        the_t=0,
        the_z=0,
        exposure_time= 100,
        exposure_time_unit='ms',
        position_x=1000,
        position_y=1000,
        position_z=0
    )
    plane3 = Plane(
        the_c=2,
        the_t=0,
        the_z=0,
        exposure_time= 100,
        exposure_time_unit='ms',
        position_x=1000,
        position_y=1000,
        position_z=0
    )
    plane4 = Plane(
        the_c=3,
        the_t=0,
        the_z=0,
        exposure_time= 100,
        exposure_time_unit='ms',
        position_x=1000,
        position_y=1000,
        position_z=0
    )

    image_pixels = Pixels(dimension_order='XYZCT', id=p_id, size_c=4, size_t=1, size_x=5056, size_y=2960, size_z=1,
                          type=p_type,
                          physical_size_x=202,
                          physical_size_x_unit='nm',
                          physical_size_y=202,
                          physical_size_y_unit='nm',
                          physical_size_z=1000,
                          physical_size_z_unit='nm',
                          metadata_only = 'True',
                          planes=[plane1, plane2, plane3, plane4],
                          channels=[channel1, channel2, channel3, channel4])


    image1 = Image(id=i_id, pixels=image_pixels)

    for x in range(0, tile_count):
        ome.images.append(image1)

    tile_counter = 0

    for tile_counter in range(0, tile_count):
        for y in range(0, 2):
            if y % 2 != 0:
                for x in range(3 - 1, -1, -1):
                    file_name = 'z_0_x' + str(x) + '_y_' + str(y) +'_c_DAPI.ome.tif'
                    image_ome = from_tiff(file_name)
                    ome.images[tile_counter].pixels.planes[0].position_x = image_ome.images[0].pixels.planes[0].position_x
                    ome.images[tile_counter].pixels.planes[0].position_y = image_ome.images[0].pixels.planes[0].position_y
                    ome.images[tile_counter].pixels.planes[1].position_x = image_ome.images[0].pixels.planes[0].position_x
                    ome.images[tile_counter].pixels.planes[1].position_y = image_ome.images[0].pixels.planes[0].position_y
                    ome.images[tile_counter].pixels.planes[2].position_x = image_ome.images[0].pixels.planes[0].position_x
                    ome.images[tile_counter].pixels.planes[2].position_y = image_ome.images[0].pixels.planes[0].position_y
                    ome.images[tile_counter].pixels.planes[3].position_x = image_ome.images[0].pixels.planes[0].position_x
                    ome.images[tile_counter].pixels.planes[3].position_y = image_ome.images[0].pixels.planes[0].position_y
                    tile_counter =+ 1


            elif y % 2 == 0:
                for x in range(0, 3):
                    file_name = 'z_0_x' + str(x) + '_y_' + str(y) +'_c_DAPI.ome.tif'
                    image_ome = from_tiff(file_name)
                    ome.images[tile_counter].pixels.planes[0].position_x = image_ome.images[0].pixels.planes[0].position_x
                    ome.images[tile_counter].pixels.planes[0].position_y = image_ome.images[0].pixels.planes[0].position_y
                    ome.images[tile_counter].pixels.planes[1].position_x = image_ome.images[0].pixels.planes[0].position_x
                    ome.images[tile_counter].pixels.planes[1].position_y = image_ome.images[0].pixels.planes[0].position_y
                    ome.images[tile_counter].pixels.planes[2].position_x = image_ome.images[0].pixels.planes[0].position_x
                    ome.images[tile_counter].pixels.planes[2].position_y = image_ome.images[0].pixels.planes[0].position_y
                    ome.images[tile_counter].pixels.planes[3].position_x = image_ome.images[0].pixels.planes[0].position_x
                    ome.images[tile_counter].pixels.planes[3].position_y = image_ome.images[0].pixels.planes[0].position_y
                    tile_counter =+ 1

    xml = to_xml(ome)

    return xml

def mcmicro_image_stack_generator(input_array, xml_metadata, cycle_number, experiment_directory):

    stack_size = input_array.size
    tile_count = input_array.shape[1]

    dapi_im_path = experiment_directory + '\DAPI\Stain\cy_' + str(cycle_number) + '\Tiles'
    a488_im_path = experiment_directory + '\A488\Stain\cy_' + str(cycle_number) + '\Tiles'
    a555_im_path = experiment_directory + '\A555\Stain\cy_' + str(cycle_number) + '\Tiles'
    a647_im_path = experiment_directory + '\A647\Stain\cy_' + str(cycle_number) + '\Tiles'

    mcmicro_path = experiment_directory + r'\mcmicro\raw'

    dapi_input_array = input_array[0]
    a488_input_array = input_array[1]
    a555_input_array = input_array[2]
    a647_input_array = input_array[3]

    mcmicro_stack = np.random.rand(stack_size, 2960, 5056).astype('float16')


    for tile in range(0, tile_count):
        for y in range(0, 2):
            if y % 2 != 0:
                for x in range(3 - 1, -1, -1):
                    best_z_index_dapi = dapi_input_array[tile]
                    best_z_index_a488 = a488_input_array[tile]
                    best_z_index_a555 = a555_input_array[tile]
                    best_z_index_a647 = a647_input_array[tile]
                    dapi_file_name = 'z_' + str(best_z_index_dapi) + '_x' + str(x) + '_y_' + str(y) +'_c_DAPI.ome.tif'
                    a488_file_name = 'z_' + str(best_z_index_a488) + '_x' + str(x) + '_y_' + str(y) +'_c_A488.ome.tif'
                    a555_file_name = 'z_' + str(best_z_index_a555) + '_x' + str(x) + '_y_' + str(y) +'_c_A555.ome.tif'
                    a647_file_name = 'z_' + str(best_z_index_a647) + '_x' + str(x) + '_y_' + str(y) +'_c_A647.ome.tif'

                    base_count_number_stack = tile * 4 - 1

                    os.chdir(dapi_im_path)
                    image = io.imread(dapi_file_name)
                    mcmicro_stack[base_count_number_stack + 1] = image

                    os.chdir(a488_im_path)
                    image = io.imread(a488_file_name)
                    mcmicro_stack[base_count_number_stack + 2] = image

                    os.chdir(a555_im_path)
                    image = io.imread(a555_file_name)
                    mcmicro_stack[base_count_number_stack + 3] = image

                    os.chdir(a647_im_path)
                    image = io.imread(a647_file_name)
                    mcmicro_stack[base_count_number_stack + 4] = image


            elif y % 2 == 0:
                for x in range(0, 3):
                    best_z_index_dapi = dapi_input_array[tile]
                    best_z_index_a488 = a488_input_array[tile]
                    best_z_index_a555 = a555_input_array[tile]
                    best_z_index_a647 = a647_input_array[tile]
                    dapi_file_name = 'z_' + str(best_z_index_dapi) + '_x' + str(x) + '_y_' + str(y) + '_c_DAPI.ome.tif'
                    a488_file_name = 'z_' + str(best_z_index_a488) + '_x' + str(x) + '_y_' + str(y) + '_c_A488.ome.tif'
                    a555_file_name = 'z_' + str(best_z_index_a555) + '_x' + str(x) + '_y_' + str(y) + '_c_A555.ome.tif'
                    a647_file_name = 'z_' + str(best_z_index_a647) + '_x' + str(x) + '_y_' + str(y) + '_c_A647.ome.tif'

                    base_count_number_stack = tile * 4 - 1

                    os.chdir(dapi_im_path)
                    image = io.imread(dapi_file_name)
                    mcmicro_stack[base_count_number_stack + 1] = image

                    os.chdir(a488_im_path)
                    image = io.imread(a488_file_name)
                    mcmicro_stack[base_count_number_stack + 2] = image

                    os.chdir(a555_im_path)
                    image = io.imread(a555_file_name)
                    mcmicro_stack[base_count_number_stack + 3] = image

                    os.chdir(a647_im_path)
                    image = io.imread(a647_file_name)
                    mcmicro_stack[base_count_number_stack + 4] = image

    os.chdir(mcmicro_path)
    mcmicro_file_name = 'cycle-0' + str(cycle_number) + '.ome.tif'
    tf.imwrite(mcmicro_file_name, mcmicro_stack, photometric='minisblack', description=xml_metadata)



def image_optimum_generator(experiment_directory, cycle_number):

    numpy_path = experiment_directory + '/' + 'np_arrays'
    os.chdir(numpy_path)
    full_array = np.load('fm_array.npy', allow_pickle=False)

    numpy_x = full_array[0]
    numpy_y = full_array[1]

    x_tile_count = np.unique(numpy_x).size
    y_tile_count = np.unique(numpy_y).size
    tile_count = x_tile_count * y_tile_count
    color_array = ['DAPI', 'A488', 'A555', 'A647']

    def im_path(index):
        experiment_directory + '/' + str(color_array[index]) + '\Stain\cy_' + str(cycle_number) + '\Tiles'


    os.chdir(im_path(0))
    z_slices = len(os.listdir())/tile_count

    optimum_array = np.random.rand(4, tile_count)
    temp_score = np.random.rand(z_slices).astype('float32')

    for index in range(0, 4):
        os.chdir(im_path(index))
        tile_counter = 0

        for y in range(0, y_tile_count):
            if y % 2 != 0:
                for x in range(x_tile_count - 1, -1, -1):

                    for z in range(0, z_slices):
                        file_name = 'z_' + str(z) + '_x' + str(x) + '_y_' + str(y) +'_c_' + color_array[index] + '.ome.tif'
                        image = io.imread(file_name)
                        temp_score[z] = cycif.focus_bin_generator(image)

                    min_score = np.min(temp_score)
                    optimal_index = np.where(temp_score == min_score)[0][0]
                    optimum_array[index][tile_counter] = temp_score[optimal_index]

                    tile_counter =+ 1


            elif y % 2 == 0:
                for x in range(0, x_tile_count):

                    for z in range(0, z_slices):
                        file_name = 'z_' + str(z) + '_x' + str(x) + '_y_' + str(y) + '_c_' + color_array[
                            index] + '.ome.tif'
                        image = io.imread(file_name)
                        temp_score[z] = cycif.focus_bin_generator(image)

                    min_score = np.min(temp_score)
                    optimal_index = np.where(temp_score == min_score)[0][0]
                    optimum_array[index][tile_counter] = temp_score[optimal_index]

                    tile_counter = + 1

    return optimum_array


experiment_directory = r'D:\Images\AutoCyPlex\folder_structure'



os.chdir(experiment_directory + '\DAPI\Stain\cy_' + str(1) + '\Tiles')



#xml = mcmicro_meta_data_generation(input_array, experiment_directory)
#stack = mcmicro_image_stack_generator(input_array, xml, 1, experiment_directory)

#os.chdir('D:\Images\AutoCyPlex/10-3-23 flat field correction')
#a647_raw = io.imread('nak_atpase_lower_right.tif')