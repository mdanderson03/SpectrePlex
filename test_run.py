#from autocyplex import *
#microscope = cycif() # initialize cycif object
#arduino = arduino()
import numpy as np
import os
from skimage import io, filters
import matplotlib.pyplot as plt


def image_percentile_level(image, cut_off_threshold):
    '''
    Takes in image and cut off threshold and finds pixel value that exists at that threshold point.

    :param numpy array image: numpy array image
    :param float cut_off_threshold: percentile for cut off. For example a 0.99 would disregaurd the top 1% of pixels from calculations
    :return: intensity og pixel that resides at the cut off fraction that was entered in the image
    :rtype: int
    '''
    pixel_values = np.sort(image, axis=None)
    pixel_count = int(np.size(pixel_values))
    cut_off_index = int(pixel_count * cut_off_threshold)
    tail_intensity = pixel_values[cut_off_index]

    return tail_intensity


def exposure_hook(image, metadata):
    '''
    Hook for expose method. Returns metric based on image_percentile_level method.

    :param numpy array image: image from expose
    :param metadata: metadata from exposure
    :return: nothing
    '''

    global level
    level = cycif.image_percentile_level(image, 0.85)

    return


def expose(seed_exposure, channel='DAPI'):
    '''
    Runs entire auto exposure algorithm in current XY position. Gives back predicted
    in exposure level via image_percentile_level method with the exposure_hook

    :param list[int, int, int] z_range: defines range and stepsize with [z start, z end, z step]
    :param str channel: channel to autofocus with

    :return: z coordinate for in focus plane
    :rtype: float
    '''

    with Acquisition(directory='C:/Users/CyCIF PC/Desktop/test_images/', name='trash', show_display=False,
                     image_process_fn=cycif.exposure_hook) as acq:
        # Create some acquisition events here:

        event = {'channel': {'group': 'Color', 'config': channel}, 'exposure': seed_exposure}
        acq.acquire(event)

    return level


def auto_expose(self, directory, seed_expose, benchmark_threshold, channels=['DAPI', 'A488', 'A555', 'A647']):
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

    numpy_path = experiment_directory + '/' + 'np_arrays'
    os.chdir(numpy_path)
    full_array = np.load('fm_array.npy', allow_pickle=False)

    new_x = full_array[0][0][0]
    new_y = full_array[1][0][0]
    z_position_channel_list = [full_array[2][0][0], full_array[3][0][0], full_array[4][0][0], full_array[5][0][0]]
    core.set_xy_position(new_x, new_y)

    bandwidth = 0.1
    sat_max = 65000
    exp_time_limit = 2000
    exposure_array = [10, 10, 10, 10]  # dapi, a488, a555, a647

    for fluor_channel in channels:

        if fluor_channel == 'DAPI':
            exp_index = 0
        if fluor_channel == 'A488':
            exp_index = 1
        if fluor_channel == 'A555':
            exp_index = 2
        if fluor_channel == 'A647':
            exp_index = 3

        z_pos = z_position_channel_list[exp_index]
        core.set_position(z_pos)

        intensity = cycif.expose(seed_expose, fluor_channel)
        new_exp = seed_expose
        while intensity < (1 - bandwidth) * benchmark_threshold or intensity > (
                1 + bandwidth) * benchmark_threshold:
            if intensity < benchmark_threshold:
                new_exp = benchmark_threshold / intensity * new_exp
                if new_exp >= exp_time_limit:
                    new_exp = exp_time_limit
                    break
                else:
                    intensity = cycif.expose(new_exp, fluor_channel)
            elif intensity > benchmark_threshold and intensity < sat_max:
                new_exp = benchmark_threshold / intensity * new_exp
                intensity = cycif.expose(new_exp, fluor_channel)
            elif intensity > sat_max:
                new_exp = new_exp / 10
                intensity = cycif.expose(new_exp, fluor_channel)
            elif new_exp >= sat_max:
                new_exp = sat_max
                break

        exposure_array[exp_index] = new_exp

    return exposure_array


def median_fm_filter(self, full_array, channel):
    if channel == 'DAPI':
        channel_index = 2
    if channel == 'A488':
        channel_index = 3
    if channel == 'A555':
        channel_index = 4
    if channel == 'A647':
        channel_index = 5

    full_array[channel_index] = median(full_array[channel_index])

    return full_array


def auto_focus(self, z_range, exposure_time, channel='DAPI'):
    '''
    Runs entire auto focus algorithm in current XY position. Gives back predicted
    in focus z position via focus_score method which is the Brenner score.

    :param list[int, int, int] z_range: defines range and stepsize with [z start, z end, z step]
    :param str channel: channel to autofocus with
    :return: z coordinate for in focus plane
    :rtype: float
    '''
    global brenner_scores
    brenner_scores = brenner()  # found using class for brenner was far superior to using it as a global variable.
    # I had issues with the image process hook function not updating brenner as a global variable

    with Acquisition(directory='C:/Users/CyCIF PC/Desktop/test_images', name='trash',
                     show_display=False,
                     image_process_fn=cycif.image_process_hook) as acq:
        events = multi_d_acquisition_events(channel_group='Color',
                                            channels=[channel],
                                            z_start=z_range[0],
                                            z_end=z_range[1],
                                            z_step=z_range[2],
                                            order='zc', channel_exposures_ms=[exposure_time])
        acq.acquire(events)

    optimal_score_array = self.score_array_generator()
    [a, b, c, derivative] = self.gauss_jordan_solver(optimal_score_array)
    z_ideal = derivative

    return z_ideal


def image_process_hook(image, metadata):
    '''
    Method that hooks from autofocus image acquistion calls. It takes image, calculates a focus score for it
    via focus_score method and exports a list that contains both the focus score and the z position it was taken at

    :param numpy image: single image from hooked from acquistion
    :param list[float] metadata: metadata for image

    :return: Nothing
    '''

    z = metadata.pop('ZPosition_um_Intended')  # moves up while taking z stack
    image_focus_score = cycif.focus_bin_generator(image)
    brenner_scores.value.append([image_focus_score, z])

    return


def score_array_generator(self):
    optimal_score_array = [[brenner_scores.value[0][0], brenner_scores.value[0][1]], [brenner_scores.value[1][0],
                                                                                      brenner_scores.value[1][1]],
                           [brenner_scores.value[2][0], brenner_scores.value[2][1]]]

    return optimal_score_array


def bin_selector(self):
    '''
    takes array of brenner scores and determines optimal bin level and outputs corresponding focus scores with z positions
    :return: list [optimal bin focus score1, z position1], [], [], ...
    '''

    bin_levels = [4, 8, 16, 32, 64, 128]
    range_array = []

    for x in range(0, len(brenner_scores.value[0][0])):
        score_array = [brenner_scores.value[0][0][x], brenner_scores.value[1][0][x], brenner_scores.value[2][0][x]]
        max_value = max(score_array)
        min_value = min(score_array)
        range_array.append(max_value - min_value)

    max_range = max(range_array)
    max_index = range_array.index(max_range)
    print(bin_levels[max_index])

    optimal_score_array = [[brenner_scores.value[0][0][max_index], brenner_scores.value[0][1]],
                           [brenner_scores.value[1][0][max_index],
                            brenner_scores.value[1][1]],
                           [brenner_scores.value[2][0][max_index], brenner_scores.value[2][1]]]

    return optimal_score_array


def focus_bin_generator(image):
    '''
    takes image and calculates brenner scores for various bin levels of 64, 32, 16, 8 and 4 and outputs them.
    :param image: numpy array 2D
    :return: list of brenner scores of binned images
    '''

    # bin_levels = [4,8,16,32,64]
    # focus_scores = [0]*len(bin_levels)

    focus_score = cycif.focus_score(image, 1)

    return focus_score


def save_optimal_quick_tile(self, image, channel, cycle, experiment_directory):
    file_name = 'quick_tile_' + str(channel) + '_' + 'cy' + str(cycle) + '.tif'

    top_path = experiment_directory + '/' + 'Quick_Tile'

    os.chdir(top_path)
    imwrite(file_name, image, photometric='minisblack')


def save_quick_tile(self, image, channel, cycle, experiment_directory, Stain_or_Bleach='Stain'):
    file_name = 'quick_tile_' + str(channel) + '_' + 'cy' + str(cycle) + '.tif'

    top_path = experiment_directory + '/' + 'Quick_Tile'
    bottom_path = experiment_directory + '/' + str(channel) + '/' + Stain_or_Bleach + '/' + 'cy_' + str(
        cycle) + '/' + 'Quick_Tile'

    os.chdir(top_path)
    imwrite(file_name, image, photometric='minisblack')

    os.chdir(bottom_path)
    imwrite(file_name, image, photometric='minisblack')


def image_metadata_generation(self, tile_x_number, tile_y_number, channel, experiment_directory):
    ome = OME()

    numpy_path = experiment_directory + '/' + 'np_arrays'
    os.chdir(numpy_path)
    full_array = np.load('fm_array.npy', allow_pickle=False)
    exp_time_array = np.load('exp_array.npy', allow_pickle=False)

    numpy_x = full_array[0]
    numpy_y = full_array[1]

    stage_x = numpy_x[tile_y_number][tile_x_number]
    stage_y = numpy_y[tile_y_number][tile_x_number]

    if channel == 'DAPI':
        channel_array_index = 2
        ex_wavelength = 405
        em_wavelength = 455
    if channel == 'A488':
        channel_array_index = 3
        ex_wavelength = 488
        em_wavelength = 525
    if channel == 'A555':
        channel_array_index = 4
        ex_wavelength = 540
        em_wavelength = 590
    if channel == 'A647':
        channel_array_index = 5
        ex_wavelength = 640
        em_wavelength = 690

    microscope_mk4 = Microscope(
        manufacturer='ASI',
        model='AutoCyPlex',
        serial_number='CFIC-1',
    )

    objective_16x = Objective(
        manufacturer='Nikon',
        model='16x water dipping',
        nominal_magnification=21.0,
    )

    instrument = Instrument(
        microscope=microscope_mk4,
        objectives=[objective_16x],
    )

    p_type = PixelType('uint16')
    p_id = PixelsID('Pixels:0')
    i_id = ImageID('Image:0')
    c_id = ChannelID('Channel:1:' + str(channel_array_index - 2))

    channel = Channel(
        id=c_id,
        emission_wavelength=em_wavelength,
        emission_wavelength_unit='nm',
        excitation_wavelength=ex_wavelength,
        excitation_wavelength_unit='nm',
        samples_per_pixel=1
    )

    plane = Plane(
        the_c=0,
        the_t=0,
        the_z=0,
        exposure_time=exp_time_array[channel_array_index - 2],
        exposure_time_unit='ms',
        position_x=stage_x,
        position_y=stage_y,
        position_z=1
    )

    image_pixels = Pixels(dimension_order='XYZCT', id=p_id, size_c=1, size_t=1, size_x=5056, size_y=2960, size_z=1,
                          type=p_type,
                          physical_size_x=202,
                          physical_size_x_unit='nm',
                          physical_size_y=202,
                          physical_size_y_unit='nm',
                          physical_size_z=1000,
                          physical_size_z_unit='nm',
                          planes=[plane],
                          channels=[channel], metadata_only='True')

    image1 = Image(id=i_id, pixels=image_pixels)
    ome.images.append(image1)
    ome.instruments.append(instrument)
    ome.images[0].instrument_ref = InstrumentRef(id=instrument.id)

    metadata = ome_types.to_xml(ome)

    return metadata


def z_scan_exposure(self, z_range, seed_exposure, channel='DAPI'):
    '''
    Goes through z scan to over z range to determine maximum intensity value in said range and step size.

    :param list[int, int, int] z_range: defines range and stepsize with [z start, z end, z step]
    :param str channel: channel to autofocus with

    :return: z coordinate for in focus plane
    :rtype: float
    '''
    global intensity
    intensity = exp_level()  # found using class for exp_level was far superior to using it as a global variable.
    # I had issues with the image process hook function not updating brenner as a global variable

    with Acquisition(directory='C:/Users/CyCIF PC/Desktop/test_images', name='trash',
                     show_display=False,
                     image_process_fn=self.z_scan_exposure_hook) as acq:
        events = multi_d_acquisition_events(channel_group='Color',
                                            channels=[channel],
                                            z_start=z_range[0],
                                            z_end=z_range[1],
                                            z_step=z_range[2],
                                            order='zc', channel_exposures_ms=[seed_exposure])
        acq.acquire(events)
    intensity_list = [x[0] for x in intensity.value]
    brightest = max(intensity_list)
    z_level_brightest_index = intensity_list.index(brightest)
    z_level_brightest = intensity.value[z_level_brightest_index][1]

    return z_level_brightest


def tile_xyz_gen(self, tile_points_xy, z_focused):
    '''
    Takes dictionary of XY coordinates applies inputted same focused z postion to all of them to make a xyz array

    :param dictionary tile_points_xy: dictionary containing all XY coordinates. In the form: {{x:(int)}, {y:(int)}}
    :param float z_focused: z position where surface is in focus

    :return: XYZ points where XY are stage coords and Z is in focus coordinate. {{x:(int)}, {y:(int)}, {z:(float)}}
    :rtype: dictionary
    '''

    z_temp = []
    num = len(tile_points_xy['x'])
    for q in range(0, num):
        z_temp.append(z_focused)
    tile_points_xy['z'] = z_temp
    surface_points_xyz = tile_points_xy

    return surface_points_xyz


def numpy_xyz_gen(self, tile_points_xy, z_focused):
    '''
    Takes dictionary of XY coordinates applies inputted same focused z postion to all of them to make a xyz array

    :param dictionary tile_points_xy: dictionary containing all XY coordinates. In the form: {{x:(int)}, {y:(int)}}
    :param float z_focused: z position where surface is in focus

    :return: XYZ points where XY are stage coords and Z is in focus coordinate. {{x:(int)}, {y:(int)}, {z:(float)}}
    :rtype: numpy array
    '''

    z_temp = []
    x_temp = np.array(tile_points_xy['x'])
    y_temp = np.array(tile_points_xy['y'])
    num = len(tile_points_xy['x'])
    for q in range(0, num):
        z_temp.append(z_focused)
    z_temp_numpy = np.array(z_temp)

    xyz = np.hstack([x_temp[:, None], y_temp[:, None], z_temp_numpy[:, None]])

    return xyz


def focused_surface_generate_xyz(self, new_surface_name, surface_points_xyz):
    '''
    Generates new micro-magellan surface with name new_surface_name and uses all points in surface_points_xyz dictionary
    as interpolation points. If surface already exists, then it checks for it and updates its xyz points.

    :param dictionary surface_points_xyz: all paired-wise points of XYZ where XY are stage coords and Z is in focus. {{x:(int)}, {y:(int)}, {z:(float)}}
    :param str new_surface_name: name that generated surface will have

    :return: Nothing
    '''
    creation_status = self.surface_exist_check(magellan,
                                               new_surface_name)  # 0 if surface with that name doesnt exist, 1 if it does
    if creation_status == 0:
        magellan.create_surface(new_surface_name)  # make surface if it doesnt already exist

    focused_surface = magellan.get_surface(new_surface_name)
    num = len(surface_points_xyz['x'])
    for q in range(0, num):
        focused_surface.add_point(surface_points_xyz['x'][q], surface_points_xyz['y'][q], surface_points_xyz['z'][q])

    return


def focused_surface_acq_settings(self, exposure, original_surface_name, surface_name, acq_surface_num, channel):
    '''
    Takes already generated micro-magellan surface with name surface_name, sets it as a 2D surface, what channel group
    to use, sets exposure levels for all 4 channels and where to make the savings directory.

    :param numpy array exposure: a numpy array of [dapi_exposure, a488_exposure, a555_exposure, a647_exposure] with exposure times in milliseconds
    :param str surface_name: name of micro-magellan surface that is to be used as the space coordinates for this acquistion event
    :param object magellan: magellan object from magellan = Magellan() in pycromanager
    :param list[int] channel_offsets: list of offsets with respect to nuclei. Order is [DAPI, A488, A555, A647]
    :param int acq_surface_num: number that corresponds to the surface number ie. 1 in 'New Surface 1' and so on

    :return: Nothing
    '''
    acquisition_name = channel + ' surface ' + str(acq_surface_num)  # make name always be channel + surface number

    i = 0
    error = 0
    acquisition_name_array = []
    while error == 0:  # create an array of all names contained in acquistion events in MM
        try:
            acquisition_name_array.append(magellan.get_acquisition_settings(i).name_)
            i += 1
        except:
            error = 1

    try:  # see if acquistion name is within that array, if not create new event
        name_index = acquisition_name_array.index(acquisition_name)
        acq_settings = magellan.get_acquisition_settings(name_index)
    except:
        magellan.create_acquisition_settings()
        acq_settings = magellan.get_acquisition_settings(i)

    acq_settings.set_acquisition_name(
        acquisition_name)  # make same name as in focused_surface_generate function (all below as well too)
    acq_settings.set_acquisition_space_type('2d_surface')
    acq_settings.set_xy_position_source(original_surface_name)
    acq_settings.set_surface(surface_name)
    acq_settings.set_channel_group('Color')
    acq_settings.set_use_channel('DAPI', False)  # channel_name, use
    acq_settings.set_use_channel('A488', False)  # channel_name, use
    acq_settings.set_use_channel('A555', False)  # channel_name, use
    acq_settings.set_use_channel('A647', False)  # channel_name, use
    acq_settings.set_use_channel('A750', False)  # channel_name, use
    acq_settings.set_use_channel(channel, True)  # channel_name, use
    acq_settings.set_channel_exposure('DAPI', int(exposure[
                                                      0]))  # channel_name, exposure in ms can auto detect channel names and iterate names with exposure times
    acq_settings.set_channel_exposure('A488', int(exposure[1]))  # channel_name, exposure in ms
    acq_settings.set_channel_exposure('A555', int(exposure[2]))  # channel_name, exposure in ms
    acq_settings.set_channel_exposure('A647', int(exposure[3]))  # channel_name, exposure in ms
    acq_settings.set_channel_exposure('A750', 10)  # channel_name, exposure in ms

    return


def surface_acquire(self, channels=['DAPI', 'A488', 'A555', 'A647']):
    '''
    Takes generated micro-magellan surface with name: surface_name and generates new micro-magellan surface with name:
    new_focus_surface_name and makes an acquistion event after latter surface and auto exposes DAPI, A488, A555 and A647 channels.
    Takes autofocus from center tile in surface and applies value to every other tile in surface

    :param MMCore_Object core: Object made from Bridge.core()
    :param object magellan: object created via = bridge.get_magellan()
    :param: list[str] channels: list that contains strings with channel names

    :return: Nothing
    '''

    num_surfaces = self.num_surfaces_count(magellan)  # checks how many 'New Surface #' surfaces exist. Not actual total

    for channel in channels:

        for x in range(1, num_surfaces + 1):
            surface_name = 'New Surface ' + str(x)
            new_focus_surface_name = 'Focused Surface ' + str(channel)

            tile_surface_xy = self.tile_xy_pos(surface_name,
                                               magellan)  # pull center tile coords from manually made surface
            auto_focus_exposure_time = self.auto_initial_expose(core, magellan, 50, 6500, channel, surface_name)

            z_center = magellan.get_surface(surface_name).get_points().get(0).z
            z_range = [z_center - 10, z_center + 10, 1]

            z_focused = self.auto_focus(z_range, auto_focus_exposure_time,
                                        channel)  # here is where autofocus results go. = auto_focus
            surface_points_xyz = self.tile_xyz_gen(tile_surface_xy,
                                                   z_focused)  # go to each tile coord and autofocus and populate associated z with result
            self.focused_surface_generate_xyz(magellan, new_focus_surface_name,
                                              surface_points_xyz)  # will generate surface if not exist, update z points if exists

            exposure_array = self.auto_expose(core, magellan, auto_focus_exposure_time, 6500, z_focused, [channel],
                                              surface_name)
            self.focused_surface_acq_settings(exposure_array, surface_name, new_focus_surface_name, magellan, x,
                                              channel)

    return


def xyz_acquire(self, xyz_array, channel, exposure_time, cycle_number, directory_name='E:/images/'):
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

    with Acquisition(directory=full_directory_path, name=channel) as acq:
        events = multi_d_acquisition_events(channel_group='Color', channels=[channel], xyz_positions=xyz_array,
                                            channel_exposures_ms=[exposure_time])
        acq.acquire(events)
        acq.await_completion()


def mm_focus_hook(self, event):
    z_center = core.get_position()
    core.snap_image()
    tagged_image = core.get_tagged_image()
    # z_range = [z_center - 50, z_center + 50, 20]
    # exposure_time = core.get_exposure()
    # z_focused_position = self.auto_focus(z_range, exposure_time, 'DAPI')
    # core.set_position(z_focused_position)
    time.sleep(0.5)
    print(z_center)

    return event


def micro_magellan_acq_auto_focus(self):
    '''
    going through micromagellan list and acquire each one while autofocusing at each tile
    '''

    for x in range(0, 100):
        try:
            acq = MagellanAcquisition(magellan_acq_index=x, post_hardware_hook_fn=self.core_snap_auto_focus)
            acq.await_completion()
            print('acq ' + str(x) + ' finished')
        except:
            continue

        print('acq ' + str(x) + ' finished')


def micro_magellan_acq(self):
    '''
    go through micromagellan list and acquire each one
    '''

    for x in range(0, 100):
        try:
            acq = MagellanAcquisition(magellan_acq_index=x)
            acq.await_completion()
            print('acq ' + str(x) + ' finished')
        except:
            continue

        print('acqs ' + ' finished')
