def z_scan_exposure_hook(image, metadata):
    '''
    Method that hooks from autofocus image acquistion calls. It takes image, calculates a focus score for it
    via focus_score method and exports a list that contains both the focus score and the z position it was taken at

    :param numpy image: single image from hooked from acquistion
    :param list[float] metadata: metadata for image

    :return: Nothing
    '''
    z = metadata.pop('ZPosition_um_Intended')  # moves up while taking z stack
    z_intensity_level = cycif.image_percentile_level(image, 0.99)
    intensity.value.append([z_intensity_level, z])

    return
def auto_initial_expose(self, seed_expose, benchmark_threshold, channel, z_range, surface_name):
    '''
    Scans z levels around surface z center and finds brightest z position via z_scan_exposure method.
    Moves machine to that z plane and executes auto_expose method to determine proper exposure. This is meant for
    an initial exposure value for autofocus pruposes.

    :param: str surface_name: string of name of magellan surface to use
    :param list z_centers: list of z points associated with xy points where the slide tilt was compensated for
    :param: str channels: list that contains strings with channel names, for example 'DAPI'

    :return: exposure time: time for inputted channels exposure to be used for autofocus
    :rtype: int
    '''

    [x_pos, y_pos] = self.tissue_center(surface_name)
    core.set_xy_position(x_pos, y_pos)

    z_brightest = z_range[0] + z_range[2]
    core.set_position(z_brightest)

    new_exp = cycif.auto_expose(seed_expose, benchmark_threshold, z_brightest, [channel])

    return new_exp

def image_percentile_level_old(self, image, cut_off_threshold=0.99):
    '''
    Takes in image and cut off threshold and finds pixel value that exists at that threshold point.

    :param numpy array image: numpy array image
    :param float cut_off_threshold: percentile for cut off. For example a 0.99 would disregaurd the top 1% of pixels from calculations
    :return: intensity og pixel that resides at the cut off fraction that was entered in the image
    :rtype: int
    '''
    # cut_off_threshold = 0.9
    threshy_image = image / 10
    thresh = filters.threshold_otsu(threshy_image)
    index = np.where(threshy_image > thresh)
    pixel_values = np.sort(image[index], axis=None)
    pixel_count = int(np.size(pixel_values))
    cut_off_index = int(pixel_count * cut_off_threshold)
    tail_intensity = pixel_values[cut_off_index]

    return tail_intensity

def one_slice_calc_array_solver(self, experiment_directory):

    numpy_path = experiment_directory + '/' + 'np_arrays'
    os.chdir(numpy_path)
    file_name = 'exp_calc_array.npy'
    calc_array = np.load(file_name, allow_pickle=False)
    file_name = 'exp_array.npy'
    exp_array = np.load(file_name, allow_pickle=False)

    y_tiles = np.shape(calc_array)[1]
    x_tiles = np.shape(calc_array)[2]

    goal_int = 65500 * 0.2

    for channel_index in range(0, 4):
        exp_time_list = np.ones([y_tiles, x_tiles])
        int_time_list = np.ones([y_tiles, x_tiles])
        for x in range(0, x_tiles):
            for y in range(0, y_tiles):
                intensity = calc_array[channel_index, y, x, 0, 0]
                exp_time_used = calc_array[channel_index, y, x, 0, 2]
                exp_time_list[y][x] = exp_time_used
                int_time_list[y][x] = intensity

        scaled_int_time_list = np.max(int_time_list) / int_time_list
        scaled_exp_time_list = exp_time_list * scaled_int_time_list
        brightest = np.max(scaled_exp_time_list)
        y_index = np.where(scaled_exp_time_list == brightest)[0][0]
        x_index = np.where(scaled_exp_time_list == brightest)[1][0]
        brightest_int = int_time_list[y_index][x_index]
        brightest_exp = exp_time_list[y_index][x_index]

        scale_factor = goal_int / brightest_int
        new_exp = scale_factor * brightest_exp

        print(channel_index, new_exp)

        exp_array[channel_index] = new_exp

    np.save('exp_array.npy', exp_array)

def calc_array_solver(self, experiment_directory):
    '''
    Uses calc array and 3 point gauss jordan reduction method to solve for projected intensity in focal plane

    :param experiment_directory:
    :return:
    '''

    numpy_path = experiment_directory + '/' + 'np_arrays'
    os.chdir(numpy_path)
    file_name = 'exp_calc_array.npy'
    calc_array = np.load(file_name, allow_pickle=False)

    y_tiles = np.shape(calc_array)[1]
    x_tiles = np.shape(calc_array)[2]

    for channel_index in range(0, 4):
        for x in range(0, x_tiles):
            for y in range(0, y_tiles):
                scores = calc_array[channel_index, y, x, 0:3, 0]
                positions = calc_array[channel_index, y, x, 0:3, 1]
                # three_point_array = np.stack((scores, positions), axis=1)

                # a, b, c, predicted_focus = self.gauss_jordan_solver(three_point_array)
                # peak_int = (-(b * b) / (4 * a) + c)
                # calc_array[channel_index][y][x][3][0] = peak_int
                # calc_array[channel_index][y][x][3][1] = predicted_focus
                calc_array[channel_index][y][x][3][0] = calc_array[channel_index][y][x][0][0]

    np.save(file_name, calc_array)

def calc_array_2_exp_array(self, experiment_directory, fraction_dynamic_range):
    '''
    Takes calc array and determines time to place into exp_array for use. In short, it scales intensities and
    find highest scaled intensity and then scales it from there again to get to the desired dynamic range occupied.
    Employs max time cut off as well

    :param experiment_directory:
    :param fraction_dynamic_range:
    :return:
    '''

    numpy_path = experiment_directory + '/' + 'np_arrays'
    os.chdir(numpy_path)
    calc_array = np.load('exp_calc_array.npy', allow_pickle=False)
    exp_array = np.load('exp_array.npy', allow_pickle=False)

    y_tiles = np.shape(calc_array)[1]
    x_tiles = np.shape(calc_array)[2]

    max_time = 2000

    desired_top_intensity = fraction_dynamic_range * 65535

    for channel_index in range(0, 4):
        predicted_int_list = np.ones([y_tiles, x_tiles])
        exp_time_list = np.ones([y_tiles, x_tiles])
        for x in range(0, x_tiles):
            for y in range(0, y_tiles):
                exp_time_list[y][x] = calc_array[channel_index][y][x][0][2]
                predicted_int_list[y][x] = calc_array[channel_index][y][x][3][0]

        lowest_exp_time = np.min(exp_time_list)
        scaled_exp_list = lowest_exp_time / exp_time_list
        scaled_int_list = predicted_int_list * scaled_exp_list
        highest_intensity = np.max(scaled_int_list)
        index = np.where(scaled_int_list == highest_intensity)
        dimensions = np.shape(index)[0]
        if dimensions == 1:
            highest_intensity = predicted_int_list[index[0][0]]
            exp_time_for_highest_int = exp_time_list[index[0][0]]
        if dimensions == 2:
            highest_intensity = predicted_int_list[index[0][0]][index[1][0]]
            exp_time_for_highest_int = exp_time_list[index[0][0]][index[1][0]]

        scale_up_factor = desired_top_intensity / highest_intensity
        new_exp_time = int(exp_time_for_highest_int * scale_up_factor)

        if new_exp_time > max_time:
            new_exp_time = max_time
        else:
            pass

        if new_exp_time < 50:
            new_exp_time = 50
        else:
            pass

        exp_array[channel_index] = new_exp_time
        print(channel_index, new_exp_time)

    np.save('exp_array.npy', exp_array)

def position_verify(self, z_position):

    difference_range = 1
    current_z = core.get_position()
    difference = abs(z_position - current_z)
    while difference > difference_range:
        core.set_position(z_position)
        current_z = core.get_position()
        difference = abs(z_position - current_z)
        time.sleep(0.5)

def tissue_center(self, mag_surface):
    '''
    take magellan surface and find the xy coordinates of the center of the surface
    :param mag_surface:
    :param magellan:
    :return: x tissue center position and y tissue center position
    :rtype: list[float, float]
    '''
    xy_pos = self.tile_xy_pos(mag_surface)
    x_center = (max(xy_pos[0]) + min(xy_pos[0])) / 2
    y_center = (max(xy_pos[1]) + min(xy_pos[1])) / 2
    return x_center, y_center

def num_surfaces_count(self):
    '''
    Looks at magellan surfaces that start with New Surface in its name, ie. 'New Surface 1' as that is the default generated prefix.

    :param object magellan: magellan object from magellan = Magellan() in pycromanager
    :return: surface_count
    :rtype: int
    '''
    x = 1
    while magellan.get_surface("New Surface " + str(x)) != None:
        x += 1
    surface_count = x - 1
    time.sleep(1)

    return surface_count

def surface_exist_check(self, surface_name):
    '''
    Checks name of surface to see if exists. If it does, returns 1, if not returns 0

    :param object magellan: magellan object from magellan = Magellan() in pycromanager
    :param str surface_name: name of surface to check if exists

    :return: status
    :rtype: int
    '''

    status = 0
    if magellan.get_surface(surface_name) != None:
        status += 1

    return status

def quick_tile_placement(self, z_tile_stack, overlap=10):

    numpy_path = 'E:/folder_structure' + '/' + 'np_arrays'
    os.chdir(numpy_path)
    full_array = np.load('fm_array.npy', allow_pickle=False)

    numpy_x = full_array[0]
    numpy_y = full_array[1]

    x_tile_count = np.unique(numpy_x).size
    y_tile_count = np.unique(numpy_y).size

    height = z_tile_stack[0].shape[0]
    width = z_tile_stack[0].shape[1]
    overlapped_height = int(height * (1 - overlap / 100))
    overlapped_width = int(width * (1 - overlap / 100))

    pna_height = int(y_tile_count * height - int((y_tile_count) * overlap / 100 * height))
    pna_width = int(x_tile_count * width - int((x_tile_count) * overlap / 100 * width))

    pna = np.random.rand(pna_height, pna_width).astype('float16')
    tile_counter = 0

    for y in range(0, y_tile_count):
        if y % 2 != 0:
            for x in range(x_tile_count - 1, -1, -1):
                pna[y * overlapped_height:(y + 1) * overlapped_height,
                x * overlapped_width:(x + 1) * overlapped_width] = z_tile_stack[tile_counter][0:overlapped_height,
                                                                   0:overlapped_width]

                tile_counter += 1


        elif y % 2 == 0:
            for x in range(0, x_tile_count):
                pna[y * overlapped_height:(y + 1) * overlapped_height,
                x * overlapped_width:(x + 1) * overlapped_width] = z_tile_stack[tile_counter][0:overlapped_height,
                                                                   0:overlapped_width]

                tile_counter += 1

    return pna

def quick_tile_optimal_z(self, z_tile_stack):

    z_slice_count = z_tile_stack.shape[0]
    tile_count = z_tile_stack[0].shape[0]

    height = z_tile_stack[0].shape[1]
    width = z_tile_stack[0].shape[2]

    optimal_stack = np.random.rand(tile_count, height, width).astype('float16')
    score_array = np.random.rand(z_slice_count, 1).astype('float32')

    for tile in range(0, tile_count):

        for z in range(0, z_slice_count):
            score_array[z] = cycif.focus_bin_generator(z_tile_stack[z][tile])

        min_score = np.min(score_array)
        optimal_index = np.where(score_array == min_score)[0][0]
        optimal_stack[tile] = z_tile_stack[optimal_index][tile]

    return optimal_stack

def optimal_quick_preview_qt(self, z_tile_stack, channel, cycle, experiment_directory, overlap=10):

    optimal_stack = self.quick_tile_optimal_z(z_tile_stack)
    optimal_qt = self.quick_tile_placement(optimal_stack, overlap)
    optimal_qt_binned = optimal_qt[0:-1:4, 0:-1:4]
    self.save_optimal_quick_tile(optimal_qt_binned, channel, cycle, experiment_directory)

def core_tile_acquire(self, experiment_directory, channel='DAPI'):
    '''
    Makes numpy files that contain all tiles and z slices. Order is z, tiles.

    :param self:
    :param channels:
    :param z_slices:
    :return:
    '''

    numpy_path = experiment_directory + '/' + 'np_arrays'
    os.chdir(numpy_path)
    full_array = np.load('fm_array.npy', allow_pickle=False)
    exp_time_array = np.load('exp_array.npy', allow_pickle=False)

    height_pixels = 2960
    width_pixels = 5056

    numpy_x = full_array[0]
    numpy_y = full_array[1]
    x_tile_count = np.unique(numpy_x).size
    y_tile_count = np.unique(numpy_y).size
    total_tile_count = x_tile_count * y_tile_count
    z_slices = full_array[3][0][0]
    slice_gap = 2

    core.set_xy_position(numpy_x[0][0], numpy_y[0][0])
    time.sleep(1)

    tif_stack = np.random.rand(int(z_slices), total_tile_count, height_pixels, width_pixels).astype('float16')

    if channel == 'DAPI':
        channel_index = 2
        tif_stack_c_index = 0
    if channel == 'A488':
        channel_index = 4
        tif_stack_c_index = 1
    if channel == 'A555':
        channel_index = 6
        tif_stack_c_index = 2
    if channel == 'A647':
        channel_index = 8
        tif_stack_c_index = 3

    numpy_z = full_array[channel_index]
    exp_time = int(exp_time_array[tif_stack_c_index])
    core.set_config("Color", channel)
    core.set_exposure(exp_time)
    tile_counter = 0

    for y in range(0, y_tile_count):
        if y % 2 != 0:
            for x in range(x_tile_count - 1, -1, -1):

                z_end = int(numpy_z[y][x])
                z_start = int(z_end - z_slices * slice_gap)
                core.set_xy_position(numpy_x[y][x], numpy_y[y][x])
                time.sleep(.5)

                z_counter = 0

                for z in range(z_start, z_end, slice_gap):
                    core.set_position(z)
                    time.sleep(0.5)
                    core.snap_image()
                    tagged_image = core.get_tagged_image()
                    pixels = np.reshape(tagged_image.pix,
                                        newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"]])
                    tif_stack[z_counter][tile_counter] = pixels
                    print(core.getRemainingImageCount())

                    z_counter += 1

                tile_counter += 1


        elif y % 2 == 0:
            for x in range(0, x_tile_count):

                z_end = int(numpy_z[y][x])
                z_start = int(z_end - z_slices * slice_gap)
                core.set_xy_position(numpy_x[y][x], numpy_y[y][x])
                time.sleep(.5)

                z_counter = 0

                for z in range(z_start, z_end, slice_gap):
                    core.set_position(z)
                    time.sleep(0.5)
                    core.snap_image()
                    tagged_image = core.get_tagged_image()
                    pixels = np.reshape(tagged_image.pix,
                                        newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"]])

                    tif_stack[z_counter][tile_counter] = pixels

                    z_counter += 1

                tile_counter += 1

    return tif_stack

def quick_tile_all_z_save(self, z_tile_stack, channel, cycle, experiment_directory, stain_bleach, overlap=0):

    z_slice_count = z_tile_stack.shape[0]
    numpy_path = experiment_directory + '/' + 'np_arrays'
    os.chdir(numpy_path)
    full_array = np.load('fm_array.npy', allow_pickle=False)

    numpy_x = full_array[0]
    numpy_y = full_array[1]

    x_tile_count = int(np.unique(numpy_x).size)
    y_tile_count = int(np.unique(numpy_y).size)

    height = int(z_tile_stack.shape[2])
    width = int(z_tile_stack.shape[3])

    pna_height = int(y_tile_count * height - int((y_tile_count) * overlap / 100 * height))
    pna_width = int(x_tile_count * width - int((x_tile_count) * overlap / 100 * width))

    pna_stack = np.random.rand(z_slice_count, pna_height, pna_width).astype('float16')

    for z in range(0, z_slice_count):
        pna = self.quick_tile_placement(z_tile_stack[z], overlap)
        pna_stack[z] = pna

    self.save_quick_tile(pna_stack, channel, cycle, experiment_directory, stain_bleach)

def save_files(self, z_tile_stack, channel, cycle, experiment_directory, Stain_or_Bleach='Stain'):

    save_directory = experiment_directory + '/' + str(channel) + '/' + Stain_or_Bleach + '/' + 'cy_' + str(
        cycle) + '/' + 'Tiles'

    numpy_path = experiment_directory + '/' + 'np_arrays'
    os.chdir(numpy_path)
    full_array = np.load('fm_array.npy', allow_pickle=False)

    numpy_x = full_array[0]
    numpy_y = full_array[1]

    x_tile_count = np.unique(numpy_x).size
    y_tile_count = np.unique(numpy_y).size

    z_tile_count = z_tile_stack.shape[0]

    for z in range(0, z_tile_count):

        tile_counter = 0

        for y in range(0, y_tile_count):
            if y % 2 != 0:
                for x in range(x_tile_count - 1, -1, -1):
                    # meta = self.image_metadata_generation(x, y, channel, experiment_directory)
                    file_name = 'z_' + str(z) + '_x' + str(x) + '_y_' + str(y) + '_c_' + str(channel) + '.tif'
                    image = z_tile_stack[z][tile_counter]
                    os.chdir(save_directory)
                    imwrite(file_name, image, photometric='minisblack')
                    tile_counter += 1

            if y % 2 == 0:
                for x in range(0, x_tile_count):
                    # meta = self.image_metadata_generation(x, y, channel, experiment_directory)
                    file_name = 'z_' + str(z) + '_x' + str(x) + '_y_' + str(y) + '_c_' + str(channel) + '.tif'
                    image = z_tile_stack[z][tile_counter]
                    os.chdir(save_directory)
                    imwrite(file_name, image, photometric='minisblack')
                    tile_counter += 1

def save_tif_stack(self, tif_stack, cycle_number, directory_name):

    add_on_folder = 'cycle_' + str(cycle_number)
    full_directory_path = directory_name + add_on_folder
    try:
        os.mkdir(full_directory_path)
    except:
        pass
    os.chdir(full_directory_path)
    file_name = 'image_array.tif'

    imwrite(file_name, tif_stack,
            bigtiff=True,
            photometric='minisblack',
            compression='zlib',
            compressionargs={'level': 8})


def DAPI_surface_autofocus(self, experiment_directory, z_slices, z_slice_gap, x_frame_size):
    numpy_path = experiment_directory + '/' + 'np_arrays'
    os.chdir(numpy_path)
    file_name = 'fm_array.npy'
    fm_array = np.load(file_name, allow_pickle=False)

    numpy_x = fm_array[0]
    numpy_y = fm_array[1]

    y_tile_count = int(np.shape(numpy_y)[0])
    x_tile_count = int(np.shape(numpy_y)[1])

    center_z = magellan.get_surface('New Surface 1').get_points().get(0).z
    #center_z = -117
    bottom_z = int(center_z - z_slices / 2 * z_slice_gap)
    top_z = int(center_z + z_slices / 2 * z_slice_gap)

    print('bottom z ', bottom_z)

    # find crop range for x dimension

    side_pixels = int(5056 - x_frame_size)

    core.set_config("Color", 'DAPI')
    core.set_exposure(50)

    self.image_capture(experiment_directory, 'DAPI', 50, 0, 0, 0)  # wake up lumencor light engine
    print('wait 10 seconds')

    core.set_xy_position(numpy_x[0][0], numpy_y[0][0])
    time.sleep(1)

    for x in range(0, x_tile_count):
        for y in range(0, y_tile_count):
            core.set_xy_position(numpy_x[y][x], numpy_y[y][x])
            z_stack = np.random.rand(z_slices, 2960, x_frame_size)
            time.sleep(1)
            stack_index = 0

            for z in range(bottom_z, top_z, z_slice_gap):
                core.set_position(z)
                time.sleep(0.5)

                core.snap_image()
                tagged_image = core.get_tagged_image()
                pixels = np.reshape(tagged_image.pix,
                                    newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"]])
                z_stack[stack_index] = pixels[::, side_pixels:x_frame_size + side_pixels]

                stack_index += 1

            print('x', x, 'y', y)
            z_index = self.highest_brenner_index_solver(z_stack)
            focus_z_position = bottom_z + z_index * z_slice_gap
            print('focus', focus_z_position)
            fm_array[2][y][x] = focus_z_position

    np.save('fm_array.npy', fm_array)

def sp_array(self, experiment_directory):
    '''
    Generate super pixel array as defined in powerpoint autofocus network
    :param string experiment_directory:
    :param string channel:
    :param y_tiles:
    :param x_tiles:
    :return:
    '''

    numpy_path = experiment_directory + '/' + 'np_arrays'
    os.chdir(numpy_path)

    fm_array = np.load('fm_array.npy', allow_pickle=False)

    channels = ['DAPI', 'A488', 'A555', 'A647']

    y_tiles = np.shape(fm_array[0])[0]
    x_tiles = np.shape(fm_array[0])[1]

    x_pixel_count = int((x_tiles) * 32)  # 32
    y_pixel_count = int((y_tiles) * 24)  # 24
    sp_array = np.random.rand(5, y_pixel_count, x_pixel_count, 2).astype('float64')
    print(np.shape(sp_array))
    for channel in channels:
        filename = channel + '_sp_array.npy'
        np.save(filename, sp_array)

def tile_subsampler(self, experiment_directory):
    '''
    Outs grid of dimensions [y_tiles, x_tiles] of 1 or 0s that indicate with 1 if the tile is chosen to be sampled. Currently, it samples all tiles.
    :return:
    '''

    numpy_path = experiment_directory + '/' + 'np_arrays'
    os.chdir(numpy_path)

    full_array = np.load('fm_array.npy', allow_pickle=False)
    numpy_x = full_array[0]
    x_tile_count = np.shape(numpy_x)[1]
    y_tile_count = np.shape(numpy_x)[0]

    subsample_grid = np.ones((y_tile_count, x_tile_count))

    return subsample_grid

def image_sub_divider(self, whole_image, y_sections, x_sections):
    '''
    takes in image and breaks into subsections of size y_sections by x_sections.
    :param whole_image:
    :param y_sections:
    :param x_sections:
    :return:
    '''

    y_pixels = np.shape(whole_image)[0]
    x_pixels = np.shape(whole_image)[1]
    sub_divided_image = np.random.rand(y_sections, x_sections, int(y_pixels / y_sections),
                                       int(x_pixels / x_sections)).astype('uint16')
    for y in range(0, y_sections):
        for x in range(0, x_sections):
            # define y and x start and ends subsection of rebuilt image

            y_start = int(y * (y_pixels / y_sections))
            y_end = y_start + int(y_pixels / y_sections)
            x_start = int(x * (x_pixels / x_sections))
            x_end = x_start + int(x_pixels / x_sections)

            sub_divided_image[y][x] = whole_image[y_start:y_end, x_start:x_end]

    return sub_divided_image

def sub_divided_2_brenner_sp(self, experiment_directory, sub_divided_image, channel, point_number, z_position,
                             x_tile_number, y_tile_number):
    '''
    Takes in subdivided image and calculated brenner score at each subsection and properly places into sp_array
    :param experiment_directory:
    :param sub_divided_image:
    :param channel:
    :param point_number:
    :param x_tile_number:
    :param y_tile_number:
    :return:
    '''

    numpy_path = experiment_directory + '/' + 'np_arrays'
    os.chdir(numpy_path)
    file_name = channel + '_sp_array.npy'
    sp_array = np.load(file_name, allow_pickle=False)
    sp_array_slice = sp_array[point_number]

    derivative_jump = 10

    y_subdivisions = 24
    x_subdivisions = 32

    y_offset = int(y_subdivisions * y_tile_number)
    x_offset = int(x_subdivisions * x_tile_number)

    for y in range(y_offset, y_subdivisions + y_offset):
        for x in range(x_offset, x_subdivisions + x_offset):
            score = self.focus_score(sub_divided_image[y - y_offset][x - x_offset], derivative_jump)
            sp_array_slice[y][x][0] = score
            sp_array_slice[y][x][1] = z_position

    np.save(file_name, sp_array)

def sp_array_focus_solver(self, experiment_directory, channel):
    '''
    takes fully populated sp point array and applied 3 point brenner solver method to populate predicted focus

    :param experiment_directory:
    :param channel:
    :return:
    '''

    numpy_path = experiment_directory + '/' + 'np_arrays'
    os.chdir(numpy_path)
    file_name = channel + '_sp_array.npy'
    sp_array = np.load(file_name, allow_pickle=False)

    y_sp_pixels = np.shape(sp_array[0])[0]
    x_sp_pixels = np.shape(sp_array[0])[1]

    for y in range(0, y_sp_pixels):
        for x in range(0, x_sp_pixels):
            scores = sp_array[0:3, y, x, 0]
            scores = scores / np.min(scores)
            positions = sp_array[0:3, y, x, 1]
            three_point_array = np.stack((scores, positions), axis=1)

            a, b, c, predicted_focus = self.gauss_jordan_solver(three_point_array)
            sp_array[3][y][x][0] = predicted_focus

        np.save(file_name, sp_array)

def sp_array_filter(self, experiment_directory, channel):
    '''
    takes sp array for a channel and generates mask that filters out nonsense answers.
    These answers are solutions that exist outside of the scan range and ones that have a low well depth
    :param experiment_directory:
    :param channel:
    :return:
    '''

    numpy_path = experiment_directory + '/' + 'np_arrays'
    os.chdir(numpy_path)
    file_name = channel + '_sp_array.npy'
    sp_array = np.load(file_name, allow_pickle=False)

    y_dim = np.shape(sp_array)[1]
    x_dim = np.shape(sp_array)[2]

    well_depth = np.random.rand(y_dim, x_dim)

    # execute otsu threshold value calc.
    for y in range(0, y_dim):
        for x in range(0, x_dim):
            score_array = sp_array[0:3, y, x, 0]
            score_array = score_array / np.min(score_array)
            depth = np.max(score_array)
            well_depth[y][x] = depth

        threshold = filters.threshold_otsu(
            well_depth)  # sets filter threshold. If dont want otsu, just chang eto number
        # threshold = 1.1
        for y in range(0, y_dim):
            for x in range(0, x_dim):

                position_array = sp_array[0:3, y, x, 1]
                highest_pos = np.max(position_array)
                lowest_pos = np.min(position_array)

                if well_depth[y][x] > threshold and lowest_pos < sp_array[3][y][x][0] < highest_pos:
                    sp_array[4][y][x][0] = 1
                else:
                    sp_array[4][y][x][0] = 0

        np.save(file_name, sp_array)

def plane_2_z(self, coefficents, xy_point):

    a = coefficents[0]
    b = coefficents[1]
    c = coefficents[2]

    z = a * xy_point[0] + b * xy_point[1] + c

    return z

def sp_array_surface_2_fm(self, experiment_directory, channel):
    '''
    Takes fully constructed sp array with mask and predicted focus position and fits plane to points.
    :param experiment_directory:
    :param channel:
    :return:
    '''

    numpy_path = experiment_directory + '/' + 'np_arrays'
    os.chdir(numpy_path)
    fm_array = np.load('fm_array.npy', allow_pickle=False)

    y_tiles = np.shape(fm_array[0])[0]
    x_tiles = np.shape(fm_array[0])[1]

    depth_of_focus = 3  # in microns

    point_list = self.map_2_points(experiment_directory, channel)
    X = point_list[:, 0:2]
    y = point_list[:, 2]

    model = HuberRegressor()
    model.fit(X, y)

    a = (model.predict([[1000, 2500]]) - model.predict([[0, 2500]])) / 1000
    b = (model.predict([[1000, 2500]]) - model.predict([[1000, 1500]])) / 1000
    c = model.predict([[2000, 2000]]) - a * 2000 - b * 2000

    coefficents = [a, b, c]
    print('coefficents', coefficents)

    if channel == 'DAPI':
        channel_index = 2
    if channel == 'A488':
        channel_index = 4
    if channel == 'A555':
        channel_index = 6
    if channel == 'A647':
        channel_index = 8

    # calc number of slices needed

    high_z = self.plane_2_z(coefficents, [0, 0])
    low_z = self.plane_2_z(coefficents, [5056, 2960])
    corner_corner_difference = math.fabs(high_z - low_z)
    # number_planes = int(corner_corner_difference/depth_of_focus) + 1
    number_planes = 9

    for y in range(0, y_tiles):
        for x in range(0, x_tiles):
            # I believe the system to count from upper left hand corner starting at 0, 0

            x_point = 5056 * x + 2528
            y_point = 2060 * y + 1480
            focus_z = self.plane_2_z(coefficents, [x_point, y_point])
            fm_array[channel_index][y][x] = focus_z + (number_planes - 1) / 2 * depth_of_focus
            fm_array[channel_index][y][x] = focus_z
            fm_array[channel_index + 1][y][x] = number_planes

    np.save('fm_array.npy', fm_array)

def map_2_points(self, experiment_directory, channel):

    numpy_path = experiment_directory + '/' + 'np_arrays'
    os.chdir(numpy_path)
    file_name = channel + '_sp_array.npy'
    sp_array = np.load(file_name, allow_pickle=False)
    fm_array = np.load('fm_array.npy', allow_pickle=False)

    y_fm_section = np.shape(sp_array)[1]
    x_fm_section = np.shape(sp_array)[2]

    y_tiles = np.shape(fm_array[0])[0]
    x_tiles = np.shape(fm_array[0])[1]

    y_pixels_per_section = int((y_tiles * 2960) / y_fm_section)
    x_pixels_per_section = int((x_tiles * 2960) / x_fm_section)

    point_list = np.array([])
    point_list = np.expand_dims(point_list, axis=0)

    first_point_counter = 0

    z_array = sp_array[4, :, :, 0] * sp_array[3, :, :, 0]

    for y in range(0, y_fm_section):
        for x in range(0, x_fm_section):

            x_coord = x * x_pixels_per_section
            y_coord = y * y_pixels_per_section
            z_coord = z_array[y][x]
            single_point = np.array([x_coord, y_coord, z_coord])
            single_point = np.expand_dims(single_point, axis=0)
            if first_point_counter == 0 and z_coord != 0:
                point_list = np.append(point_list, single_point, axis=1)
                first_point_counter = 1
            elif first_point_counter == 1 and z_coord != 0:
                point_list = np.append(point_list, single_point, axis=0)

    point_list = Points(point_list)

    return point_list

def gauss_jordan_solver(self, three_point_array):
    '''
    Takes 3 points and solves quadratic equation in a generic fashion and returns constants and
    solves for x in its derivative=0 equation

    :param numpy[float, float] three_point_array: numpy array that contains pairs of [focus_score, z]
    :results: z coordinate for in focus plane
    :rtype: float
    '''

    x1 = three_point_array[0][1]
    x2 = three_point_array[1][1]
    x3 = three_point_array[2][1]
    score_0 = three_point_array[0][0]
    score_1 = three_point_array[1][0]
    score_2 = three_point_array[2][0]

    aug_matrix = np.array([[x1 * x1, x1, 1, score_0], [x2 * x2, x2, 1, score_1], [x3 * x3, x3, 1, score_2]])

    aug_matrix[0] = aug_matrix[0] / (aug_matrix[0, 0] + 0.000001)
    aug_matrix[1] = -(aug_matrix[1, 0]) * aug_matrix[0] + aug_matrix[1]
    aug_matrix[2] = -(aug_matrix[2, 0]) * aug_matrix[0] + aug_matrix[2]

    aug_matrix[1] = -(aug_matrix[1, 1] - 1) / (aug_matrix[2, 1] + 0.0000001) * aug_matrix[2] + aug_matrix[1]
    aug_matrix[0] = -aug_matrix[0, 1] * aug_matrix[1] + aug_matrix[0]
    aug_matrix[2] = -aug_matrix[2, 1] * aug_matrix[1] + aug_matrix[2]

    aug_matrix[2] = aug_matrix[2] / (aug_matrix[2, 2] + 0.000001)
    aug_matrix[0] = -aug_matrix[0, 2] * aug_matrix[2] + aug_matrix[0]
    aug_matrix[1] = -aug_matrix[1, 2] * aug_matrix[2] + aug_matrix[1]

    a = aug_matrix[0, 3]
    b = aug_matrix[1, 3]
    c = aug_matrix[2, 3]

    derivative = -b / (2 * (a + 0.00001))

    return a, b, c, derivative

def highest_brenner_index_solver(self, image_stack):

    slice_count = int(np.shape(image_stack)[0])
    scores = np.random.rand(slice_count)
    slice_array = np.random.rand(slice_count)

    for x in range(0, slice_count):
        image = image_stack[x]
        score = self.focus_score(image, 17)
        scores[x] = score
        slice_array[x] = x

    highest_score = np.max(scores)
    index = np.where(scores == highest_score)[0][0]

    return index
def image_exp_scorer(self, experiment_directory, image_stack, channel, percentage_cut_off, target_percentage):

    # import exp_calc_array
    numpy_path = experiment_directory + '/' + 'np_arrays'
    os.chdir(numpy_path)
    exp_calc_array = np.load('exp_calc_array.npy', allow_pickle=False)
    fm_array == np.load('fm_array.npy', allow_pickle=False)

    x_tiles = np.shape(exp_calc_array[0][0])[1]
    y_tiles = np.shape(exp_calc_array[0][0])[0]
    tissue_fm = fm_array[10]

    # find desired intensity (highest)
    desired_int = 65000 * target_percentage

    if channel == 'DAPI':
        channel_index = 0
    if channel == 'A488':
        channel_index = 1
    if channel == 'A555':
        channel_index = 2
    if channel == 'A647':
        channel_index = 3

    for y in range(0, y_tiles):
        for x in range(0, x_tiles):

            if tissue_fm[y][x] ==  1:
                score = self.image_percentile_level(image_stack[y][x],
                                               cut_off_threshold=percentage_cut_off)
                exp_calc_array[channel_index][1][y][x] = score

                original_exp_time = exp_calc_array[channel_index][0][y][x]
                score_per_millisecond_exp = score / original_exp_time
                projected_exp_time = int(desired_int / score_per_millisecond_exp)
                exp_calc_array[channel_index][2][y][x] = projected_exp_time

            if tissue_fm[y][x] == 0:
                exp_calc_array[channel_index][2][y][x] = 5000

    os.chdir(numpy_path)
    np.save('exp_calc_array.npy', exp_calc_array)

def calculate_exp_array(self, experiment_directory):

    # import exp_calc_array
    numpy_path = experiment_directory + '/' + 'np_arrays'
    os.chdir(numpy_path)
    exp_calc_array = np.load('exp_calc_array.npy', allow_pickle=False)
    exp_array = np.load('exp_array.npy', allow_pickle=False)

    for channel_index in range(1, 4):
        # find lowest exp time
        lowest_exp = np.min(exp_calc_array[channel_index, 2, ::, ::])

        if lowest_exp < 2000:
            exp_array[channel_index] = int(lowest_exp)
        if lowest_exp > 2000:
            exp_array[channel_index] = int(2000)
        if lowest_exp < 50:
            exp_array[channel_index] = int(50)



        print('channel_index', channel_index, 'time', exp_array[channel_index])

    exp_array[0] = 230
    np.save('exp_array.npy', exp_array)