from autocyplex import *
from pycromanager import Core, Studio, Magellan
microscope = cycif() # initialize cycif object



experiment_directory = r'E:\auto'



microscope.file_structure(experiment_directory, 1)
xy_points = microscope.tile_xy_pos('New Surface 1')
xyz_points = microscope.nonfocus_tile_DAPI(xy_points)
array = microscope.tile_pattern(xyz_points)

off_array = [-3,-5,-6]
fm_array = microscope.fm_channel_initial(experiment_directory, array, off_array)
microscope.sp_array(experiment_directory, 'DAPI')
sample_grid = microscope.tile_subsampler(experiment_directory)
print(fm_array[2])

sample_mid_z = fm_array[2][0][0]


scan_range = 20
sample_span = [sample_mid_z - scan_range/2, sample_mid_z , sample_mid_z + scan_range/2]


for x in range(0, 3):
    for y in range(0, 3):
        for points in range(0, 3):
            z_slice = int(sample_span[points])
            im = microscope.image_capture(experiment_directory, 'DAPI',25, y, x, z_slice)
            #cycif.auto_exposure_calculation(im, 0.99, 'DAPI', points, z_slice, x, y)
            div_im = microscope.image_sub_divider(im, 2, 2)
            microscope.sub_divided_2_brenner_sp(experiment_directory, div_im, 'DAPI', points, z_slice, x, y)

#cycif.calc_array_solver(experiment_directory, 'DAPI')
#cycif.calc_array_2_exp_array(experiment_directory, 'DAPI', 0.5)
microscope.sp_array_focus_solver(experiment_directory, 'DAPI')
#microscope.sp_array_filter(experiment_directory, 'DAPI')
microscope.sp_array_surface_2_fm(experiment_directory, 'DAPI')

numpy_path = experiment_directory + '/' + 'np_arrays'

os.chdir(numpy_path)
fm_array = np.load('fm_array.npy', allow_pickle=False)

print(fm_array[2])

'''
def focus_expose(self, experiment_directory, channel, ):
    experiment_directory = r'D:\Images\AutoCyPlex\7_18_23 focus z stack sample set\dapi_full_frame_good_tissue_1um_1'
    os.chdir(experiment_directory)
    filename = 'dapi_full_frame_good_tissue_1um_1_MMStack_3-Pos000_000.ome.tif'
    stack = io.imread(filename, plugin="tifffile")
    start_time = time.time()

    cycif.sp_array(experiment_directory, 'DAPI', 3, 4)  # pass
    cycif.exp_calc_array(experiment_directory, 'DAPI', 3, 4)
    sample_grid = cycif.tile_subsampler(experiment_directory)  # pass

    sample_high_z = 25
    scan_range = 17
    sample_span = [sample_high_z - scan_range, sample_high_z - scan_range / 2, sample_high_z]

    for points in range(0, 3):
        z_slice = int(sample_span[points])
        for y in range(0, 3):
            for x in range(0, 4):
                im = image_capture(stack, y, x, z_slice)
                cycif.auto_exposure_calculation(im, 0.99, 'DAPI', points, z_slice, x, y)
                div_im = cycif.image_sub_divider(im, 24, 32)
                cycif.sub_divided_2_brenner_sp(experiment_directory, div_im, 'DAPI', points, z_slice, x, y)

    cycif.calc_array_solver(experiment_directory, 'DAPI')
    cycif.calc_array_2_exp_array(experiment_directory, 'DAPI', 0.5)
    cycif.sp_array_focus_solver(experiment_directory, 'DAPI')
    cycif.sp_array_filter(experiment_directory, 'DAPI')
    cycif.sp_array_surface_2_fm(experiment_directory, 'DAPI')

    stop_time = time.time()
    print(stop_time - start_time)

    re_image = cycif.fm_rebuilt_image(stack)

    os.chdir(r'D:\Images\AutoCyPlex\7_18_23 focus z stack sample set\dapi_full_frame_good_tissue_1um_1')
    io.imsave('dapi_rebuild.tif', re_image)

'''












