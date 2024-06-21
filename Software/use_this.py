from autocyplex import *
from optparse import OptionParser
microscope = cycif() # initialize cycif object
experiment_directory = r'E:\6-6-24 marco'
pump = fluidics(experiment_directory, 6, 13, flow_control=1)
core = Core()

z_slices = 5
x_frame_size = 2960
offset_array = [0, -7, -7, -6]
focus_position = 155



#for cycle in range(6, 7):
#    microscope.full_cycle(experiment_directory, cycle, offset_array, cycle, pump, z_slices, focus_position=focus_position)

#microscope.post_acquisition_processor(experiment_directory, x_frame_size, rolling_ball=0)
#microscope.brightness_uniformer(experiment_directory, cycle_number=1)


times = np.array([10, 20, 30, 50, 100, 150, 250, 400, 800, 2000])
number_times = np.shape(times)[0]
print(times)

side_pixel_count = int((5056 - x_frame_size)/2)

time_img_array = np.random.rand(number_times, 2960, x_frame_size).astype('uint16')

for x in range(0, number_times):
    print('pic_', str(x), '_taken')
    core.set_exposure(int(times[x]))
    time.sleep(0.1)
    core.set_config("amp", 'high')
    core.snap_image()
    tagged_image = core.get_tagged_image()
    pixels = np.reshape(tagged_image.pix, newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"]])
    pixels = np.nan_to_num(pixels, posinf=0)
    #pixels[pixels > (65234)] = 0
    time_img_array [x] = pixels[::, side_pixel_count:side_pixel_count + x_frame_size]

os.chdir(r'C:\Users\CyCIF PC\Desktop\linearity')
io.imsave('linearity.tif', time_img_array)

os.chdir(r'C:\Users\CyCIF PC\Desktop\linearity')
time_img_array = io.imread('linearity.tif')
times = np.array([10, 20, 30, 50, 100, 150, 250, 400, 800, 2000])
number_times = np.shape(times)[0]

wb = Workbook()
ws = wb.active



y_axis = np.zeros(number_times)
ws.cell(row=1, column=1).value = 'Time(ms)'
ws.cell(row=1, column=2).value = 'Average Pixel Intensity(AU)'
non_zero_indicies = np.nonzero(time_img_array[9])

for y in range(0, number_times):

    average_intensity = np.mean(time_img_array[y])
    y_axis[y] = average_intensity

    ws.cell(row=y+2, column =1).value = times[y]
    ws.cell(row=y+2, column=2).value = average_intensity

wb.save('linearity.xlsx')
plt.scatter(times, y_axis)
plt.show()
