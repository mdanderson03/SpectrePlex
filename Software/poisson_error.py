import os
from skimage import io, util
import tifffile
import cv2
import shutil
import numpy as np
import math
from openpyxl import Workbook, load_workbook
from matplotlib import pyplot as plt
import pylab
from pylab import xticks




folder = r'D:\Images\poisson_noise\dapi_25_same_frame_200ms_1'
os.chdir(folder)
images_name = 'dapi_25_same_frame_200ms_1_MMStack_Pos0.ome.tif'
images = io.imread(images_name)

min_exp = images[0]
max_exp = images[1]

exp_offet = 27.27
offset_error = 1.2
e0 = 200
e1 = 200
proportionality_factor = 110

error_results = np.zeros(8761600)
error_diff_results = np.zeros(8761600)
pixel_number_list = np.linspace(0,8761600, 8761600)
diff_list = np.zeros(8761600)
intensity_list = np.zeros(8761600)
pixel_counter = 0



def avg_error_curve_generator(int_array,diff_array, bins):


    max_val = np.max(int_array)
    min_val = np.min(int_array)
    interval_gap = int((max_val-min_val)/bins)

    averages = np.zeros(bins)
    intensity = np.zeros(bins)
    stnd_dev = np.zeros(bins)

    for bin in range(1,bins + 1):

        lower_bound = min_val + (bin - 1)*interval_gap
        upper_bound = lower_bound + interval_gap

        bin_indicies = np.where((int_array >= lower_bound) & (int_array <= upper_bound))
        average_value = np.mean(diff_array[bin_indicies])

        averages[bin - 1] = average_value
        intensity[bin - 1] = (upper_bound + lower_bound)/2
        stnd_dev[bin - 1] = np.std(diff_array[bin_indicies])

    return averages, intensity, stnd_dev

def avg_extrapolation_error_curve_generator(images, ref_frame_number, min_range, max_range, start_frame_number, end_frame_number):



    interval_gap = max_range - min_range
    exposure_times = np.logspace(0,3.5, 75, base=10)
    exp_offset = 27.27
    ref_exp = exposure_times[ref_frame_number]
    ref_im  = images[ref_frame_number]
    flattened_ref_frame = ref_im.flatten()

    averages = np.zeros(end_frame_number - start_frame_number)
    comp_exp_time = np.zeros(end_frame_number - start_frame_number)
    stnd_dev = np.zeros(end_frame_number - start_frame_number)

    array_counter = 0

    for im in range(start_frame_number,end_frame_number):

        comp_frame = images[im]
        flattened_comp_frame = comp_frame.flatten()
        comp_exp = exposure_times[im]
        diff_image = np.abs((ref_im - comp_frame * ((ref_exp + exp_offset) / (comp_exp + exp_offset))) / ref_im * 100)
        flattened_diff_image = diff_image.flatten()

        bin_indicies = np.where((flattened_comp_frame >= min_range) & (flattened_comp_frame<= max_range))
        average_value = np.mean(flattened_diff_image[bin_indicies])
        avg_intensity = np.mean(flattened_comp_frame[bin_indicies])
        avg_ref_intensity = np.mean(flattened_ref_frame[bin_indicies])

        averages[array_counter] = average_value
        #comp_exp_time[array_counter] = (ref_exp + exp_offset)/ (comp_exp + exp_offset)
        comp_exp_time[array_counter] = (ref_exp + exp_offset)/(comp_exp + exp_offset)
        stnd_dev[array_counter] = np.std(flattened_diff_image[bin_indicies])

        array_counter += 1

    return averages,  comp_exp_time, stnd_dev

def avg_poisson_band_curve_generator(images, ref_frame_number, min_int_range, max_int_range, start_frame_number, end_frame_number):


    images = images.astype('int32') - 300
    ref_im  = images[ref_frame_number]
    flattened_ref = ref_im.flatten()
    frame_count = end_frame_number - start_frame_number


    all_image_diff_array = np.zeros((frame_count, 2960, 5056))

    for im in range(start_frame_number,end_frame_number):

        all_image_diff_array[im - start_frame_number] = ((images[im] - ref_im)*100)/ref_im
        #all_image_diff_array[im - start_frame_number] = images[im] - ref_im

    partially_flattened_array = np.reshape(all_image_diff_array, (frame_count, 14965760))
    intensity_indicies  = np.where((flattened_ref >= min_int_range - 300) & (flattened_ref <= max_int_range - 300))[0]
    intensity_screened = partially_flattened_array[::,intensity_indicies]
    intensity_screened = intensity_screened.flatten()



    return intensity_screened

def poisson_error_vs_intensity_curve(images, ref_frame_number, min_int_range, max_int_range, start_frame_number, end_frame_number):

    int_gap = 500
    number_entries = int((max_int_range - min_int_range)/int_gap)
    std_dev = np.zeros(number_entries)
    intensities = np.zeros(number_entries)

    entry_counter = 0
    for intensity in range(min_int_range, max_int_range, int_gap):

        intensities[entry_counter] = math.sqrt(intensity + int_gap/2 - 300)
        std_dev[entry_counter] = np.sqrt(np.std(avg_poisson_band_curve_generator(images, ref_frame_number, intensity, intensity + gap, start_frame_number, end_frame_number)))
        entry_counter += 1

    return std_dev, intensities


#diff_image = (max_exp - min_exp * ((e1 + exp_offet)/(e0 + exp_offet)))/max_exp*100
diff_image = max_exp - min_exp
flattened_array = np.abs(diff_image.flatten())
min_flattened_array = min_exp.flatten()
#plt.hist(flattened_array, bins=100, color='skyblue', edgecolor='black', range=(-15, 15))
#plt.show()

#averages, intensity, stnd_dev = avg_error_curve_generator(min_flattened_array, flattened_array, 15)
#plt.scatter(intensity, averages)
#plt.errorbar(intensity, averages, yerr=stnd_dev, fmt="o")
#plt.show()
#print(intensity, 'intensity')
#print(averages, 'averages')

start_val = 5000
gap = 500

#percent_differences = avg_poisson_band_curve_generator(images, 8, start_val, start_val + gap, 9, 10)
#print('mean', np.std(percent_differences))


#plt.hist(percent_differences, bins=100, color='skyblue', edgecolor='black')
#plt.show()
std_dev, intensities = poisson_error_vs_intensity_curve(images, 8, 1000, 35000, 9, 10)

plt.scatter( intensities, std_dev)
plt.show()


'''

averages, comp_exp_time, stnd_dev = avg_extrapolation_error_curve_generator(images, 0, start_val, start_val + gap, 1, 24)

plt.scatter(comp_exp_time, averages)
#plt.errorbar(comp_exp_time, averages, yerr=stnd_dev, fmt="o")
plt.show()
print(comp_exp_time, 'intensity')
print(averages, 'averages')
'''






