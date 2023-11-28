import os

import cv2

from autocyplex import *
from optparse import OptionParser
microscope = cycif() # initialize cycif object
pump = fluidics(6, 3)






experiment_directory = r'E:\28_11_23 trial'
offset_array = [0, -8, -7, -7]
z_slices = 15
x_frame_size = 2960
cycle = 0



#microscope.full_cycle(experiment_directory, cycle, offset_array, cycle, pump, z_slices)


#pump.liquid_action('Stain', stain_valve = 1, incub_val=45)


#microscope.establish_fm_array(experiment_directory, 2, z_slices, offset_array, initialize=0,x_frame_size=x_frame_size, autofocus=1, auto_expose=0)

#print(core.get_position())
#microscope.image_cycle_acquire( 0, experiment_directory, 7, 'Bleach', offset_array, x_frame_size=x_frame_size, establish_fm_array=0, auto_focus_run=0, auto_expose_run=0)

#for cycle in range(2,4):
#    microscope.full_cycle(experiment_directory, cycle, offset_array, cycle, pump, z_slices)

#microscope.post_acquisition_processor(experiment_directory, x_frame_size)
#microscope.stage_placement(experiment_directory, 1, x_frame_size)

#microscope.mcmicro_image_stack_generator(1, experiment_directory, x_frame_size)

'''

def DAPI_surface_autofocus(x_frame_size):



        bottom_z = -131
        top_z = -101

        z_slice_gap = 1


        # find crop range for x dimension

        side_pixels = int(5056 - x_frame_size)

        core.set_config("Color", 'DAPI')
        core.set_exposure(50)

        z_stack = np.random.rand(30, 2960, x_frame_size)

        stack_index = 0



        for z in range(bottom_z, top_z, z_slice_gap):
            core.set_position(z)
            time.sleep(0.5)

            core.snap_image()
            tagged_image = core.get_tagged_image()
            pixels = np.reshape(tagged_image.pix,newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"]])
            z_stack[stack_index] = pixels[::, side_pixels:x_frame_size + side_pixels]

            stack_index += 1

        z_index = microscope.highest_brenner_index_solver(z_stack)
        print('index', z_index)
        focus_z_position = bottom_z + z_index * z_slice_gap
        print('focus position', focus_z_position)

#DAPI_surface_autofocus(2960)


core.snap_image()
tagged_image = core.get_tagged_image()
image = np.reshape(tagged_image.pix, newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"]])

score = microscope.focus_score(image, 15)

print(score)
'''