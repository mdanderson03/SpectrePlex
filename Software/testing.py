import numpy as np
import os
from skimage import io, morphology
from matplotlib import pyplot as plt
import os
from autocyplex import *

project_path = r'E:\30-1-24 flat field imaging'
color = 'A488'

scale_path = project_path + '/scale/' + str(color)
ff_path = project_path + '/corrections/' + str(color)
os.chdir(scale_path)

core.set_config("Color", color)

image_stack = np.full((50, 2960, 5056), 0)

for x in range (0, 50):

    exp_time = .12 * (x + 1)
    core.set_exposure(exp_time)

    core.snap_image()
    tagged_image = core.get_tagged_image()
    pixels = np.reshape(tagged_image.pix,newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"]])

    image_stack[x] = pixels

    time.sleep(0.5)


io.imsave('scale.tif', image_stack)

for x in range(0, 50):

    exp_time = 3
    core.set_exposure(exp_time)

    core.snap_image()
    tagged_image = core.get_tagged_image()
    pixels = np.reshape(tagged_image.pix, newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"]])

    image_stack[x] = pixels

    time.sleep(0.5)



os.chdir(ff_path)
io.imsave('Flat_field.tif', image_stack)