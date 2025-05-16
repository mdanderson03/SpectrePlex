import numpy as np
from skimage import io, util
from matplotlib import pyplot
import os
import math
from skimage import transform, morphology

from stardist.models import StarDist2D
from csbdeep.utils import normalize

model = StarDist2D.from_pretrained('2D_versatile_fluo')

os.chdir(r'E:\integreity_star_dist')
img = io.imread(r'dapi_stack_20_41.tif')
#img = io.imread(r'1_20_dapi_label_stack.tif')

cycles = np.shape(img)[0]
y_pix = np.shape(img)[1]
x_pix = np.shape(img)[2]


label_stack = np.zeros((cycles, y_pix, x_pix))
for cycle in range(0, cycles):

    dapi_cycle_index = cycle*1
    labels, _ = model.predict_instances(normalize(img[dapi_cycle_index]))
    label_stack[cycle] = labels
    print(str(cycles))

io.imsave('20_41_dapi_label_stack.tif', label_stack)
#labels[labels > 0] = 1
#io.imsave('final_dapi_binary.tif', labels)

'''
y_axis = np.linspace(0,20)
x_axis = np.linspace(0,20)

for x in range(0,20):
    x_axis[0] = np.max(img[x])

pyplot.scatter(y_axis, x_axis)
pyplot.show()
'''