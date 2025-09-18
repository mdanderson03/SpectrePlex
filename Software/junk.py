import numpy as np
import os
from skimage import io, filters, morphology, transform, util
#from skimage.filters import rank
#from matplotlib import pyplot as plt
import cv2
import math
import time
#from skimage.morphology import disk
from stardist.models import StarDist2D
from csbdeep.utils import normalize
import tensorflow as tf




model = StarDist2D.from_pretrained('2D_versatile_fluo')

os.chdir(r'F://23_6_25_casey_SP23_12585//archive\DAPI\Stain\cy_0\Tiles')
file_name = 'x' + str(3) + '_y_' + str(1) + '_c_DAPI.tif'
#labelled_file_name = 'x' + str(0) + '_y_' + str(2) + '_c_DAPI.tif'
img = io.imread(file_name)
img = normalize(img)
start = time.time()
labels, _ = model.predict_instances(normalize(img))
end = time.time()
print(end - start)
labels[labels > 0] = 1
io.imshow(labels)
io.show()
