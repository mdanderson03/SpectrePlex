from kasa import Discover
from KasaSmartPowerStrip import SmartPowerStrip
import asyncio
import binascii
import ipaddress
import logging
import socket
#found_devices = asyncio.run(Discover.discover(target="192.168.0.1"))
#print(found_devices)
import numpy as np
from copy import deepcopy
from csbdeep.utils import normalize
from stardist.models import StarDist2D
import os
from skimage import io

model = StarDist2D.from_pretrained('2D_versatile_fluo')


experiment_directory = r'E:\3-7-24 marco'
z_slices = 3
x_frame_size = 2960
offset_array = [0, -7, -7, -6]
focus_position = 258

labelled_path = experiment_directory + '/Labelled_Nuc'
dapi_im_path = r'E:\3-7-24 marco\DAPI\Bleach\cy_0\Tiles'
z_center_index = 1


for x in range(0, 1):
    for y in range(5, 6):
        os.chdir(dapi_im_path)
        file_name = 'z_' + str(z_center_index) + '_x' + str(x) + '_y_' + str(y) + '_c_DAPI.tif'
        labelled_file_name = 'x' + str(x) + '_y_' + str(y) + '_c_DAPI.tif'
        img = io.imread(file_name)
        img = img.astype('int32')
        print(np.min(img))
        img[img < 0] = 0
        #img = skimage.util.img_as_uint(img)
        labels, _ = model.predict_instances(normalize(img))
        labels[labels > 0] = 1
        print('x', x, 'y', y)
        io.imshow(img)
        io.show()
        io.imshow(labels)
        io.show()