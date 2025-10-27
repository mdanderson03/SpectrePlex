import numpy as np
import os
from copy import deepcopy

from ome_types import from_xml, to_xml
from ome_types.model import Channel, Plane, TiffData






os.chdir(r'C:\Users\ch199435\Documents\GitHub\SpectrePlex')
ome = from_xml('image.xml')


for im_num in range(0, 6):

    id_name = deepcopy(ome.images[im_num].pixels.channels[0].id)
    id_length = len(ome.images[im_num].pixels.channels[0].id)
    id_name = id_name[0:id_length-1] + '4'

    # 2. Create a new Channel
    new_channel = Channel(
        id= id_name,  # next channel ID
        samples_per_pixel=1,
        emission_wavelength=800.0,  # nm, example value
        excitation_wavelength=750.0,
        excitation_wavelength_unit="nm",
        emission_wavelength_unit = "nm",
        nd_filter = "0.0"
    )

    ome.images[im_num].pixels.channels.append(new_channel)
    ome.images[im_num].pixels.size_c = 5

    #new plane
    new_plane = Plane(
        the_z= '0',
        the_c= '4',  # index of new channel (0-based)
        the_t='0',
        exposure_time=0.20000000298023224,
        position_x=25498.0,
        position_y=12615.0,
        position_z=0.06201172,
        delta_t=1166.76904296875,
        delta_t_unit="s",
        exposure_time_unit="s",
        position_x_unit="reference frame",
        position_y_unit="reference frame",
        position_z_unit="reference frame"

    )# optional metadata

    ome.images[im_num].pixels.planes.append(new_plane)


    #new TIFF Data

    new_data = TiffData(
        uuid={'value': 'urn:uuid:27f14ce6-cf37-4ec8-9913-018a402a9ad4', 'file_name': 'exemplar-001-cycle-06.ome.tiff'},
        ifd= 4,
        first_c=4,
        first_t=0,
        first_z=0,
        plane_count=1
    )

    ome.images[im_num].pixels.tiff_data_blocks.append(new_data)


#ome.images.append(ome.images[0].pixels.channels[0])
#ome.images[0].pixels.size_c = 5
#ome.images[0].pixels.append(ome_1)


xml_string = ome.to_xml()
file_path = "image_5th_channel.xml"
with open(file_path, "w", encoding="utf-8") as f:
    f.write(xml_string)


ome = from_xml('image_5th_channel.xml')

print(ome)


