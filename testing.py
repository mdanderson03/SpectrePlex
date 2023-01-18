from autocyplex import *


microscope = cycif()

max_z = 5415
min_z = 5380

z_range = [min_z, max_z, (max_z-min_z)/2]
z_ideal = microscope.auto_focus(z_range, 25, 'DAPI')
print(z_ideal)


