from autocyplex import *
microscope = cycif() # initialize cycif object
import numpy as np

def take_image():

    core.snap_image()
    tagged_image = core.get_tagged_image()
    pixels = np.reshape(tagged_image.pix, newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"]])

    return pixels



time_points = 160
time_sampling = 1 #seconds between images

image_stack = np.random.rand(time_points, 740, 1264).astype('float16')



stain_valve = 3

time_total_flow_on = 80
time_stain_on = 12
time_pbs_on = time_total_flow_on - time_stain_on


on_pbs_command = (8 * 100) + 10  # multiplication by 100 forces x value into the 1st spot on a 3 digit code.
off_pbs_command = (8 * 100) + 00


arduino.flow(stain_valve, run_time=time_stain_on)
arduino.mqtt_publish(on_pbs_command, 'valve')
arduino.mqtt_publish(170, 'peristaltic')

for x in range(0, time_points):

    image = arduino.flow(stain_valve, run_time = time_stain_on)
    image_stack[x] = image

arduino.mqtt_publish(off_pbs_command, 'valve')
arduino.mqtt_publish(0o70, 'peristaltic')

microscope.save_tif_stack(image_stack, 1, directory_name=)
