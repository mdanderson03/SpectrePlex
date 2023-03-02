from autocyplex import *
microscope = cycif() # initialize cycif object
arduino = arduino()
import numpy as np

def take_image():

    core.snap_image()
    tagged_image = core.get_tagged_image()
    pixels = np.reshape(tagged_image.pix, newshape=[tagged_image.tags["Height"], tagged_image.tags["Width"]])

    return pixels



time_points = 160
time_sampling = 1 #seconds between images

#image_stack = np.random.rand(time_points, 740, 1264).astype('float16')



stain_valve = 1

time_total_flow_on = 80
time_stain_on = 30
time_pbs_on = time_total_flow_on - time_stain_on


on_pbs_command = (8 * 100) + 10  # multiplication by 100 forces x value into the 1st spot on a 3 digit code.
off_pbs_command = (8 * 100) + 00


for x in range(0,9):
    print(x+1)
    time.sleep(1)


arduino.flow(stain_valve, run_time=time_stain_on)
time.sleep(0.25)
arduino.flow(8, run_time=time_pbs_on)
#time.sleep(0.5)
#arduino.mqtt_publish(on_pbs_command, 'valve')
#arduino.mqtt_publish(170, 'peristaltic')

#for x in range(0, time_points):

#    image = take_image()
#    image_stack[x] = image
#    time.sleep(time_sampling)

#arduino.mqtt_publish(off_pbs_command, 'valve')
#arduino.mqtt_publish(0o70, 'peristaltic')

#microscope.save_tif_stack(image_stack, time_stain_on, directory_name= 'E:/23_23_2 bracketed flow test/')
