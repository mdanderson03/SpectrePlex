from autocif import *
import time

bridge = Bridge()
core = bridge.get_core()
magellan = bridge.get_magellan()
microscope = cycif() # initialize cycif object


with Acquisition(directory='C:/Users/CyCIF PC/Desktop/test_images',
                 name='z_stack_DAPI',
                 show_display=False,
                 image_process_fn=microscope.image_process_hook) as acq:
    events = multi_d_acquisition_events(channel_group='Color',
                                        channels=['DAPI'],
                                        z_start=20900,
                                        z_end=20902,
                                        z_step=1,
                                        order='zc')

    acq.acquire(events)


