from autocif import *
from pycromanager import Core, Magellan, MagellanAcquisition
import numpy as np

core = Core() # initialize core object
magellan = Magellan() # initialize magellan object
microscope = cycif() # initialize cycif object
magellan_acq = MagellanAcquisition() # intialize mag acq object
robotics = arduino('COM5') # initialize arduino controlled robotics object at defined COM terminal number

client = mqtt.Client('autocyplex_server')

client.connect('192.168.1.232', 1883)

client.loop_start()
client.publish("control/valve", 214)
client.loop_stop()