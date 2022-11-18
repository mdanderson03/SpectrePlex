from autocif import *
from pycromanager import Core, Magellan, MagellanAcquisition
import numpy as np

#core = Core() # initialize core object
#magellan = Magellan() # initialize magellan object
#microscope = cycif() # initialize cycif object
#magellan_acq = MagellanAcquisition() # intialize mag acq object



client = mqtt.Client('autocyplex_server')

client.connect('10.3.141.1', 1883)
'''
client.loop_start()
client.publish("control/valve", 804)
client.loop_stop()
'''
'''
arduino.mqtt_publish(310, 'valve', client)
arduino.mqtt_publish(170, 'peristaltic', client)
time.sleep(18)
arduino.mqtt_publish(810, 'valve', client)
arduino.mqtt_publish(300, 'valve', client)
time.sleep(18)
arduino.mqtt_publish(0o60, 'peristaltic', client)
time.sleep(2700)
arduino.mqtt_publish(170, 'peristaltic', client)
time.sleep(60)
arduino.mqtt_publish(800, 'valve', client)
arduino.mqtt_publish(610, 'valve', client)
time.sleep(60)
arduino.mqtt_publish(600, 'valve', client)
arduino.mqtt_publish(0o60, 'peristaltic', client)
time.sleep(300)
arduino.mqtt_publish(810, 'valve', client)
arduino.mqtt_publish(170, 'peristaltic', client)
time.sleep(120)
arduino.mqtt_publish(0o60, 'peristaltic', client)
arduino.mqtt_publish(800, 'valve', client)

'''

'''
arduino.mqtt_publish(170, 'peristaltic', client)
time.sleep(1)
arduino.mqtt_publish(0o60, 'peristaltic', client)
time.sleep(2)
'''


arduino.mqtt_publish(170, 'peristaltic', client)
time.sleep(60)
arduino.mqtt_publish(0o60, 'peristaltic', client)
