import ob1
import time
import sys
from KasaSmartPowerStrip import SmartPowerStrip


# OB1 initialize
experiment_path = sys.argv[1]
ob1_com_port = 13
flow_control = 1


pump = ob1.fluidics(experiment_path, ob1_com_port, flow_control = 1)
power_strip = SmartPowerStrip('10.3.141.157')

#turn on upper fluid filter pump
power_strip.toggle_plug('on', plug_num=2)
#run actions
pump.flow('ON')
time.sleep(150)
pump.flow('OFF')

#turn off upper fluid filter pump
power_strip.toggle_plug('off', plug_num=2)

#end communication
pump.ob1_end()
