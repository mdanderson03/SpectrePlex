import os
import ob1
import time
import sys
import numpy as np
from KasaSmartPowerStrip import SmartPowerStrip


# OB1 initialize
experiment_path = sys.argv[1]
ob1_com_port = 7
flow_control = 1


#save array that confirms that file was run
numpy_path = experiment_path + '/' + 'np_arrays'
os.chdir(numpy_path)
np_file_name = 'fluid_info_array.npy'
fluid_array = np.load(np_file_name, allow_pickle=False)

os.chdir(numpy_path)
fluid_array[1] = 1
np.save(np_file_name, fluid_array)



pump = ob1.fluidics(experiment_path, ob1_com_port, flow_control = 1)
power_strip = SmartPowerStrip('10.3.141.157')

#turn on upper fluid filter pump
#power_strip.toggle_plug('on', plug_num=2)
#run actions
pump.flow('ON HIGH')
time.sleep(180)
#time.sleep(30)
pump.flow('OFF')

'''
#save array that confirms that file was run
numpy_path = experiment_path + '/' + 'np_arrays'
os.chdir(numpy_path)
np_file_name = 'fluid_info_array.npy'
fluid_array = np.load(np_file_name, allow_pickle=False)

os.chdir(numpy_path)
fluid_array[1] = 1
np.save(np_file_name, fluid_array)
'''
#turn off upper fluid filter pump
#power_strip.toggle_plug('off', plug_num=2)

#end communication
pump.ob1_end()
