import os
import ob1
import time
import sys
import numpy as np
from KasaSmartPowerStrip import SmartPowerStrip


# OB1 initialize
experiment_path = sys.argv[1]
ob1_com_port = 13
flow_control = 1

pump = ob1.fluidics(experiment_path, ob1_com_port, flow_control = 1)

#run actions
pump.flow('OFF')

#save array that confirms that file was run
numpy_path = experiment_path + '/' + 'np_arrays'
os.chdir(numpy_path)
np_file_name = 'fluid_info_array.npy'
fluid_array = np.load(np_file_name, allow_pickle=False)

os.chdir(numpy_path)
fluid_array[1] = 1
np.save(np_file_name, fluid_array)


#end communication
pump.ob1_end()
