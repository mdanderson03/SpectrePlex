import ob1
import time
import sys
import os
import numpy as np


# OB1 initialize
experiment_path = sys.argv[1]
ob1_com_port = 13
flow_control = 1

numpy_path = experiment_path + '/' + 'np_arrays'
os.chdir(numpy_path)
np_file_name = 'fluid_info_array.npy'
fluid_array = np.load(np_file_name, allow_pickle=False)

pump = ob1.fluidics(experiment_path, ob1_com_port, flow_control = 1)

pump.flow_check()

#save array that confirms that file was run
os.chdir(numpy_path)
fluid_array[1] = 1
np.save(np_file_name, fluid_array)

#end communication
pump.ob1_end()