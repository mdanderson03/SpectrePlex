import ob1
import time
import sys


# OB1 initialize
experiment_path = sys.argv[1]
ob1_com_port = 13
flow_control = 1

pump = ob1.fluidics(experiment_path, ob1_com_port, flow_control = 1)

pump.flow_check()

#end communication
pump.ob1_end()