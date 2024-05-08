import ob1
import time

# OB1 initialize
experiment_path = r'E:\8-5-24 Healthy'
ob1_com_port = 13
flow_control = 1

pump = ob1.fluidics(experiment_path, ob1_com_port, flow_control = 1)


#run actions
pump.flow('ON')
time.sleep(150)
pump.flow('OFF')

#end communication
pump.ob1_end()
