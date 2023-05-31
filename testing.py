from autocyplex import *


pump = fluidics(6, 3)

pump.valve_select(3)

#time.sleep(2)
pump.flow(0)

time.sleep(10)

#pump.flow(0)
#pump.flow_recorder(0.1, 30)

#print('stop')
pump.ob1_end()
pump.mux_end()


