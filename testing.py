from autocyplex import *


pump = fluidics(6, 3)

pump.valve_select(8)

time.sleep(2)
pump.flow(500)
time.sleep(60)



pump.ob1_end()
pump.mux_end()



