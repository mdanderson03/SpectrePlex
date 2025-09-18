import time

from hamilton_psd4 import APump
from fluidics_v4_hamilton import fluidics
from hamilton_mvp import AValveChain




parameters = {"pump_com_port": "COM7","pump_ID": "PSD4","verbose": False,"simulate_pump": False,"serial_verbose": False,"high_res": True,"syringe_volume": 12.5,"syringe_type": "smooth_flow"}
sy = APump(parameters=parameters)

sy.initializePump()
#print(sy.getStatus())
sy.load_syringe(500)
sy.volume_dispense(500, 500)
#pump = fluidics(6,5,7)

#valves = [8]
#for valve in valves:
#    pump.liquid_action('Stain', valve, incub_val=1)
#    pump.liquid_action('Bleach')
#pump.load_stain( 1500, 12)
#pump.deposit_stain(1300, 0)
#pump.load_stain( 3000, 3)
#pump.valve_prime()




#pump.load_stain( 3000, 12)
#pump.load_stain( 3000, 12)


