import time

from hamilton_psd4 import APump
#from fluidics_v4_hamilton import fluidics
from hamilton_mvp import AValveChain




parameters = {"pump_com_port": "COM7","pump_ID": "PSD4","verbose": False,"simulate_pump": False,"serial_verbose": False,"high_res": True,"syringe_volume": 12.5,"syringe_type": "smooth_flow"}
sy = APump(parameters=parameters)

sy.initializePump()

#print(sy.getStatus())
sy.load_syringe(1000)

sy.volume_dispense(900, 500, device_port=3)
#pump = fluidics(6,5,7)

#valves = [8]
#for valve in valves:
#pump.liquid_action('Stain', stain_valve=1, incub_val=5)
#    pump.liquid_action('Bleach')
#pump.load_stain( 1500, 12)
#pump.deposit_stain(1300, 0)
#pump.load_stain( 3000, 3)
#pump.valve_prime()




#pump.load_stain( 3000, 12)
#pump.load_stain( 3000, 12)


