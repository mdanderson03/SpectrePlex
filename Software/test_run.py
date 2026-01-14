import time

from hamilton_psd4 import APump
#from fluidics_v4_hamilton import fluidics
from hamilton_mvp import AValveChain




parameters = {"pump_com_port": "COM7","pump_ID": "PSD4","verbose": False,"simulate_pump": False,"serial_verbose": False,"high_res": True,"syringe_volume": 12.5,"syringe_type": "smooth_flow"}
sy = APump(parameters=parameters)

sy.initializePump()


print('loading')
#sy.startFill(500)
sy.load_syringe(800)
print('dispensing')
#sy.startFill(500)
sy.volume_dispense(800, 500, device_port=3)
#x=0
#for x in range(0,50):
#    time.sleep(0.5)
#    (is_moving, pos_in_uL, vel_in_mLmin, valve_pos) = sy.getStatus()
#    print(pos_in_uL)
#    x =+1
#sy.swish(1, 100, 2)
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


