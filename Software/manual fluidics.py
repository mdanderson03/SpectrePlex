import time

from hamilton_psd4 import APump
#from fluidics_v4_hamilton import fluidics
from hamilton_mvp import AValveChain




parameters = {"pump_com_port": "COM7","pump_ID": "PSD4","verbose": False,"simulate_pump": False,"serial_verbose": False,"high_res": True,"syringe_volume": 12.5,"syringe_type": "smooth_flow"}
sy = APump(parameters=parameters)
sy.write('/1T\r')
sy.initializePump()

print('loading')

sy.load_syringe(1000)
print('dispensing')
#sy.setPort(3)
sy.volume_dispense(800, 500, device_port=3)

