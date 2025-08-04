import time

from hamilton_psd4 import APump



parameters = {
  "pump_com_port": "COM4",
  "pump_ID": "PSD4",
  "verbose": False,
  "simulate_pump": False,
  "serial_verbose": False,
  "high_res": True,
  "syringe_volume": 12.5,
  "syringe_type": "smooth_flow"
}

sy = APump(parameters=parameters)
#sy.setPort(6)
#sy.write('/1h3000R\r')
#sy.read()
#for x in range(0, 4):
sy.initializePump()
sy.load_syringe(350)

#time.sleep(90)

#sy.load_syringe(350)
#sy.volume_dispense(350, 500, wait_until_flow_done=True)
#sy.setPort(3)

#print('start time lapse')
#time.sleep(15)

#for x in range(1, 15):
#  sy.volume_dispense(50, 500, device_port=3, wait_until_flow_done=False)
#  time.sleep(60)
#sy.startFill(-350)
#print('get ready')
#time.sleep(10)
#sy.volume_dispense(-1500, 2000, wait_until_flow_done=True)
#time.sleep(20)
#sy.stopFill()
#sy.empty_syringe(wait_until_flow_done=False)
#for x in range(0,10):
#  time.sleep(5)
(is_moving, pos_in_mL, vel_in_mLmin, valve_pos) = sy.getStatus()
print(is_moving, vel_in_mLmin, valve_pos, pos_in_mL)



