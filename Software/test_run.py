import time

from hamilton_psd4 import APump



parameters = {
  "pump_com_port": "COM3",
  "pump_ID": "PSD4",
  "verbose": False,
  "simulate_pump": False,
  "serial_verbose": False,
  "high_res": True,
  "syringe_volume": 12.5,
  "syringe_type": "smooth_flow"
}

sy = APump(parameters=parameters)
#sy.setPort(4)
#sy.initializePump()
#sy.setSpeed(1500)
#sy.startFill(2500)
(is_moving, pos_in_mL, vel_in_mLmin, valve_pos) = sy.getStatus()
print(is_moving, vel_in_mLmin, valve_pos, pos_in_mL)

sy.volume_dispense(1000, 150, wait_until_flow_done=False)
time.sleep(20)
sy.stopFill()
#sy.empty_syringe(wait_until_flow_done=True)
#for x in range(0,10):
#  time.sleep(5)
(is_moving, pos_in_mL, vel_in_mLmin, valve_pos) = sy.getStatus()
print(is_moving, vel_in_mLmin, valve_pos, pos_in_mL)



