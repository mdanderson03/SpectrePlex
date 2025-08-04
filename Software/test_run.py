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
#sy.load_syringe(350)


#(is_moving, pos_in_mL, vel_in_mLmin, valve_pos) = sy.getStatus()
#print(is_moving, vel_in_mLmin, valve_pos, pos_in_mL)


def load_stain(pre_expel_vol, load_volume, stain_valve):
    '''
    Loads stain into syringe. First a volume (pre-expel) is taken in and expelled.
    Next, the stain is pulled into the syringe in the indicated volume.
    Parameters
    ----------
    pre_expel_vol
    load_volume
    stain_valve

    Returns
    -------

    '''

    pbs_valve = 12
    load_syringe_valve = 2

    # set to PBS valve and load syringe valves
    # self.valve_select(pbs_valve)
    sy.setPort(load_syringe_valve)

    # load syringe of initial volume and expel
    sy.load_syringe(pre_expel_vol, wait_until_flow_done=True)
    sy.initializePump()
    # move to stain valve and bring stain solution in
    # self.valve_select(stain_valve)
    sy.load_syringe(load_volume)

def deposit_stain(expel_volume, post_expel_volume, device_valve=3):
    '''
    Pushes out volume of stain into device and then sucks PBS back into syringe
    and follows by pushing fluid front post_expel_volume more.
    Parameters
    ----------
    expel_volume
    post_expel_volume
    device_valve

    Returns
    -------

    '''

    pbs_valve = 12
    load_syringe_valve = 2

    sy.volume_dispense(expel_volume, dispense_vel_ulmin=500, device_port=device_valve,
                            wait_until_flow_done=True)

    if post_expel_volume > 0:
        # if volume is non zero, then expel syringe, swap to PBS valve and fill
        sy.initializePump()
        sy.setPort(load_syringe_valve)
        #self.valve_select(pbs_valve)
        sy.load_syringe(post_expel_volume * 2)

        sy.volume_dispense(post_expel_volume, dispense_vel_ulmin=500, device_port=device_valve,
                                wait_until_flow_done=True)


#load_stain(30, 350, 3)
#deposit_stain(350, 30, device_valve=3)