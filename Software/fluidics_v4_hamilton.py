import datetime
import os
import numpy as np
import time
from openpyxl import Workbook, load_workbook
from hamilton_mvp import AValveChain
from hamilton_psd4 import APump

class fluidics:

    def __init__(self, mux_valve1_ID, mux_valve2_ID, syringe_com_port):


        #self.experiment_path = experiment_path
        #self.experiment_directory = experiment_path

        #initialize distribution valves
        ###############################################

        self.valve1_ID = 'COM' + str(mux_valve1_ID)
        self.valve2_ID = 'COM' + str(mux_valve2_ID)

        self.mux1 = AValveChain(parameters=self.valve1_ID)
        self.mux2 = AValveChain(parameters=self.valve2_ID)

        #   initialize syringe pump
        ###########################################
        parameters = {
            "pump_com_port": 'COM' + str(syringe_com_port),
            "pump_ID": "PSD4",
            "verbose": False,
            "simulate_pump": False,
            "serial_verbose": False,
            "high_res": True,
            "syringe_volume": 12.5,
            "syringe_type": "smooth_flow"
        }
        self.sy = APump(parameters=parameters)
        self.device_number = 2 #makes valve 1 only connected to device

        #recalibrate volume and expel into waste reservior
        #self.sy.setPort(5) #5 is waste port
        self.sy.initializePump()

        return

    def fluidics_logger(self, function_used_string, error_code, value_sr):
        experiment_directory = self.experiment_path
        filename = 'logger.xlsx'
        logger_path = experiment_directory + '/fluidics data logger'
        os.chdir(experiment_directory)
        try:
            os.mkdir('fluidics data logger')
            os.chdir(logger_path)
        except:
            os.chdir(logger_path)

        if os.path.isfile(filename) == True:
            wb = load_workbook(filename)
            ws = wb.active
        elif os.path.isfile(filename) == False:
            wb = Workbook()
            ws = wb.active
            # add headers
            ws.cell(row=1, column=1).value = 'Time Stamp'
            ws.cell(row=1, column=2).value = 'Function Used'
            ws.cell(row=1, column=3).value = 'Error Code'
            ws.cell(row=1, column=4).value = 'Value Sent/Recieved'

        # determine row number (add to next line as a logged event)
        current_max_row = ws.max_row
        row_select = current_max_row + 1

        # add in values
        ws.cell(row=row_select, column=1).value = datetime.datetime.now()
        ws.cell(row=row_select, column=2).value = function_used_string
        ws.cell(row=row_select, column=3).value = error_code
        ws.cell(row=row_select, column=4).value = value_sr

        wb.save(filename)

    def wait_for_readiness(self, device):
        while not device.get_status()[1]:
            time.sleep(0.1)

    def device_valve_select(self):
        '''
        Chooses which device is connected to the syringe pump
        Parameters
        ----------
        device_number

        Returns
        -------

        '''

        self.sy.setPort(self.device_number)

    def valve_select(self, valve_number):
        '''
        Moves to desired vial number via 2 hamilton valves
        Parameters
        ----------
        valve_number

        Returns
        -------

        '''

        #remap from physical valve number to reference valve number
        #physical = how eppendorfs are organized, interal = valve on dist valve module

        remap_vector = [0,8,9,10,7,1,2,3,4,5,6,12,13,14,11]
        valve_number = int(remap_vector[valve_number])

        if valve_number <= 7:
            self.mux1.changePort(0, valve_number - 1, direction=0, wait_until_done=True)
        elif valve_number >= 8 and valve_number <= 15:
            self.mux1.changePort(0, 7, direction=0, wait_until_done=True)
            self.mux2.changePort(0, valve_number - 8, direction=0, wait_until_done=True)

        '''
        valve1 = MVPvalve("MVP/4", "0", self.valve1_ID)
        valve2 = MVPvalve("MVP/4", "1", self.valve2_ID)

        valve1.initialize("left")
        valve2.initialize("left")

        vial_ID = valve_number

        if 1 <= vial_ID <= 8:
            valve1_port_ID = vial_ID
            valve2_port_ID = 1

            valve1.valve_input(position=valve1_port_ID)
            valve1.run_command()
            self.wait_for_readiness(valve1)

            valve2.run_command(position=valve2_port_ID)
            self.wait_for_readiness(valve2)

        elif 9 <= vial_ID <= 15:
            valve2_port_ID = vial_ID - 7

            valve2.run_command(position=valve2_port_ID)
            self.wait_for_readiness(valve2)

        elif vial_ID > 15:
            print('error: vial_ID out of range. Please select option between 1-15')

        return
        '''

    def valve_prime(self):

        valves = [1,2,3,4,5,6,7,8,9,10, 11, 12]
        #valves = [14]


        #for valve in range(1,12):
        for valve in valves:
            self.valve_select(valve)
            time.sleep(0.5)
            self.sy.load_syringe(350, wait_until_flow_done=True)


        self.sy.initializePump()

    def load_stain(self, load_volume, stain_valve):
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

        #set to PBS valve and load syringe valves
        #self.valve_select(pbs_valve)
        #self.sy.setPort(load_syringe_valve)

        #load syringe of initial volume and expel
        #self.sy.load_syringe(pre_expel_vol, wait_until_flow_done=True)
        #self.sy.initializePump()
        #move to stain valve and bring stain solution in
        self.valve_select(stain_valve)
        self.sy.load_syringe(load_volume)

    def deposit_stain(self, expel_volume, post_expel_volume, device_valve = 3):
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

        self.sy.volume_dispense(expel_volume, dispense_vel_ulmin=500, device_port=device_valve, wait_until_flow_done=True)

        if post_expel_volume > 0:
            #if volume is non zero, then expel syringe, swap to PBS valve and fill
            self.sy.initializePump()
            self.sy.setPort(load_syringe_valve)
            self.valve_select(pbs_valve)
            self.sy.load_syringe(post_expel_volume*2)

            self.sy.volume_dispense(post_expel_volume, dispense_vel_ulmin=500, device_port=device_valve, wait_until_flow_done=True)

    def back_flush(self, valve_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
        '''

        Parameters
        ----------
        valve_list(list, int): list of valve numbers to back flush into

        Returns
        -------

        '''

        # Suck in water
        water_port = 4
        self.sy.setPort(water_port)

        # suck in enough volume to flush with
        self.sy.setSpeed(6000)
        fill_volume = len(valve_list) * 500  # 500uL per valve to flush
        self.sy.startFill(fill_volume)

        wait_until_flow_done = True
        while wait_until_flow_done == True:
            time.sleep(1)
            (is_moving, pos_in_uL, vel_in_mLmin, valve_pos) = self.sy.getStatus()

            wait_until_flow_done = is_moving

        # loop through each intended valve and dispense 500uL

        for valve in valve_list:
            self.valve_select(valve)
            self.sy.volume_dispense(500, 1000, 5)

    def clean_valve(self):

        water_valve = 4
        device_valve = 3
        load_valve = 5

        self.valve_select(water_valve)
        self.sy.load_syringe(1000)

        self.sy.volume_dispense(400, dispense_vel_ulmin=500, device_port=device_valve,wait_until_flow_done=True)
        self.sy.volume_dispense(100, dispense_vel_ulmin=500, device_port=load_valve, wait_until_flow_done=True)

        self.sy.initializePump()

    def stop_syringe(self):
        self.sy.write('/1T\r')

    def bleach(self, volume_2_dispense, flow_rate):
        '''
        volume 2 dispense is in uL and flow_rate is uL/min
        Parameters
        ----------
        volume_2_dispense
        flow_rate

        Returns
        -------

        '''

        self.sy.load_syringe(volume_2_dispense + 150)
        self.sy.volume_dispense(volume_2_dispense, flow_rate, wait_until_flow_done=True)
        self.sy.volume_dispense(150, flow_rate, device_port=1, wait_until_flow_done=True)

    def wash(self, volume_2_dispense, flow_rate):
        '''
        volume 2 dispense is in uL and flow_rate is uL/min
        Parameters
        ----------
        volume_2_dispense
        flow_rate

        Returns
        -------

        '''

        self.sy.load_syringe(volume_2_dispense + 150)
        self.sy.volume_dispense(volume_2_dispense, flow_rate, device_port=3, wait_until_flow_done=True)

    def low_flow(self, volume_2_dispense, flow_rate):
        self.sy.load_syringe(volume_2_dispense)
        print('dispensing')
        self.sy.volume_dispense(volume_2_dispense, flow_rate, device_port=3, wait_until_flow_done=False)

    def stain(self, flow_rate):
        '''
        volume 2 dispense is in uL and flow_rate is uL/min
        Parameters
        ----------
        volume_2_dispense
        flow_rate

        Returns
        -------

        '''
        self.sy.initializePump()
        self.sy.load_syringe(160)
        self.sy.volume_dispense(160, flow_rate, device_port=1, wait_until_flow_done=True)
        print('loading in rest of stain')
        self.sy.initializePump()
        self.sy.load_syringe(300)
        print('loading in dead vol')
        self.valve_select(13)
        self.sy.load_syringe(160)
        print('dispensing')
        self.sy.volume_dispense(410, flow_rate, wait_until_flow_done=True)


    def liquid_action(self, action_type, stain_valve=0, incub_val=45, heater_state=0):

        bleach_valve = 14
        pbs_valve = 13
        bleach_time = 7  # minutes
        bleach_flow_rate = 500
        wash_flow_rate = 500
        stain_flow_rate = 500
        if heater_state == 0:
            stain_inc_time = incub_val  # minutes
        if heater_state == 1:
            stain_inc_time = 45  # minutes

        self.device_valve_select()

        if action_type == 'Bleach':

            self.valve_select(bleach_valve)
            self.bleach(700, flow_rate=500)

            for x in range(0, bleach_time):
                time.sleep(60)

            self.valve_select(pbs_valve)
            self.wash(700, flow_rate=500)
            self.wash(500, flow_rate=500)

        elif action_type == 'Stain':

            self.valve_select(stain_valve)
            self.stain(flow_rate=500)

            self.sy.swish(1, 100, swish_duration=incub_val)

            self.valve_select(pbs_valve)
            self.wash(700, flow_rate=500)
            self.wash(500, flow_rate=500)
            self.low_flow(800, 100)



        elif action_type == "Wash":

            self.valve_select(pbs_valve)
            time.sleep(2)
            self.flow_checker()
            self.file_run('wash.py')

        elif action_type == 'low flow on':

            self.valve_select(pbs_valve)
            self.sy.empty_syringe(wait_until_flow_done=True)
            self.device_valve_select()
            self.sy.volume_dispense(10000, 150, wait_until_flow_done=False)

        elif action_type == 'flow off':
            self.sy.stopFill()

        elif action_type == 'empty':
            self.sy.empty_syringe(wait_until_flow_done=True)
            self.device_valve_select()




