import datetime
import os
import numpy as np
import time
from openpyxl import Workbook, load_workbook
from hamilton_mvp import AValveChain
from hamilton_psd4 import APump

class fluidics:

    def __init__(self, experiment_path, mux_valve1_ID, mux_valve2_ID, syringe_com_port):


        self.experiment_path = experiment_path
        self.experiment_directory = experiment_path

        #initialize distribution valves
        ###############################################

        self.valve1_ID = 'COM' + str(mux_valve1_ID)
        self.valve2_ID = 'COM' + str(mux_valve2_ID)

        #self.mux1 = AValveChain(parameters=self.valve1_ID)
        #self.mux2 = AValveChain(parameters=self.valve2_ID)

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
        self.device_number = 1 #makes valve 1 only connected to device

        #recalibrate volume and expel into waste reservior
        self.sy.setPort(5) #5 is waste port
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

        for valve in range(1,12):
            self.valve_select(valve)
            time.sleep(0.5)
            self.sy.volume_dispense(250, 500, wait_until_flow_done=True)

    def flow(self, volume_2_dispense, flow_rate):
        '''
        volume 2 dispense is in uL and flow_rate is uL/min
        Parameters
        ----------
        volume_2_dispense
        flow_rate

        Returns
        -------

        '''

        self.sy.volume_dispense(volume_2_dispense, flow_rate, wait_until_flow_done=True)

    def liquid_action(self, action_type, stain_valve=0, incub_val=45, heater_state=0):

        bleach_valve = 11
        pbs_valve = 12
        bleach_time = 10  # minutes
        bleach_flow_rate = 1000
        wash_flow_rate = 1000
        stain_flow_rate = 500
        if heater_state == 0:
            stain_inc_time = incub_val  # minutes
        if heater_state == 1:
            stain_inc_time = 45  # minutes

        self.device_valve_select()

        if action_type == 'Bleach':

            self.valve_select(bleach_valve)
            self.flow(500, bleach_flow_rate)

            for x in range(0, bleach_time):
                time.sleep(60)

            self.valve_select(pbs_valve)
            self.flow(1500, wash_flow_rate)

        elif action_type == 'Stain':

            self.valve_select(stain_valve)
            self.flow(400, stain_flow_rate)

            self.valve_select(pbs_valve)
            self.valve_select(pbs_valve)
            self.flow(1500, wash_flow_rate)

            for x in range(0, stain_inc_time):
                time.sleep(60)
                print('Staining Time Elapsed ', x)

            self.valve_select(pbs_valve)
            self.flow(1500, wash_flow_rate)

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




