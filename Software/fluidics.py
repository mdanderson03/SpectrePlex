import numpy as np
import time
import matplotlib.pyplot as plt
from openpyxl import Workbook
import sys
from ctypes import *


sys.path.append(
    r'C:\Users\CyCIF PC\Documents\GitHub\AutoCIF\Python_64_elveflow\DLL64')  # add the path of the library here
sys.path.append(r'C:\Users\CyCIF PC\Documents\GitHub\AutoCIF\Python_64_elveflow')  # add the path of the LoadElveflow.py
sys.path.append(r'C:\Users\mike\Documents\GitHub\AutoCIF\Python_64_elveflow\DLL64')  # add the path of the library here
sys.path.append(r'C:\Users\mike\Documents\GitHub\AutoCIF\Python_64_elveflow')  # add the path of the LoadElveflow.py

from array import array
from Elveflow64 import *

class fluidics:

    def __init__(self, mux_com_port, ob1_com_port):

        # OB1 initialize
        ob1_path = 'ASRL' + str(ob1_com_port) + '::INSTR'
        Instr_ID = c_int32()
        pump = OB1_Initialization(ob1_path.encode('ascii'), 0, 0, 0, 0, byref(Instr_ID))
        pump = OB1_Add_Sens(Instr_ID, 1, 5, 1, 0, 7,
                            0)  # 16bit working range between 0-1000uL/min, also what are CustomSens_Voltage_5_to_25 and can I really choose any digital range?

        Calib_path = r'C:\Users\CyCIF PC\Documents\GitHub\AutoCIF\Python_64_elveflow\calibration\1_12_24_cal.txt'
        Calib = (c_double * 1000)()
        Elveflow_Calibration_Load(Calib_path.encode('ascii'), byref(Calib), 1000)
        #Elveflow_Calibration_Default(byref(Calib), 1000)
        OB1_Start_Remote_Measurement(Instr_ID.value, byref(Calib), 1000)
        self.calibration_array = byref(Calib)

        set_channel_regulator = int(1)  # convert to int
        set_channel_regulator = c_int32(set_channel_regulator)  # convert to c_int32
        set_channel_sensor = int(1)
        set_channel_sensor = c_int32(set_channel_sensor)  # convert to c_int32
        PID_Add_Remote(Instr_ID.value, set_channel_regulator, Instr_ID.value, set_channel_sensor, 0.9, 0.004, 1)

        # MUX intiialize
        path = 'ASRL' + str(mux_com_port) + '::INSTR'
        mux_Instr_ID = c_int32()
        MUX_DRI_Initialization(path.encode('ascii'), byref(
            mux_Instr_ID))  # choose the COM port, it can be ASRLXXX::INSTR (where XXX=port number)

        # home
        # answer = (c_char * 40)()
        self.mux_ID = mux_Instr_ID.value
        # MUX_DRI_Send_Command(self.mux_ID, 0, answer, 40)

        self.pump_ID = Instr_ID.value

        return

    def mux_end(self):

        MUX_DRI_Destructor(self.mux_ID)

    def valve_select(self, valve_number):
        '''
        Selects valve in mux unit with associated mux_id to the valve_number declared.
        :param c_int32 mux_id: mux_id given from mux_initialization method
        :param int valve_number: number of desired valve to be selected
        :return: Nothing
        '''

        desired_valve = valve_number
        valve_number = c_int32(valve_number)
        MUX_DRI_Set_Valve(self.mux_ID, valve_number, 0)  # 0 is shortest path. clockwise and cc are also options

        valve = c_int32(-1)
        MUX_DRI_Get_Valve(self.mux_ID, byref(valve))
        current_valve = int(valve.value)

        while current_valve != desired_valve:
            MUX_DRI_Get_Valve(self.mux_ID, byref(valve))
            current_valve = int(valve.value)
            # print('valve', current_valve, 'deired valve', desired_valve)
            time.sleep(1)

        time.sleep(1)

    def flow(self, flow_rate):

        set_channel = int(1)  # convert to int
        set_channel = c_int32(set_channel)  # convert to c_int32

        set_target = float(flow_rate)  # in uL/min for flow
        set_target = c_double(set_target)  # convert to c_double

        # OB1_Start_Remote_Measurement(self.pump_ID, self.calibration_array, 1000)
        OB1_Set_Remote_Target(self.pump_ID, set_channel, set_target)

        data_sens = c_double()
        data_reg = c_double()
        set_channel = int(1)  # convert to int
        set_channel = c_int32(set_channel)  # convert to c_int32
        time.sleep(3)
        error = OB1_Get_Remote_Data(self.pump_ID, set_channel, byref(data_reg), byref(data_sens))
        current_flow_rate = data_sens.value
        current_pressure = int(data_reg.value)
        print('current flow rate', int(current_flow_rate))
        print('error: ', error)


    def ob1_end(self):

        set_channel = int(1)  # convert to int
        set_channel = c_int32(set_channel)  # convert to c_int32

        data_sens = c_double()
        data_reg = c_double()

        x = 0
        self.flow(0)

        while x == 0:
            OB1_Get_Remote_Data(self.pump_ID, set_channel, byref(data_reg), byref(data_sens))
            flow_rate = data_sens.value

            if flow_rate < 10:
                x = 1
            if flow_rate > 10:
                x = 0
            time.sleep(1)

        OB1_Stop_Remote_Measurement(self.pump_ID)
        OB1_Destructor(self.pump_ID)

    def measure(self):

        set_channel = int(1)  # convert to int
        set_channel = c_int32(set_channel)  # convert to c_int32

        data_sens = c_double()
        data_reg = c_double()

        OB1_Get_Remote_Data(self.pump_ID, set_channel, byref(data_reg), byref(data_sens))

        pressure = data_reg.value
        flow_rate = data_sens.value
        time_stamp = time.time()

        return pressure, flow_rate, time_stamp

    def flow_recorder(self, time_step, total_time, file_name='none', plot=1):

        wb = Workbook()
        ws = wb.active
        ws.cell(row=1, column=1).value = 'Time'
        ws.cell(row=1, column=2).value = 'Flow Rate'
        ws.cell(row=1, column=3).value = 'Pressure'

        total_steps = int(total_time / time_step)
        pressure_points = np.random.rand(total_steps).astype('float16')
        time_points = np.random.rand(total_steps).astype('float16')
        flow_points = np.random.rand(total_steps).astype('float16')

        for t in range(0, total_steps):
            pressure_point, flow_point, time_point = self.measure()
            pressure_points[t] = pressure_point
            time_points[t] = t * time_step
            flow_points[t] = flow_point

            ws.cell(row=t + 2, column=1).value = t * time_step
            ws.cell(row=t + 2, column=2).value = flow_point
            ws.cell(row=t + 2, column=3).value = pressure_point

            time.sleep(time_step)

        # wb.save(filename = file_name)

        if plot == 1:
            plt.plot(time_points, flow_points, 'o', color='black')
            plt.show()

    def liquid_action(self, action_type, stain_valve=0, incub_val=45, heater_state=0):

        bleach_valve = 11
        pbs_valve = 12
        bleach_time = 3  # minutes
        stain_flow_time = 45  # seconds
        if heater_state == 0:
            stain_inc_time = incub_val  # minutes
        if heater_state == 1:
            stain_inc_time = 45  # minutes
        nuc_valve = 4
        nuc_flow_time = 45  # seconds
        nuc_inc_time = 3  # minutes

        if action_type == 'Bleach':

            self.valve_select(bleach_valve)
            self.flow(500)
            time.sleep(70)
            self.flow(-3)
            # time.sleep(bleach_time*60)
            self.valve_select(pbs_valve)

            for x in range(0, bleach_time):
                time.sleep(60)

            self.flow(500)
            time.sleep(80)
            self.flow(-3)
            time.sleep(5)

        elif action_type == 'Stain':

            if heater_state == 1:
                arduino.heater_state(1)
                arduino.chamber('drain')
            else:
                pass

            time.sleep(4)
            self.valve_select(stain_valve)
            self.flow(500)
            time.sleep(stain_flow_time)
            self.flow(-3)
            self.valve_select(pbs_valve)

            for x in range(0, stain_inc_time):
                time.sleep(60)
                print('Staining Time Elapsed ', x)

            # if heater_state == 1:
            #    arduino.heater_state(0)
            #    arduino.chamber('fill')
            # else:
            #    pass

            self.valve_select(pbs_valve)
            time.sleep(30)
            self.flow(500)
            time.sleep(80)
            self.flow(-3)
            time.sleep(5)


        elif action_type == "Wash":

            self.valve_select(pbs_valve)
            self.flow(500)
            time.sleep(70)
            self.flow(-3)


        elif action_type == 'Nuc_Touchup':

            self.valve_select(nuc_valve)
            self.flow(500)
            time.sleep(nuc_flow_time)
            self.flow(0)
            time.sleep(nuc_inc_time * 60)

            self.valve_select(pbs_valve)
            self.flow(450)
            time.sleep(70)
            self.flow(0)

        elif action_type == 'PBS_flow_on':

            self.valve_select(pbs_valve)
            self.flow(500)
            time.sleep(10)

        elif action_type == 'PBS_flow_off':

            self.valve_select(pbs_valve)
            self.flow(0)
            time.sleep(10)