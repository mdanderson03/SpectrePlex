# Python wrapper for ElveFlow64DLL
# Version : 3.10
# Author : Elveflow
# Copyright : Copyright � 2025

# this python routine load the ElveflowDLL.
# It defines all function prototype for use with python lib

from ctypes import *
ElveflowDLL=CDLL(r'C:\Users\CyCIF PC\Documents\GitHub\AutoCIF\Python_64_elveflow\DLL64/Elveflow64.dll')# change this path 


 # Elveflow Library
 # Mux Device
 # 
 # Initiate the MUX device using device name (could be obtained in NI MAX). It 
 # return the F_S_R ID (number >=0) to be used with other function
 #
def MUX_Initialization (Device_Name, MUX_ID_out):
	X_MUX_Initialization=ElveflowDLL.MUX_Initialization
	X_MUX_Initialization.argtypes=[c_char_p, POINTER(c_int32)]
	return X_MUX_Initialization (Device_Name, MUX_ID_out)



 # Elveflow Library
 # Mux Device
 # 
 # Valves are set by a array of 16 element. If the valve value is equal or 
 # below 0, valve is close, if it's equal or above 1 the valve is open. The 
 # index in the array indicate the selected  valve as shown below : 
 # 0   1   2   3
 # 4   5   6   7
 # 8   9   10  11
 # 12  13  14  15
 # If the array does not contain exactly 16 element nothing happened
 # 
 #
# use ctypes c_int32*16 for array_valve_in
def MUX_Set_all_valves (MUX_ID_in, array_valve_in, len):
	X_MUX_Set_all_valves=ElveflowDLL.MUX_Set_all_valves
	X_MUX_Set_all_valves.argtypes=[c_int32, POINTER(c_int32), c_int32]
	return X_MUX_Set_all_valves (MUX_ID_in, array_valve_in, len)



 # Elveflow Library
 # MUX D-R-I Device
 # 
 # Initiate the MUX Distribution, Distributor, Recirculation or Injection 
 # device using device COM port (ASRLXXX::INSTR where XXX is usually the COM 
 # port that could be found in Windows device manager). It returns the MUX 
 # D-R-I ID (number >=0) to be used with other function.
 #
def MUX_DRI_Initialization (Visa_COM, MUX_DRI_ID_out):
	X_MUX_DRI_Initialization=ElveflowDLL.MUX_DRI_Initialization
	X_MUX_DRI_Initialization.argtypes=[c_char_p, POINTER(c_int32)]
	return X_MUX_DRI_Initialization (Visa_COM, MUX_DRI_ID_out)



 # Elveflow Library
 # OB1 Device
 # 
 # Initialize the OB1 device using device name and regulators type (see SDK 
 # Z_regulator_type for corresponding numbers). It modify the OB1 ID (number 
 # >=0). This ID can be used be used with other function to identify the 
 # targed OB1. If an error occurs during the initialization process, the OB1 
 # ID value will be -1. 
 #
def OB1_Initialization (Device_Name, Reg_Ch_1, Reg_Ch_2, Reg_Ch_3, Reg_Ch_4, OB1_ID_out):
	X_OB1_Initialization=ElveflowDLL.OB1_Initialization
	X_OB1_Initialization.argtypes=[c_char_p, c_uint16, c_uint16, c_uint16, c_uint16, POINTER(c_int32)]
	return X_OB1_Initialization (Device_Name, Reg_Ch_1, Reg_Ch_2, Reg_Ch_3, Reg_Ch_4, OB1_ID_out)



 # Elveflow Library
 # OB1 Device
 # 
 # Launch OB1 calibration and return the calibration array. Before 
 # Calibration, ensure that ALL channels are proprely closed with adequate 
 # caps. 
 # Len correspond to the Calib_array_out length.
 #
# use ctypes c_double*1000 for calibration array
def OB1_Calib (OB1_ID_in):
	X_OB1_Calib=ElveflowDLL.OB1_Calib
	X_OB1_Calib.argtypes=[c_int32]
	return X_OB1_Calib (OB1_ID_in)



 # Elveflow Library
 # OB1 Device
 # 
 # Set the pressure of the OB1 selected channel, Calibration array is required 
 # (use Set_Default_Calib if required). Len correspond to the Calib_array_in 
 # length.
 #
# use ctypes c_double*1000 for calibration array
def OB1_Set_Press (OB1_ID, Channel_1_to_4, PressureTarget):
	X_OB1_Set_Press=ElveflowDLL.OB1_Set_Press
	X_OB1_Set_Press.argtypes=[c_int32, c_int32, c_double]
	return X_OB1_Set_Press (OB1_ID, Channel_1_to_4, PressureTarget)



 # Elveflow Library
 # OB1 Device
 # 
 # Close communication with OB1
 #
def OB1_Destructor (OB1_ID):
	X_OB1_Destructor=ElveflowDLL.OB1_Destructor
	X_OB1_Destructor.argtypes=[c_int32]
	return X_OB1_Destructor (OB1_ID)



 # Elveflow Library
 # OB1 Device
 # 
 # Get the trigger of the OB1 (0 = 0V, 1 =3,3V for MK3 devices, 5V for MK4 
 # devices)
 #
def OB1_Get_Trig (OB1_ID, Trigger):
	X_OB1_Get_Trig=ElveflowDLL.OB1_Get_Trig
	X_OB1_Get_Trig.argtypes=[c_int32, POINTER(c_int32)]
	return X_OB1_Get_Trig (OB1_ID, Trigger)



 # Elveflow Library
 # Mux Device
 # 
 # Close the communication of the MUX device
 #
def MUX_Destructor (MUX_ID_in):
	X_MUX_Destructor=ElveflowDLL.MUX_Destructor
	X_MUX_Destructor.argtypes=[c_int32]
	return X_MUX_Destructor (MUX_ID_in)



 # Elveflow Library
 # Mux Device
 # 
 # Get the trigger of the MUX device (0=0V, 1=5V).
 #
def MUX_Get_Trig (MUX_ID_in, Trigger):
	X_MUX_Get_Trig=ElveflowDLL.MUX_Get_Trig
	X_MUX_Get_Trig.argtypes=[c_int32, POINTER(c_int32)]
	return X_MUX_Get_Trig (MUX_ID_in, Trigger)



 # Elveflow Library
 # Mux Device
 # 
 # Set the state of one valve of the instrument. The desired valve is 
 # addressed using Input and Output parameter which corresponds to the 
 # fluidics inputs and outputs of the instrument. 
 #
def MUX_Set_indiv_valve (MUX_ID_in, Input, Ouput, OpenClose):
	X_MUX_Set_indiv_valve=ElveflowDLL.MUX_Set_indiv_valve
	X_MUX_Set_indiv_valve.argtypes=[c_int32, c_int32, c_int32, c_int32]
	return X_MUX_Set_indiv_valve (MUX_ID_in, Input, Ouput, OpenClose)



 # Elveflow Library
 # Mux Device
 # 
 # Set the Trigger of the MUX device (0=0V, 1=5V).
 #
def MUX_Set_Trig (MUX_ID_in, Trigger):
	X_MUX_Set_Trig=ElveflowDLL.MUX_Set_Trig
	X_MUX_Set_Trig.argtypes=[c_int32, c_int32]
	return X_MUX_Set_Trig (MUX_ID_in, Trigger)



 # Elveflow Library
 # MUX D-R-I Device
 # 
 # Close Communication with MUX Distribution, Distributor, Recirculation or 
 # Injection device.
 #
def MUX_DRI_Destructor (MUX_DRI_ID_in):
	X_MUX_DRI_Destructor=ElveflowDLL.MUX_DRI_Destructor
	X_MUX_DRI_Destructor.argtypes=[c_int32]
	return X_MUX_DRI_Destructor (MUX_DRI_ID_in)



 # Elveflow Library
 # MUX D-R-I Device
 # 
 # Get the current valve number. If the valve is changing, function returns 0.
 #
def MUX_DRI_Get_Valve (MUX_DRI_ID_in, selected_Valve):
	X_MUX_DRI_Get_Valve=ElveflowDLL.MUX_DRI_Get_Valve
	X_MUX_DRI_Get_Valve.argtypes=[c_int32, POINTER(c_int32)]
	return X_MUX_DRI_Get_Valve (MUX_DRI_ID_in, selected_Valve)



 # Elveflow Library
 # MUX D-R-I Device
 # 
 # Switch the MUX Distribution, Distributor, Recirculation or Injection to the 
 # desired valve.
 # For MUX Distribution 12, between 1-12.
 # For MUX Distributor (6 or 10 valves), between 1-6 or 1-10.
 # For MUX Recirculation 6 or MUX Injection (6 valves), the two states are 1 
 # or 2.
 # 
 # Rotation indicates the path the valve will perform to select a valve, 
 # either shortest 0, clockwise 1 or counter clockwise 2.
 #
def MUX_DRI_Set_Valve (MUX_DRI_ID_in, selected_Valve, Rotation):
	X_MUX_DRI_Set_Valve=ElveflowDLL.MUX_DRI_Set_Valve
	X_MUX_DRI_Set_Valve.argtypes=[c_int32, c_int32, c_uint16]
	return X_MUX_DRI_Set_Valve (MUX_DRI_ID_in, selected_Valve, Rotation)



 # Elveflow Library
 # OB1 Device
 # 
 # Add sensor to OB1 device. Select the channel n� (1-4) the sensor type. 
 # 
 # For Flow sensor, the type of communication (Analog/Digital), the 
 # Calibration for digital version (H20 or IPA) should be specify as well as 
 # digital resolution (9 to 16 bits). (see SDK user guide,  Z_sensor_type_type 
 # , Z_sensor_digit_analog, Z_Sensor_FSD_Calib and Z_D_F_S_Resolution for 
 # number correspondance)
 # 
 # For digital version, the sensor type is automatically detected during this 
 # function call. 
 # 
 # For Analog sensor, the calibration parameters is not taken into account. 
 # 
 # If the sensor is not compatible with the OB1 version, or no digital sensor 
 # are detected an error will be thrown as output of the function.
 #
def OB1_Add_Sens (OB1_ID, Channel_1_to_4, DigitalAnalog, SensorType, FSens_Digit_Calib, FSens_Digit_Resolution, CustomSens_Voltage_5_to_25):
	X_OB1_Add_Sens=ElveflowDLL.OB1_Add_Sens
	X_OB1_Add_Sens.argtypes=[c_int32, c_int32, c_uint16, c_uint16, c_uint16, c_uint16, c_double]
	return X_OB1_Add_Sens (OB1_ID, Channel_1_to_4, DigitalAnalog, SensorType, FSens_Digit_Calib, FSens_Digit_Resolution, CustomSens_Voltage_5_to_25)



 # Elveflow Library
 # BFS Device
 # 
 # Close Communication with BFS device
 #
def BFS_Destructor (BFS_ID_in):
	X_BFS_Destructor=ElveflowDLL.BFS_Destructor
	X_BFS_Destructor.argtypes=[c_int32]
	return X_BFS_Destructor (BFS_ID_in)



 # Elveflow Library
 # BFS Device
 # 
 # Initiate the BFS device using device com port (ASRLXXX::INSTR where XXX is 
 # the com port that could be found in windows device manager). It return the 
 # BFS ID (number >=0) to be used with other function 
 #
def BFS_Initialization (Visa_COM, Filter, M_temp, M_density, BFS_ID_out):
	X_BFS_Initialization=ElveflowDLL.BFS_Initialization
	X_BFS_Initialization.argtypes=[c_char_p, c_double, c_int32, c_int32, POINTER(c_int32)]
	return X_BFS_Initialization (Visa_COM, Filter, M_temp, M_density, BFS_ID_out)



 # Elveflow Library - ONLY FOR ILLUSTRATION - 
 # OB1 Devices
 # 
 # This function is only provided for illustration purpose, to explain how to 
 # do your own feedback loop. Elveflow does not guarante neither efficient nor 
 # optimum regulation with this illustration of PI regulator . With this 
 # function the PI parameters have to be tuned for every regulator and every 
 # microfluidic circuit.   
 # 
 # In this function need to be initiate with a first call where PID_ID =-1. 
 # The PID_out will provide the new created PID_ID. This ID should be use in 
 # further call. 
 # 
 # General remarks of this PI regulator :
 # 
 # The error "e" is calculate for every step as e=target value-actual value
 # There are 2 contributions to a PI regulator: proportional contribution 
 # which only depend on this step and  Prop=e#P and integral part which is the 
 # "memory" of the regulator. This value is calculated as 
 # Integ=integral(I#e#dt) and can be reset. 
 #   
 #
def Elveflow_EXAMPLE_PID (PID_ID_in, currentValue, Reset, P, I, targetValue, PID_ID_out, value):
	X_Elveflow_EXAMPLE_PID=ElveflowDLL.Elveflow_EXAMPLE_PID
	X_Elveflow_EXAMPLE_PID.argtypes=[c_int32, c_double, c_int32, c_double, c_double, c_double, POINTER(c_int32), POINTER(c_double)]
	return X_Elveflow_EXAMPLE_PID (PID_ID_in, currentValue, Reset, P, I, targetValue, PID_ID_out, value)



 # Elveflow Library
 # Mux Device
 # 
 # Valves are set by a array of 16 element. If the valve value is equal or 
 # below 0, valve is close, if it's equal or above 1 the valve is open. If the 
 # array does not contain exactly 16 element nothing happened
 # 
 #
def MUX_Wire_Set_all_valves (MUX_ID_in, array_valve_in, len):
	X_MUX_Wire_Set_all_valves=ElveflowDLL.MUX_Wire_Set_all_valves
	X_MUX_Wire_Set_all_valves.argtypes=[c_int32, POINTER(c_int32), c_int32]
	return X_MUX_Wire_Set_all_valves (MUX_ID_in, array_valve_in, len)



 # Elveflow Library
 # OB1 Device
 # 
 # Returns all the pressure and sensor measurements on all 4 channels of the 
 # OB1 device
 #
def OB1_Get_All_Data (OB1_ID, PressureChannel1, SensorChannel1, PressureChannel2, SensorChannel2, PressureChannel3, SensorChannel3, PressureChannel4, SensorChannel4):
	X_OB1_Get_All_Data=ElveflowDLL.OB1_Get_All_Data
	X_OB1_Get_All_Data.argtypes=[c_int32, POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double)]
	return X_OB1_Get_All_Data (OB1_ID, PressureChannel1, SensorChannel1, PressureChannel2, SensorChannel2, PressureChannel3, SensorChannel3, PressureChannel4, SensorChannel4)



 # Elveflow Library
 # BFS Device
 # 
 # Perform zero calibration of the BFS. Ensure that there is no flow when 
 # performed; it is advised to use valves.
 # The calibration procedure is finished when the green LED stop blinking.
 #
def BFS_Zero (BFS_ID_in):
	X_BFS_Zero=ElveflowDLL.BFS_Zero
	X_BFS_Zero.argtypes=[c_int32]
	return X_BFS_Zero (BFS_ID_in)



 # Warning: advanced feature. Reset OB1 communication for pressure and flow.
 #
def OB1_Reset_Instr (OB1_ID):
	X_OB1_Reset_Instr=ElveflowDLL.OB1_Reset_Instr
	X_OB1_Reset_Instr.argtypes=[c_int32]
	return X_OB1_Reset_Instr (OB1_ID)



 # Warning: advanced feature. Reset digital sensor communication from the 
 # selected channel. Select again resolution and calibration type 
 # (H2O/Isopro).
 #
def OB1_Reset_Digit_Sens (OB1_ID, Channel_1_to_4):
	X_OB1_Reset_Digit_Sens=ElveflowDLL.OB1_Reset_Digit_Sens
	X_OB1_Reset_Digit_Sens.argtypes=[c_int32, c_int32]
	return X_OB1_Reset_Digit_Sens (OB1_ID, Channel_1_to_4)



 # Elveflow Library
 # MSRD Device
 # 
 # Initialize the Sensor Reader device able to read digital sensors (MSRD) 
 # using device name and sensors type (see SDK Z_sensor_type for corresponding 
 # numbers). It modify the MSRD ID (number >=0). This ID can be used with 
 # other function to identify the targeted MSRD. If an error occurs during the 
 # initialization process, the MSRD ID value will be -1. 
 # 
 # Initiate the communication with the Sensor Reader able to read digital 
 # sensors (MSRD). This VI generates an identification cluster of the 
 # instrument to be used with other VIs.
 # 
 # NB: Sensor type has to be written here in addition to the "Add_Sens".
 # NB 2: Sensor connected to channel 1-2 and 3-4 have to be the same type 
 # otherwise they will not be taken into account. 
 #
def M_S_R_D_Initialization (Device_Name, Sens_Ch_1, Sens_Ch_2, Sens_Ch_3, Sens_Ch_4, CustomSens_Voltage_Ch12, CustomSens_Voltage_Ch34, MSRD_ID_out):
	X_M_S_R_D_Initialization=ElveflowDLL.M_S_R_D_Initialization
	X_M_S_R_D_Initialization.argtypes=[c_char_p, c_uint16, c_uint16, c_uint16, c_uint16, c_double, c_double, POINTER(c_int32)]
	return X_M_S_R_D_Initialization (Device_Name, Sens_Ch_1, Sens_Ch_2, Sens_Ch_3, Sens_Ch_4, CustomSens_Voltage_Ch12, CustomSens_Voltage_Ch34, MSRD_ID_out)



 # Elveflow Library
 # MSRD Device
 # 
 # Add sensor to MSRD device. Select the channel n� (1-4) the sensor type. 
 # 
 # For Flow sensor, the type of communication (Analog/Digital), the 
 # Calibration for digital version (H20 or IPA) should be specify as well as 
 # digital resolution (9 to 16 bits). (see SDK user guide,  Z_sensor_type_type 
 # , Z_sensor_digit_analog, Z_Sensor_FSD_Calib and Z_D_F_S_Resolution for 
 # number correspondance)
 # 
 # For digital version, the sensor type is automatically detected during this 
 # function call. 
 # 
 # For Analog sensor, the calibration parameters is not taken into account. 
 # 
 # If the sensor is not compatible with the MSRD version, or no digital sensor 
 # are detected an error will be thrown as output of the function.
 # 
 # NB: Sensor type has to be the same as in the "Initialization" step.
 #
def M_S_R_D_Add_Sens (M_S_R_D_ID, Channel_1_to_4, SensorType, DigitalAnalog, FSens_Digit_Calib, FSens_Digit_Resolution, Custom_Sensor_VoltageIn525VOptional):
	X_M_S_R_D_Add_Sens=ElveflowDLL.M_S_R_D_Add_Sens
	X_M_S_R_D_Add_Sens.argtypes=[c_int32, c_int32, c_uint16, c_uint16, c_uint16, c_uint16, c_double]
	return X_M_S_R_D_Add_Sens (M_S_R_D_ID, Channel_1_to_4, SensorType, DigitalAnalog, FSens_Digit_Calib, FSens_Digit_Resolution, Custom_Sensor_VoltageIn525VOptional)



 # Elveflow Library
 # MSRD Device
 # 
 # Close communication with MSRD
 #
def M_S_R_D_Destructor (M_S_R_D_ID):
	X_M_S_R_D_Destructor=ElveflowDLL.M_S_R_D_Destructor
	X_M_S_R_D_Destructor.argtypes=[c_int32]
	return X_M_S_R_D_Destructor (M_S_R_D_ID)



 # Elveflow Library
 # MSRD Device
 # 
 # Read the sensor of the requested channel.s
 # Units: Flow sensor: �l/min
 # Pressure: mbar
 # 
 # NB: For Digital Flow Senor, If the connection is lost, MSRD will be reseted 
 # and the return value will be zero
 #
def M_S_R_D_Get_Data (M_S_R_D_ID, Channel_1_to_4, Sens_Data):
	X_M_S_R_D_Get_Data=ElveflowDLL.M_S_R_D_Get_Data
	X_M_S_R_D_Get_Data.argtypes=[c_int32, c_int32, POINTER(c_double)]
	return X_M_S_R_D_Get_Data (M_S_R_D_ID, Channel_1_to_4, Sens_Data)



 # Elveflow Library
 # MUX D-R-I Device
 # 
 # !This function only works for MUX Distribution 12 or Recirculation 6!
 # 
 # Get the Serial Number or Home the valve. len is the length of the Answer. 
 # Remember that Home the valve take several seconds.
 # 
 # Home the valve is necessary as an initialization step before using the 
 # valve for a session.
 #
def MUX_DRI_Send_Command (MUX_DRI_ID_in, Action, Answer, len):
	X_MUX_DRI_Send_Command=ElveflowDLL.MUX_DRI_Send_Command
	X_MUX_DRI_Send_Command.argtypes=[c_int32, c_uint16, c_char_p, c_int32]
	return X_MUX_DRI_Send_Command (MUX_DRI_ID_in, Action, Answer, len)



 # Elveflow Library
 # MSRD Device
 # 
 # Set filter for the corresponding channel.
 #
def M_S_R_D_Set_Filt (M_S_R_D_ID, Channel_1_to_4, FilterRunning):
	X_M_S_R_D_Set_Filt=ElveflowDLL.M_S_R_D_Set_Filt
	X_M_S_R_D_Set_Filt.argtypes=[c_int32, c_int32, c_int32]
	return X_M_S_R_D_Set_Filt (M_S_R_D_ID, Channel_1_to_4, FilterRunning)



 # Elveflow Library
 # OB1 Device
 # 
 # Read the sensor and regulator values of the requested channel.
 # Warning: This Function only extracts data obtained in the remote 
 # measurement loop
 # Sensor unit : mbar if pressure sensor, �l/min if flow sensor
 # Regulator unit : mbar
 # 
 # NB: For Digital Flow Senor, If the connection is lost, OB1 will be reseted 
 # and the return value will be zero
 #
def OB1_Get_Data (OB1_ID, Channel_1_to_4, Reg_Data, Sens_Data):
	X_OB1_Get_Data=ElveflowDLL.OB1_Get_Data
	X_OB1_Get_Data.argtypes=[c_int32, c_int32, POINTER(c_double), POINTER(c_double)]
	return X_OB1_Get_Data (OB1_ID, Channel_1_to_4, Reg_Data, Sens_Data)



 # Elveflow Library
 # PID Module
 # 
 # Set to run/pause the selected PID loop. Only works when using the remote 
 # measurement functions.
 #
def PID_Set_Running_Remote (Regulator_ID, Channel_1_to_4, Running):
	X_PID_Set_Running_Remote=ElveflowDLL.PID_Set_Running_Remote
	X_PID_Set_Running_Remote.argtypes=[c_int32, c_int32, c_int32]
	return X_PID_Set_Running_Remote (Regulator_ID, Channel_1_to_4, Running)



 # Elveflow Library
 # PID Module
 # 
 # Configure a PID loop between a regulator and a sensor. Only works when 
 # using the remote measurement functions.
 #
def PID_Add_Remote (Regulator_ID, Regulator_Channel_1_to_4, ID_Sensor, Sensor_Channel_1_to_4, P, I, Running):
	X_PID_Add_Remote=ElveflowDLL.PID_Add_Remote
	X_PID_Add_Remote.argtypes=[c_int32, c_int32, c_int32, c_int32, c_double, c_double, c_int32]
	return X_PID_Add_Remote (Regulator_ID, Regulator_Channel_1_to_4, ID_Sensor, Sensor_Channel_1_to_4, P, I, Running)



 # Elveflow Library
 # BFS Device
 # 
 # Read the sensors from the remote monitoring loop:
 # Units: Flow sensor: �l/min
 #            Density: g/m3
 #            Temperature: Celcius
 #
def BFS_Get_Data (BFS_ID, Flow, Temperature, Density):
	X_BFS_Get_Data=ElveflowDLL.BFS_Get_Data
	X_BFS_Get_Data.argtypes=[c_int32, POINTER(c_double), POINTER(c_double), POINTER(c_double)]
	return X_BFS_Get_Data (BFS_ID, Flow, Temperature, Density)



 # Elveflow Library
 # BFS Device
 # 
 # Modify the parameters of the remote monitoring loop:
 # M_density: a new measure of the density will be taken before each flow 
 # measurement
 # M_temp: a new temperature measurement will be taken after each flow 
 # measurement
 # Filter: change the filter used to measure the flow
 #
def BFS_Set_Params (BFS_ID, Filter, M_temp, M_density):
	X_BFS_Set_Params=ElveflowDLL.BFS_Set_Params
	X_BFS_Set_Params.argtypes=[c_int32, c_double, c_int32, c_int32]
	return X_BFS_Set_Params (BFS_ID, Filter, M_temp, M_density)



 # Elveflow Library
 # OB1 Device
 # 
 # Set the trigger of the OB1 (0 = 0V, 1 =3,3V for MK3 devices, 5V for MK4 
 # devices)
 #
def OB1_Set_Trig (OB1_ID, TriggerIn):
	X_OB1_Set_Trig=ElveflowDLL.OB1_Set_Trig
	X_OB1_Set_Trig.argtypes=[c_int32, c_int32]
	return X_OB1_Set_Trig (OB1_ID, TriggerIn)



 # Elveflow Library
 # PID Module
 # 
 # Change the parameters of the selected PID loop. Only works when using the 
 # remote measurement functions.
 #
def PID_Set_Params_Remote (Regulator_ID, Channel_1_to_4, Reset, P, I):
	X_PID_Set_Params_Remote=ElveflowDLL.PID_Set_Params_Remote
	X_PID_Set_Params_Remote.argtypes=[c_int32, c_int32, c_int32, c_double, c_double]
	return X_PID_Set_Params_Remote (Regulator_ID, Channel_1_to_4, Reset, P, I)



 # Warning: advanced feature. Reset MSRD communication.
 #
def M_S_R_D_Reset_Instr (M_S_R_D_ID):
	X_M_S_R_D_Reset_Instr=ElveflowDLL.M_S_R_D_Reset_Instr
	X_M_S_R_D_Reset_Instr.argtypes=[c_int32]
	return X_M_S_R_D_Reset_Instr (M_S_R_D_ID)



 # Warning: advanced feature. Reset digital sensor communication from the 
 # selected channel. Select again resolution and calibration type 
 # (H2O/Isopro).
 #
def M_S_R_D_Reset_Sens (M_S_R_D_ID):
	X_M_S_R_D_Reset_Sens=ElveflowDLL.M_S_R_D_Reset_Sens
	X_M_S_R_D_Reset_Sens.argtypes=[c_int32]
	return X_M_S_R_D_Reset_Sens (M_S_R_D_ID)



 # Elveflow_Checkaliases
 #
def Elveflow_Checkaliases ():
	X_Elveflow_Checkaliases=ElveflowDLL.Elveflow_Checkaliases
	X_Elveflow_Checkaliases.argtypes=[]
	return X_Elveflow_Checkaliases ()



 # Elveflow Library
 # MSR Device
 # 
 # Get the state of the INT trigger (or Trigger IN). If nothing is connected 
 # it returns a High state. 
 # Low=>0V
 # High=>5V
 # 
 # NB: For Digital Flow Senor, If the connection is lost, MSRD will be reseted 
 # and the return value will be zero
 #
def M_S_R_D_Get_Trig (M_S_R_D_ID, GetTriggerInput):
	X_M_S_R_D_Get_Trig=ElveflowDLL.M_S_R_D_Get_Trig
	X_M_S_R_D_Get_Trig.argtypes=[c_int32, ]
	return X_M_S_R_D_Get_Trig (M_S_R_D_ID, GetTriggerInput)



 # Elveflow Library
 # MSR Device
 # 
 # Set the trigger OUT of the MSR (HIGH =>5V, LOW => 0V)
 # 
 # NB: only works for the MSR MCU device
 #
def M_S_R_D_Set_Trig (M_S_R_D_ID, SetTriggerOutput):
	X_M_S_R_D_Set_Trig=ElveflowDLL.M_S_R_D_Set_Trig
	X_M_S_R_D_Set_Trig.argtypes=[c_int32, ]
	return X_M_S_R_D_Set_Trig (M_S_R_D_ID, SetTriggerOutput)



 # Elveflow Library
 # Mux Device
 # 
 # Get valve type plugged into your MUX Wire
 # 
 # 
 #
def MUX_Get_valves_Type (MUX_ID_in, Types_array, len):
	X_MUX_Get_valves_Type=ElveflowDLL.MUX_Get_valves_Type
	X_MUX_Get_valves_Type.argtypes=[c_int32, POINTER(c_int32), c_int32]
	return X_MUX_Get_valves_Type (MUX_ID_in, Types_array, len)



 # Elveflow Library
 # Mux Device
 # 
 # Set the vavle type.
 # This function is available for MUX Wire V3 using custom Valves or Valve V2.
 # Valve V3 type are automatically recognized by the MUX
 # 
 # ValveNB (MUX port where your valve is plugged)
 # Type (Type of valve plugged)
 # 
 #
def MUX_Set_valves_Type (MUX_ID_in, ValveNb, Type):
	X_MUX_Set_valves_Type=ElveflowDLL.MUX_Set_valves_Type
	X_MUX_Set_valves_Type.argtypes=[c_int32, c_int32, ]
	return X_MUX_Set_valves_Type (MUX_ID_in, ValveNb, Type)



 # Elveflow Library
 # BFS Device
 # 
 # Set the instruement Filter. 0.000001= maximum filter -> slow change but 
 # very low noise.  1= no filter-> fast change but noisy. 
 # 
 # Default value is 0.1  
 #
def BFS_Set_Filter (BFS_ID_in, Filter_value):
	X_BFS_Set_Filter=ElveflowDLL.BFS_Set_Filter
	X_BFS_Set_Filter.argtypes=[c_int32, c_double]
	return X_BFS_Set_Filter (BFS_ID_in, Filter_value)



 # Elveflow Library
 # OB1 Device
 # 
 # Saves the actual calibration to the desired path. The function prompts the 
 # user to choose a path if no path is specified.
 #
def OB1_Calib_Save (OB1_ID_in, Path):
	X_OB1_Calib_Save=ElveflowDLL.OB1_Calib_Save
	X_OB1_Calib_Save.argtypes=[c_int32, c_char_p]
	return X_OB1_Calib_Save (OB1_ID_in, Path)



 # Elveflow Library
 # OB1 Device
 # 
 # Launch OB1 calibration and return the calibration array. Before 
 # Calibration, ensure that ALL channels are proprely closed with adequate 
 # caps. 
 # Len correspond to the Calib_array_out length.
 #
def OB1_Calib_Load (OB1_ID_in, Path):
	X_OB1_Calib_Load=ElveflowDLL.OB1_Calib_Load
	X_OB1_Calib_Load.argtypes=[c_int32, c_char_p]
	return X_OB1_Calib_Load (OB1_ID_in, Path)



 # Elveflow Library
 # OB1 Device
 # 
 # Set the Target of the OB1 selected channel. Modify the pressure if the PID 
 # is off, or the sensor is a pressure sensor. Modify a flow if the sensor is 
 # a flow sensor and the PID is on.
 #
def OB1_Set_Sens (OB1_ID, Channel_1_to_4, Target):
	X_OB1_Set_Sens=ElveflowDLL.OB1_Set_Sens
	X_OB1_Set_Sens.argtypes=[c_int32, c_int32, c_double]
	return X_OB1_Set_Sens (OB1_ID, Channel_1_to_4, Target)




def DEBUG_status():
	Status = ElveflowDLL.LVDLLStatus
	Status.argtypes=[c_char_p, c_int32, c_void_p]
	# Prepare the parameters
	err_str = create_string_buffer(256)  # Buffer for error message
	err_str_len = c_int(len(err_str))  # Length of error message buffer
	module = None  # Modify this depending on what "module" should reference
	# Call the function
	result = Status(err_str, err_str_len, module)

	# Print the result
	print(f"Function returned: {result}")
	print(f"Error message: {err_str.value.decode()}")