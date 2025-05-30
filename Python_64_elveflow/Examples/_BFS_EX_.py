#tested with Python 3.6.10
#add python_xx and python_xx/DLL to the project path

import sys
from _ast import Load
sys.path.append('D:/dev/SDK/DLL64/DLL64')#add the path of the library here
sys.path.append('D:/dev/SDK/Python_64')#add the path of the LoadElveflow.py

from ctypes import *

from array import array

from Elveflow64 import *

#
# Initialization of BFS ( ! ! ! REMEMBER TO USE .encode('ascii') ! ! ! )
#
Instr_ID=c_int32()
print("Instrument name is hardcoded in the Python script")
#see User Guide and NIMAX to determine the instrument name 
error=BFS_Initialization("ASRL5::INSTR".encode('ascii'),0.001,1,1,byref(Instr_ID))#choose the com port, it can be ASRLXXX::INSTR (where XXX=port number)
#all functions will return error codes to help you to debug your code, for further information refer to User Guide
print('error:%d' % error)
print("Instr ID: %d" % Instr_ID.value)

#
#Main loop 
#
    
repeat=True
while repeat:
    answer=input('What to do (set_filter, do_zero, read_channel, or exit) : ')

    if answer=="set_filter":
        filter=0.001
        filter=input("select filter(1= minimum filter, 0,00001 maximum filter) : ")
        filter=float(filter)
        filter=c_double(filter)#convert to c_double
        error=BFS_Set_Filter(Instr_ID.value,filter)
        
    if answer=="do_zero":#put valves to stop the flow before performing zeroing
        print('Zeroing last approximately 10 sec. Wait for the LED to stop blinking before sending an other command. Read corresponding User Guide to perform correctly the zeroing procedure.')
        error=BFS_Zero(Instr_ID.value)
           
            
    if answer=="read_channel":
        data_sens=c_double()
        data_dens=c_double()
        data_temp=c_double()
        error=BFS_Get_Data(Instr_ID.value, byref(data_sens),byref(data_temp),byref(data_dens))
        print('Sensor/temperature/density: ', data_sens.value,'; ',data_temp.value,'; ',data_dens.value)
           
    if answer=='exit':
        repeat=False
    
    print( 'error :', error)
    error=0
        

error=BFS_Destructor(Instr_ID.value)