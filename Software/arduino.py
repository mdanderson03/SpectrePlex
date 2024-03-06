import serial



class arduino:

    def __init__(self):


    def heater_state(self, state):

        heater = serial.Serial(port='COM7', baudrate=9600)
        if state ==1:
            heater.write(b'ON\r')
        elif state == 0:
            heater.write(b'OFF\r')

