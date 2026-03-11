#!/usr/bin/python
# ----------------------------------------------------------------------------------------
# A basic class for the control of the PSD4 syringe pump from Hamilton
# ----------------------------------------------------------------------------------------
# Jeff Moffitt
# 11/20/21
# jeffrey.moffitt@childrens.harvard.edu
#
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# Import
# ----------------------------------------------------------------------------------------
import sys
import time
import serial




class APump():
    def __init__(self, parameters = False):

        # Define attributes
        self.com_port = parameters.get("pump_com_port", "COM3")
        self.pump_ID = parameters.get("pump_ID", "PSD4")
        self.verbose = parameters.get("verbose", True)
        self.simulate = parameters.get("simulate_pump", False)
        self.serial_verbose = parameters.get("serial_verbose", False)
        self.high_res_mode = parameters.get("high_res", True)
        self.syringe_volume = parameters.get("syringe_volume", 12.5)
        self.syringe_type = parameters.get("syringe_type", "smooth_flow")
        self.return_step_count = parameters.get("return_step_count", 1000)
        self.backoff_step_count = parameters.get("backoff_step_count", 2000)
        
        self.min_velocity_in_steps_s = 2
        self.max_velocity_in_steps_s = 10000
        self.min_stroke_in_steps = 0
        self.max_stroke_in_steps = None

        # Check syringe type
        if self.syringe_type == "standard":
            self.high_res_step = 24000.0
            self.low_res_step = 3000.0
        elif self.syringe_type == "smooth_flow":
            self.high_res_step = 192000.0
            self.low_res_step = 24000.0
        else:
            print("The provided syringe type for the PSD4 is not valid")
            assert False

        # Define the resolution mode
        if self.high_res_mode:
            self.max_stroke_in_steps = self.high_res_step
        else:
            self.max_stroke_in_steps = self.low_res_step
           
        self.steps_to_volume = self.syringe_volume/self.max_stroke_in_steps
        
        # Create serial port
        self.serial = serial.Serial(port = self.com_port, 
                                    timeout=0.5)

        # Define initial pump status
        self.flow_status = "Stopped"
        self.speed = 0.0
        self.num_ports = 0
        
        # Configure pump
        self.configurePump()
        self.identification = "PSD4"
        
        # Report configuration
        print("--------------------------------")
        print("Configured PSD4 Syringe Pump")
        print("   PSD4 Type: " + str(self.syringe_type))
        print("   Syringe Volume: " + str(self.syringe_volume) + " mL")
        print("   High Res Mode: " + str(self.high_res_mode))
        print("   Steps for Full Fill: " + str(self.max_stroke_in_steps))
        print("   Minimum Speed: " + str(self.min_velocity_in_steps_s * 1000 * self.syringe_volume * (4/self.high_res_step) * 60 ) + " uL/min")

    def initializePump(self):

        message = "/1k" + str(self.backoff_step_count) + "YR\r"
        self.write(message)
        response = self.read()
        
        if len(response) < 2:
            print("PSD4 not found")
            assert False

        is_moving = True
        while is_moving == True:
            time.sleep(1)
            try:
                (is_moving, pos_in_uL, vel_in_mLmin, valve_pos) = self.getStatus()
            except:
                is_moving = True

        #set return step count (must be <=6400 in high res smooth mode)
        message ='/1K' + str(self.return_step_count) +'R\r'
        print(message)

        self.write(message)
        response = self.read()
        if len(response) < 2:
            assert False

    def configurePump(self):
        
        # Determine the port configuration
        message = "/1?21000R\r"
        self.write(message)
        response = self.read()
        
        if len(response) < 2:
            assert False
        
        num_ports_dict = {'0': 3,
                          '1': 4,
                          '2': 3,
                          '3': 8,
                          '4': 4,
                          '6': 6}
        
        self.num_ports = num_ports_dict[chr(response[3])]
                
        # Set the resolution
        if self.high_res_mode:
            message = '/1N1R\r'
        else:
            message = '/1N0R\r'

        self.write(message)
        response = self.read()
        if len(response) < 2:
            assert False

    def getStatus(self):
        # Determine if is moving
        message = "/1Q\r"
        
        self.write(message)
        response = self.read()
        
        if len(response)<2:
            print("Unknown response from PSD4")
            assert False
        
        response_char = chr(response[2])
        if response_char == '@':
            is_moving = True
        elif response_char == "`":
            is_moving = False
        else:
            print("Unknown response from PSD4")
            print(response)
            assert False
        
        # Determine the syringe position and return in mL
        message = '/1?R\r'
        
        self.write(message)
        response = self.read()
        
        if len(response)<2:
            print("Unknown response from PSD4")
            assert False

        start_pos = 3
        end_pos = response.find('\x03'.encode())
        pos_in_units = float(response[start_pos:end_pos].decode())
        pos_in_uL = pos_in_units*self.steps_to_volume * 1000
        
        # Determine current speed
        message = '/1?2R\r'
        
        self.write(message)
        response = self.read()
        
        if len(response)<2:
            print("Unknown response from PSD4")
            assert False

        start_pos = 3
        end_pos = response.find('\x03'.encode())
        vel_in_units = float(response[start_pos:end_pos].decode())
        vel_in_uLmin = vel_in_units*self.syringe_volume*60/(self.high_res_step/4)*1000
    
        # Determine valve numerical position
        message = '/1?24000R\r'
        
        self.write(message)
        response = self.read()
    
        if len(response)<2:
            print("Unknown response from PSD4")
            assert False

        start_pos = 3
        end_pos = response.find('\x03'.encode())
        valve_pos = int(response[start_pos:end_pos].decode())
        
        return (is_moving, pos_in_uL, vel_in_uLmin, valve_pos)
            
    def setPort(self, port_id, wait = True):
        # Check to see if it is within the number of ports
        if port_id > self.num_ports:
            print("An invalid port was requested for the PSD4")
            assert False
        
        # Set the port
        message = "/1h2500" + str(port_id) + "R\r"
        self.write(message)
        response = self.read()
        
        if len(response)<2:
            print("Unknown response from PSD4")
            assert False

        if wait == True:
            (is_moving, pos_in_uL, vel_in_mLmin, valve_pos) = self.getStatus()
            while valve_pos != port_id:
                time.sleep(1)
                (is_moving, pos_in_uL, vel_in_mLmin, valve_pos) = self.getStatus()
        else:
            pass

    def close(self):
        self.serial.close()
    
    def setSpeed(self, fill_speed_in_uLmin):
        
        # Convert the requested speed to steps per s
        fill_speed_in_uLs = fill_speed_in_uLmin/60
        new_speed_value = int((fill_speed_in_uLs/(self.syringe_volume * 1000)) * (self.high_res_step/4))
        
        # Coerce to the hardware limits
        if new_speed_value < self.min_velocity_in_steps_s:
            new_speed_value = self.min_velocity_in_steps_s
            print("Coerced pump speed to lowest value")
        
        if new_speed_value > self.max_velocity_in_steps_s:
            new_speed_value = self.max_velocity_in_steps_s
            print("Coerced pump speed to highest value")
        
        message = '/1V' + str(new_speed_value) + 'R\r'
        
        self.write(message)
        response = self.read()
        
        if len(response)<2:
            print("Unknown response from PSD4")
            assert False

    def startFill(self, new_volume_uL):
        #update current volume
        self.curr_vol = new_volume_uL

        # Define the volume in uL
        new_step_pos = int(new_volume_uL/(self.steps_to_volume * 1000))
        #print('new steps: ', new_step_pos)
        
        # Coerce to the hardware limits
        if new_step_pos < self.min_stroke_in_steps:
            new_step_pos = self.min_stroke_in_steps
            print("Coerced pump fill to lowest value")
        
        if new_step_pos > self.max_stroke_in_steps:
            new_step_pos = self.max_stroke_in_steps
            print("Coerced pump fill to highest value")

        # Define and write the message
        message = '/1A' + str(new_step_pos) + 'R\r'
        
        self.write(message)
        response = self.read()
        
        if len(response)<2:
            print("Unknown response from PSD4")
            assert False

    def stopFill(self):
        message = '/1TR\r'
        
        self.write(message)
        response = self.read()
        
        if len(response)<2:
            print("Unknown response from PSD4")
            assert False

    def load_syringe(self, volume, wait_until_flow_done = True):

        #set port to distribution valve
        dist_port = 6
        self.setPort(dist_port)

        #set fill speed
        self.setSpeed(750)

        #find current fill position
        (is_moving, pos_in_uL, vel_in_mLmin, valve_pos) = self.getStatus()

        if volume + pos_in_uL > self.syringe_volume * 1000:
            print('Error: More Volume than Syringe Capacity Requested')
        else:
            self.startFill(volume + pos_in_uL)

        while wait_until_flow_done == True:
            time.sleep(1)
            (is_moving, pos_in_uL, vel_in_mLmin, valve_pos) = self.getStatus()
            wait_until_flow_done = is_moving

    def volume_dispense(self, vol_2_dispense, dispense_vel_ulmin, device_port = 3, wait_until_flow_done = True):
        '''
        Dispenses additional volume in uLs. Gives back error if ask
        for more than max possible. Only pushes out of syringe, will not fill

        Parameters
        ----------
        vol_2_dispense
        dispense_vel_ulmin
        wait_until_flow_done

        Returns
        -------

        '''

        #set to device port
        self.setPort(device_port)

        #set flow velcity in uL/min
        self.setSpeed(dispense_vel_ulmin)

        #determine new volume to move to
        (is_moving, pos_in_uL, vel_in_mLmin, valve_pos) = self.getStatus()
        new_vol = pos_in_uL + (-1 * abs(vol_2_dispense)) #mandate negative amount
        if new_vol > self.syringe_volume * 1000:
            print('Error: More Volume than Syringe Capacity Requested')
        else:
            self.startFill(new_vol)

        while wait_until_flow_done == True:
            time.sleep(1)
            (is_moving, pos_in_uL, vel_in_mLmin, valve_pos) = self.getStatus()
            wait_until_flow_done = is_moving

    def swish(self, swish_volume, swish_vel_ulmin, swish_duration, device_port=3):
        '''

        Parameters
        ----------
        swish_volume(int):
        swish_vel_ulmin (int): in uL/min
        swish_duration (float): in minutes
        device_port(int):

        Returns
        -------

        '''

        # set port to distribution valve
        dist_port = device_port
        self.setPort(dist_port)

        #set Swish speed
        self.setSpeed(swish_vel_ulmin)


        self.startFill(swish_volume)
        #wait until syringe is at swish volume or higher
        wait_until_flow_done = True
        while wait_until_flow_done == True:
            time.sleep(1)
            (is_moving, pos_in_uL, vel_in_mLmin, valve_pos) = self.getStatus()
            wait_until_flow_done = is_moving



        self.startFill(0)
        #wait until syringe is at swish volume or higher
        wait_until_flow_done = True
        while wait_until_flow_done == True:
            time.sleep(1)
            (is_moving, pos_in_uL, vel_in_mLmin, valve_pos) = self.getStatus()
            wait_until_flow_done = is_moving

        #determine starting time for swish cycles
        starting_time = time.time() #starting time in seconds
        end_time = swish_duration * 60 # convert to seconds
        current_elapsed_time = time.time() - starting_time

        while current_elapsed_time < end_time:

            #push out phase
            #print('push', str(time.time()-starting_time))
            #self.startFill(pos_in_uL - swish_volume)

            #wait_until_flow_done = True
            #while wait_until_flow_done == True:
            #    time.sleep(1)
            #    (is_moving, pos_in_uL, vel_in_mLmin, valve_pos) = self.getStatus()
            #    print('position', pos_in_uL)
            #    wait_until_flow_done = is_moving


            #pull in phase
            self.startFill(pos_in_uL + swish_volume)

            wait_until_flow_done = True
            while wait_until_flow_done == True:
                time.sleep(1)
                (is_moving, pos_in_uL, vel_in_mLmin, valve_pos) = self.getStatus()
                wait_until_flow_done = is_moving

            time.sleep(10)

            #update time
            current_elapsed_time = time.time() - starting_time
            print('Staining Time Elapsed ', current_elapsed_time/60)

    
    def read(self):
       # response = self.serial.readline().decode()
        self.serial.flushOutput()
        response = self.serial.readline()

        if self.verbose:
            print("Received: " + str((response, "")))
        return response

    def write(self, message):
        self.serial.flushInput()
        self.serial.write(message.encode())
        if self.verbose:
            print("Wrote: " + message[:-1]) # Display all but final carriage return


#
# The MIT License
#
# Copyright (c) 2021 Moffitt Laboratory, Boston Children's Hospital
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#

