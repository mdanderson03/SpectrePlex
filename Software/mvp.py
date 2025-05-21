import serial
import time
import serial.tools.list_ports  # Only used for example code at end of page.


serial.tools


STX = b'\x02'
ETX = b'\x03'
EOT = b'\x04'
ENQ = b'\x05'
ACK = b'\x06'
NAK = b'\x15'


class Mvp:
    def __init__(self, serial_port, mvp_address="01", debug=True):
        self.debug = debug
        self.addr = mvp_address.encode(encoding='ascii')
        if len(self.addr) != 2:
            raise Exception("MVP address must be two digits. e.g. '01'. \
            Hardwired address is always one higher than the DIP switch binary value '00' -> '01'.")
        if not isinstance(serial_port, str):
            raise Exception("MVP serial_port must be a string!")
        self.port = serial.Serial(
            port=serial_port,
            baudrate=9600,
            timeout=1,
            parity=serial.PARITY_EVEN,
            bytesize=serial.SEVENBITS,
            stopbits=2,
            rtscts=False)
        if self.port.is_open:
            self._log("Serial port is open.")
        else:
            raise Exception("Serial port couldn't be opened!")
        # self.send_cmd(b'\x02I1G\x03')

    def __del__(self):
        self.port.close()
        self._log("Serial port is closed.")

    def _log(self, *args):
        if self.debug:
            print("Mvp: ", *args)

    def _send_cmd(self, cmdbytes, response_length=16, retries=5, has_bcc=False):
        # self.log(bytes)
        bcc = cmdbytes[0]
        # self.log(hex(bcc))
        for i in cmdbytes[1:]:
            bcc ^= i
        bcc ^= int.from_bytes(ETX)
        bcc ^= 0xff  # Invert
        bcc &= 0x7F  # Truncate to 7 bits
        output = STX + cmdbytes + ETX + bcc.to_bytes()
        resp = b''
        while retries >= 0:
            self._log("send_cmd: \tSending: ", output)
            self.port.write(output)
            if response_length == 0:
                return b''
            resp = self.port.read(response_length)
            self._log("send_cmd Resp: ", resp)
            if len(resp) == 0:
                self._log("senc_cmd empty response.")
            elif resp[0:1] == ACK:
                if response_length == 1:
                    return
                break
            elif resp[0:1] == NAK:
                self._log("send_cmd response is NAK.")
            retries -= 1
        if len(resp) != response_length:
            self._log("send_cmd Response length mismatch.")
        if resp[0:1] != ACK:
            raise Exception("send_cmd response had no ACK.")
        if resp[1:2] != STX:
            raise Exception("send_cmd Packet didn't contain STX.")
        retval = b''
        bcc = 0x00
        for i in resp[2:]:
            b = i.to_bytes()
            if b == ETX:
                bcc ^= i
                break
            retval += b
            bcc ^= i
        bcc ^= 0xff  # Invert
        bcc &= 0x7f  # Truncate to 7 bits
        if has_bcc and resp[-1] != bcc:
            self._log("send_cmd BCC is invalid.")
        return retval

    def open_session(self):
        """Send (address + ENQ) to MVP and check response.
        This is the first thing to run after power-on or reset. No commands will work until this function is ran.
        Return codes:
        0: Success
        1: Response wrong length
        2: Response wrong address
        3: Response is missing ACK"""
        self.port.write(self.addr + ENQ)
        resp = self.port.read(3)
        self._log("open_session resp: ", resp)
        if len(resp) != 3:
            self._log("MVP did not respond to open_session() with the correct length packet.")
            if resp == NAK:
                self._log("MVP sent a NAK.")
            if len(resp) == 0:
                self._log("MVP sent empty response, session is likely already open.")
            return 1
        elif resp[0:2] != self.addr:
            self._log("MVP did not respond to open_session() with the correct address.")
            return 2
        elif resp[2:] != ACK:
            self._log("MVP did not respond to open_session() with an ACK.")
            return 3
        return 0

    def initialize(self):
        self._send_cmd(b'I1G', response_length=1)

    def close_session(self):
        """Send (EOT) to MVP. Must run open_session before more commands will be processed."""
        self.port.write(EOT)

    def reset(self):
        """Send reset (R) to the MVP.
        MUST re-address and re-initialize before sending other commands."""
        self._send_cmd(b'R', response_length=1)

    def poll_query(self):
        resp = self._send_cmd(b'Q', 6, has_bcc=True)
        if len(resp) != 2:
            self._log("Query Response wrong length.")
            return -1
        if resp[0:1] != b'Q':
            self._log("Query response is invalid. Command packet didn't arrive properly.")
            return -2
        status = resp[1]
        if resp[1:2] == b'@':
            self._log("Query: No errors reported.")
            return status
        if status & 0x01:
            self._log("Query: Bit 0: Instrument received command, but not executed")
        if status & 0x04:
            self._log("Query: Bit 2: Valve drive busy")
        if status & 0x08:
            self._log("Query: Bit 3: Syntax error")
        if status & 0x10:
            self._log("Query: Bit 4: Instrument Error (valve error)")
        # Bit 7 is parity, but we aren't checking it.
        return status

    def poll_error(self):
        resp = self._send_cmd(b'E', 6, has_bcc=False)
        if len(resp) != 3:
            self._log("Poll for Error Response wrong length.")
            return -1
        if resp[0:1] != b'E':
            self._log("Poll for Error response is invalid. Command packet didn't arrive properly.")
            return -2
        status = resp[2]  # Two bytes of status are returned, but I've only seen the second one change.
        if resp[2:] == b'@':
            self._log("Poll for Error: No errors reported.")
            return status
        if status & 0x01:
            self._log("Error: Bit 0: Valve not initialized")
        if status & 0x02:
            self._log("Error: Bit 1: Valve initialization error")
        if status & 0x04:
            self._log("Error: Bit 2: Valve overload error")
        # Bit 7 is parity, but we aren't checking it.
        return status

    def poll_status(self):
        resp = self._send_cmd(b'Xs', 6, has_bcc=False)
        if len(resp) != 3:
            self._log("Poll for Status Response wrong length.")
            return -1
        if resp[0:2] != b'Xs':
            self._log("Poll for Status response is invalid. Command packet didn't arrive properly.")
            return -2
        status = resp[2]  # Two bytes of status are returned, but I've only seen the second one change.
        if resp[2:] == b'@':
            self._log("Poll for Status: No issues reported.")
            return status
        if status & 0x01:
            self._log("Status: Bit 0: Timer busy")
        if status & 0x02:
            self._log("Status: Bit 1: Diagnostic mode busy")
        if status & 0x10:
            self._log("Status: Bit 4: Over temperature error")
        # Bit 7 is parity, but we aren't checking it.
        return status

    def poll_valve_position(self):
        """Position returned is an int and can be 1-8"""
        resp = self._send_cmd(b'Ap', 7, has_bcc=True)
        if len(resp) != 3:
            self._log("Poll for Valve Position Response wrong length.")
            if resp == b'Ap':
                self._log("No position given, valve may need to be initialized first, or is busy moving.")
            return -1
        if resp[0:2] != b'Ap':
            self._log("Poll for Valve Position response is invalid. Command packet didn't arrive properly.")
            return -2
        return int.from_bytes(resp[2:])

    def poll_valve_angle(self):
        """Angle returned is an int: 0-359 degrees."""
        resp = self._send_cmd(b'Aa', 9, has_bcc=True)
        if len(resp) < 3 or len(resp) > 5:
            self._log("Poll for Valve Angle Response wrong length.")
            if resp == b'Aa':
                self._log("No angle given, valve may need to be initialized first, or is busy moving.")
            return -1
        if resp[0:2] != b'Aa':
            self._log("Poll for Valve Angle response is invalid. Command packet didn't arrive properly.")
            return -2
        return int.from_bytes(resp[2:])

    def poll_valve_type(self):
        """Valve type returned is an int: 2-7
            x = 2: 8 ports
            x = 3: 6 ports
            x = 4: 3 ports
            x = 5: 2 ports @ 180 degrees apart
            x = 6: 2 ports @ 90 degrees apart
            x = 7: 4 ports"""
        resp = self._send_cmd(b'Av', 7, has_bcc=True)
        if len(resp) != 3:
            self._log("Poll for Valve Type Response wrong length.")
            if resp == b'Av':
                self._log("No position given, valve may need to be initialized first!")
            return -1
        if resp[0:2] != b'Av':
            self._log("Poll for Valve Type response is invalid. Command packet didn't arrive properly.")
            return -2
        return int.from_bytes(resp[2:])

    def poll_valve_speed(self):
        """Valve speed/response returned is an int: 0-9
            y = 0 30 Hz
            y = 1 40 Hz
            y = 2 50 Hz * for y = 2-9, valve speed only applies to Protocol 1/RNO+
            y = 3 60 Hz
            y = 4 70 Hz
            y = 5 80 Hz
            y = 6 90 Hz
            y = 7 100 Hz
            y = 8 110 Hz
            y = 9 120 Hz"""
        resp = self._send_cmd(b'Az', 7, has_bcc=True)
        if len(resp) != 3:
            self._log("Poll for Speed Response wrong length.")
            if resp == b'Az':
                self._log("No speed given, valve may need to be initialized first!")
            return -1
        if resp[0:2] != b'Az':
            self._log("Poll for Speed response is invalid. Command packet didn't arrive properly.")
            return -2
        return int.from_bytes(resp[2:])

    def poll_firmware(self):
        """Firmware returned is a bytestring."""
        resp = self._send_cmd(b'F', 15, has_bcc=False)
        if resp[0:2] != b'F':
            self._log("Poll for Firmware response is invalid. Command packet didn't arrive properly.")
            return -2
        return resp[2:]

    def set_valve_type(self, valve_type: int):
        """Set Valve Type - valve_type is int
            x = 2 8 ports @ 45 degrees apart
            x = 3 6 ports @ 60 degrees apart
            x = 4 3 ports @ 90 degrees apart
            x = 5 2 ports @ 180 degrees apart
            x = 6 2 ports @ 90 degrees apart
            x = 7 4 ports @ 90 degrees apart (default)"""
        self._send_cmd(b'Sv' + str(valve_type).encode(encoding='ascii'), 1, has_bcc=False)

    def set_valve_speed(self, valve_speed: int):
        """Set Valve Motor Speed. valve_speed must be a string!
            y = 0 30 Hz
            y = 1 40 Hz
            y = 2 50 Hz
            y = 3 60 Hz
            y = 4 *70 Hz
            y = 5 *80 Hz
            y = 6 *90 Hz
            y = 7 *100 Hz
            y = 8 *110 Hz
            y = 9 *120 Hz
            * If you plan to operate at >60 Hz, please contact Hamilton Company prior to use."""
        self._send_cmd(b'Sz' + str(valve_speed).encode(encoding='ascii'), 1, has_bcc=False)

    def set_diagnostic_mode(self):
        """Put instrument in diagnostic mode
        Diagnostic is halted by the 'Reset' command"""
        self._send_cmd(b'Ut', 1, has_bcc=False)

    def halt_all(self):
        """Halt all device commands in progress"""
        self._send_cmd(b'Uk', 1, has_bcc=False)

    def resume_all(self):
        """Resume all device commands"""
        self._send_cmd(b'Ur', 1, has_bcc=False)

    def clear_all(self):
        """Clear all pending device commands"""
        self._send_cmd(b'Uc', 1, has_bcc=False)

    def clear_addressing(self):
        """Clear Hardware-addressing
        Configure communication for auto-addressing"""
        self._send_cmd(b'Y', 1, has_bcc=False)

    def set_valve_position(self, valve_position: int, counter_clockwise: int = 0):
        """Valve Positioning
        d = 0, CW
        d = 1, CCW
        pp = 1-8, valve positions"""
        self._send_cmd((b'Vv' + str(counter_clockwise).encode(encoding='ascii') +
                        b'n' + str(valve_position).encode(encoding='ascii')) + b'G',
                       1, has_bcc=False)

    def set_valve_angle(self, valve_angle: int, counter_clockwise: int = 0):
        """Valve Positioning
        d = 0, CW
        d = 1, CCW
        aaa = 0-345 degrees, absolute
        angles from 0 degrees @ 15
        degree increments"""
        self._send_cmd((b'Vv' + str(counter_clockwise).encode(encoding='ascii') +
                        b'w' + str(valve_angle).encode(encoding='ascii')) + b'G',
                       1, has_bcc=False)

    def wait_for_valve(self, timeout=10):
        """Helper function that will pause for up to timeout seconds for valve motor to stop moving."""
        start_s = time.time()
        while (time.time() - start_s) < timeout:
            resp = self.poll_query()
            if resp & 0x04 == 0:
                self._log("wait_for_valve drive busy bit clear, continuing.")
                break
            time.sleep(0.25)




def example_usage():
    found_ports = serial.tools.list_ports.comports()
    if len(found_ports) == 0:
        print("No serial devices found!")
        return
    for index, port in enumerate(found_ports):
        print(f"Found port: [{index}], {port.device}: \"{port.manufacturer}: {port.product}\" {port.vid}:{port.pid}")
    port_pick = int(input("Enter the index number (in [ ]) of the port to use: "))
    saline_valve = Mvp(found_ports[port_pick].device)
    # saline_valve = Mvp('/dev/ttyUSB0')  # OR manually. Find serial port with `ls /dev/tty*`
    saline_valve.open_session()
    saline_valve.initialize()
    saline_valve.wait_for_valve()  # Wait for valve motion to stop, or the next command will be ignored.
    existing_valve_type = saline_valve.poll_valve_type()
    print("Current valve type: ", existing_valve_type)
    if existing_valve_type != 6:
        saline_valve.set_valve_type(6)  # Set valve to 2-position, 90-degrees.
    saline_valve.set_valve_position(valve_position=2, counter_clockwise=0)
    saline_valve.wait_for_valve()
    time.sleep(0.5)
    saline_valve.set_valve_position(valve_position=1, counter_clockwise=1)
    saline_valve.wait_for_valve()
    time.sleep(0.5)
    saline_valve.set_valve_angle(valve_angle=90, counter_clockwise=1)
    saline_valve.wait_for_valve()
    time.sleep(0.5)
    saline_valve.set_valve_angle(valve_angle=0, counter_clockwise=0)
    saline_valve.wait_for_valve()
    saline_valve.set_valve_type(existing_valve_type)  # Shouldn't matter, valve settings will reset with power cycle.
    saline_valve.close_session()
    print("done")


if __name__ == "__main__":
    example_usage()
