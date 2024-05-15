from kasa import Discover
from KasaSmartPowerStrip import SmartPowerStrip
import asyncio
import binascii
import ipaddress
import logging
import socket
#found_devices = asyncio.run(Discover.discover(target="192.168.0.1"))
#print(found_devices)

power_strip = SmartPowerStrip('10.3.141.157')
print(power_strip.toggle_plug('on', plug_num=5))