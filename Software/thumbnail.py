from kasa import Discover
from KasaSmartPowerStrip import SmartPowerStrip
import asyncio
import binascii
import ipaddress
import logging
import socket
#found_devices = asyncio.run(Discover.discover(target="192.168.0.1"))
#print(found_devices)
import numpy as np
from copy import deepcopy


a = np.array((((1,5,100,2), (0,1,2,3))))
print(a)
b = deepcopy(np.sort(a, kind = 'stable'))
x = 0
while x  < np.shape(a)[1]:

    area = b[0][x]
    x_index = np.where(a[0] == area)
    index = a[1][x_index]
    b[1][x:x+np.shape(x_index)[0]] = index
    x += np.shape(x_index)[0]
b = np.fliplr(b)
b[0][b[0]<5] = 0
index_smallest = np.where(b[0] == 0)[0][0]
b = b[::, 0:index_smallest]
print(b)
