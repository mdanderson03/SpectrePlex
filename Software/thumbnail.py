from kasa import Discover
from KasaSmartPowerStrip import SmartPowerStrip
import asyncio
import binascii
import ipaddress
import logging
import socket
import numpy as np
from copy import deepcopy
from csbdeep.utils import normalize

import os
from skimage import io

a= np.array(([1,1], [0,0]))
b= np.array(([0,1], [0,1]))
non_zeros = np.nonzero(b)
#print(a, b)
c = np.subtract(a, b)
d = np.where(c[non_zeros] == 0)

a = np.array([1,2,3,4,5])
b = np.array([1,2,4,7,9])
disjoint_a_b = np.setdiff1d(a, b)
disjoint_b_a = np.setdiff1d(b, a)
print(disjoint_a_b, disjoint_b_a)

for x in range(0, len(disjoint_a_b)):
    indicies = np.where(b == disjoint_b_a[x] )
    b[indicies] = disjoint_a_b[x]

print(b)