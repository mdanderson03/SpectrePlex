from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support
import numpy as np
import os
from skimage import io, filters, util
import time



times = np.array([10, 20, 30, 50, 100, 150, 250, 400, 800, 2000])
hdr_indicies = [0,3,9]

os.chdir(r'C:\Users\CyCIF PC\Desktop\linearity')

array = io.imread('linearity.tif')
#array = array.astype('float32')

hdr_array = array[hdr_indicies]
print('x')

start_time = time.time()


weight_array = np.random.rand(len(hdr_indicies), 2960, 2960).astype('float32')

#subtract linear offset
array = array - 300

#populate weight array
index_counter = 0
for index in hdr_indicies:

    im = array[index]
    im[im >65234] = 0
    im = np.sqrt(im)
    weight_array[index_counter] = im
    index_counter += 1

def func(a, b):
    return a + b

def main():
    a_args = [1,2,3]
    second_arg = 1
    with Pool() as pool:
        L = pool.starmap(func, [(1, 1), (2, 1), (3, 1)])
        #M = pool.starmap(func, zip(a_args, repeat(second_arg)))
        #N = pool.map(partial(func, b=second_arg), a_args)
        #assert L == M == N

if __name__=="__main__":
    main()

