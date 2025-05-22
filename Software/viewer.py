from hamilton_mvp import AValveChain
import time

mux = AValveChain(parameters='COM3')
time.sleep(5)
mux.changePort(0, 1, direction=0, wait_until_done=True)
time.sleep(10)
mux.changePort(0, 2, direction=0, wait_until_done=True)




