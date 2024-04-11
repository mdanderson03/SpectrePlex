import time
from subprocess import call

def open_py_file():
    call(["python", "bleach.py"])

open_py_file()
time.sleep(5)
print('second start')
open_py_file()