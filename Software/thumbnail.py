from shutil import make_archive
import os
import time

start = time.time()

#os.chdir('E:/')
#folder = '1_5_25_S24_7971A_Casey'
#make_archive(folder, 'tar', 'E:/', 'E:/' + folder)
ed = r'E:\03_6_25_test_double_hamilton_wirth_flow_meter'
print(ed.split('\\')[-1])
end = time.time()
print(end - start)