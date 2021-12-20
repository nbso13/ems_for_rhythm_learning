from math import fabs
from subprocess import Popen
import numpy as np
import time


#### read and write to arduino ###
import serial
import time
# port = '/dev/ttys000'
port = '/dev/tty.usbserial-18DNB483'
bluetooth = serial.Serial(port, 115200)
bluetooth.flushInput()
bluetooth.write(b"2")

#### test serial delay ####




def pingtest():
    for i in range(20):
        time.sleep(0.1)
        if(bluetooth.in_waiting):
            print(bluetooth.readline())

    

    act_lists = []
    std_dvs = []
    means = []
    time_delays = [0.01]

    for j in time_delays:
        act_list = []
        for i in range(100):
            # tic = time.time()
            bluetooth.write(b"C1I100T1G \n")
            tic = time.time()
            count = 0
            for k in range(300):
                # print(k)
                if(bluetooth.in_waiting):
                    count = count + 1
                    out = bluetooth.readline().decode('utf-8')
                    # print(out + " " + str(count))
                    if count == 1:
                        toc = time.time()
                    if count == 2:
                        tec = time.time()
                    if count == 3:
                        tuc = time.time()
                    if count == 4:
                        tac = time.time()  
                time.sleep(0.001)  
            time.sleep(j)

            elapsed_to_received = (toc-tic)
            elapsed_to_applied = (tuc-tic)
            elapsed_to_activated = (tac-tic)
            elapsed_to_act_com = (tec-tic)
            # elapsed_to_applied = float(out) + elapsed_to_received
            act_list.append(elapsed_to_activated)
            # print("elapsed to received: " + str(elapsed_to_received) + ", to activated: " + str (elapsed_to_activated)+ ", to applied: " + str(elapsed_to_applied))
            print("to received: " + str(elapsed_to_received) + ", to act_com: " + str(elapsed_to_act_com) + ", to applied: " + str(elapsed_to_applied) + ", to activated: " + str(elapsed_to_activated))
        # print("delay: " +str(j) + ", average activated lag: " + str(sum(act_list)/len(act_list)))
        act_lists.append(act_list)
        means.append(np.mean(np.array(act_list)))
        std_dvs.append(np.std(np.array(act_list)))

    print("mean delay (seconds) +/- std (n=100): " + str(means[0]) + " +/- " + str(std_dvs[0]))
    return

pingtest()