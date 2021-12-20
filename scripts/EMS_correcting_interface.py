from subprocess import Popen, PIPE
import subprocess
import shlex
import numpy as np
import math
import pygame
import time
import threading

def metronome(bpm, length_of_stim):  #in beats per minute and lengthofstim is seconds
#max bpm is __ as given actual stim length of ___ms as you can only fit around 6 stims in a 
# second.
    actual_stim_length = 300 #ms
    max_bpm = 60*math.floor(1000/actual_stim_length)

    if (bpm > max_bpm):
        print("max metronome bpm is " + str(max_bpm))
        return

    # determine wait time between pulses
    milliseconds_per_beat = 60000/bpm
    milliseconds_wait = milliseconds_per_beat - actual_stim_length

    total_repeats = math.floor(length_of_stim*1000/milliseconds_wait) # det

    for i in range(total_repeats):
        # start stim command
        print("stim started")
        # wait for actual_stim_length
        # turn off stim        
        print("stim stopped")
        #wait for milliseconds_wait




def play_rhythm(bluetooth, actual_stim_length, rhythm_substr, repeats, bpm, ems_flag, audio_flag, metronome_intro_flag, audio_pre_display_flag, delay_val):
    
    max_bpm = math.floor(30000/actual_stim_length) #how many eighthnote pulses could you fit into a minute without overlapping?
    audio_repeats = 2

    if (bpm > max_bpm):
        print("max metronome bpm is " + str(max_bpm))
        return
    if delay_val > actual_stim_length:
        print("delay_value more than stim length")

    #determine pulse+wait length
    milliseconds_per_eighthnote = 30000/bpm
    milliseconds_wait = milliseconds_per_eighthnote - actual_stim_length

    ### metronome count in ###
    if metronome_intro_flag:
        count_in_substr = '10101010'
        for i in range(len(count_in_substr)):
            if (count_in_substr[i] == '1'): # this is a note
                if audio_flag:
                    time.sleep(delay_val/1000)
                    pygame.mixer.music.play()
                    time.sleep((actual_stim_length-delay_val)/1000)
                    time.sleep(milliseconds_wait/1000)
            elif(count_in_substr[i] == '0'): # rest
                if audio_flag:
                    pygame.mixer.music.stop()
                    time.sleep(milliseconds_per_eighthnote/1000)

### rhythm audio display ###
    if audio_pre_display_flag:
        for i in range(audio_repeats): # present the rhythm with appropriate number of repeats
            for j in range(len(rhythm_substr)):  # go through each eighthnote in the pattern
                if (rhythm_substr[j] == '1'): # this is a note
                    time.sleep(delay_val/1000)
                    pygame.mixer.music.play()
                    time.sleep((actual_stim_length-delay_val)/1000)
                    time.sleep(milliseconds_wait/1000)
                elif(rhythm_substr[j] == '0'): # rest
                    pygame.mixer.music.stop()
                    time.sleep(milliseconds_per_eighthnote/1000)
                else:
                    print("malformed rhythm pattern: " + rhythm_substr)
                    break


    for i in range(repeats): # present the rhythm with appropriate number of repeats
        for j in range(len(rhythm_substr)):  # go through each eighthnote in the pattern
            if (rhythm_substr[j] == '1'): # this is a note
                if ems_flag:
                    command_bytes = "xC1I100T" +str(actual_stim_length) + "G \n"
                    byt_com = bytes(command_bytes, encoding='utf8')
                    bluetooth.write(byt_com)
                    print("stim on")
                time.sleep(delay_val/1000)
                if audio_flag:
                    pygame.mixer.music.play()
                time.sleep((actual_stim_length-delay_val)/1000)
                time.sleep(milliseconds_wait/1000)
            elif(rhythm_substr[j] == '0'): # rest
                pygame.mixer.music.stop()
                time.sleep(milliseconds_per_eighthnote/1000)
            else:
                print("malformed rhythm pattern: " + rhythm_substr)
                break

def read_contact_trace(ser,  len_rhythm_presentation_ms, check_interval, readings_list):
    check_repeats = len_rhythm_presentation_ms/check_interval
    for i in range(check_repeats):
        if(ser.in_waiting):
            out = bluetooth.readline().decode('utf-8')
            readings_list.append(out)
        time.sleep(check_interval)
    return(readings_list)


def eval_rhythm(actual_stim_length, rhythm_substr, repeats, bpm,  metronome_intro_flag, delay_val, read_thread):    
    max_bpm = math.floor(30000/actual_stim_length) #how many eighthnote pulses could you fit into a minute without overlapping?
    if (bpm > max_bpm):
        print("max metronome bpm is " + str(max_bpm))
        return
    if delay_val > actual_stim_length:
        print("delay_value more than stim length")

    #determine pulse+wait length
    milliseconds_per_eighthnote = 30000/bpm
    milliseconds_wait = milliseconds_per_eighthnote - actual_stim_length
    
    ### metronome count in ###
    if metronome_intro_flag:
        count_in_substr = '10101010'
        for i in range(len(count_in_substr)):
            if (count_in_substr[i] == '1'): # this is a note
                time.sleep(delay_val/1000)
                pygame.mixer.music.play()
                time.sleep((actual_stim_length-delay_val)/1000)
                time.sleep(milliseconds_wait/1000)
                pygame.mixer.music.stop()
            elif(count_in_substr[i] == '0'): # rest
                if audio_flag:
                    time.sleep(milliseconds_per_eighthnote/1000)

    read_thread.start()

    for i in range(repeats): # present the rhythm with appropriate number of repeats
        for j in range(len(rhythm_substr)):  # go through each eighthnote in the pattern
            if (rhythm_substr[j] == '1'): # this is a note
                time.sleep(delay_val/1000)
                pygame.mixer.music.play()
                time.sleep((actual_stim_length-delay_val)/1000)
                time.sleep(milliseconds_wait/1000)
                pygame.mixer.music.stop()
            elif(rhythm_substr[j] == '0'): # rest
                time.sleep(milliseconds_per_eighthnote/1000)
            else:
                print("malformed rhythm pattern: " + rhythm_substr)
                break

    read_thread.join()




pygame.mixer.init()
pygame.mixer.music.load("440Hz_44100Hz_16bit_05sec.mp3")

#### read and write to arduino ###
import serial
import time
# port = '/dev/ttys000'
port = '/dev/tty.usbserial-18DNB483'
bluetooth = serial.Serial(port, 115200) # EMS arduino
bluetooth.flushInput()
bluetooth.write(b"2")

port = '/dev/tty.usbserial-18DNB483'
ser = serial.Serial(port, 115200) # reader arduino

######## calibrate EMS timing #######

# set up participant in tapping position

# send 10 tap commands and record relative times of initial contact

# compute mean and variance of delays (ideally low variance (and low mean))

# save mean delay value as 
MEASURED_DELAY = 20 #ms

##### TRAINING PROCEDURE WITH EMS #######

# audio metronome playing

# introduce rhythm in context of metronome


# rhythm audio

while(bluetooth.in_waiting):
    print(bluetooth.readline().decode("utf-8"))



rhythm_substr = "010010011010100"
bpm = 100
delay_val = MEASURED_DELAY
actual_stim_length = 140

# user trains with rhythm for 5 repeats
repeats = 6
ems_flag = 0
audio_flag = 1
audio_pre_display_flag = 0
metronome_intro_flag = 1
play_rhythm(bluetooth, actual_stim_length, rhythm_substr, repeats, bpm, ems_flag, audio_flag, metronome_intro_flag, audio_pre_display_flag, delay_val)

# user is measured on rhythm for 3 repeats
repeats = 3
audio_flag = 1
check_interval = 0.5 #ms/ 2000 HZ
audio_pre_display_flag = 0
metronome_intro_flag = 1
len_pres = (30000/bpm) * len(rhythm_substr)
all_readings = []
for i in range(repeats):
    read_results = []
    read_thread = threading.Thread(target=read_contact_trace, args= (ser, len_pres, check_interval, read_results))
    rep = 1
    eval_rhythm(actual_stim_length, rhythm_substr, rep, bpm,  metronome_intro_flag, delay_val, read_thread)
    all_readings.append(read_results)

# correct contact traces to sync up

# determine which notes were not played correctly

# repeat and actuate the user on those notes


bluetooth.close()
print("done")