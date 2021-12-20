import numpy as np
import math
import time
import xlsxwriter
import matplotlib.pyplot as plt
import datetime
import serial
import time

LENGTH_BEGIN = 165
LENGTH_END = 270
LENGTH_STEP = 15
INTENSITY_BEGIN = 60
INTENSITY_END = 100
INTENSITY_STEP = 5
BPM = 70
BAUD_RATE = 115200
REPEATS = 1
RHYTH_STR = "101010"
PARTICIPANT_NUMBERS = [0, 1]

# initial results indicate that after finding max level they can take with this max length, max stim length


# notes: we need to think about where we're actuating the muscle along its length - there is a standard for electrode placement. Talk to 
# Lewis about this. This already exists. Jakob might know about this. Probably will.


def play_rhythm(bluetooth, actual_stim_length, rhythm_substr, repeats, bpm, intensity_level):
    
    max_bpm = math.floor(30000/actual_stim_length) #how many eighthnote pulses could you fit into a minute without overlapping?

    if (bpm > max_bpm):
        print("max metronome bpm is " + str(max_bpm))
        return

    #determine pulse+wait length
    milliseconds_per_eighthnote = 30000/bpm # if you think about it this math works
    milliseconds_wait = milliseconds_per_eighthnote - actual_stim_length

    for i in range(repeats): # present the rhythm with appropriate number of repeats
        for j in range(len(rhythm_substr)):  # go through each eighthnote in the pattern
            if (rhythm_substr[j] == '1'): # this is a note
                command_bytes = "xC1I" + str(intensity_level) + "T" + str(actual_stim_length) + "G \n"
                byt_com = bytes(command_bytes, encoding='utf8')
                bluetooth.write(byt_com)
                print("stim on")
                time.sleep(actual_stim_length/1000)
                time.sleep(milliseconds_wait/1000)
            elif(rhythm_substr[j] == '0'): # rest
                time.sleep(milliseconds_per_eighthnote/1000)
            else:
                print("malformed rhythm pattern: " + rhythm_substr)
                break


#### read and write to arduino ###

# port = '/dev/ttys000'
port = '/dev/tty.usbserial-18DNB483'
bluetooth = serial.Serial(port, BAUD_RATE)
bluetooth.flushInput()
bluetooth.write(b"2")

input("adjust ems channel intensity to 8!") #my max was16

lengths = range(LENGTH_BEGIN, LENGTH_END, LENGTH_STEP) #length in ms of stimulation pulse -- watch out for mystery numbers - using 130 to 290 as range -- 130 being minimum length of stim and 290 as max necessary for low intensity
# perhaps better to set up a 'settings file'
 # make it global variables
intensities = range(INTENSITY_BEGIN, INTENSITY_END, INTENSITY_STEP) 

rhythm_substr = RHYTH_STR
repeats = REPEATS
bpm = BPM


### calibrate user to strongest impulses ###
input("For your reference: this is strongest and longest stim: (enter to play it) - let's determine how high we can set the ems")
keep_going = True
while keep_going:
    play_rhythm(bluetooth, lengths[-1], rhythm_substr, repeats, bpm, intensities[-1]) # revisit this with Lewis, bring back to group to discuss re procedure
    result = input("IF: 'I can go higher. I've upped the stim max on stimulator. Let's go again.' type y. IF: 'I can go no further!! It hurts a lot.' type n.")
    if result == "n":
        keep_going = False


input("By contrast, this is weakest and shortest stim: (enter to play it)")
play_rhythm(bluetooth, lengths[0], rhythm_substr, repeats, bpm, intensities[0])

### Gathering subject info ###

participant_number = input("participant number?")
if not participant_number in PARTICIPANT_NUMBERS:
    participant_number = PARTICIPANT_NUMBERS[-1]+1
    PARTICIPANT_NUMBERS.append(participant_number)
now = datetime.datetime.now()
test_time = now.strftime("%Y_%m_%d_%H_%M_%S")
subject_arm = input("subject arm?")
electrode_config = input("electrode config?") #first pair of numbers is coordinates of 1, x and y, second is coordinates of 2. x and y
max_ems_stim_intensity = input("max ems stim intensity?")
pulse_width = input("pulse width?")
pulse_frequency = input("frequency?") #these may be found on the stimulator and are not usually iterated on (using lit values)


label_header = ["pp number and random seed", "test time", "subject arm", "electrode config", "rhythm pattern", "bpm", "max_stim_intensity", "pulse width (microsecs)", "frequency (Hz)"]
header_values = [participant_number, test_time, subject_arm, electrode_config, rhythm_substr, bpm, max_ems_stim_intensity, pulse_width, pulse_frequency]
data_header = ["y ax = intensities (%, " + max_ems_stim_intensity + " is max on toolkit stim), x ax = stim lengths (microsecs)"] + [*lengths]

 
### open workbook, define worksheets ###

workbook = xlsxwriter.Workbook(test_time + '_' +  "pp" + str(participant_number) + '.xlsx')
bold = workbook.add_format({'bold': True})
pain_worksheet = workbook.add_worksheet("Pain level") # judged subjectively, 1-10, 1 is barely perceivable sensation, 10 is intense pain, 0 is no sensation.
act_worksheet = workbook.add_worksheet("Actuation level") # judged subjectively, 1-10, 1 is barely visible movement, 10 is complete flexion. 0 is no visible actuation.
speed_worksheet = workbook.add_worksheet("Actuation speed") # judged subjectively, 1-10, 1 is very slow, 10 is very fast. 0 is no actuation.
worksheets = [pain_worksheet, act_worksheet, speed_worksheet]

## write header values ##

for worksheet in worksheets:
    for i in range(len(label_header)):
        worksheet.write(0, i, label_header[i], bold) # write in header values
        worksheet.write(1, i, header_values[i], bold)

    for i in range(len(data_header)):
        worksheet.write(2, i, data_header[i])
    for i in range(len(intensities)):
        worksheet.write(i+3, 0, intensities[i])

# workbook.close()
worksheet_data_begin_indices = [3, 1] # where empty data space begins in each worksheet


## scramble length and intensity test order ##

# NOTE FROM MATTI: use a random seed for reproducibility - maybe everyone gets a different permutation - same seed but different permutations for each participant
## Discuss this more with lewis - how often do we sample? Does it make sense to do this this way?
# talk to Fiona and Jesse re IRB.

lens = np.array(lengths) 
len_inds = np.arange(len(lengths))
big_lens = np.tile(lens, len(intensities)) # repeat len array
big_lens_inds = np.tile(len_inds, len(intensities)) # do the same for len_indices

intents = np.array(intensities)
int_inds = np.arange(len(intensities))
big_ints = np.repeat(intents, len(lens))
big_ints_inds = np.repeat(int_inds, len(lens)) # repeat individual elements of intensities and same for inds

param_array = np.vstack([big_lens, big_ints])
param_inds_array = np.vstack([big_lens_inds, big_ints_inds]) #concatenate them

np.random.seed(participant_number)
shuffled_vals = np.transpose(np.random.permutation(np.transpose(param_array))) #randomly permute all indices
np.random.seed(participant_number)
shuffled_inds = np.transpose(np.random.permutation(np.transpose(param_inds_array)))


# np.random.seed(participant_number)
# shuffled_lengths = np.random.permutation(lengths)
# np.random.seed(participant_number)
# shuffled_length_indices= np.random.permutation(np.arange(len(lengths)))

# np.random.seed(participant_number)
# shuffled_intensities = np.random.permutation(intensities)
# np.random.seed(participant_number)
# shuffled_intensity_indices= np.random.permutation(np.arange(len(intensities)))

arr = np.empty((len(intensities), len(lengths)))
arr[:] = np.NaN
data_arr = [arr, arr, arr]


### run testing loop ###

input("Stimulations will now be presented, four at a time, sampling stim_length and intensity space. (enter to continue)") 
input("You are to rate: (1) the pain of the stimulation (0 is no sensation, 10 is as painful as the example stimulation), (enter to continue)")
input("(2) the extent to which your finger moved, i.e., 'actuation' (0 is no movement, 10 is the extent to which it moved on the example), (enter to continue)")
input("(3) the speed of actuation (0 is no movement, 1 is very slow, 10 is almost instantaneous activation) (enter to begin trials)")

n = 0
for i in range(len(shuffled_vals[0])): # go through whole list of shuffled values
    n = n +1
    actual_stim_length = shuffled_vals[0][i] #ms
    data_index_x = shuffled_inds[0][i]
    actual_intensity = shuffled_vals[1][i]
    data_index_y = shuffled_inds[1][i]

    #print(" Length: "+ str(actual_stim_length) + ", intensity: " + str(actual_intensity))
    play_rhythm(bluetooth, actual_stim_length, rhythm_substr, repeats, bpm, actual_intensity)
    print(str(n) + " of "+ str(len(lengths)*len(intensities)))
    pain_val = input("pain? 0-10: ")
    actuation_val = input("actuation? 0-10: ")
    speed_val = input("speed? 0-10: ")
    try:
        values_to_write = [float(pain_val), float(actuation_val), float(speed_val)]
    except ValueError:
        print("You must enter some real value between [0 and 10] for each question.")
        pain_val = input("pain? 0-10: ")
        actuation_val = input("actuation? 0-10: ")
        speed_val = input("speed? 0-10: ")
        values_to_write = [float(pain_val), float(actuation_val), float(speed_val)]



    for k in range(len(worksheets)): # writing
        try_again = False
        worksheets[k].write(worksheet_data_begin_indices[0] + data_index_y, worksheet_data_begin_indices[1] + data_index_x, values_to_write[k])
        try:
            data_arr[k][data_index_y, data_index_x] = values_to_write[k]
        except ValueError:
            try_again = True
            break


    if(try_again == True or max(values_to_write) > 10 or min(values_to_write) < 0):
        print("Please be sure to enter a value between 0 and 10 for each inquiry.")
        pain_val = input("pain? 0-10")
        actuation_val = input("actuation? 0-10")
        speed_val = input("speed? 0-10")
        values_to_write = [pain_val, actuation_val, speed_val]
        for k in range(len(worksheets)):
            worksheets[k].write(worksheet_data_begin_indices[0] + data_index_y, worksheet_data_begin_indices[1] + data_index_x, values_to_write[k])
            data_arr[k][data_index_y, data_index_x] = values_to_write[k]

workbook.close()


# calibration index

data_arr[2] = np.abs(data_arr[2]-6) # distance from ideal actuation
index_map = np.subtract(np.add(data_arr[1],2*data_arr[2]), 2*data_arr[0])

bluetooth.close()
print("done")


### visualize as heat map at the end ###
len_labs = []
for i in lengths:
    len_labs.append(str(i))
int_labs = []
for i in intensities:
    int_labs.append(str(i))

fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(data_arr[0], cmap='hot', interpolation='nearest')
axs[0, 0].set_title('Pain map')
axs[0, 1].imshow(data_arr[1], cmap='hot', interpolation='nearest')
axs[0, 1].set_title('Actuation map')
axs[1, 0].imshow(data_arr[2], cmap='hot', interpolation='nearest')
axs[1, 0].set_title('Speed map')
axs[1,1].imshow(index_map, cmap='hot', interpolation='nearest')
axs[1, 1].set_title('Index Map')

for ax in axs.flat:
    ax.set(xlabel='Stim length (ms)', ylabel='Stim intensity (%)')
    ax.set_xticklabels(len_labs) 
    ax.set_yticklabels(int_labs)

plt.show()

