
# set vars
to_the_beat_substr = "1000100010001000"
lunch_room_beat = "10001000101010001010101000101000"
clave_substr = "0010100010010010"
five_to_four_substr = "10000100001000010000"
three_to_four_substr = "100100100100"
seven_to_four = "1000000100000010000001000000"
syncopated_substr = "00100010000100010000"
bass_drum_pattern = "10100000100000101000000010000000"
flip_the_beat = "1000100100010000"
telescoping = "1000001000010001001000"
count_in_substr = '1000100010001000'
rhythm_strings = [to_the_beat_substr]
# clave_substr, five_to_four_substr, three_to_four_substr,  \
#     syncopated_substr, bass_drum_pattern, flip_the_beat
rhythm_strings_names = ["to_the_beat"]
# , "clave", "five_to_four", "three_to_four", \
#     "syncopated_substr", "bass_drum_pattern", "flip_the_beat"
bpms_ordered = [75, 85, 95, 105, 115, 125, 135] #bpms to try
bpms = [125, 85, 105, 115, 75, 135, 95] # shuffled
repeats = 3 # ems and audio period repeats
audio_repeats = 3
post_ems_repeats = 3 # how many post ems repeats
no_audio_repeats = 3
audio_delay = 0.0023 # but why
# bpm = 120 # beats per minute
ems_flag = 1 # turn ems on
audio_flag = 1 # turn audio on
audio_pre_display_flag = 1 # include audio pre-ems display for training purposes
post_ems_test_flag = 1 # include audio only post-ems display for testing purposes
no_audio_flag = 1 # include metronome only post-post-ems display for testing purposes
metronome_intro_flag = 1 # if we want a count in
samp_period_ms = 2 # milliseconds
delay_trial_num = 10 # if measuring delay, this is how many trials we use
sleep_len = 2 # seconds of waiting while zeroing sensor
sd_more_than_mult = 7 # deprecated
actual_stim_length = 155 # actual stim length
baseline_subtractor = 55 # this is the noise threshold for contact trace BACK THIS UP EXPERIMENTALLY?
surpression_window = 250 # ms BACK THIS UP EXPERIMENTALLY?
contact_spike_time_width = 2 # ms
double_stroke_rhythm = "101001010010101001010010100"
interval_tolerance = 40 #ms
phase_name_strs = ["pre_ems", "ems_audio", "post_ems", "no_audio"] # name phases
num_phases = len(phase_name_strs) # number of phases is number of names of phases
port_contact = '/dev/cu.usbmodem1101'
port_ems = '/dev/tty.usbserial-18DNB483'
# port_ems = '/dev/ttys000' for bluetooth
worksheet_data_begin_indices = [1, 0] # where empty data space begins in each worksheet
verbose_mode = 1