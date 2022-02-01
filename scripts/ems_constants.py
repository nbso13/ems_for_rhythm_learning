import numpy as np
# set vars


# CURRENT PARAMS FOR NICK: 7 intensity 150 ms.
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
nothing_str = '00000000'
all_str = '11111111'
almost_all = '11111011'
rhythm_strings = [to_the_beat_substr]
# lunch_room_beat, seven_to_four, three_to_four_substr, bass_drum_pattern, five_to_four_substr, syncopated_substr, telescoping, flip_the_beat, clave_substr
# clave_substr, five_to_four_substr, three_to_four_substr,  \
#     syncopated_substr, bass_drum_pattern, flip_the_beat
rhythm_strings_names = ["to_the_beat_substr"]
# "lunch_room_beat", "seven_to_four", "three_to_four_substr", "bass_drum_pattern", "five_to_four_substr", "syncopated_substr", "telescoping", "flip_the_beat", "clave_substr"
# , "clave", "five_to_four", "three_to_four", \
#     "syncopated_substr", "bass_drum_pattern", "flip_the_beat"
bpms_ordered = [85, 100, 115, 130, 145, 160, 175] #bpms to try
bpms = [175] # shuffled 100, 160, 130, 85, 115, 145, 
repeats = 1 # ems and audio period repeats
audio_repeats = 1
post_ems_repeats = 1 # how many post ems repeats
no_audio_repeats = 1
audio_delay = 0.0023 # but why
delay_mode = 'contact' # could be contact or key for keyboard p key sensitivity for measuring delay
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
actual_stim_length = 150 # actual stim length
baseline_subtractor = 15 # this is the noise threshold for contact trace BACK THIS UP EXPERIMENTALLY?
surpression_window = 250 # ms BACK THIS UP EXPERIMENTALLY?
contact_spike_time_width = 2 # ms
double_stroke_rhythm = "101001010010101001010010100"
interval_tolerance = 40 #ms
phase_name_strs = ["pre_ems", "ems_audio", "post_ems", "no_audio"] # name phases
num_phases = len(phase_name_strs) # number of phases is number of names of phases
port_contact = '/dev/cu.usbmodem11401'
port_ems = '/dev/tty.usbserial-18DNB483'
# port_ems = '/dev/ttys000' for bluetooth
worksheet_data_begin_indices = [1, 0] # where empty data space begins in each worksheet
verbose_mode = 1
channel = 0


level_zero_probability = 0.891 # Hayden motzart
distances_to_next_level = [8, 4, 2, 1]
metrical_position_probabilities = [0.988, 0.092, 0.584, 0.309, 0.842, 0.126, 0.794, 0.313]
anchor_stat = {
    # whole notes, half notes, quarter notes, eighthnotes
    'both_anchored':[np.nan, 0.738, 0.68, 0.576],
    'post_anchored':[np.nan, 0.685, 0.51, 0.359],
    'pre_anchored':[np.nan, 0.525, 0.239, 0.109],
    'not_anchored':[np.nan,  0.5, 0.398, 0.12 ]
}



runtime_parameters = {
    "to_the_beat_substr" : to_the_beat_substr,
    "lunch_room_beat" : lunch_room_beat,
    "clave_substr" : clave_substr,
    "five_to_four_substr" : five_to_four_substr,
    "three_to_four_substr" : three_to_four_substr,
    "seven_to_four_substr" : seven_to_four,
    "syncopated_substr" : syncopated_substr,
    "bass_drum_pattern" : bass_drum_pattern,
    "flip_the_beat" : flip_the_beat,
    "telescoping" : telescoping,
    "count_in_substr" : count_in_substr,
    "nothing_str" : nothing_str,
    "all_str" : all_str,
    "almost_all" : almost_all,
    "rhythm_strings" : rhythm_strings,
    # clave_substr, five_to_four_substr, three_to_four_substr,  \
    #     syncopated_substr, bass_drum_pattern, flip_the_beat
    "rhythm_strings_names" : rhythm_strings_names,
    # , "clave", "five_to_four", "three_to_four", \
    #     "syncopated_substr", "bass_drum_pattern", "flip_the_beat"
    "bpms_ordered" : bpms_ordered, #bpms to try
    "bpms" : bpms, # shuffled
    "repeats" : repeats, # ems and audio period repeats
    "audio_repeats" : audio_repeats,
    "post_ems_repeats" : post_ems_repeats, # how many post ems repeats
    "no_audio_repeats" : no_audio_repeats,
    "audio_delay" : audio_delay, # but why
    "delay_mode" : delay_mode, # could be contact or key for keyboard p key sensitivity for measuring delay
    # bpm : 120 # beats per minute
    "ems_flag" : ems_flag, # turn ems on
    "audio_flag" : audio_flag, # turn audio on
    "audio_pre_display_flag" : audio_pre_display_flag, # include audio pre-ems display for training purposes
    "post_ems_test_flag" : post_ems_test_flag, # include audio only post-ems display for testing purposes
    "no_audio_flag" : no_audio_flag,# include metronome only post-post-ems display for testing purposes
    "metronome_intro_flag" : metronome_intro_flag, # if we want a count in
    "samp_period_ms" : samp_period_ms, # milliseconds
    "delay_trial_num" : delay_trial_num, # if measuring delay, this is how many trials we use
    "sleep_len" : sleep_len, # seconds of waiting while zeroing sensor
    "actual_stim_length" : actual_stim_length, # actual stim length
    "baseline_subtractor" : baseline_subtractor, # this is the noise threshold for contact trace BACK THIS UP EXPERIMENTALLY?
    "surpression_window" : surpression_window, # ms BACK THIS UP EXPERIMENTALLY?
    "contact_spike_time_width" : contact_spike_time_width, # ms
    "double_stroke_rhythm" : double_stroke_rhythm,
    "interval_tolerance" : interval_tolerance, #ms
    "phase_name_strs" : phase_name_strs, # name phases
    "num_phases" : num_phases, # number of phases is number of names of phases
    "worksheet_data_begin_indices" : worksheet_data_begin_indices# where empty data space begins in each worksheet
}
