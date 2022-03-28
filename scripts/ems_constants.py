import numpy as np
# set vars


# CURRENT PARAMS FOR NICK: 7 intensity 150 ms.
practice_rhythm = "1000100000100000"
demo_rhythm = "1000100010010010"
practice_bpm = 130
to_the_beat_substr = "1000100010001000"
lunch_room_beat = "1000100010100010"
clave_substr = "0010100010010010"
five_to_four_substr = "10000100001000010000"
three_to_four_substr = "1001001001001000"
seven_to_four = "1000000100000010000001000000"
syncopated_substr = "0010001000100010"
bass_drum_pattern = "1010010100100100"
flip_the_beat = "1000100100010000"
telescoping = "1000100101001000"
count_in_substr = '1000100010001000'
random_generated_one = "1000100010100100"
random_generated_two = "0100010001010010"
random_one_march_22 = '0001000100010001'
random_two_march_22 = '1000100010010010'
random_three_march_22 = '1001000100010010'
random_four_march_22 = '0010000100010001'
random_one_march_23 = "0000100010000010"
random_two_march_23 = "1000100010010010"
nothing_str = '00000000'
all_str = '11111111'
almost_all = '11111011'
random_three = '0010001001000001'
random_four = '0100100001001000'
random_five = '0100100010000010'
fandango = '100000100000100000100000'
solea = '000010000010001000100010'
buleria = '000010000000101000100010' 
seguiria = '100010001000001000001000'
guajira = '100000100000100010001000'
alegria = '000010000010001000100010' # guajira", "seguiria", "buleria", "solea", "fandango", 
# rhythm_strings = [to_the_beat_substr, random_generated_one,  three_to_four_substr, syncopated_substr, random_generated_two, telescoping] # 
# rhythm_strings = [to_the_beat_substr, syncopated_substr, three_to_four_substr, random_three, random_four, lunch_room_beat, random_five] # 
rhythm_strings = [syncopated_substr, flip_the_beat, random_one_march_22, random_two_march_22, random_three_march_22, random_four_march_22, random_five] 
# random_one_march_22, random_two_march_22, random_three_march_22, random_four_march_22
# lunch_room_beat, seven_to_four, three_to_four_substr, bass_drum_pattern, five_to_four_substr, syncopated_substr, telescoping, flip_the_beat, clave_substr
# clave_substr, five_to_four_substr, three_to_four_substr,  \
#     syncopated_substr, bass_drum_pattern, flip_the_beat
rhythm_strings_names = ['syncopated_substr', 'flip_the_beat', 'random_one_march_22', 'random_two_march_22', 'random_three_march_22', 'random_four_march_22', 'random_five']  # , 
# "random_one_march_22", "random_two_march_22", "random_three_march_22", "random_four_march_22"
# rhythm_strings_names = ["to_the_beat_substr", "random_generated_one", "three_to_four_substr", "syncopated_substr", "random_generated_two", "telescoping"] 
# "lunch_room_beat", "seven_to_four", "three_to_four_substr", "bass_drum_pattern", "five_to_four_substr", "syncopated_substr", "telescoping", "flip_the_beat", "clave_substr"
# , "clave", "five_to_four", "three_to_four", \
#     "syncopated_substr", "bass_drum_pattern", "flip_the_beat"
# bpms_ordered = [100, 115, 130, 145, 160, 175] #bpms to try 
# bpms = [100, 160, 140, 120] # shuffled  
bpms = [110, 150]
# repeats = 4 # ems and audio period repeats
# audio_repeats = 4
# post_ems_repeats = 4 # how many post ems repeats
# no_audio_repeats = 4
prep_time_ms = 100
update_period_ms = 2 
refractory_period_ms = 250 
audio_delay = 0.0023 # but why
delay_mode = 'contact' # could be contact or key for keyboard p key sensitivity for measuring delay
# bpm = 120 # beats per minute
phase_flags_list = [1, 1, 1, 1, 1] # turns a phase on or off
phase_repeats_list = [8, 4, 7, 1, 4] # repeats at each phase
demo_phase_repeats_list = [5, 3, 4, 1, 3] # repeats at each phase
phase_name_strs = ["pre ems audio", "pre ems no audio", "ems", "post ems audio", "post ems no audio"] # name phases
phase_warning_strs =  ["TRAINING", "METRONOME TESTING", "EXPERIMENTAL TRAINING", "AUDIO ADJUST PHASE", "METRONOME TESTING"] # name phases
num_phases = len(phase_name_strs) # number of phases is number of names of phases
audio_on_flags = [1, 0, 1, 1, 0] # at each phase, whether the audio is on
ems_on_flags = [0, 0, 1, 0, 0]
metronome_intro_flag = 1 # if we want a count in
samp_period_ms = 5 # milliseconds
delay_trial_num = 12 # if measuring delay, this is how many trials we use
sleep_len_ms = 1000 # seconds of waiting while zeroing sensor
sd_more_than_mult = 7 # deprecated
actual_stim_length = 155 # actual stim length
baseline_subtractor = 10 # this is the noise threshold for contact trace BACK THIS UP EXPERIMENTALLY?
surpression_window = 300 # ms BACK THIS UP EXPERIMENTALLY?
contact_spike_time_width = 2 # ms
double_stroke_rhythm = "1010010100101010"
interval_tolerance = 100 #ms
delay_reduction = 120 # ms
port_contact = '/dev/cu.usbmodem11201'
port_ems = '/dev/tty.usbserial-18DNB483'
# port_ems = '/dev/ttys000' for bluetooth
worksheet_data_begin_indices = [1, 0] # where empty data space begins in each worksheet
phase_change_warnings_delay = 3000 # ms before phase change to warn participant
verbose_mode = 0
channel = 0
read_buffer_time_val = 2000 # ms added to read length in order to capture the end of the contact trace
chopping_buffer = 150 # ms, amount of time before first audio pulse and after last audio pulse where contact is recorded.
cutoff_freq_low = 0.005 # 100 ms longest wave length component before attenuation during high pass filtering
cutoff_freq_high = 0.1 # 20 ms shortest allowed wave 
level_zero_probability = 0.891 # Hayden motzart
distances_to_next_level = [8, 4, 2, 1]
metrical_position_probabilities = [0.988, 0.092, 0.584, 0.309, 0.842, 0.126, 0.794, 0.313]
analysis_sample_period = 6 # ms
anchor_stat = {
    # whole notes, half notes, quarter notes, eighthnotes
    'both_anchored':[np.nan, 0.738, 0.68, 0.576],
    'post_anchored':[np.nan, 0.685, 0.51, 0.359],
    'pre_anchored':[np.nan, 0.525, 0.239, 0.109],
    'not_anchored':[np.nan,  0.5, 0.398, 0.12 ]
}

counter_balanced_number_dict = {
    "1" : "actuating-EMS",
    "2" : "tactile-EMS",
    "3" : "no-EMS"
}

buffer_time_after = 50 # ms. the time after a given audio onset to look for contact onsets to estimate user performance of that interval.


runtime_parameters = {
    "to_the_beat_substr" : to_the_beat_substr,
    "random_three" : random_three,
    "random_four" : random_four,
    "random_five" : random_five,
    "random_one_march_22" : random_one_march_22,
    "random_two_march_22" : random_two_march_22,
    "random_three_march_22" : random_three_march_22,
    "random_four_march_22" : random_four_march_22,
    "random_one_march_23" : random_one_march_23,
    "random_two_march_23" : random_two_march_23,
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
    "random_generated_one" : random_generated_one,
    "random_generated_two" : random_generated_two,
    # clave_substr, five_to_four_substr, three_to_four_substr,  \
    #     syncopated_substr, bass_drum_pattern, flip_the_beat
    "rhythm_strings_names" : rhythm_strings_names,
    # , "clave", "five_to_four", "three_to_four", \
    #     "syncopated_substr", "bass_drum_pattern", "flip_the_beat"
    "update_period_ms" : update_period_ms,
    "refractory_period_ms" : refractory_period_ms,
    "prep_time_ms" : prep_time_ms,
    "bpms" : bpms, # shuffled
    "audio_on_flags" : audio_on_flags,
    "ems_on_flags" : ems_on_flags,
    "phase_repeats_list" : phase_repeats_list,
    "delay_reduction" : delay_reduction,
    "audio_delay" : audio_delay, # but why
    "delay_mode" : delay_mode, # could be contact or key for keyboard p key sensitivity for measuring delay
    # bpm : 120 # beats per minute
    "phase_flags_list" : phase_flags_list,
    "metronome_intro_flag" : metronome_intro_flag, # if we want a count in
    "samp_period_ms" : samp_period_ms, # milliseconds
    "delay_trial_num" : delay_trial_num, # if measuring delay, this is how many trials we use
    "sleep_len_ms" : sleep_len_ms, # seconds of waiting while zeroing sensor
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
