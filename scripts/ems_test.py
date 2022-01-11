import numpy as np
import math
import time
import threading
import matplotlib.pyplot as plt
import vlc
import datetime
import xlsxwriter
import ems_constants
import serial
import time

# current best settings: 155 ms. 10 intensity. bpm 110. never double up strokes direct. You can triple stroke indirect tho.

def play_rhythm(ems_serial, contact_ser, actual_stim_length, count_in_substr, rhythm_substr, \
    repeats, bpm, metronome_intro_flag, audio_pre_display_flag, no_audio_flag, audio_repeats, post_ems_test_flag, post_ems_repeats,  \
        no_audio_repeats, samp_period_ms, delay_val):
        # mammoth function that should be broken up. takes in serial objects, rhythm parameters,
        # whether the EMS and the audio should be turned on, whether there should be a metronome intro,
        # sample period, and measured delay value. Plays the rhythm in audio and EMS and runs threading
        # to listen to user results.

    reading_results = [] # a list that is written to by the thread that is the contact
    x_value_results = [] # a list of time values corresponding to when each contact data point was recorded
    
    max_bpm = math.floor(30000/actual_stim_length) #how many eighthnote pulses could you fit into a 
    #minute without overlapping?
    if (bpm > max_bpm):
        print("max metronome bpm is " + str(max_bpm))
        return

    #determine pulse+wait length
    milliseconds_per_eighthnote = 30000/bpm

    ## start reading thread ##
    len_pres = 3000 + milliseconds_per_eighthnote*(len(count_in_substr) + (audio_repeats+repeats+post_ems_repeats+no_audio_repeats) *  \
        len(rhythm_substr)) + delay_val # length of rhythm presentation
    
    audio_onset_times = [] # when list of times audio began to play
    stim_onset_times = []  # list of times when stim command was sent
    time_naught_thread = time.time() # beginning of time for contact tracing thread. 
    # SHOULD THIS BE DIFFERENT FROM OTHER BEGINNING TIME COUNT?
    read_thread = threading.Thread(target=read_contact_trace, args= (contact_ser, len_pres,  \
        samp_period_ms, reading_results, x_value_results, time_naught_thread)) # creating read thread
    read_thread.start() 

    rhythm_display_flag = 1 # SHOULD display EMS and audio together after audio presentation.
    post_ems_test_flag = 1
    # total eighthnotes in count in, audio display, and rhythm display (EMS+audio)
    total_eighthnotes = len(count_in_substr) + (repeats+audio_repeats+post_ems_repeats+no_audio_repeats)*len(rhythm_substr) 
    time_naught_main = time.time() # beginning time for EMS and audio threads. 
    #WHY SHOULD THIS BE DIFFERENT THAN CONTACT THREAD?

    ## creating EMS and audio and metronome threads ##
    ems_thread = threading.Thread(target=run_rhythm_ems, args= (rhythm_display_flag, ems_serial, time_naught_main,  \
        stim_onset_times, repeats, rhythm_substr, actual_stim_length, milliseconds_per_eighthnote, \
             metronome_intro_flag, count_in_substr, audio_pre_display_flag, audio_repeats)) 
    audio_thread = threading.Thread(target=run_rhythm_audio, args= (rhythm_display_flag, post_ems_test_flag, audio_onset_times, time_naught_main, repeats, rhythm_substr, \
    milliseconds_per_eighthnote, metronome_intro_flag, count_in_substr, audio_pre_display_flag, no_audio_flag, audio_repeats, post_ems_repeats, no_audio_repeats))
    metronome_thread = threading.Thread(target=metronome_tone, args= (milliseconds_per_eighthnote, total_eighthnotes))

    print("rhythm in 3")
    time.sleep(1)
    print("rhythm  in 2")
    time.sleep(1)
    print("rhythm  in 1")
    time.sleep(1)
    ems_thread.start()

    time.sleep(delay_val/1000) # implements delay between EMS and audio timelines.
    
    metronome_thread.start()

    audio_thread.start()

    ems_thread.join()
    metronome_thread.join()
    audio_thread.join()
    read_thread.join()
    audio_onset_times_ms = [1000 * item for item in audio_onset_times] # take list of onset times from seconds to ms.
    stim_onset_times_ms = [1000 * item for item in stim_onset_times]

    # reading results is the contact trace list. x value results are time of recording each data point.
    return reading_results, x_value_results, audio_onset_times_ms, stim_onset_times_ms 

def run_rhythm_ems(rhythm_display_flag, ems_serial, time_naught, stim_onset_times, repeats, rhythm_substr, actual_stim_length, \
    milliseconds_per_eighthnote, metronome_intro_flag, count_in_substr, audio_pre_display_flag, pre_repeats):
    # runs ems. vars:
    # rhythm display flag: if 1, display EMS and audio after audio_only presentation. if 0, no rhythm display at all
    # ems_serial: serial object to write commands to
    # time naught is x axis origin (times recorded relative to that)
    # stim onset times is passed as an empty list and appended to (accessed after thread is joined.)
    # repeats: number of RHYTHM DISPLAY repeats
    # rhythm_substr - the string to be played. Each 1 or 0 is an eightnote.
    # actual stim length - how long to tell EMS to stimulate user. in ms. on the order of 100 to 200.
    # ms per eighthnote - self explan
    # metronome_intro_flag: if 1, do metronome count in according to count in str. else skip
    # count_in_substr - the string used to count in (usually hits on beats)
    # audio_pre_display_flag - if yes, play rhythm without ems (probably with audio) before playing rhythm with ems and after count in.
    # prerepeats - this is audio repeats
    # print("ems repeats: " + str(repeats) + "prerepeats: " +str(pre_repeats))
    last_time = time.time() # this last time technique aims to take out processing time. Then we only wait for
    # the perscribed millisecond per eighthnote MINUS processing time. might just be milliseconds but i suspect this builds up
    # problematically.
    # print("EMS THREAD: Beginning metronome: time since start: " + str(time.time()-time_naught))
    if metronome_intro_flag:
        for j in range(len(count_in_substr)):  # go through each eighthnote in the pattern
            if (count_in_substr[j] == '1'): # this is a note
                command_bytes = "xC1I100T" +str(actual_stim_length) + "G \n"
                byt_com = bytes(command_bytes, encoding='utf8')
                stim_onset_times.append(time.time() - time_naught)
                ems_serial.write(byt_com)
                print("stim on")
                time.sleep((milliseconds_per_eighthnote/1000) - time.time() + last_time)
                last_time = time.time()
            elif(count_in_substr[j] == '0'): # rest
                time.sleep((milliseconds_per_eighthnote/1000) - time.time() + last_time)
                last_time = time.time()
            else:
                print("malformed rhythm pattern: " + rhythm_substr)
                break

    # print("EMS THREAD: Beginning predisplay: time since start: " + str(time.time()-time_naught))
    if audio_pre_display_flag:
        for i in range(pre_repeats): # present the rhythm with appropriate number of repeats
            for j in range(len(rhythm_substr)):  # go through each eighthnote in the pattern
                time.sleep((milliseconds_per_eighthnote/1000) - time.time() + last_time)
                last_time = time.time()

    # print("EMS THREAD: Beginning ems display: time since start: " + str(time.time()-time_naught))
    if rhythm_display_flag:
        for i in range(repeats): # present the rhythm with appropriate number of repeats
            for j in range(len(rhythm_substr)):  # go through each eighthnote in the pattern
                if (rhythm_substr[j] == '1'): # this is a note
                    command_bytes = "xC1I100T" +str(actual_stim_length) + "G \n"
                    byt_com = bytes(command_bytes, encoding='utf8')
                    stim_onset_times.append(time.time() - time_naught)
                    ems_serial.write(byt_com)
                    print("stim on")
                    time.sleep((milliseconds_per_eighthnote/1000) - time.time() + last_time)
                    last_time = time.time()
                elif(rhythm_substr[j] == '0'): # rest
                    time.sleep((milliseconds_per_eighthnote/1000) - time.time() + last_time)
                    last_time = time.time()
                else:
                    print("malformed rhythm pattern: " + count_in_substr)
                    break
    # print("EMS THREAD: Beginning post display: time since start: " + str(time.time()-time_naught))

def run_rhythm_audio(rhythm_display_flag, post_ems_test_flag, audio_onset_times, time_naught, repeats, rhythm_substr, \
    milliseconds_per_eighthnote, metronome_intro_flag, count_in_substr, audio_predisplay_flag, no_audio_flag, pre_repeats, post_ems_repeats, no_audio_repeats):
    # runs audio.
    # rhythm display flag: if 1, display EMS and audio after audio_only presentation.
    # audio onset times, passed as an empty list and written to during thread.
    # time naught is x axis origin (times recorded relative to that)
    # repeats: number of RHYTHM DISPLAY repeats
    # rhythm_substr - the string to be played. Each 1 or 0 is an eightnote.
    # ms per eighthnote - self explan
    # metronome_intro_flag: if 1, do metronome count in according to count in str. else skip
    # count_in_substr - the string used to count in (usually hits on beats)
    # audio_pre_display_flag - if yes, play rhythm without ems (probably with audio) before playing rhythm with ems and after count in.
    # prerepeats - this is audio repeats
    # print("repeats: " + str(repeats) + ", prerepeats: " + str(pre_repeats) + ", post_ems repeats: " + str(post_ems_repeats))
    # print("AUDIO THREAD: Beginning metronome: " + str(time.time()-time_naught))
    last_time = time.time()
    first_time = np.copy(last_time)
    if metronome_intro_flag:
        for j in range(len(count_in_substr)):
            fork_time = time.time()
            if int(count_in_substr[j]): # this is a note
                audio_onset_times.append(time.time() - time_naught)
                eighteighty_tone.play()
                time.sleep((milliseconds_per_eighthnote/1000) - time.time() + last_time)
                last_time = time.time()
                eighteighty_tone.stop()
                eight_tone_stop_time = time.time()
            else: # rest
                eighteighty_tone.stop()
                now = time.time()
                time.sleep((milliseconds_per_eighthnote/1000) - now + last_time)
                last_time = time.time()

    # print("AUDIO THREAD: Beginning predisplay: time since start: " + str(time.time()-time_naught))
    if audio_predisplay_flag:
        for i in range(pre_repeats): # present the rhythm with appropriate number of repeats
            for j in range(len(rhythm_substr)):  # go through each eighthnote in the pattern
                if (rhythm_substr[j] == '1'): # this is a note
                    audio_onset_times.append(time.time() - time_naught)
                    eighteighty_tone.play()
                    time.sleep((milliseconds_per_eighthnote/1000) - time.time() + last_time)
                    last_time = time.time()
                    eighteighty_tone.stop()
                elif(rhythm_substr[j] == '0'): # rest
                    eighteighty_tone.stop()
                    time.sleep((milliseconds_per_eighthnote/1000) - time.time() + last_time)
                    last_time = time.time()
                else:
                    print("malformed rhythm pattern: " + rhythm_substr)
                    break
    # print("AUDIO THREAD: Beginning rhythm display: time since start: " + str(time.time()-time_naught))
    if rhythm_display_flag:
        for i in range(repeats): # present the rhythm with appropriate number of repeats
            for j in range(len(rhythm_substr)):  # go through each eighthnote in the pattern
                if (rhythm_substr[j] == '1'): # this is a note
                    audio_onset_times.append(time.time() - time_naught)
                    eighteighty_tone.play()
                    time.sleep((milliseconds_per_eighthnote/1000) - time.time() + last_time)
                    last_time = time.time()
                    eighteighty_tone.stop()
                elif(rhythm_substr[j] == '0'): # rest
                    eighteighty_tone.stop()
                    time.sleep((milliseconds_per_eighthnote/1000) - time.time() + last_time)
                    last_time = time.time()
                else:
                    print("malformed rhythm pattern: " + rhythm_substr)
                    break
    # print("AUDIO THREAD: Beginning post_ems display: time since start: " + str(time.time()-time_naught))
    if post_ems_test_flag:
        for i in range(post_ems_repeats): # present the rhythm with appropriate number of repeats
            for j in range(len(rhythm_substr)):  # go through each eighthnote in the pattern
                if (rhythm_substr[j] == '1'): # this is a note
                    audio_onset_times.append(time.time() - time_naught)
                    eighteighty_tone.play()
                    time.sleep((milliseconds_per_eighthnote/1000) - time.time() + last_time)
                    last_time = time.time()
                    eighteighty_tone.stop()
                elif(rhythm_substr[j] == '0'): # rest
                    eighteighty_tone.stop()
                    time.sleep((milliseconds_per_eighthnote/1000) - time.time() + last_time)
                    last_time = time.time()
                else:
                    print("malformed rhythm pattern: " + rhythm_substr)
                    break
    # print("AUDIO THREAD: Beginning no_audio display: time since start: " + str(time.time()-time_naught))
    # this section is to account for ground truth rhythm trace even when no audio is being played so we can compare
    # against user performance in the no audio (but with metronome) section.

    if no_audio_flag: 
        for i in range(no_audio_repeats): # present the rhythm with appropriate number of repeats
            for j in range(len(rhythm_substr)):  # go through each eighthnote in the pattern
                if (rhythm_substr[j] == '1'): # this is a note
                    audio_onset_times.append(time.time() - time_naught)
                    # eighteighty_tone.play()
                    time.sleep((milliseconds_per_eighthnote/1000) - time.time() + last_time)
                    last_time = time.time()
                    # eighteighty_tone.stop()
                elif(rhythm_substr[j] == '0'): # rest
                    # eighteighty_tone.stop()
                    time.sleep((milliseconds_per_eighthnote/1000) - time.time() + last_time)
                    last_time = time.time()
                else:
                    print("malformed rhythm pattern: " + rhythm_substr)
                    break

def metronome_tone(milliseconds_per_eighthnote, total_str_len):
    # plays tone on the beat repeatedly
    AUDIO_DELAY = ems_constants.audio_delay
    time.sleep(AUDIO_DELAY) # sleep for 2 ms to let audio catch up
    last_time = time.time()
    counter = 0 
    for i in range(total_str_len):
        counter = counter + 1
        if counter == 1:
            fourfourty_tone.play()
            time.sleep((milliseconds_per_eighthnote/1000) - time.time() + last_time)
            last_time = time.time()
            fourfourty_tone.stop()
        else:
            if counter == 8:
                counter = 0
            time.sleep((milliseconds_per_eighthnote/1000) - time.time() + last_time)
            last_time = time.time()

def read_contact_trace(ser,  len_rhythm_presentation_ms, samp_period_ms, readings_list, x_values_list, time_naught_contact_trace):
    # reads from contact detection serial object every sample period. Saves results to a list
    # time.sleep(1)
    # print("thread time since start " + str(time.time()- time_naught))
    check_repeats = int(np.floor((len_rhythm_presentation_ms/samp_period_ms)))
    print("read thread begun")
    while (time.time()-time_naught_contact_trace)*1000 < len_rhythm_presentation_ms:
        if ser.in_waiting:
            out = ser.readline().decode('utf-8')
            time_measured = time.time()
            # if int(out[:-2]) > 5:
            #     print(int(out[:-2]))
            readings_list.append(int(out[:-2]))
            x_values_list.append(1000*(time_measured-time_naught_contact_trace)) #from seconds to milliseconds
    print("done reading trace")
    # print("mean samp period and stdv: " + str(mean_contact_samp_period) + " +/- " + str(stdv_contact_samp_period))
    return readings_list, x_values_list

def rhythm_string_to_stim_trace_and_audio_trace(count_in_substr, rhythm_substr,  actual_stim_length, bpm, repeats,  \
    samp_period, delay, audio_repeats, post_ems_repeats):
    # takes in the count-in string, the actual rhythm string, the length of stimulation in ms, beats per minute,
    # stim repeats number, requested sample period of resulting trace (in ms). Returns stim_trace numpy array
    # with 0 values for time points of no stim and 1000 values for stim. This is offset /delay/ amount in ms
    # from audio stimulus (also returned in same size array). Final value returned is a time array, steps in 
    # samp_period.
    milliseconds_per_eighthnote = 30000/bpm
    array_len_per_eighthnote = int(np.floor(milliseconds_per_eighthnote/samp_period))
    delay_array_len = int(np.floor(delay/samp_period))
    actual_stim_len_array_indices = int(np.floor(actual_stim_length/samp_period))
    eighthnotes_pres = len(count_in_substr) + (audio_repeats+repeats+post_ems_repeats) * len(rhythm_substr)
    trace_array_len = array_len_per_eighthnote * eighthnotes_pres + delay_array_len
    stim_trace = np.zeros((trace_array_len,))
    audio_trace = np.zeros((trace_array_len,))
    x_array = np.arange(0, trace_array_len) * samp_period

    for i in range(len(count_in_substr)): # write in count-in traces.
        if count_in_substr[i] == '1':
            stim_begin_ind = i * array_len_per_eighthnote
            stim_end_ind = stim_begin_ind + actual_stim_len_array_indices
            stim_trace[stim_begin_ind:stim_end_ind] = 1
            audio_begin_ind = stim_begin_ind+delay_array_len
            audio_end_ind = audio_begin_ind + array_len_per_eighthnote
            audio_trace[audio_begin_ind:audio_end_ind] = 1


    start_index_audio = len(count_in_substr) * array_len_per_eighthnote + delay_array_len

    if audio_repeats > 0:
        for i in range(audio_repeats): # write the audio trace for any audio pre-stim presentation
            for j in range(len(rhythm_substr)):
                if rhythm_substr[j] == '1':
                    audio_begin_ind = start_index_audio + (j * array_len_per_eighthnote)
                    audio_end_ind = audio_begin_ind + array_len_per_eighthnote
                    audio_trace[audio_begin_ind:audio_end_ind] = 1
            start_index_audio = start_index_audio + (array_len_per_eighthnote * len(rhythm_substr))
    
    start_index_stim = array_len_per_eighthnote * (len(count_in_substr) + (audio_repeats * len(rhythm_substr)))

    for i in range(repeats): # now writing for actual rhythm display and actuation
        for j in range(len(rhythm_substr)):
            if rhythm_substr[j] == '1':
                stim_begin_ind = start_index_stim + (j * array_len_per_eighthnote)
                stim_end_ind = stim_begin_ind + actual_stim_len_array_indices
                stim_trace[stim_begin_ind:stim_end_ind] = 1
                audio_begin_ind = stim_begin_ind+delay_array_len
                audio_end_ind = audio_begin_ind + array_len_per_eighthnote
                audio_trace[audio_begin_ind:audio_end_ind] = 1
                audio_trace[audio_end_ind] = 0
        start_index_stim = start_index_stim + (array_len_per_eighthnote * len(rhythm_substr))
    
    return stim_trace, audio_trace, x_array

def plot_contact_trace_and_rhythm(reading_list, contact_x_values, stim_trace, audio_trace, x_array, samp_period, legend_labels, input_title):
    fig, ax = plt.subplots()
    ax.plot(contact_x_values, reading_list)
    ax.set_yticks(np.arange(0, 500, 100))
    ax.set_xticks(np.arange(0, (len(reading_list) * samp_period), 10000))
    ax.set_title(input_title)
    ax.plot(x_array, stim_trace*np.max(reading_list))
    ax.plot(x_array, audio_trace*np.max(reading_list))
    ax.legend(legend_labels)
    plt.ion()
    plt.show()
    plt.draw()
    plt.pause(0.01)

def onset_times_to_traces(audio_onset_times, audio_hold_ms, stim_onset_times, stim_hold_ms, samp_period):
    # take a series of onset time points and craft plottable traces.
    array_value_audio_hold = int(np.floor(audio_hold_ms/samp_period))
    array_value_stim_time = int(np.floor(stim_hold_ms/samp_period))
    final_time_point = int(np.floor(np.max(audio_onset_times) + audio_hold_ms))
    x_vec = np.arange(0, final_time_point, samp_period)
    audio_trace = np.zeros_like(x_vec)
    stim_trace = np.zeros_like(x_vec)
    for time_val in audio_onset_times:
        array_ind_begin = int(np.floor(time_val/samp_period))
        array_ind_end = array_ind_begin + array_value_audio_hold
        audio_trace[array_ind_begin:array_ind_end] = 1
    for time_val in stim_onset_times:
        array_ind_begin = int(np.floor(time_val/samp_period))
        array_ind_end = array_ind_begin + array_value_stim_time
        stim_trace[array_ind_begin:array_ind_end] = 1
    return x_vec, audio_trace, stim_trace

def spike_times_to_traces(onset_times, hold_length, x_vector, samp_period):
    # take a series of onset time points and craft plottable traces.
    array_value_stim_time = int(np.floor(hold_length/samp_period))
    trace = np.zeros_like(x_vector)
    for time_val in onset_times:
        array_ind_begin = int(np.floor(time_val/samp_period))
        array_ind_end = array_ind_begin + array_value_stim_time
        trace[array_ind_begin:array_ind_end] = 1
    return trace

def trace_to_spike_times(baseline_mean, baseline_sd, reading_results_list, x_values, sd_more_than_multiplier, baseline_subtractor):
    # take a trace and threshold to pull spike times.
    reading_results_array = np.array(reading_results_list)
    x_vals_array = np.array(x_values)
    bool_list = reading_results_array < baseline_subtractor
    reading_results_array[bool_list] = 0 # anything below this baseline is 0'd out
    bool_selector = reading_results_array > baseline_mean + baseline_sd*sd_more_than_multiplier
    time_points = x_vals_array[bool_selector]
    return time_points

def zero_sensor(contact_ser, sleep_len_ms, samp_period_ms):
    print("DON't TOUCH - zeroing")
    time.sleep(0.5)
    initial_outlist = []
    initial_x_results = []
    first_time_naught = time.time()
    read_contact_trace(contact_ser, sleep_len_ms, samp_period_ms, initial_outlist, initial_x_results, \
        first_time_naught)
    baseline_mean = np.mean(np.array(initial_outlist))
    baseline_sd = np.std(np.array(initial_outlist))
    print("Mean basline was "  + str(baseline_mean) + " +/- " + str(baseline_sd))
    print("DONE ZEROING")
    return baseline_mean, baseline_sd

def measure_delay(ems_serial, contact_ser, actual_stim_length, trial_num, sleep_len, samp_period_ms, sd_more_than_mult, baseline_subtractor, baseline_mean, baseline_sd):
    # uses a set of trials and random stims and determines the average delay from EMS command to contact registration.
    

    times_stimmed = []
    reading_results = []
    x_value_results = []
    rand_values = np.divide(np.random.rand(trial_num), 2) #between 0 and 0.5 second random delay
    len_pres = 3000 + (trial_num * sleep_len + np.sum(rand_values)) * 1000 # ms
    time_naught_delay = time.time()
    print("time naught delay: " + str(time_naught_delay))
    read_thread = threading.Thread(target=read_contact_trace, args= (contact_ser, len_pres,  \
        samp_period_ms, reading_results, x_value_results, time_naught_delay))
    time_naught_main = time.time()
    print("time naught main thread: " + str(time_naught_main))
    read_thread.start()
    # time.sleep(1)
    # print("time since start: " + str(time.time() - time_naught_main))
    print("calibrating delay in 3")
    time.sleep(1)
    print("calibrating delay in 2")
    time.sleep(1)
    print("calibrating delay in 1")
    time.sleep(1)
    for i in range(trial_num):
        command_bytes = "xC1I100T" + str(actual_stim_length) + "G \n" # metronome intro
        byt_com = bytes(command_bytes, encoding='utf8')
        ems_serial.write(byt_com)
        times_stimmed.append(time.time()-time_naught_main)
        print("STIM " + str(i))
        time.sleep(sleep_len)
        time.sleep(rand_values[i])
    read_thread.join()

    times_responded_ms = trace_to_spike_times(baseline_mean, baseline_sd, reading_results, x_value_results,  sd_more_than_mult, baseline_subtractor)
    times_stimmed_ms = 1000*np.array(times_stimmed)
    first_responses_post_stim = []
    diffs = []
    for i in range(len(times_stimmed_ms)):
        # get earliest response threshold crossing
        temp = np.copy(times_responded_ms)
        before_bool = np.subtract(times_responded_ms, times_stimmed_ms[i]) < 0 # subtract stimmed time from response times to find
        # only responses after stim. then get bools above 0.
        temp[before_bool] = np.max(times_responded_ms) # set befores to maximum to avoid finding a close one before stim
        first_threshold_cross_post_stim = np.argmin(temp)
        first_responses_post_stim.append(times_responded_ms[first_threshold_cross_post_stim])
        diffs.append(times_responded_ms[first_threshold_cross_post_stim] - times_stimmed_ms[i])
    first_responses_post_stim = np.array(first_responses_post_stim)
    mean_delay = np.mean(diffs)
    std_delay = np.std(diffs)
    return mean_delay, std_delay, first_responses_post_stim, times_stimmed_ms, reading_results, x_value_results

def test_double_stroke(ems_serial, actual_stim_length, bpm, double_stroke_rhythm):
    # tests sensation of double and triple strokes. This depends on stim length, stim intensity, and bom.
    temp = 0
    reps = 1
    metronome = 0
    milliseconds_per_eighthnote = 30000/bpm
    milliseconds_wait = milliseconds_per_eighthnote - actual_stim_length
    rhythm_display_flag = 1
    audio_pre_display_flag = 0
    pre_repeats = 0
    run_rhythm_ems(rhythm_display_flag, ems_serial, temp, [], reps, double_stroke_rhythm, actual_stim_length,  \
        milliseconds_per_eighthnote, metronome, [], audio_pre_display_flag, pre_repeats)

def process_contact_trace_to_hit_times(contact_trace_array, x_values_array, threshold, surpression_window):
    bool_list = contact_trace_array > threshold # find indices of contact trace array that exceed threshold
    time_points = x_values_array[bool_list] # get the time points of those points in trace
    time_points_cop = np.copy(time_points) # make a shallow copy so as not to modify the array we are looping through (i think...)
    for i in range(len(time_points)): 
        if np.isnan(time_points_cop[i]): # if the point is nan do not surpress.
            continue
        max_time_surpress = time_points[i] + surpression_window
        indices_to_surpress_bools = np.logical_and((time_points > time_points[i]), (time_points <= max_time_surpress))
        time_points_cop[indices_to_surpress_bools] = np.nan
    nonsurpressedselector_bool = np.logical_not(np.isnan(time_points_cop))
    time_points_out = time_points[nonsurpressedselector_bool]
    return time_points_out

def wrapper_test_double_stroke(ems_serial):
    out = input("test double stroke sensation?")
    if out == 'y':
        contin = True
        while contin:
            test_double_stroke(ems_serial, ems_constants.actual_stim_length, ems_constants.bpm, ems_constants.double_stroke_rhythm)
            out = input("adjust? a / continue? c")
            if out == 'c':
                contin = False

def test_sounds():
    fourfourty_tone.play()
    eighteighty_tone.play()
    time.sleep(0.3)
    time_before = time.time()
    fourfourty_tone.stop()
    eighteighty_tone.stop()
    time_to_stop_tones = time.time() - time_before
    print("time to stop tones: " + str(time_to_stop_tones))

def load_sounds():
    global fourfourty_tone
    fourfourty_tone = vlc.MediaPlayer("tones/440Hz_44100Hz_16bit_05sec.mp3")
    global eighteighty_tone
    eighteighty_tone = vlc.MediaPlayer("tones/880hz.mp3")
    test_sounds()
    return

def listen_serial(ems_serial):
    for k in range(300):
        if(ems_serial.in_waiting):
            out = ems_serial.readline().decode('utf-8')
            print(out)  
        time.sleep(0.001) 

def test_contact_trace_read(contact_serial):
    while True:
        out = contact_serial.readline().decode('utf-8')
        if int(out)>0:
            print(int(out[:-2]))

def set_up_serials(port_ems, port_contact):
    ems_serial = serial.Serial(port_ems, 115200) # baud number
    ems_serial.flushInput()
    ems_serial.write(b"2")
    
    ## read all setup from EMS
    listen_serial(ems_serial)
    contact_serial = serial.Serial(port_contact, 9600) # baud number 
    #IF THIS IS THROWING A could not open port ERROR look for correct port at /dev/cu.usbmodem____ on mac.
    
    ## testing contact trace read
    # test_contact_trace_read(contact_serial)
    
    return ems_serial, contact_serial

def write_headers(worksheet, label_header, header_values, rotated_bpm_list):
    for i in range(len(label_header)):
        # first entry to write is the data row number
        worksheet.write(0, i, label_header[i], bold) # write in header values
        worksheet.write(1, i, header_values[i], bold)

    for i in range(len(ems_constants.rhythm_strings_names)):
        worksheet.write(2, i, ems_constants.rhythm_strings_names[i])
        worksheet.write(3, i, ems_constants.rhythm_strings[i])

    worksheet.write(4, 0, "bpms: ")
    for i in range(len(rotated_bpm_list)):
        worksheet.write(4, i+1, rotated_bpm_list[i])
    return

def measure_delay_loop(ems_serial, contact_serial, baseline_mean, baseline_sd):
    repeat_bool = True
    while(repeat_bool):
        delay_mean, delay_std, reaction_onsets, stim_onsets, \
        reading_results, contact_x_values = measure_delay(ems_serial, contact_serial, ems_constants.actual_stim_length, \
            ems_constants.delay_trial_num, ems_constants.sleep_len, ems_constants.samp_period_ms,  \
            ems_constants.sd_more_than_mult, ems_constants.baseline_subtractor, baseline_mean, baseline_sd)
        
        x_vec = np.arange(0, np.max(contact_x_values), ems_constants.samp_period_ms)
        stim_trace = spike_times_to_traces(stim_onsets, ems_constants.actual_stim_length, x_vec, ems_constants.samp_period_ms)
        reaction_trace = spike_times_to_traces(reaction_onsets, ems_constants.actual_stim_length, x_vec, ems_constants.samp_period_ms)

        # x_vec, reaction_trace, stim_trace = onset_times_to_traces(reaction_onsets, ems_constants.contact_spike_time_width, stim_onsets, ems_constants.actual_stim_length, ems_constants.samp_period_ms)
        
        legend_labels = ["raw response trace", "stim trace",  "filtered response trace"]
        plot_contact_trace_and_rhythm(reading_results, contact_x_values, stim_trace, reaction_trace, x_vec,  \
        ems_constants.samp_period_ms, legend_labels, "Delay Measurement")
        
        print(f"Measured delay was {delay_mean} +/- {delay_std}.")
        
        # out = input("recalibrate double stroke intensity? y/n")
        # if out == 'y':
        #     test_double_stroke(ems_serial)
        
        out = input("y to proceed, n to try again, control C to quit.")
        if out == 'y':
            repeat_bool = False
    return delay_mean, delay_std

def gather_subject_info():
    participant_number = input("participant number?")
    now = datetime.datetime.now()
    test_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    subject_arm = input("subject arm?")
    electrode_config = input("electrode config?") #first pair of numbers is coordinates of 1, x and y, second is coordinates of 2. x and y
    max_ems_stim_intensity = input("max ems stim intensity?")
    pulse_width = input("pulse width?")
    pulse_frequency = input("frequency?") #these may be found on the stimulator and are not usually iterated on (using lit values)
    return participant_number,now, test_time, subject_arm, electrode_config, max_ems_stim_intensity, pulse_width, pulse_frequency

def rotate_list(l, n):
    return l[-n:] + l[:-n]
# example: command_str = "C0I100T750G \n"
# _____________________________________________________
# _____________________### MAIN ###____________________

if __name__ == '__main__':
    tic = time.time()

    ## load sound
    load_sounds()

    #### read and write to arduino ###
    ems_serial, contact_serial, = set_up_serials(ems_constants.port_ems, ems_constants.port_contact)

    ### testing double stroke ###
    wrapper_test_double_stroke(ems_serial)

    ## zero ##
    sleep_len_ms = ems_constants.sleep_len * 1000 
    baseline_mean, baseline_sd = zero_sensor(contact_serial, sleep_len_ms, ems_constants.samp_period_ms)


    ### MEASURE DELAY # TUNE THIS TO KASAHARA RESPONSE TIME, GET RESULTS REGARDING AGENCY AND MEASURE TRAINING RESULT
    out = input("measure delay? 'y' to measure, enter number otherwise in milliseconds.")
    if out == 'y':
        delay_mean, delay_std = measure_delay_loop(ems_serial, contact_serial, baseline_mean, baseline_sd)         
    else:
        delay_mean = int(out)
        delay_std = 0 # if we entered the delay manually then there is no known std.

    MEASURED_DELAY = delay_mean

    ### Gathering subject info ###
    participant_number, now, test_time, subject_arm, electrode_config, max_ems_stim_intensity, pulse_width, pulse_frequency = gather_subject_info()

    ### open workbook, define worksheets ###
    workbook = xlsxwriter.Workbook(f"{test_time}_pp{participant_number}.xlsx")
    bold = workbook.add_format({'bold': True})

    # play rhythm and conduct test
    shuffled_bpm_list = ems_constants.bpms

    # write header worksheet

    label_header = ["pp number", "test time", "subject arm", "electrode config", \
                "max_stim_intensity", "pulse width (microsecs)", "frequency (Hz)", "measured delay mean",  \
                    "measured delay std", "pre-ems repeats", "with ems repeats", "post ems repeats", \
                        "no audio repeats", "zeroed mean", "zeroed sd", "sampling period", "actual stimulation length milliseconds"]
    header_values = [participant_number, test_time, subject_arm, electrode_config, \
        max_ems_stim_intensity, pulse_width, pulse_frequency, MEASURED_DELAY, \
            delay_std, ems_constants.audio_repeats, ems_constants.repeats, ems_constants.post_ems_repeats, \
                ems_constants.no_audio_repeats, baseline_mean, baseline_sd, ems_constants.samp_period_ms, ems_constants.actual_stim_length]

    worksheet = workbook.add_worksheet("header") 
    write_headers(worksheet, label_header, header_values, shuffled_bpm_list)

    for rhythm_index in range(len(ems_constants.rhythm_strings)): # for each of the different rhythms
        rotated_bpm_list = rotate_list(shuffled_bpm_list, rhythm_index)
        for bpm_index in range(len(rotated_bpm_list)): # at each bpm MUST COUNTERBALANCE THIS SOMEHOW (by rotating list of bpms for each rhythm)
            if ems_constants.verbose_mode:
                print(f"rhythm: {ems_constants.rhythm_strings_names[rhythm_index]}, bpm: {rotated_bpm_list[bpm_index]}")
            time.sleep(1) #
            rhythm_substr = ems_constants.rhythm_strings[rhythm_index]
            
            reading_list, contact_x_values, audio_onset_times, stim_onset_times = play_rhythm(ems_serial, \
                contact_serial, ems_constants.actual_stim_length, ems_constants.count_in_substr, rhythm_substr, ems_constants.repeats, rotated_bpm_list[bpm_index], \
                ems_constants.metronome_intro_flag, ems_constants.audio_pre_display_flag, ems_constants.no_audio_flag, ems_constants.audio_repeats,  \
                ems_constants.post_ems_test_flag, ems_constants.post_ems_repeats, \
                ems_constants.no_audio_repeats, ems_constants.samp_period_ms, MEASURED_DELAY) # gives contact trace, audio onset, stim onset.

            reading_list = np.array(reading_list) # raw recorded touch values

            contact_x_values = np.array(contact_x_values) # timestamps

            audio_hold = 30000/ems_constants.bpms[bpm_index] # this math gives the length of time in milliseconds for eighthnote duration
            x_vec = np.arange(0, np.max(contact_x_values), ems_constants.samp_period_ms)
            stim_trace = spike_times_to_traces(stim_onset_times, ems_constants.actual_stim_length,  \
                x_vec, ems_constants.samp_period_ms)
            audio_trace = spike_times_to_traces(audio_onset_times, audio_hold, x_vec, ems_constants.samp_period_ms)

            legend_labels = ["contact trace", "stim trace", "audio trace"]
            input_title = f"{ems_constants.rhythm_strings_names[rhythm_index]}: {ems_constants.rhythm_strings[rhythm_index]}," + \
                f"bpm: {rotated_bpm_list[bpm_index]}"

            plot_contact_trace_and_rhythm(reading_list, contact_x_values, stim_trace, audio_trace,  \
                x_vec, ems_constants.samp_period_ms, legend_labels, input_title)

            # print("done")

            ### SAVE DATA ### ### CONVERT THIS TO JSON FORMAT

            

            data_header = ["time values (ms)", "contact trace", "stim time onsets", "audio time onsets"]
            
            worksheet = workbook.add_worksheet(f"{ems_constants.rhythm_strings_names[rhythm_index]}_bpm{rotated_bpm_list[bpm_index]}") 

            ## write header values ##
            for i in range(len(data_header)):
                worksheet.write(0, i, data_header[i], bold)            

            arrs_to_write = [contact_x_values, reading_list, stim_onset_times, audio_onset_times]

            for i in range(len(arrs_to_write)):
                for row_num, data in enumerate(arrs_to_write[i]):
                    worksheet.write(row_num + ems_constants.worksheet_data_begin_indices[0], ems_constants.worksheet_data_begin_indices[1] + i, data)
            
    toc = time.time()
    diff = toc-tic
    print("Time elapsed: " + str(diff))
    workbook.close()

    contact_serial.close()
    ems_serial.close()







    ### this all is included in the separate EMS TEST ANALYSIS SCRIPT in directory.

#     ## further processing
#     surpressed_contact_onset_times = process_contact_trace_to_hit_times(reading_list, contact_x_values, ems_constants.baseline_subtractor, ems_constants.surpression_window)

#     contact_hold = ems_constants.contact_spike_time_width
#     surpressed_contact_trace = spike_times_to_traces(surpressed_contact_onset_times, contact_hold, x_vec, ems_constants.samp_period_ms)

#     legend_labels = ["surpressed contact trace", "stim trace", "audio trace"]

#     plot_contact_trace_and_rhythm(surpressed_contact_trace, x_vec, stim_trace, audio_trace, x_vec, ems_constants.samp_period_ms, legend_labels)




#     ### Process data ###

#     len_rhythm_ms = len(rhythm_substr) * ems_constants.milliseconds_per_eighthnote
#     len_count_off_ms = len(ems_constants.count_in_substr) * ems_constants.milliseconds_per_eighthnote 
#     len_count_off_and_audio_display_ms = len_count_off_ms + ems_constants.audio_repeats*len_rhythm_ms
#     len_count_off_and_audio_display_and_ems_ms = len_count_off_ms + ems_constants.audio_repeats*len_rhythm_ms + ems_constants.repeats*len_rhythm_ms

#     delays_list = [len_count_off_ms, len_count_off_and_audio_display_ms, len_count_off_and_audio_display_and_ems_ms]
#     audio_repeats_distances = []
#     ems_repeats_distances = []
#     post_ems_repeats_distances = []
#     distances_list = []
#     repeat_list = [ems_constants.audio_repeats, ems_constants.repeats, ems_constants.post_ems_repeats]

#     # for each loop of rhythm, calculate EMD for contact trace vs real audio rhythm

#     for i in range(3): # for each condition (audio only, ems and audio, post_ems audio only test)
#         for j in range(repeat_list[i]): # for each repeat of the rhythm in this condition
#             loop_begin = delays_list[i] + j * len_rhythm_ms - ems_constants.milliseconds_per_eighthnote #include one eighthnote before
#             loop_end = loop_begin + len_rhythm_ms + 2 * ems_constants.milliseconds_per_eighthnote #include one eighthnote after as well
#             contact_bool = np.logical_and((surpressed_contact_onset_times >= loop_begin), (surpressed_contact_onset_times <= loop_end)) # select contact onset times during this loop of rhythm
#             audio_bool = np.logical_and((audio_onset_times >= loop_begin), (audio_onset_times <= loop_end)) # select audio onset times during this loop of rhythm
#             total_spikes_contact = sum(contact_bool) # how many spikes total?
#             total_spikes_audio = sum(audio_bool)
#             trace_selector_bool = np.logical_and((x_vec >= loop_begin), (x_vec <= loop_end)) # which indices in traces are during this loop?
#             contact_trace_selected = surpressed_contact_onset_times[trace_selector_bool] # pick those data points from suprpressed contact trace
#             audio_trace_selected = audio_trace[trace_selector_bool] # pick those data points from audio trace
#             emd = earth_movers_distance(contact_trace_selected, audio_trace_selected, total_spikes_contact, total_spikes_audio) # run emd
#             distances_list.append(emd) # add to appropriate list.
        

#     fig, ax = plt.subplots()
#     ax.plot(np.arange(len(distances_list)), distances_list)
#     ax.set_title("EMD from surpressed contact to audio ground truth for each rhythm repeat")
#     plt.ion()
#     plt.show()
#     plt.draw()
#     plt.pause(0.01)

#     # 



# # subprocess.call(["blueutil", "-p", "0"])
# # subprocess.call(["blueutil", "-p", "1"])

# # #reset ems_serial
# # subprocess.call(["blueutil", "-p", "0"])
# # subprocess.call(["blueutil", "-p", "1"])

# # command_str = "ble-serial -d 2001D755-B5B0-4253-A363-3132B0F93E71 -w 454d532d-5374-6575-6572-756e672d4348 -r 454d532d-5374-6575-6572-756e672d4348"
# # # command_str = "ls -l"
# # # connect using ble-serial script
# # #second number is write characteristic and third is read. Find these
# #    # by calling ble-scan -d DEVICE_ID and look for notify service/characteristic.

# # print(command_str)
# # process = Popen(shlex.split(command_str)) #stdout=PIPE, stderr=None, shell=True
# # text = process.communicate()[0]
# # print(text)

# # time.sleep(3) #wait for connection to work


# for i in range(5):
#     # print("ping")
#     ems_serial.write(b"e")
#     time.sleep(1)
#     ems_serial.write(b"r")
#     time.sleep(1)
    # input_data = ems_serial.read(8)
    # print(input_data.decode())




# address = "00-1E-C0-42-85-FF"
# perif_id = '2001D755-B5B0-4253-A363-3132B0F93E71'
# service = '454D532D-5365-7276-6963-652D424C4531' #read write?

### test play rhythm