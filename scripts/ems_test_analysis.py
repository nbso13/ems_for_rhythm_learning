
import warnings
import copy
import pickle
import ems_constants
import numpy as np
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from ems_test import process_contact_trace_to_hit_times
from ems_test import plot_contact_trace_and_rhythm
import glob
import quantities as pq
from elephant.spike_train_dissimilarity import victor_purpura_distance
from neo import SpikeTrain
import scipy
import numpy as np
import pandas as pd
from scipy import signal
import os
import time
from math import log10, floor
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def dtw_trace_distance(query, template):
    distance, path = fastdtw(query, template, dist=euclidean)
    return distance

def round_sig(x, sig=2):
    if np.isnan(x):
        return(x)
    else:
        return round(x, sig-int(floor(log10(abs(x))))-1)
def sine_generator(fs, sinefreq, duration):
    T = duration
    nsamples = fs * T
    w = 2. * np.pi * sinefreq
    t_sine = np.linspace(0, T, nsamples, endpoint=False)
    y_sine = np.sin(w * t_sine)
    result = pd.DataFrame({ 
        'data' : y_sine} ,index=t_sine)
    return result

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_lowpass(cutOff, fs, order=5):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = signal.butter(order, normalCutoff, btype='low', analog = True)
    return b, a

def butter_lowpass_filter(data, cutOff, fs, order=4):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def butter_band_pass_filter(x_vec, data, cutoff_low, cutoff_high, fs, plot_flag, order=5):
    b, a = butter_highpass(cutoff_low, fs, order=order)
    y = signal.filtfilt(b, a, data)
    # y = butter_lowpass_filter(y, cutoff_high, fs)
    if plot_flag == 1:
        plt.figure(figsize=(20,10))
        plt.subplot(211)
        plt.plot(x_vec, data)
        plt.title('input signal')
        plt.subplot(212)
        plt.plot(x_vec, y)
        plt.title('filtered signal')
        plt.show()
    return y

def spike_times_to_traces(onset_times, hold_length, x_vector, samp_period):
    # take a series of onset time points and craft plottable traces.
    ### XVECTOR SHOULD HAVE PERIOD OF SAMP PERIOD
    array_value_stim_time = int(np.floor(hold_length/samp_period))
    trace = np.zeros_like(x_vector)
    for time_val in onset_times: # for each onset time
        array_ind_begin = int(np.floor((time_val-x_vector[0])/samp_period)) 
        array_ind_end = array_ind_begin + array_value_stim_time
        trace[array_ind_begin:array_ind_end] = 1
    return trace

def plot_w_k(axes_array, means, sds, title_list):
    axes = sum([list(axes_array[0]), list(axes_array[1])], [])
    slopes = []
    intercepts = []
    r_vals = []
    p_vals = []
    for i in range(len(means)):
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(means[i], sds[i])
        slopes.append(slope)
        intercepts.append(intercept)
        r_vals.append(r_value)
        p_vals.append(p_value)
        axes[i].scatter(means[i], sds[i])
        axes[i].plot(means[i], slope * np.array(means[i]) + intercept)
        axes[i].set_title(title_list[i])
        axes[i].text(min(means[i]), max(sds[i]), "R^2: " + str(r_value**2))
        axes[i].set_ylabel("Standard deviation (ms)")
        axes[i].set_xlabel("Mean Produced Interval (ms)")
    plt.ion()
    plt.tight_layout()
    plt.show()
    plt.draw()
    plt.pause(0.01)
    return slopes, intercepts, r_vals, p_vals

def interpolate(reading_list, contact_x, samp_period):
    contact_x = contact_x - contact_x[0]
    contact_x_interped = np.arange(0, max(contact_x), samp_period)
    f = scipy.interpolate.interp1d(contact_x, reading_list)
    reading_interped = f(contact_x_interped)
    return contact_x_interped, reading_interped



def plot_traces(x_array, trace_list, samp_period, legend_labels, title):
    fig, ax = plt.subplots()
    ax.set_yticks(np.arange(0, 500, 100))
    ax.set_xticks(np.arange(np.min(x_array), np.max(x_array), 1000))
    ax.plot(x_array, trace_list[0])
    for i in range(len(trace_list) - 1):
        ax.plot(x_array, trace_list[i+1]*np.max(trace_list[0])/2)
    ax.legend(legend_labels)
    ax.set_title(title)
    ax.set_xlabel("time, ms")
    ax.set_ylabel("onset")
    plt.ion()
    plt.show()
    plt.draw()
    plt.pause(0.01)

def earth_movers_distance(spike_times_a, spike_times_b):
    
    # rhythm_total_spike_times_a = len(spike_times_a)
    # rhythm_total_spike_times_b = len(spike_times_b)
    # if rhythm_total_spike_times_a == 0:
    #     return -1
    # cumulative_a = np.cumsum(np.divide(rhythm_trace_a, rhythm_total_spike_times_a))
    # cumulative_b = np.cumsum(np.divide(rhythm_trace_b, rhythm_total_spike_times_b))
    # same thing as np.sum(np.abs(np.subtract(cumulative_a, cumulative_b))),
    if len(spike_times_a) == 0 or len(spike_times_b) == 0:
        return np.nan
    return  scipy.stats.wasserstein_distance(spike_times_a, spike_times_b)

def victor_purp(onset_times_1, onset_times_2, loop_begin, loop_end):
    q = 1.0 / (10.0 * pq.ms)
    st_a = SpikeTrain(onset_times_1, units='ms', t_start = 0, t_stop= loop_end-loop_begin)
    st_b = SpikeTrain(onset_times_2, units='ms', t_start = 0, t_stop= loop_end-loop_begin)
    vp_f = victor_purpura_distance([st_a, st_b], q)[1][0] # give only the distance between the two different trains (not diagonal)
    return vp_f


# TEST THIS FUNCTION
def compile_unique_interval_list(intervals, tolerance):
    interval_index_lib = np.arange(0, len(intervals))
    unique_interval_list = []
    index = 0
    indices = []
    for interval in intervals: # for every interval
        # if this interval is within tolerance of any item on the list,
        bool_list = [(interval+tolerance > item and interval-tolerance < item) for item in unique_interval_list]
        if any(bool_list):
            indices.append(bool_list.index(True))
            continue # we consider it on the list.
        else: #otherwise
            unique_interval_list.append(interval) # add it to list
            indices.append(len(unique_interval_list) - 1)
    return unique_interval_list, indices

def determine_delays_list(rhythm_substr, bpm, header_dict, first_audio):
    # determine the exact beginning timestamp (delays_list) for each phase (pre-ems audio, ems, post-ems audio, no audio)
    len_rhythm_ms = len(rhythm_substr) * 30000/bpm #+ 75 ### tHIS IS SOMEHOW OFF BU ABOUT 75 MS?????
    # warnings.warn("Adding 75 ms to rhythm length for unknown reason?")
    list_of_delays = [first_audio] # first phase begins at first_audio
    for i in range(header_dict['num_phases']): # for every phase, find the end of the phase and tack it on.
        last_time = list_of_delays[-1] # get the last time (end of last phase, beginning of this one)
        list_of_delays.append(last_time + header_dict['phase_repeats_list'][i] * len_rhythm_ms) #get the number of repeats and find the length in time in ms and add it on
    return list_of_delays, len_rhythm_ms
        
def chop_traces(k, contact_trace,  audio_trace, x_vec, times_list, bpm):
    loop_begin = times_list[k] - 0.5*30000/bpm #include half an eighthnote before
    loop_end = times_list[k+1] + 0.5 * 30000/bpm #include half eigthnote after as well
    select_bool = np.logical_and((x_vec >= loop_begin), (x_vec <= loop_end)) # select contact onset times during this loop of rhythm
    audio_selected_trace = audio_trace[select_bool]
    contact_selected_trace = contact_trace[select_bool]
    x_selected_trace = x_vec[select_bool]
    return contact_selected_trace, audio_selected_trace, x_selected_trace

def chop_onsets(k, contact_times, audio_times, delays_list, bpm):
    loop_begin = delays_list[k] - 0.5*30000/bpm #include half an eighthnote before
    loop_end = delays_list[k+1] - 0.5*30000/bpm #include half an eighthnote after as well
    contact_bool = np.logical_and((contact_times >= loop_begin), (contact_times <= loop_end)) # select contact onset times during this loop of rhythm
    audio_bool = np.logical_and((audio_times >= loop_begin), (audio_times <= loop_end)) # select audio onset times during this loop of rhythm
    spike_times_contact = contact_times[contact_bool] # how many spikes total?
    spike_times_audio = audio_times[audio_bool]
    return spike_times_contact, spike_times_audio 

def load_headers(file_stems):
    header_dicts = []
    for file_stem in file_stems:
        pkl_file = f"data/{file_stem}_header_info.pkl"
        with open(pkl_file, "rb") as pkl_handle:
            header_dict = pickle.load(pkl_handle)
        header_dicts.append(header_dict)
    return header_dicts

def histo_intervals(gt_intervals):
    # q25, q75 = np.percentile(gt_intervals, [25, 75])
    # bin_width = 2 * (q75 - q25) * len(gt_intervals) ** (-1/3)
    # bins = round((np.max(gt_intervals) - np.min(gt_intervals)) / bin_width)
    fig, ax = plt.subplots()
    ax.hist(gt_intervals, bins=40)  # density=False would make counts
    ax.set_ylabel("count")
    ax.set_xlabel('Interval length, ms')
    ax.set_title("interval frequency")
    return

def pull_repeat_times(first_audio, rhythm, bpm, block_repeats_list, block_flags):
    # given the first audio onset time, the rhythm substr, the bpm, the blocks repeat list
    # from the header dictionary, and the block_flags (usually all 1), returns 
    # the repeat times where the rhythm is repeated.
    summer = 0 # total number of repeats
    for i in range(len(block_repeats_list)):
        if block_flags[i]:
            summer = summer + block_repeats_list[i]
    eighthnote_length = 30000/bpm
    rhythm_time = len(rhythm)*eighthnote_length
    repeat_times = []
    time_var = first_audio
    for i in range(summer):
        repeat_times.append(time_var)
        time_var = time_var + rhythm_time
    repeat_times.append(time_var) # book end
    return repeat_times

def pull_repeat_times_by_block(block_times_list, rhythm, bpm, block_repeats_list, block_flags):
    repeat_times_by_block = []
    for i in range(len(block_times_list) - 1): # bookended times
        if block_flags[i]:
            repeats = block_repeats_list[i]
            this_block_repeat_times = pull_repeat_times(block_times_list[i], rhythm, bpm, [repeats], [1]) # pull times within this block
            repeat_times_by_block.append(this_block_repeat_times)
    return repeat_times_by_block

def plot_each_block(rhythm, rhythm_name, bpm, block_repeats_list, block_flags, block_times_list, \
    surpressed_contact_onset_times, audio_onset_times, surpressed_contact_trace, \
        audio_trace, x_vec, raw_contact_trace, raw_x_vec, processed_contact_trace, processed_x_vec, plot_flag, y_axis_mins=0, y_axis_maxes=0):

    mad_per_block, vad_per_block, = MAD_VAD_per_time_calc(rhythm, bpm, block_times_list, audio_onset_times, surpressed_contact_onset_times)
    emd_per_block = emd_per_time_calc(surpressed_contact_onset_times, audio_onset_times, block_times_list, bpm)
    twd_per_block = twd_per_time_calc(surpressed_contact_trace, audio_trace, x_vec, block_times_list, bpm)
    block_names = ems_constants.phase_warning_strs
    repeat_times_by_block = pull_repeat_times_by_block(block_times_list, rhythm, bpm, block_repeats_list, block_flags)


    plot_each_block_dict = {
        'emds' : [],
        'twds' : [],
        'mads' : [],
        'vads' : []
    } # four rhythms, three days, 8 presentations of each rhythm in that day, mean then sd



    for i in range(len(block_repeats_list)):
        if block_flags[i]:
            if plot_flag: 
                fig, axes = plt.subplots(7,1)
                fig.set_size_inches(13, 8)
                title = f"{rhythm_name}: {rhythm}," + f"bpm: {bpm}, block: {block_names[i]}"
                fig.suptitle(title)
            
            raw_contact_trace_selected, _, raw_x_trace_selected = chop_traces(i, raw_contact_trace,  \
                raw_contact_trace, raw_x_vec, block_times_list, bpm) # raw contact is double input because not used and need to fit the dimensions.
            processed_contact_trace_selected, _, processed_x_vec_selected = chop_traces(i, processed_contact_trace,  \
                processed_contact_trace, processed_x_vec, block_times_list, bpm) # raw contact is double input because not used and need to fit the dimensions.
            
            ind = 0
            if plot_flag:
                axes[ind].plot(raw_x_trace_selected, raw_contact_trace_selected)
                axes[ind].set_ylabel("contact trace value")
                axes[ind].set_title("Raw recorded contact trace")

                ind+=1
                axes[ind].plot(processed_x_vec_selected, processed_contact_trace_selected)
                # axes[ind].set_ylabel("contact trace value")
                axes[ind].set_title("Processed contact trace")
                axes[ind].sharex(axes[ind-1])
                ind+=1

            spikes_contact_trace_selected, spikes_audio_trace_selected, x_vec_selected = chop_traces(i, surpressed_contact_trace,  \
                audio_trace, x_vec, block_times_list, bpm)
            repeat_times = repeat_times_by_block[i]
            if plot_flag:
                axes[ind].set_title("Processed spikes versus audio onsets")
                axes[ind].set_yticks([])
                axes[ind].plot(x_vec_selected, (spikes_contact_trace_selected + 1))
                axes[ind].plot(x_vec_selected, spikes_audio_trace_selected)
                axes[ind].legend("contact spikes", "audio spikes")
                for time in repeat_times:
                    axes[ind].axvline(time, linewidth=2, color='g')

            contact_trace_selected, audio_trace_selected, x_trace_selected = chop_traces(i,surpressed_contact_trace, audio_trace, x_vec, block_times_list, bpm)
            contact_onsets_selected, audio_onsets_selected = chop_onsets(i,surpressed_contact_onset_times, audio_onset_times, block_times_list, bpm)
            mad_per_repeat, vad_per_repeat = MAD_VAD_per_time_calc(rhythm, bpm, repeat_times, audio_onsets_selected, contact_onsets_selected)
            emd_per_repeat = emd_per_time_calc(contact_onsets_selected, audio_onsets_selected, repeat_times, bpm)
            twd_per_repeat = twd_per_time_calc(contact_trace_selected, audio_trace_selected, x_trace_selected, repeat_times, bpm)
            
            plot_each_block_dict['emds'].append(emd_per_repeat)
            plot_each_block_dict['twds'].append(twd_per_repeat)
            plot_each_block_dict['mads'].append(mad_per_repeat)
            plot_each_block_dict['vads'].append(vad_per_repeat)

            mid_repeat_x_val = []
            for j in range(len(repeat_times)-1):
                x_val = (repeat_times[j] + repeat_times[j+1])/2
                mid_repeat_x_val.append(x_val)
            
            # normed_emd = emd_per_block[i]/block_repeats_list[i] 
            ind+=1
            if plot_flag:
                dict_keys = plot_each_block_dict.keys()
                titles = ["Earth Mover's Distance per repeat", "Time Warp Distance per repeat", "Mean of Asynch Distribution per repeat", "Variance of Asynch Distribution per repeat"]
                colors = ["r", "orange", "blue", "green"]
                y_labels = ["", "", "Fraction of interval", ""]
                for key_ind in range(len(dict_keys)):
                    key = dict_keys[key_ind]
                    axes[ind + key_ind].set_title(titles[key_ind])
                    axes[ind + key_ind].plot(mid_repeat_x_val, plot_each_block_dict[key][-1], color=colors[key_ind])
                    axes[ind+ key_ind].set_ylabel(y_labels[key_ind])
                    # axes[ind + key_ind].axhline(normed_emd, color='r')
                    axes[ind + key_ind].sharex(axes[ind-1])
                    if key_ind == len(dict_keys) - 1:
                        axes[ind + key_ind].set_xlabel("time (ms)")
                plt.tight_layout()
                plt.show()
                input("continue?")
    return plot_each_block_dict 
    

def load_data(file_stems):
    vars_and_headers = []
    for file_stem in file_stems:
        print(f"loading {file_stem}...")
        pkl_file = f"data/processed_{file_stem}.pkl"
        with open(pkl_file, "rb") as pkl_handle:
            saved_data = pickle.load(pkl_handle)
        vars_and_headers.append(saved_data)
        print("done.")
    return vars_and_headers

def plot_emds(emds, header_dict, rhyth_index):
    mean_distances = np.mean(np.vstack(emds), 0)
    sd_distances = np.std(np.vstack(emds), 0)
    fig, ax = plt.subplots() # change this according to num phase?
    legend_labels = ["earth mover's distances mean", "std+", "std-"]
    ax.plot(np.arange(len(mean_distances)), mean_distances,'b')
    ax.plot(np.arange(len(mean_distances)), mean_distances+sd_distances, 'r')
    ax.plot(np.arange(len(mean_distances)), mean_distances-sd_distances, 'r')
    ax.set_title(f"mean earth mover's distance across bpms for each phase, {header_dict['rhythm_strings_names'][rhyth_index]}")
    ax.legend(legend_labels)
    plt.ion()
    plt.show()
    plt.draw()
    plt.pause(0.01)
    return

def craft_x_vec(list_of_onsets):
    mini = np.min(list_of_onsets)
    maxi = np.max(list_of_onsets)
    stepsize = (maxi-mini)/(len(list_of_onsets))
    return np.arange(mini, maxi, stepsize)

def count_intervals(rhyth_string):
    zero_counter = 0
    intervs = []
    for i in range(len(rhyth_string)):
        if rhyth_string[i] == '0':
            zero_counter += 1 # add to interval
        if rhyth_string[i] == '1':
            zero_counter += 1 # add to interval
            intervs.append(zero_counter) # save interval
            zero_counter = 0 # reset counter
    if zero_counter == len(rhyth_string): # no intervals, empty string
        unique_intervals = []
        num_unique = 0
        return intervs, unique_intervals, num_unique
    intervs[0] += zero_counter # add trailing zeros to first measured interval
    unique_intervals = list(set(intervs))
    num_unique = len(unique_intervals)
    return intervs, unique_intervals, num_unique
    

def get_scaled_asynchronies_by_interval_by_time(rhythm_string, bpm, cut_times, audio_onset_times, contact_onset_times):
    # get unique intervals in this rhythm
    _, unique_intervals, num_unique  = count_intervals(rhythm_string)
    ms_per_eighthnote = 30000/bpm
    unique_intervals_ms = [i*ms_per_eighthnote for i in unique_intervals]
    # make a 3d matrix: first dim, block, second, interval.  third, user performance instance.
    user_performance_matrix = [[[] for i in range(num_unique)] for j in range(len(cut_times)-1)]

    for i in range(len(cut_times)-1): # for every phase -1 for bookend times
        # get the audio and the contact times in that phase
        contact_times_chopped, audio_times_chopped = chop_onsets(i, contact_onset_times, audio_onset_times, cut_times, bpm)
        # get the ground truth intervals and the user produced intervals in that phase
        ground_truth_intervals, user_intervals = accumulate_intervals(audio_times_chopped, contact_times_chopped)
        for j in range(len(ground_truth_intervals)): # for every ground truth interval
            # get the closest unique interval
            index = np.argmin(np.abs(unique_intervals_ms - ground_truth_intervals[j]))
            # if difference between unique interval in the rhythm and the ground truth interval is more than 10% of ground truth interval that's weird. throw an error.
            if ((unique_intervals_ms[index] - ground_truth_intervals[j])/unique_intervals_ms[index]) > 0.1:
                warnings.warn("difference between measured ground truth interval and known unique interval is more than 10%??")

            # calculate scaled (normalized) asnychrony as a signed fraction of the ground truth interval. If -0.1, user was 10% too soon.
            else:
                scaled_asynchrony = (user_intervals[j] - ground_truth_intervals[j])/ground_truth_intervals[j]
                user_performance_matrix[i][index].append(scaled_asynchrony) # append to correct list (first index, which phase, second index, which interval.)

    return user_performance_matrix, unique_intervals_ms


def mad_vad(times_a, times_b, unique_intervals):
    # make a 2d matrix: first dim,  interval. second, user performance instance.
    user_performance_matrix = [[] for i in range(len(unique_intervals))]

    # get the ground truth intervals and the user produced intervals in that phase
    ground_truth_intervals, user_intervals = accumulate_intervals(times_a, times_b)
    for j in range(len(ground_truth_intervals)): # for every ground truth interval
        # get the closest unique interval
        index = np.argmin(np.abs(np.array(unique_intervals)- ground_truth_intervals[j]))
        # if difference between unique interval in the rhythm and the ground truth interval is more than 10% of ground truth interval that's weird. throw an error.
        if ((unique_intervals[index] - ground_truth_intervals[j])/unique_intervals[index]) > 0.1:
            warnings.warn("difference between measured ground truth interval and known unique interval is more than 10%??")
        # calculate scaled (normalized) asnychrony as a signed fraction of the ground truth interval. If -0.1, user was 10% too soon.
        else:
            scaled_asynchrony = (user_intervals[j] - ground_truth_intervals[j])/ground_truth_intervals[j]
            user_performance_matrix[index].append(scaled_asynchrony) # append to correct list (first index, which phase, second index, which interval.)

    MAD_across_intervals = []
    VAD_across_intervals = []
    for i in range(len(user_performance_matrix)):
        asynchs_by_interval = user_performance_matrix[i]
        MAD_across_intervals.append(np.mean(np.abs(asynchs_by_interval)))
        VAD_across_intervals.append(np.std(asynchs_by_interval))
    
    return MAD_across_intervals, VAD_across_intervals


def emd_mad_vad_test(title, times_a, times_b, mini, maxi, samp_period, unique_intervals):
    xvec = np.arange(mini, maxi, samp_period)
    query = spike_times_to_traces(times_a, samp_period, xvec, samp_period)
    template = spike_times_to_traces(times_b, samp_period, xvec, samp_period)
    twd = dtw_trace_distance(query, template)
    emd = earth_movers_distance(times_a, times_b)
    mad, vad = mad_vad(times_a, times_b, unique_intervals)
    mad_mean = np.mean(mad)
    vad_mean = np.mean(vad)
    title = title + f", emd = {round_sig(emd, 3)}, \n twd = {round_sig(twd, 3)}, \n mad = {round_sig(mad_mean, 3)}, \n vad = {round_sig(vad_mean, 3)}"
    plot_traces(xvec, [query, template], samp_period, ["a", "b"], title)
    plt.tight_layout()
    return emd, twd, mad, vad

def filter_test():
    fps = 30
    sine_fq = 10 #Hz
    duration = 10 #seconds
    sine_5Hz = sine_generator(fps,sine_fq,duration)
    sine_fq = 1 #Hz
    duration = 10 #seconds
    sine_1Hz = sine_generator(fps,sine_fq,duration)
    sine_fq = 50 #Hz
    duration = 10 #seconds
    sine_50Hz = sine_generator(fps,sine_fq,duration)

    sine = sine_5Hz + sine_1Hz + sine_50Hz
    x = np.arange(0, duration, 1/fps)
    filtered_sine = butter_band_pass_filter(x, sine.data, 9, 20, fps, plot_flag=1)



def teaserfigure():
    mini = 0
    maxi = 6
    samp_period = 0.01
    # titles = ["times b just after", "times b just before", "times b mixed", \
    #     "times b missing beat", "times b missing two beats", "times b extra beat", \
    #         "times b extra two beats", "times b extra beat end range", "times b extra beat mid range"]

    # times_a = [1, 2, 3, 4, 5]
    # times_bs = [ [1.1, 2.1, 3.1, 4.1, 5.1], [0.9, 1.9, 2.9, 3.9, 4.9], [1.1, 1.9, 3.1, 3.9, 5.1], \
    #     [1.1, 2.1, 3.1, 5.1], [1.1, 3.1, 5.1], [1.1, 2.1, 3.1, 4.1, 4.5, 5.1],  \
    #         [1.1, 2.1, 3.1, 3.5, 4.1, 4.5, 5.1 ], [1.1, 2.1, 3.1, 4.1, 4.9, 5.1], [1.1, 2.1, 3.1, 3.2, 4.1, 5.1]]

    titles = ["Before EMS", "During EMS", "After EMS"]

    times_a = [1, 2, 3, 4, 5]
    times_bs = [ [0.5, 0.8, 2.2, 2.9, 3.7, 3.8, 5.1], [1, 2, 3.1, 4, 4.9], [0.8, 2.3, 3.1, 3.8, 5.1]]

    # legend_labels = ["Subject performance", "Target pattern"]
    fig, axes = plt.subplots(1, 4)
    emds = []

    hold_length = 0.05 # ms
    for i in range(len(times_bs)):
        times_b = times_bs[i]
        # title = titles[i]
        
        xvec = np.arange(mini, maxi, samp_period)
        trace_a = spike_times_to_traces(times_a, hold_length, xvec, samp_period)
        trace_b = spike_times_to_traces(times_b, hold_length, xvec, samp_period)
        emds.append(earth_movers_distance(times_a, times_b))
        axes[i].set_yticks([])
        axes[i].set_xticks([])
        axes[i].plot(xvec, trace_b+1)
        axes[i].plot(xvec, trace_a)
        if i == 0:
            # axes[i].legend(legend_labels, loc='best')
            axes[i].set_yticks([0.5, 1.5])
            axes[i].set_yticklabels(['ground truth \n pattern', 'user produced \n pattern'], fontsize=10)
        axes[i].set_title(titles[i])
        axes[i].set_xlabel("time")


    
    axes[i+1].plot([0,3,6], emds, '-o', color='green')
    axes[i+1].set_yticks([0.05, 0.4])
    axes[i+1].set_yticklabels(['high \nprecision', 'low \nprecision'], fontsize=10)
    axes[i+1].set_xticks([0,3,6])
    axes[i+1].set_xticklabels(titles)
    # plt.tight_layout()
    plt.xticks(rotation=10, ha="right")
    axes[i+1].set_title("Temporal precision by phase")
    l, b, w, h = axes[i+1].get_position().bounds
    axes[i+1].set_position([l*1.05, b, w, h])



def emd_tests():
    mini = 0
    maxi = 6
    samp_period = 0.01
    titles = ["times b just after", "times b just before", "times b mixed", \
        "times b missing beat", "times b missing two beats", "times b extra beat", \
            "times b extra two beats", "times b extra beat end range", "times b extra beat mid range"]

    times_a = [1, 2, 3, 4, 5]
    times_bs = [ [1.1, 2.1, 3.1, 4.1, 5.1], [0.9, 1.9, 2.9, 3.9, 4.9], [1.1, 1.9, 3.1, 3.9, 5.1], \
        [1.1, 2.1, 3.1, 5.1], [1.1, 3.1, 5.1], [1.1, 2.1, 3.1, 4.1, 4.5, 5.1],  \
            [1.1, 2.1, 3.1, 3.5, 4.1, 4.5, 5.1 ], [1.1, 2.1, 3.1, 4.1, 4.9, 5.1], [1.1, 2.1, 3.1, 3.2, 4.1, 5.1]]

    emds = []
    twds = []
    MADs = []
    VADs = []
    unique_intervals = [1]
    for i in range(len(times_bs)):
        times_b = times_bs[i]
        title = titles[i]
        emd, twd, MAD, VAD = emd_mad_vad_test(title, times_a, times_b, mini, maxi, samp_period, unique_intervals)
        emds.append(emd)
        twds.append(twd)
        
        MADs.append(MAD[0])
        VADs.append(VAD[0])

    fig, axes = plt.subplots(3,1)
    axes[0].set_title("Bar plot for EMDS across tests")
    axes[0].set_ylabel("EMD")
    axes[0].bar(np.arange(len(titles)), emds, align='center')

    axes[1].set_title("Bar plot for MADs across tests")
    axes[1].set_ylabel("MAD")
    axes[1].bar(np.arange(len(titles)), MADs, align='center')

    axes[2].set_xticks(np.arange(len(titles)))
    axes[2].set_xticklabels(titles)
    axes[2].set_title("Bar plot for VADs across tests")
    axes[2].set_ylabel("VAD")

    axes[3].set_xticks(np.arange(len(titles)))
    axes[3].set_xticklabels(titles)
    axes[3].set_title("Bar plot for TWDs across tests")
    axes[3].set_ylabel("TWD")
    plt.xticks(rotation=30, ha="right")
    axes[2].bar(np.arange(len(titles)), twds, align='center')
    plt.tight_layout()
    return 


def plot_test_blocks(emds, mads, vads, twds, name, rhythm, bpm, block_labels, y_axis_maxes=0, y_axis_mins=0):
    # mean_emds = [np.nanmean(repeat_scores) for repeat_scores in emds]
    # std_emd = [np.nanstd(repeat_scores) for repeat_scores in emds]
    # mean_twds = [np.nanmean(repeat_scores) for repeat_scores in twds]
    # std_twd = [np.nanstd(repeat_scores) for repeat_scores in twds]
    # mean_mad = [np.nanmean(repeat_scores) for repeat_scores in mads]
    # std_mad = [np.nanstd(repeat_scores) for repeat_scores in mads]
    # mean_vad =  [np.nanmean(repeat_scores) for repeat_scores in vads]
    # std_vad = [np.nanstd(repeat_scores) for repeat_scores in vads]
    mean_emds = [np.nanmean(emds[1]), np.nanmean(emds[4])]
    std_emd = [np.nanstd(emds[1]), np.nanstd(emds[4])]
    mean_twds = [np.nanmean(twds[1]), np.nanmean(twds[4])]
    std_twd = [np.nanstd(twds[1]), np.nanstd(twds[4])]
    mean_mad = [np.nanmean(mads[1]), np.nanmean(mads[4])]
    std_mad = [np.nanstd(mads[1]), np.nanstd(mads[4])]
    mean_vad =  [np.nanmean(vads[1]), np.nanmean(vads[4])]
    std_vad = [np.nanstd(vads[1]), np.nanstd(vads[4])]
    block_labels = (block_labels[1], block_labels[4])
    fig, axes = plt.subplots(4,1)
    title = f"{name}: {rhythm}," + f" bpm: {bpm}"
    fig.suptitle(title)
    axes[0].scatter(np.arange(len(mean_emds)), mean_emds, color='r')
    axes[0].set_ylabel("Mean EMD")
    axes[0].set_xticks(np.arange(len(mean_emds)))
    axes[0].set_xticklabels([])
    axes[0].errorbar(np.arange(len(mean_emds)), mean_emds, yerr=std_emd, fmt="o", color='r')
    if not type(y_axis_maxes) == int and not type(y_axis_mins) == int:
        axes[0].set_ylim([y_axis_mins['emds'],y_axis_maxes['emds']])
    axes[1].scatter(np.arange(len(mean_twds)), mean_twds, color='orange')
    axes[1].set_ylabel("Mean TWD")
    axes[1].set_xticks(np.arange(len(mean_twds)))
    axes[1].set_xticklabels([])
    axes[1].errorbar(np.arange(len(mean_twds)), mean_twds, yerr=std_twd, fmt="o", color='orange')
    if not type(y_axis_maxes) == int and not type(y_axis_mins) == int:
        axes[1].set_ylim([y_axis_mins['twds'],y_axis_maxes['twds']])
    axes[2].scatter(np.arange(len(mean_mad)), mean_mad, color='b')
    axes[2].set_ylabel("Mean MAD")
    axes[2].set_xticks(np.arange(len(mean_mad)))
    axes[2].errorbar(np.arange(len(mean_mad)), mean_mad, yerr=std_mad, fmt="o", color='b')
    axes[2].set_xticklabels([])
    if not type(y_axis_maxes) == int and not type(y_axis_mins) == int:
        axes[2].set_ylim([y_axis_mins['mads'],y_axis_maxes['mads']])
    axes[3].scatter(np.arange(len(mean_vad)), mean_vad, color='g')
    axes[3].errorbar(np.arange(len(mean_vad)), mean_vad, yerr=std_vad, fmt="o", color='g')
    axes[3].set_ylabel("Mean VAD")
    axes[3].set_xticks(np.arange(len(mean_vad)))
    axes[3].set_xticklabels(block_labels, rotation=45, ha='right')
    axes[3].set_xlabel("Experimental block")
    if not type(y_axis_maxes) == int and not type(y_axis_mins) == int:
        axes[3].set_ylim([y_axis_mins['vads'],y_axis_maxes['vads']])

    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)
    return [axes[0].get_ylim(), axes[1].get_ylim(), axes[2].get_ylim(), axes[3].get_ylim()]



def plot_all_blocks(performance_dictionary, rhythm_name_list, pt_name, condition):

    keys = ["emds","mads", "vads"]
    ylims = [[0, 600], [0, 0.6], [0, 0.3]]
    day_dividers = [7.5, 15.5]
    repeat_dividers = [3.5, 11.5, 19.5]
    colors = ['r',  'b', 'g']
    labels = ["Mean EMD",  "Mean MAD", "Mean VAD"]
    test_performance_dictionary = performance_dictionary['test_performance']
    initial_array = test_performance_dictionary[keys[0]] # num rhythms by num sessions by num presentations per session, mean then sd
    change_in_performance_dictionary = performance_dictionary['change_in_performance']
    
    for rhythm in range(np.size(initial_array, axis=0)):
        fig, axes = plt.subplots(3,1)
        title = f"ppt_num: {pt_name}, condition: {condition}, rhythm: {rhythm_name_list[rhythm]},"
        fig.suptitle(title)
        key_count = -1
        for key in keys:
            key_count += 1
            m = change_in_performance_dictionary[key][rhythm][3]
            b = change_in_performance_dictionary[key][rhythm][4]
            r = change_in_performance_dictionary[key][rhythm][5]
            p = change_in_performance_dictionary[key][rhythm][6]
            x = np.arange(24)
            axes[key_count].plot(x, m*x + b , color=colors[key_count])
            axes[key_count].set_title(f"{key}: y = {round_sig(m, 3)}x + {round_sig(b,3)}, r = {round_sig(r, 3)}, p = {round_sig(p, 3)}")
            test_array = test_performance_dictionary[key][rhythm]# num sessions by num presentations per session
            for session in range(np.size(test_array, axis = 0)):
                session_means = test_array[session, :,0]
                session_sds = test_array[session, :, 1]
                x_ax = np.arange(len(session_means)) + 8 * session
                axes[key_count].scatter(x_ax, session_means, color=colors[key_count])
                axes[key_count].set_ylabel(labels[key_count])
                # axes[key_count].set_xticks())
                # axes[key_count].set_xticklabels([])    
                # axes[key_count].errorbar(x_ax, session_means, yerr=session_sds, fmt="o", color=colors[key_count])
                axes[key_count].vlines(day_dividers, ylims[key_count][0], ylims[key_count][1]/5, linewidth=3, color='k')
                axes[key_count].vlines(repeat_dividers, ylims[key_count][0], ylims[key_count][1]/5, linewidth=1, color='k')
            plt.ylim(ylims[key_count][0],ylims[key_count][1])
            plt.tight_layout()
            plt.draw()
            plt.pause(0.001)


def accumulate_intervals(phase_audio_onsets, surpressed_contact_onset_times):
    ground_truth_intervals = []
    user_intervals = []
    user_error = []
    for j in range(len(phase_audio_onsets)-1):
        # get the interval between them
        gt_interval = phase_audio_onsets[j+1] - phase_audio_onsets[j]
        # get the index of the nearest response pulse to the j+1 audio that is more than buffer ms after the earlier audio.
        if type(surpressed_contact_onset_times) == list:
            surpressed_contact_onset_times = np.array(surpressed_contact_onset_times)
        contact_onset_times_after = surpressed_contact_onset_times[surpressed_contact_onset_times >  phase_audio_onsets[j]]
        if len(contact_onset_times_after) == 0: # If no contacts after audio, skip this audio.
            # raise ValueError("no contacts after this audio!")
            warnings.warn(f'no contact at all after audio !!')
        else:
            arg_min = np.argmin(np.abs(np.subtract(contact_onset_times_after, phase_audio_onsets[j+1]))) # throws out the first one...
            # get the user interval
            user_interval = contact_onset_times_after[arg_min] - phase_audio_onsets[j]#response time user - previous audio pulse
            ground_truth_intervals.append(gt_interval)
            user_intervals.append(user_interval)
    return ground_truth_intervals, user_intervals

def get_all_intervals(header_dict, audio_onset_times, delays_list, surpressed_contact_onset_times):
    var_lists = []
    for k in range(header_dict['num_phases']): # for every phase
        # array of 1s for audio onset times in this phase
        this_phase_bools = np.logical_and((audio_onset_times > delays_list[k]), (audio_onset_times < delays_list[k+1])) 
        #contact trace selection
        this_phase_cont_bools = np.logical_and((surpressed_contact_onset_times > delays_list[k]), (surpressed_contact_onset_times < delays_list[k+1]))
        # get those onset times
        phase_audio_onsets = audio_onset_times[this_phase_bools]
        phase_contact_onsets = surpressed_contact_onset_times[this_phase_cont_bools]
        total_onsets = np.hstack([phase_audio_onsets, phase_contact_onsets])
        # xvec = np.arange(np.min(total_onsets), np.max(total_onsets), header_dict['samp_period_ms'])
        #plot the phase
        # contact_trace = spike_times_to_traces(phase_contact_onsets, 10, xvec, header_dict['samp_period_ms'])
        # audio_trace = spike_times_to_traces(phase_audio_onsets, audio_hold, xvec, header_dict['samp_period_ms'])
        # labels = ["audio", "contact"]
        # title = f"{header_dict['rhythm_strings_names'][rhythm_index]}, {header_dict['bpms'][bpm_index]} bpm, {header_dict['phase_name_strs'][k]}"
        # plot_traces(xvec, [audio_trace, contact_trace], header_dict['samp_period_ms'], labels, title)

        # for each onset time
        ground_truth_intervals, user_intervals = accumulate_intervals(phase_audio_onsets, surpressed_contact_onset_times)
        var_list = [ground_truth_intervals, user_intervals]
        var_lists.append(var_list) # now has each WK relevant variable for each phase
    return var_lists

def MAD_VAD_per_time_calc(rhythm, bpm, cut_times, audio_onset_times, contact_onset_times):
    user_performance_matrix, unique_intervals_ms = get_scaled_asynchronies_by_interval_by_time(rhythm, bpm, cut_times, audio_onset_times, contact_onset_times)
    MAD_by_time = []
    VAD_by_time = []
    for i in range(len(user_performance_matrix)):
        asynchs_by_interval = user_performance_matrix[i]
        #flatten
        flat_asynchs_by_interval = [item for sublist in asynchs_by_interval for item in sublist]
        MAD_by_time.append(np.mean(np.abs(flat_asynchs_by_interval)))
        VAD_by_time.append(np.std(flat_asynchs_by_interval))
    return MAD_by_time, VAD_by_time    

# def emd_per_phase_calc(surpressed_contact_onset_times, surpressed_contact_trace, audio_onset_times, audio_trace, delays_list, x_values, bpm):
#     distances_list = []
#     for k in range(len(delays_list)-1): # for each phase
#         spike_times_contact, spike_times_audio = chop_traces(k, \
#             surpressed_contact_onset_times, surpressed_contact_trace, audio_onset_times, audio_trace, delays_list, x_values, bpm)
#         emd = earth_movers_distance(spike_times_contact, spike_times_audio) # run emd
#         distances_list.append(emd) # add to appropriate list.
#         title = f"total spikes contact: {len(spike_times_contact)}, total_spikes audio: {len(spike_times_audio)}, emd = {str(emd)}" # vic purp: {str(vp_dist)}"
#         # if count == 2:
#             # plot_traces(x_vec[trace_selector_bool], [contact_trace_selected, audio_trace_selected], header_dict["samp_period_ms"], ["contact", "audio"], title)
#     return np.array(distances_list)

def emd_per_time_calc(contact_times, audio_times, cut_times, bpm):
    distances_list = []
    for k in range(len(cut_times)-1): # for each rime (-1 because two bookend times)
        spike_times_contact, spike_times_audio = chop_onsets(k, \
            contact_times, audio_times, cut_times, bpm)
        if len(spike_times_audio) == 0 or len(spike_times_contact) == 0:
            pause_here = True
        emd = earth_movers_distance(spike_times_contact, spike_times_audio) # run emd
        distances_list.append(emd) # add to appropriate list.
        # title = f"total spikes contact: {len(spike_times_contact)}, total_spikes audio: {len(spike_times_audio)}, emd = {str(emd)}" # vic purp: {str(vp_dist)}"
        # if count == 2:
            # plot_traces(x_vec[trace_selector_bool], [contact_trace_selected, audio_trace_selected], header_dict["samp_period_ms"], ["contact", "audio"], title)
    return np.array(distances_list)

def twd_per_time_calc(contact_trace, audio_trace, x_vec, cut_times, bpm):
    distances_list = []
    for k in range(len(cut_times)-1): # for each rime (-1 because two bookend times)
        spike_trace_contact, spike_trace_audio, _ = chop_traces(k, contact_trace, audio_trace, x_vec, cut_times, bpm)
        if len(spike_trace_audio) == 0 or len(spike_trace_contact) == 0:
            pause_here = True
        # twd = dtw_trace_distance(spike_trace_contact, spike_trace_audio) # run emd
        twd = 0
        distances_list.append(twd) # add to appropriate list.
        # title = f"total spikes contact: {len(spike_times_contact)}, total_spikes audio: {len(spike_times_audio)}, emd = {str(emd)}" # vic purp: {str(vp_dist)}"
        # if count == 2:
            # plot_traces(x_vec[trace_selector_bool], [contact_trace_selected, audio_trace_selected], header_dict["samp_period_ms"], ["contact", "audio"], title)
    return np.array(distances_list)

def bar_plot_scores(names, scores, errors, title, ylabel):
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(len(scores)))
    ax.set_xticklabels(names)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    ax.bar(np.arange(len(scores)), scores, yerr=errors, align='center')
    plt.axhline(np.mean(scores), color='r')
    plt.tight_layout()


def rotate_list(l, n):
    return l[-n:] + l[:-n]


def plot_scores(test_EMDs, test_MADs, test_VADs, bpm_labels, title_name, header_dict, rhythm_num, v_line_1=0, v_line_2=0):
    _, axes = plt.subplots(3,1)
    axes[0].set_title(title_name)
    axes[0].plot(np.arange(len(test_EMDs)), test_EMDs, color='r')
    axes[0].set_ylabel("EMD")
    axes[0].set_xticks(np.arange(len(test_EMDs)))
    axes[0].set_xticklabels(bpm_labels, rotation=45, ha='right')
    axes[1].plot(np.arange(len(test_MADs)), test_MADs, color='b')
    axes[1].set_ylabel("MAD")
    axes[1].set_xticks(np.arange(len(test_EMDs)))
    axes[1].set_xticklabels(bpm_labels, rotation=45, ha='right')
    axes[2].plot(np.arange(len(test_VADs)), test_VADs, color='g')
    axes[2].set_ylabel("VAD")
    axes[2].set_xticks(np.arange(len(test_EMDs)))
    axes[2].set_xticklabels(bpm_labels, rotation=45, ha='right')
    axes[2].set_xlabel("BPM")
    if v_line_1 > 0 and v_line_2 > 0:
        axes[0].axvline(v_line_1)
        axes[0].axvline(v_line_2)
        axes[1].axvline(v_line_1)
        axes[1].axvline(v_line_2)
        axes[2].axvline(v_line_1)
        axes[2].axvline(v_line_2)
    plt.tight_layout()
    return

def plot_scores_mean(emd_arr, mads_arr, vads_arr, bpm_labels, title_name, v_line_1, v_line_2):
    mean_emd = np.mean(emd_arr, axis=0)
    std_emd = np.std(emd_arr, axis=0)
    mean_mad = np.mean(mads_arr, axis=0)
    std_mad = np.std(mads_arr, axis=0)
    mean_vad = np.mean(vads_arr, axis=0)
    std_vad = np.std(vads_arr, axis=0)
    _, axes = plt.subplots(3,1)
    axes[0].set_title(title_name)
    axes[0].plot(np.arange(len(mean_emd)), mean_emd, color='r')
    axes[0].fill_between(np.arange(len(mean_emd)), mean_emd+std_emd, mean_emd-std_emd, color = 'r', alpha = 0.4)
    axes[0].set_ylabel("EMD")
    axes[1].plot(np.arange(len(mean_mad)), mean_mad, color='b')
    axes[1].fill_between(np.arange(len(mean_mad)), mean_mad+std_mad, mean_mad-std_mad, color = 'b', alpha = 0.4)
    axes[1].set_ylabel("MAD")
    axes[2].plot(np.arange(len(mean_vad)), mean_vad, color='g')
    axes[2].fill_between(np.arange(len(mean_vad)), mean_vad+std_vad, mean_vad-std_vad, color = 'g', alpha = 0.4)
    axes[2].set_ylabel("VAD")
    axes[2].set_xlabel("Phases")
    if v_line_1 > 0 and v_line_2 > 0:
        axes[0].axvline(v_line_1)
        axes[0].axvline(v_line_2)
        axes[1].axvline(v_line_1)
        axes[1].axvline(v_line_2)
        axes[2].axvline(v_line_1)
        axes[2].axvline(v_line_2)
    plt.tight_layout()
    return


def plot_learning_traces(all_rhythms_mean_flag, list_of_phases_to_plot, title_name, scores_by_rhythm, scores_by_session, header_dict, v_line_1=0, v_line_2=0):
    all_EMDs = []
    all_VADs = []
    all_MADs = []
    for rhythm_num in range(len(scores_by_rhythm)): # going by ACTUAL BPM PRESENTATION ORDER
        bpm_presentation_list_short = bpm_presentations[rhythm_num]
        bpm_presentation_list_single = bpm_presentation_list_short*len(scores_by_session)
        bpm_presentation_list = list(np.repeat(bpm_presentation_list_single, len(list_of_phases_to_plot)))
        bpm_axes = [bpms[i] for i in bpm_presentation_list]
        test_EMDs = []
        test_MADs = []
        test_VADs = []
        actual_bpms = []
        for session_num in range(len(scores_by_session)):
            scores_by_rhythm = scores_by_session[session_num][rhythm_num]
            for index in range(len(bpm_presentation_list_short)):
                bpm_index = bpm_presentation_list_short[index]
                scores_list = scores_by_rhythm[bpm_index]
                bpm_label = bpms[bpm_index]
                for phase in list_of_phases_to_plot:
                    test_EMDs.append(scores_list["EMD"][phase])
                    test_MADs.append(scores_list["MAD"][phase])
                    test_VADs.append(scores_list["VAD"][phase])
        all_MADs.append(test_MADs)
        all_VADs.append(test_VADs)
        all_EMDs.append(test_EMDs)
        if all_rhythms_mean_flag == 0:
            new_title_name = title_name + f" {header_dict['rhythm_strings_names'][rhythm_num]}"
            plot_scores(test_EMDs, test_MADs, test_VADs, bpm_axes, new_title_name, header_dict, rhythm_num, v_line_1, v_line_2)
    mads_arr = np.array(all_MADs)
    vads_arr = np.array(all_VADs)
    emd_arr = np.array(all_EMDs)
    if all_rhythms_mean_flag == 1:
        plot_scores_mean(emd_arr, mads_arr, vads_arr, bpm_axes, title_name, v_line_1, v_line_2)

def cumulative_plot(participant_results_list):
    # drop participant 7
    keys = ["emds", "mads", "vads"]
    colors = ['r',  'b', 'g']
    rhythm_name_list = ["rhythm_1", "rhythm_2", "rhythm_3", "rhythm_4"]

   

    slope_and_correlation_by_rhythm_and_participant = {
        'emds' : np.zeros([4, 6, 2]),
        'twds' : np.zeros([4, 6, 2]),
        'mads' : np.zeros([4, 6, 2]),
        'vads' : np.zeros([4, 6, 2])
    }

    for key_index in range(len(keys)):
        fig, axes = plt.subplots(1,4)
        fig.suptitle(f"{keys[key_index]} slope vs corcoeff for each rhythm across ppts")
        for rhythm_index in range(len(rhythm_name_list)):
            
            
            for participant_index in range(len(participant_results_list)):
                if (rhythm_index == 0 or rhythm_index == 1) and (participant_index == 1): # if heiko, pp7, and first two rhythms, drop
                    continue
                pp_results = participant_results_list[participant_index]
                change_in_perf_dict = pp_results["change_in_performance"]
                this_rhythm = change_in_perf_dict[keys[key_index]][rhythm_index]
                slope_and_correlation_by_rhythm_and_participant[keys[key_index]][rhythm_index, participant_index, 0] = this_rhythm[3] # slope
                slope_and_correlation_by_rhythm_and_participant[keys[key_index]][rhythm_index, participant_index, 1] = this_rhythm[5] # correlation coefficient

            slopes = slope_and_correlation_by_rhythm_and_participant[keys[key_index]][rhythm_index, :, 0]
            corrcoeff = slope_and_correlation_by_rhythm_and_participant[keys[key_index]][rhythm_index, :, 1]
            mask = np.logical_or(~(slopes==0), ~(corrcoeff==0))
            slopes = slopes[mask]
            corrcoeff = corrcoeff[mask]
            axes[rhythm_index].set_title(rhythm_name_list[rhythm_index])
            axes[rhythm_index].scatter(slopes, \
                corrcoeff, color = colors[key_index])
            axes[rhythm_index].set_xlabel("slope of trendline")
            axes[rhythm_index].set_ylabel("corrcoeff of trendline")
            plt.tight_layout()
            plt.draw()
            plt.pause(0.001)






# def cluster_intervals(num_intervals, gt_intervals)
# _____________________________________________________
# _____________________### MAIN ###____________________

if __name__ == '__main__':

    plt.style.use('ggplot')

    # teaserfigure()

    ### run tests ###
    # emd_tests()
    # filter_test()

    # file_stems = ['2022_03_24_16_43_14_pp4', '2022_03_25_15_46_43_pp4']
    # file_stems = ['2022_03_23_13_46_12_pp4', '2022_03_24_16_20_37_pp4', '2022_03_25_15_29_53_pp4']
    # file_stems =  ['2022_03_27_13_56_12_pp5']
    file_stems_all = [['2022_03_29_14_47_38_pp6', '2022_03_30_14_29_49_pp6'], \
        ['2022_03_30_13_23_09_pp7', '2022_03_31_13_19_05_pp7', '2022_04_01_13_30_32_pp7'], \
        ['2022_03_30_16_18_53_pp8', '2022_03_31_16_09_58_pp8', '2022_04_01_16_17_46_pp8'], \
        ['2022_04_01_19_39_42_pp9', '2022_04_02_16_43_08_pp9', '2022_04_03_13_51_14_pp98'], \
        ['2022_04_09_12_08_53_pp10', '2022_04_10_12_23_50_pp10', '2022_04_11_18_40_41_pp10'], \
        ['2022_04_11_15_56_10_pp11','2022_04_12_15_27_18_pp11', '2022_04_13_13_37_03_pp11']]
    names = ["pt6", "pt7", "pt8", "pt9", "pt10", "pt11"]
    conditions = ["audio", "audio", "audio", "tactile", "ems", "ems"]


    num_unique_rhythms = 4
    num_unique_bpms = 2
    test_performance_dictionary = {
        'emds' : np.zeros([num_unique_rhythms,3,8,2]),
        'twds' : np.zeros([num_unique_rhythms,3,8,2]),
        'mads' : np.zeros([num_unique_rhythms,3,8,2]),
        'vads' : np.zeros([num_unique_rhythms,3,8,2])
    } 
    change_in_performance_dictionary = {
        'emds' : np.zeros([num_unique_rhythms, 7]),
        'twds' : np.zeros([num_unique_rhythms, 7]),
        'mads' : np.zeros([num_unique_rhythms, 7]),
        'vads' : np.zeros([num_unique_rhythms, 7])
    }
    participant_results_dict = {
        "test_performance" : test_performance_dictionary,
        "change_in_performance" : change_in_performance_dictionary
    }
     # four rhythms, seven data points: difference from beginning to end of first day, difference beginning first day end of second, different beginning first end third
        # also slope, intercept of line of best fit and correlation coefficient, and p val on corr coef
    # four rhythms, three days, 8 presentations of each rhythm in that day, mean then sd
    metric_strings = test_performance_dictionary.keys() # get names of each metric (emd, twd, mad vad)

    participant_results_list = []
    for participant in range(len(names)):
        participant_results_list.append(copy.deepcopy(participant_results_dict))


    for pt_index in range(len(names)):
        file_stems = file_stems_all[pt_index]
        pt_name = names[pt_index]
        condition = conditions[pt_index]


        ### load data ###
        print(f"Loading data for {pt_name}.")
        t1 = time.time()
        vars_dicts = load_data(file_stems)
        end = time.time() - t1
        print("Data loaded. time taken: " + str(end))

        scores_by_session = []

        # each rhythm is tested twice per presentation. Two presentations. Then later another two for four presentations in the day. So far a total of 8 tests. Then there are three days. So 24 tests.
        

        for session_number in range(len(vars_dicts)):

            header_dict = vars_dicts[session_number]["header_dict"]
            var_lists = vars_dicts[session_number]["vars_by_rhythm_and_bpm"]

            shuffled_bpm_indexes = np.arange(len(header_dict['bpms']))
            bpm_presentations = []
            
            for i in range(len(header_dict['rhythm_strings'])):
                bpm_order = rotate_list(list(shuffled_bpm_indexes), i)
                bpm_presentations.append(bpm_order)

            # # pull experimental condition string
            # condition = ems_constants.counter_balanced_number_dict[header_dict["counter-balanced-number"]]
            # conditions.append(condition)

            count = 0

            repeat_list = header_dict['phase_repeats_list']
            
            rhythm_strings = header_dict['rhythm_strings']
            for rhythm_index in range(len(rhythm_strings)): # for each string
                rhythm_substr = rhythm_strings[rhythm_index]
                count = count + 1

                bpms = header_dict['bpms'] # list_of_bpms
                presentation_order = bpm_presentations[rhythm_index] # for this rhythm, in what order were the bpms presented?
                for i in range(len(presentation_order)): # for every bpm presented
                    bpm_index = presentation_order[i] # get the index of that bpm in the master list
                    bpm = bpms[bpm_index]  # index the list to get the bpm.


                    # pull vars from list
                    contact_x_interped = var_lists[rhythm_index][bpm_index]["contact_x_interped"]
                    reading_list_interped = var_lists[rhythm_index][bpm_index]["reading_list_interped"]
                    surpressed_contact_trace = var_lists[rhythm_index][bpm_index]["surpressed_contact_trace"]
                    surpressed_contact_onset_times = var_lists[rhythm_index][bpm_index]["surpressed_contact_onset_times"]
                    stim_onset_times = var_lists[rhythm_index][bpm_index]["stim_onset_times"]
                    audio_onset_times = var_lists[rhythm_index][bpm_index]["audio_onset_times"]
                    stim_trace = var_lists[rhythm_index][bpm_index]["stim_trace"]
                    audio_trace = var_lists[rhythm_index][bpm_index]["audio_trace"]

                    audio_hold = 30000/bpm 
                    first_audio = audio_onset_times[0]
                    last_audio = audio_onset_times[-1]
                    # EXAMINE DATA

                    # interpolated
                    # legend_labels = ['contact', 'stim', 'audio']
                    # title_str = f"{header_dict['rhythm_strings_names'][rhythm_index]}, {bpm} interpolated"
                    # view_window_begin = 5000
                    # view_window_end = -1
                    # plot_contact_trace_and_rhythm(reading_list_interped[view_window_begin:view_window_end], \
                    #     contact_x_interped[view_window_begin:view_window_end], stim_trace[view_window_begin:view_window_end],  \
                    #         audio_trace[view_window_begin:view_window_end], contact_x_interped[view_window_begin:view_window_end], header_dict['samp_period_ms'], legend_labels, title_str)

                    # # examine spiking
                    # title_str = f"{header_dict['rhythm_strings_names'][rhythm_index]}, {bpm} spikes"
                    # plot_contact_trace_and_rhythm(surpressed_contact_trace[view_window_begin:view_window_end], \
                    #    contact_x_interped[view_window_begin:view_window_end], stim_trace[view_window_begin:view_window_end],  \
                    #         audio_trace[view_window_begin:view_window_end], contact_x_interped[view_window_begin:view_window_end], header_dict['samp_period_ms'], legend_labels, title_str)

                    delays_list, len_rhythm_ms = determine_delays_list(rhythm_substr, bpm, header_dict, first_audio)
                    plot_flag = 0
                    # get metrics for each repeat by block 
                    plot_blocks_results_dict = plot_each_block(rhythm_substr, header_dict['rhythm_strings_names'][rhythm_index], bpm, header_dict['phase_repeats_list'], header_dict['phase_flags_list'], delays_list, \
                    surpressed_contact_onset_times, audio_onset_times, surpressed_contact_trace, audio_trace, contact_x_interped, reading_list_interped, contact_x_interped, reading_list_interped, contact_x_interped, plot_flag)

                    pres_index = int(np.floor((rhythm_index)/num_unique_rhythms)) # first set of rhythms or second
                    for strang in metric_strings: # for every metric
                        # put first test block mean/median into dictionary
                        first_test_block = np.nanmedian(plot_blocks_results_dict[strang][1][0:1])
                        if np.isnan(first_test_block):
                            stop = 1
                        participant_results_list[pt_index]["test_performance"][strang][rhythm_index%num_unique_rhythms][session_number][pres_index*num_unique_rhythms + \
                            bpm_index*num_unique_bpms][0] = first_test_block
                        
                        # put second test block mean/median into dict
                        second_test_block = np.nanmedian(plot_blocks_results_dict[strang][4][0:1])
                        if np.isnan(second_test_block):
                            stop = 1
                        participant_results_list[pt_index]["test_performance"][strang][rhythm_index%num_unique_rhythms][session_number][pres_index*num_unique_rhythms + \
                            bpm_index*num_unique_bpms + 1][0] = second_test_block
                        # put first test block standard dev into dict
                        participant_results_list[pt_index]["test_performance"][strang][rhythm_index%num_unique_rhythms][session_number][pres_index*num_unique_rhythms + \
                            bpm_index*num_unique_bpms][1] = np.nanstd(plot_blocks_results_dict[strang][1])
                        # put second test block standard dev into dict
                        participant_results_list[pt_index]["test_performance"][strang][rhythm_index%num_unique_rhythms][session_number][pres_index*num_unique_rhythms + \
                            bpm_index*num_unique_bpms + 1][1] = np.nanstd(plot_blocks_results_dict[strang][4])
        
        for rhythm_index in range(num_unique_rhythms):
            for strang in metric_strings: # for every metric   
                sessions = [0, 1, 2]
                first_session = 0
                # for this rhythm, all sessions together, all data points, only the means.
                all_sessions_array = participant_results_list[pt_index]["test_performance"][strang][rhythm_index][:,:,0]
                if sum(all_sessions_array[2]) == 0:
                    all_sessions_array = all_sessions_array[0:2]
                all_sesh_array = np.hstack(all_sessions_array)
                x = np.arange(len(all_sesh_array))
                nan_mask = ~np.isnan(all_sesh_array)
                all_sesh_array = all_sesh_array[nan_mask]
                x = x[nan_mask]
                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, all_sesh_array)
                participant_results_list[pt_index]["change_in_performance"][strang][rhythm_index][3] = slope
                participant_results_list[pt_index]["change_in_performance"][strang][rhythm_index][4] = intercept
                participant_results_list[pt_index]["change_in_performance"][strang][rhythm_index][5] = r_value
                participant_results_list[pt_index]["change_in_performance"][strang][rhythm_index][6] = p_value
                
                for session_num in sessions:
                    # get for [each index] at [each rhythm] for [the first session], [the first two datapoints], [their means].
                    change_in_performance = np.mean(participant_results_list[pt_index]["test_performance"][strang][rhythm_index][first_session][0:1][0]) - \
                        np.mean(participant_results_list[pt_index]["test_performance"][strang][rhythm_index][session_num][7:8][0])
                    #first two data points of first session - last two of first second and third
                    participant_results_list[pt_index]["change_in_performance"][strang][rhythm_index][session_num] = change_in_performance
                    # y_lims_list = plot_test_blocks(emds, mads, vads, twds, header_dict['rhythm_strings_names'][rhythm_index], rhythm_substr, bpm, ems_constants.phase_warning_strs)
            
        plot_all_blocks(participant_results_list[pt_index], header_dict['rhythm_strings_names'], pt_name, condition)
    # cumulative_plot(participant_results_list)

                    ## check delays markers
                    # plot_contact_trace_and_rhythm(surpressed_contact_trace, x_vec, stim_trace,  audio_trace, \
                    #     x_vec, header_dict['samp_period_ms'], legend_labels, title_str)
                    # ax = plt.gca()
                    # ax.scatter(delays_list, np.ones_like(delays_list), s=20)

                    # check stim accuracy
                    # if bpm_index == 0:

                    #     x_vec_array = np.array(contact_x_interped)
                    #     bool_selector = np.logical_and((x_vec_array > delays_list[2]), (x_vec_array < delays_list[3]))
                    #     title_str = f"{header_dict['rhythm_strings_names'][rhythm_index]}, bpm: {bpm}, stim phase check raw trace"
                    #     plot_contact_trace_and_rhythm(reading_list_interped[bool_selector], contact_x_interped[bool_selector], stim_trace[bool_selector],  audio_trace[bool_selector], \
                    #         contact_x_interped[bool_selector], header_dict['samp_period_ms'], legend_labels, title_str)
                    #     title_str = f"{header_dict['rhythm_strings_names'][rhythm_index]}, bpm: {bpm}, stim phase check spikes"
                    #     plot_contact_trace_and_rhythm(surpressed_contact_trace[bool_selector],contact_x_interped[bool_selector], stim_trace[bool_selector],  audio_trace[bool_selector], \
                    #         contact_x_interped[bool_selector], header_dict['samp_period_ms'], legend_labels, title_str)
                    
                    # MAD_by_phase, VAD_by_phase = MAD_VAD_per_time_calc(rhythm_substr, bpm, delays_list, audio_onset_times, surpressed_contact_onset_times)
                    
                    # distances_array = emd_per_time_calc(surpressed_contact_onset_times, audio_onset_times, delays_list, bpm)
                    # dist_array = np.copy(distances_array)


                    # phase_dict = {
                    #     "MAD" : MAD_by_phase,
                    #     "VAD" : VAD_by_phase,
                    #     "EMD" : distances_array
                    # }

                    # scores_by_bpm.append(phase_dict)
                #______________________________________#
                ###### END OF PER TRIAL ANALYSIS ####### (still per rhythm)

                # all_emds = all_emds + emds
                # plot_emds(emds, header_dict, rhythm_index)

        

        

    #     # print(f"mean improvement EMD: {np.mean(difference_EMD)} +/- {np.std(difference_EMD)} \n  mean improvement MAD: {np.mean(difference_MAD)} +/- {np.std(difference_MAD)} \n mean improvement VAD: {np.mean(difference_VAD)} +/- {np.std(difference_VAD)} \n")
    # title_name_mean = f"Last test block, all rhythms mean \n EMDs MADs and VADs"
    # title_name_rhythms = f"All test blocks \n EMDs MADs and VADs for"
    # all_rhythms_mean_flag = 1
    # list_of_phases_to_plot = [4]
    # title_name = title_name_mean

    # plot_learning_traces(all_rhythms_mean_flag, list_of_phases_to_plot, title_name, scores_by_rhythm, scores_by_session, header_dict, v_line_1=0, v_line_2=0)
    

        
        

end = 3




                # fig, ax = plt.subplots()
                # label_list = ["EMD", "VPD"]
                # normalize EMD

                # distances_array = np.divide(distances_array, np.max(distances_array)) why would we norm?

                # get intervals in this rhythm
                # wk_var_lists = get_all_intervals(header_dict, audio_onset_times, delays_list, surpressed_contact_onset_times)
                # list_of_WK_var_lists_by_bpm.append(wk_var_lists)
                




# get mean differences by rhythm
        # difference_EMD = []
        # difference_MAD = []
        # difference_VAD = []
        # for rhythm_number in range(len(scores_by_rhythm)):
        #     bpm_scores = scores_by_rhythm[rhythm_number]
        #     diff_EMD_rhythm = []
        #     difference_EMD.append(diff_EMD_rhythm)
        #     diff_MAD_rhythm = []
        #     difference_MAD.append(diff_MAD_rhythm)
        #     diff_VAD_rhythm = []
        #     difference_VAD.append(diff_VAD_rhythm)
        #     for bpm_number in range(len(bpm_scores)):
        #         scores_dict = bpm_scores[bpm_number]
        #         diff_EMD = scores_dict['EMD'][1] - scores_dict['EMD'][4]
        #         diff_EMD_rhythm.append(diff_EMD)
        #         diff_VAD = scores_dict['VAD'][1] - scores_dict['VAD'][4]
        #         diff_VAD_rhythm.append(diff_VAD)
        #         diff_MAD = scores_dict['MAD'][1] - scores_dict['MAD'][4]
        #         diff_MAD_rhythm.append(diff_MAD)


        # titles = [f"EMD differences by rhythm, {condition}, \n date: {header_dict['test time']}, \n mean: {round_sig(np.mean(difference_EMD), 3)} +/- {round_sig(np.std(difference_EMD), 3)}", \
        #     f"MAD differences by rhythm, {condition}, \n date: {header_dict['test time']}, \n mean: {round_sig(np.mean(difference_MAD), 3)} +/- {round_sig(np.std(difference_MAD), 3)}", \
        #     f"VAD differences by rhythm, {condition}, \n date: {header_dict['test time']}, \n mean: {round_sig(np.mean(difference_VAD), 3)} +/- {round_sig(np.std(difference_VAD), 3)}"] 
        # ylabels = ["Baseline test EMD - post-test EMD", "Baseline test MAD - post-test MAD", "Baseline test VAD - post-test VAD"]
        # names = header_dict['rhythm_strings_names']
        # for metric in range(0, 2*len(titles), 2):
        #     scores = []
        #     errors = []
        #     for improvement_vec in improvement_index_matrix:
        #         # for every rhythm, add to the scores vector the mean of, for every index in the improvement vector,
        #         scores.append(np.mean([i[metric] - i[metric+1] for i in improvement_vec]))
        #         errors.append(np.std([i[metric] - i[metric+1] for i in improvement_vec]))
        
        #     bar_plot_scores(names, scores, errors, titles[round(metric/2)], ylabels[round(metric/2)])

        # for i in range(len(improvement_index_matrix)):
        #     improvements_vec = improvement_index_matrix[i]
        #     rhythm_improvements_vec = [i[0] - i[1] for i in improvements_vec]
        #     name = header_dict['rhythm_strings_names'][i]
        #     meann = np.mean(rhythm_improvements_vec)
        #     stdd = np.std(rhythm_improvements_vec)
        #     print(name + f", improvement: {meann}, +/- {stdd}")

        # for i in range(len(improvement_index_matrix[0])):
        #     improvements_vec = improvement_index_matrix[:][i]
        #     rhythm_improvements_vec = [i[0] - i[1] for i in improvements_vec]
        #     name = str(header_dict['bpms'][i])
        #     meann = np.mean(rhythm_improvements_vec)
        #     stdd = np.std(rhythm_improvements_vec)
        #     print(name + f", improvement: {meann}, +/- {stdd}")


    # perf_array = np.array(first_performances)
    # means_first_perf_emd = np.mean(first_performances[:][:][0], axis = 0)
    # std_first_perf_emd = np.std(first_performances[:][:][0], axis = 0)
    # means_first_perf_mad = np.mean(first_performances[:][:][1], axis = 0)
    # std_first_perf_mad = np.std(first_performances[:][:][1], axis = 0)
    # means_first_perf_vad = np.mean(first_performances[:][:][2], axis = 0)
    # std_first_perf_vad = np.std(first_performances[:][:][2], axis = 0)

    # means_first = [means_first_perf_emd, means_first_perf_mad, means_first_perf_vad]
    # std_first = [std_first_perf_emd, std_first_perf_mad, std_first_perf_vad]
    # ylabels = ["EMD", "MAD", "VAD"]
    # plt.figure()
    # fig, axes = plt.subplots(3,1)
    # for i in range(len(means_first)):
    #     axes[i].scatter(np.arange(len(means_first[i])), means_first[i] )
    #     axes[i].errorbar(np.arange(len(means_first[i])), means_first[i], yerr = std_first[i]) 
    #     axes[i].set_ylabel(ylabels[i])
        

    # axes[i].set_xticks(np.arange(len(means_first[i])))
    # axes[i].set_xlabel("experimental condition")
    # axes[i].set_xticklabels(conditions)
    # axes[0].set_title("EMD, MAD, VAD over first tests for each condition")


print("done")
        # # organize intervals
        # intervs, unique_intervals, num_unique = count_intervals(header_dict['rhythm_strings'][rhythm_index]) 
        # means = []
        # sds = []
        # # wing kris across bpms for each phase
        # for i in range(header_dict['num_phases']): # for each phase
        #     # list_of_vars_for_phase = [] # first dim, num bpms long. second dim, phase, third, ground truth, user, user_error.
        #     interval_means_list = []
        #     interval_sd_list = []
        #     for j in range(len(header_dict['bpms'])): # grab the data for this phase at each tempo
        #         variables_for_this_phase_at_this_tempo = list_of_WK_var_lists_by_bpm[j][i]
        #         # list_of_vars_for_phase.append(variables_for_this_phase_at_this_tempo)
        #         ground_truth_intervs = variables_for_this_phase_at_this_tempo[0] # get all gt intervals
        #         # ground_truth_intervs_across_rhythms_flat = sum(ground_truth_intervs_across_rhythms, [])
        #         # histo_intervals(ground_truth_intervs)
        #         # get list of unique intervals and indices that map all intervals to their assigned unique interval
        #         unique_gt_intervals, indices = compile_unique_interval_list(ground_truth_intervs, ems_constants.interval_tolerance)
        #         # unique_gt_intervals, indices = cluster_intervals(ground_truth_intervs, num_intervals)
        #         user_intervals_across_rhythms = variables_for_this_phase_at_this_tempo[1] # get all user intervasl
        #         # user_intervals_across_rhythms_flat = sum(user_intervals_across_rhythms, [])
        #         list_of_user_intervals_by_target_interval = [[] for _ in range(len(unique_gt_intervals))] # make a list of lists as long as unique intervals
        #         for k in range(len(user_intervals_across_rhythms)): # for every user interval
        #             target_interval_index = indices[k] # find its unique target 
        #             # now add it to the list of intervals produced for that target
        #             # within the list of unique intervals
        #             list_of_user_intervals_by_target_interval[target_interval_index].append(user_intervals_across_rhythms[k])
        #         interval_means_list = interval_means_list + [np.nanmean(np.array(item_list)) for item_list in list_of_user_intervals_by_target_interval]
        #         interval_sd_list = interval_sd_list + [np.nanstd(np.array(item_list)) for item_list in list_of_user_intervals_by_target_interval]
        #     means.append(interval_means_list)
        #     sds.append(interval_sd_list)

        # fig, axes = plt.subplots(2,3) # change this according to num phase?
        # fig.suptitle(f"{header_dict['rhythm_strings_names'][rhythm_index]} Wing Kris", fontsize=16)

        # titles = header_dict['phase_name_strs']
        # slopes, intercepts, r_values, p_values = plot_w_k(axes, means, sds, titles)

            


        # fig, ax = plt.subplots() # change this according to num phase?
        # legend_labels = ["clock var", "motor var"]
        # ax.plot(np.arange(len(slopes)), slopes/max(slopes))
        # ax.plot(np.arange(len(intercepts)), intercepts/max(intercepts))
        # ax.set_title(header_dict['rhythm_strings_names'][rhythm_index])
        # ax.legend(legend_labels)

        # ax.set_title("normalized variances for clock and motor across epochs")
        # plt.ion()
        # plt.show()
        # plt.draw()
        # plt.pause(0.01)
   

#notes

# turn off graphs

# pull emds for test phases and graph across each tempo and each rhythm.



# stop graphing vp_dist

# double check graph labels and fix axes and make them fucking pretty

# combine all repeats into just phase Values/averages
# investigate cut offs at the end?? total record time must be off

# compute histogram of onsets for each phase for each rhythm for each tempo then scaled by Tempo

# investigate timing errors in metronome and presentations ??? 

# investigate negative times in x axis for wing kris plots -- find a way to kick them

# stop plotting so much

# slow tempo is broken? analysis or recording? on flip beat





            # title_str = f"raw contact trace, {header_dict['rhythm_strings_names'][rhythm_index]}, {bpm}"
            # legend_labels = ["contact trace", "stim trace", "audio trace"]
            # plot_contact_trace_and_rhythm(reading_list, contact_x_values, stim_trace, audio_trace, x_vec, header_dict['samp_period_ms'], legend_labels, title_str)

            # legend_labels = ["surpressed contact trace", "stim trace", "audio trace"]
            # title_str = f"surpressed contact trace, {header_dict['rhythm_strings_names'][rhythm_index]}, {bpm}"
            # # plot_traces(x_vec, [surpressed_contact_trace, audio_trace, stim_trace], header_dict['samp_period_ms'], legend_labels, header_dict['rhythm_strings_names'][rhythm_index])
            # plot_contact_trace_and_rhythm(surpressed_contact_trace, x_vec, stim_trace, audio_trace, x_vec, header_dict['samp_period_ms'], legend_labels, title_str)


            ### analysis ###
             
            
            # ax.plot(np.arange(len(distances_array)), distances_array)
            # vp_array = np.array(vp_dist_list)
            # vp_array = np.divide(vp_array, np.max(vp_array))
            # ax.plot(np.arange(len(vp_array)), vp_array)
            # ax.legend(label_list)

            # ax.set_title(f"normalized Earth Mover's Distance for {rhythm_substr}, at {bpm} bpm")
            # plt.ion()
            # plt.show()
            # plt.draw()
            # plt.pause(0.01)

            # store interval information for wing kris analysis

            # user interval is the time between the nearest contact spike to an audio spike and the previous audio spike.
            # user error is the absolute difference between user interval and the ground truth audio interval.
