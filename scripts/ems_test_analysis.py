
import warnings
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

def earth_movers_distance(spike_times_a, spike_times_b, rhythm_trace_a, rhythm_trace_b):
    
    rhythm_total_spike_times_a = len(spike_times_a)
    rhythm_total_spike_times_b = len(spike_times_b)
    if rhythm_total_spike_times_a == 0:
        return -1
    cumulative_a = np.cumsum(np.divide(rhythm_trace_a, rhythm_total_spike_times_a))
    cumulative_b = np.cumsum(np.divide(rhythm_trace_b, rhythm_total_spike_times_b))
    # same thing as np.sum(np.abs(np.subtract(cumulative_a, cumulative_b))),
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

def count_time_readings(worksheet):
## how many time readings are there?
    counter = 0
    val = 1.0
    while type(val) is float or type(val) is int: #while we're reading floats and not nans
        val = worksheet.cell(row=counter + worksheet_data_begin_indices[0], column=1).value
        counter = counter+1
    time_readings_length = counter-2
    return time_readings_length

def read_array_values(time_readings_length, variables_number, worksheet):
    arr = np.empty([time_readings_length, variables_number]) 
    arr[:] = np.NaN
    for r in range(time_readings_length):
        for c in range(variables_number):
            arr[r][c] = worksheet.cell(row=r + worksheet_data_begin_indices[0], column=c + worksheet_data_begin_indices[1]).value
    contact_x_values = arr[:, 0]
    reading_list = arr[:, 1]
    stim_onsets_temp = arr[:, 2]
    stim_onset_times = stim_onsets_temp[~np.isnan(stim_onsets_temp)] # take care of nans that make array work
    audio_onsets_temp = arr[:, 3]
    audio_onset_times = audio_onsets_temp[~np.isnan(audio_onsets_temp)]
    return contact_x_values, reading_list, stim_onset_times, audio_onset_times

def determine_delays_list(rhythm_substr, bpm, header_dict):
    # determine the exact beginning timestamp (delays_list) for each phase (pre-ems audio, ems, post-ems audio, no audio)
    len_rhythm_ms = len(rhythm_substr) * 30000/bpm #+ 75 ### tHIS IS SOMEHOW OFF BU ABOUT 75 MS?????
    # warnings.warn("Adding 75 ms to rhythm length for unknown reason?")
    if header_dict['metronome_intro_flag'] == 1:
        len_count_off_ms = len(header_dict['count_in_substr']) * 30000/bpm  # this is the three second count in
    else:
        len_count_off_ms = 0 # 

    list_of_delays = [len_count_off_ms] # first phase begins after count off
    for i in range(header_dict['num_phases']): # for every phase, find the end of the phase and tack it on.
        last_time = list_of_delays[-1] # get the last time (end of last phase, beginning of this one)
        list_of_delays.append(last_time + header_dict['phase_repeats_list'][i] * len_rhythm_ms) #get the number of repeats and find the length in time in ms and add it on
    return list_of_delays, len_rhythm_ms
        
def chop_traces(k, surpressed_contact_onset_times, surpressed_contact_trace, audio_onset_times, audio_trace, delays_list):
    loop_begin = delays_list[k] - 0.5*30000/bpm #include half an eighthnote before
    loop_end = delays_list[k+1] + 1 * 30000/bpm #include half an eighthnote after as well
    contact_bool = np.logical_and((surpressed_contact_onset_times >= loop_begin), (surpressed_contact_onset_times <= loop_end)) # select contact onset times during this loop of rhythm
    audio_bool = np.logical_and((audio_onset_times >= loop_begin), (audio_onset_times <= loop_end)) # select audio onset times during this loop of rhythm
    spike_times_contact = surpressed_contact_onset_times[contact_bool] - loop_begin # how many spikes total?
    spike_times_audio = audio_onset_times[audio_bool] - loop_begin
    trace_selector_bool = np.logical_and((x_vec >= loop_begin), (x_vec <= loop_end)) # which indices in traces are during this loop?
    contact_trace_selected = surpressed_contact_trace[trace_selector_bool] # pick those data points from suprpressed contact trace
    audio_trace_selected = audio_trace[trace_selector_bool] # pick those data points from audio trace
    return spike_times_contact, spike_times_audio, contact_trace_selected, audio_trace_selected, trace_selector_bool

def load_header():
    pkl_files = []
    for file in glob.glob("data/*.pkl"):   
        if file[0:6] == 'data/2': # starts with the date, so the type of file we want, i.e., 2022_...
            pkl_files.append(file)
    pkl_files.sort(reverse=True)
    with open(pkl_files[0], "rb") as pkl_handle:
        header_dict = pickle.load(pkl_handle)
    return header_dict

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

def load_data():
    xlsxfiles = []
    for file in glob.glob("data/*.xlsx"):   
        if file[0:6] == 'data/2': # starts with the date, so the type of file we want, i.e., 2022_...
            xlsxfiles.append(file)
    xlsxfiles.sort(reverse=True)
    # ## open workbook, define worksheets ###
    wb = load_workbook(xlsxfiles[0]) # most recent participant file
    return wb

def plot_emds(emds_normed, header_dict):
    mean_distances = np.mean(np.vstack(emds_normed), 0)
    sd_distances = np.std(np.vstack(emds_normed), 0)
    fig, ax = plt.subplots() # change this according to num phase?
    legend_labels = ["earth mover's distances mean", "std+", "std-"]
    ax.plot(np.arange(len(mean_distances)), mean_distances,'b')
    ax.plot(np.arange(len(mean_distances)), mean_distances+sd_distances, 'r')
    ax.plot(np.arange(len(mean_distances)), mean_distances-sd_distances, 'r')
    ax.set_title(f"mean normalized earth mover's distance across bpms for each phase, {header_dict['rhythm_strings_names'][rhythm_index]}")
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
    

def emd_test(title, times_a, times_b, mini, maxi, samp_period):
    xvec = np.arange(mini, maxi, samp_period)
    trace_a = spike_times_to_traces(times_a, samp_period, xvec, samp_period)
    trace_b = spike_times_to_traces(times_b, samp_period, xvec, samp_period)
    emd = earth_movers_distance(times_a, times_b, trace_a, trace_b)
    title = title + f", emd = {emd}"
    plot_traces(xvec, [trace_a, trace_b], samp_period, ["a", "b"], title)
    return emd

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
        emds.append(earth_movers_distance(times_a, times_b, trace_a, trace_b))
        axes[i].set_yticks([])
        axes[i].set_xticks([])
        axes[i].plot(xvec, trace_b+1)
        axes[i].plot(xvec, trace_a)
        if i == 0:
            # axes[i].legend(legend_labels, loc='best')
            axes[i].set_yticks([0.5, 1.5])
            axes[i].set_yticklabels(['ground truth \n pattern', 'user produced \n pattern'], fontsize=10)
        axes[i].set_title(titles[i])
        if i == 1:
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
    for i in range(len(times_bs)):
        times_b = times_bs[i]
        title = titles[i]
        emds.append(emd_test(title, times_a, times_b, mini, maxi, samp_period))

    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(len(titles)))
    ax.set_xticklabels(titles)
    ax.set_title("Bar plot for EMDS across tests")
    ax.set_ylabel("EMD")
    plt.xticks(rotation=30, ha="right")
    ax.bar(np.arange(len(titles)), emds, align='center')
    return 


def accumulate_intervals(phase_audio_onsets, surpressed_contact_onset_times):
    ground_truth_intervals = []
    user_intervals = []
    user_error = []
    for j in range(len(phase_audio_onsets)-1):
        # get the interval between them
        gt_interval = phase_audio_onsets[j+1] - phase_audio_onsets[j]
        # get the index of the nearest response pulse to the j+1 audio
        arg_min = np.argmin(np.abs(np.subtract(surpressed_contact_onset_times, phase_audio_onsets[j+1]))) # throws out the first one...
        # get the user interval
        user_interval = surpressed_contact_onset_times[arg_min] - phase_audio_onsets[j]#response time user - previous audio pulse
        if user_interval <= 0:
            user_interval = np.nan
            # raise ValueError("user interval less than 0")
        ground_truth_intervals.append(gt_interval)
        user_intervals.append(user_interval)
        user_error.append(np.abs(gt_interval-user_interval))
    return ground_truth_intervals, user_intervals, user_error

def get_all_intervals(header_dict, audio_onset_times, delays_list, surpressed_contact_onset_times):
    var_lists = []
    for k in range(header_dict['num_phases']):
        # array of 1s for audio onset times in this phase
        this_phase_bools = np.logical_and((audio_onset_times > delays_list[k]), (audio_onset_times < delays_list[k+1])) 
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
        ground_truth_intervals, user_intervals, user_error = accumulate_intervals(phase_audio_onsets, surpressed_contact_onset_times)
        var_list = [ground_truth_intervals, user_intervals, user_error]
        var_lists.append(var_list) # now has each WK relevant variable for each phase
    return var_lists

def surpress(audio_trace, surpressed_contact_trace):
# change to pure spikes (complete surround surpression)
    audio_trace_copy = np.copy(audio_trace)
    for j in range(len(audio_trace)-1):
        if audio_trace_copy[j] == 1:
            audio_trace[j+1] = 0
    
    contact_trace_copy = np.copy(surpressed_contact_trace)
    for j in range(len(surpressed_contact_trace)-1):
        if contact_trace_copy[j] == 1:
            surpressed_contact_trace[j+1] = 0
    return surpressed_contact_trace, audio_trace

def emd_per_phase_calc(surpressed_contact_onset_times, surpressed_contact_trace, audio_onset_times, audio_trace, delays_list):
    distances_list = []
    for k in range(len(delays_list)-1): # for each phase
        spike_times_contact, spike_times_audio, contact_trace_selected, audio_trace_selected, trace_selector_bool = chop_traces(k, \
            surpressed_contact_onset_times, surpressed_contact_trace, audio_onset_times, audio_trace, delays_list)
        emd = earth_movers_distance(spike_times_contact, spike_times_audio, contact_trace_selected, audio_trace_selected) # run emd
        distances_list.append(emd) # add to appropriate list.
        title = f"total spikes contact: {len(spike_times_contact)}, total_spikes audio: {len(spike_times_audio)}, emd = {str(emd)}" # vic purp: {str(vp_dist)}"
        # if count == 2:
            # plot_traces(x_vec[trace_selector_bool], [contact_trace_selected, audio_trace_selected], header_dict["samp_period_ms"], ["contact", "audio"], title)
    return np.array(distances_list)

# def cluster_intervals(num_intervals, gt_intervals)
# _____________________________________________________
# _____________________### MAIN ###____________________

if __name__ == '__main__':

    plt.style.use('ggplot')

    teaserfigure()

    ### run tests ###
    # emd_tests()
    # filter_test()

    ### load header
    header_dict = load_header()

    ### load data ###
    wb = load_data()

    # xlsx parser does not do 0-start counting
    worksheet_data_begin_indices = [val + 1 for val in header_dict["worksheet_data_begin_indices" ]]
    variables_number = 4   # time_readings, contact trace, stim_onsets, audio_onsets

    count = 0
    all_emds_normed = []
    # doing analyses FOR EACH RHYTHM separately.
    rhythm_strings = header_dict['rhythm_strings']
    for rhythm_index in range(len(rhythm_strings)): # for each string
        rhythm_substr = rhythm_strings[rhythm_index]
        count = count + 1

        list_of_WK_var_lists_by_bpm = []
        list_of_var_lists_by_bpm = []
        list_of_emd_distances_by_bpm = []
        list_of_vp_distances_by_bpm = []
        emds_normed = []

        bpms = header_dict['bpms'] # for each bpm
        for bpm_index in range(len(bpms)):
            bpm = bpms[bpm_index]
            worksheet = wb[f"{header_dict['rhythm_strings_names'][rhythm_index]}_bpm{bpm}"]
            time_readings_length = count_time_readings(worksheet)

            ## make the appropriate size data array and read in values.
            contact_x_values, reading_list, stim_onset_times, audio_onset_times = read_array_values(time_readings_length, variables_number, worksheet)
            cutoff_freq_low = ems_constants.cutoff_freq_low # cutoff period Any component with a longer period is attenuated
            cutoff_freq_high = ems_constants.cutoff_freq_high
            contact_x_interped, reading_list_interped = interpolate(reading_list, contact_x_values, header_dict['samp_period_ms'])
            reading_list_filtered = butter_band_pass_filter(contact_x_interped, reading_list_interped, cutoff_freq_low, cutoff_freq_high, header_dict['samp_period_ms'], plot_flag=0)
            
            ## preprocessing
            audio_hold = 30000/bpm 
            x_vec = contact_x_interped
            stim_trace = spike_times_to_traces(stim_onset_times, header_dict['actual_stim_length'], x_vec, header_dict['samp_period_ms'])
            audio_trace = spike_times_to_traces(audio_onset_times, audio_hold, x_vec, header_dict['samp_period_ms'])

            surpressed_contact_onset_times = process_contact_trace_to_hit_times(reading_list_filtered, contact_x_interped, ems_constants.baseline_subtractor, ems_constants.surpression_window)
            surpressed_contact_trace = spike_times_to_traces(surpressed_contact_onset_times, ems_constants.contact_spike_time_width, x_vec, header_dict['samp_period_ms'])

            var_list = [contact_x_values, reading_list, stim_onset_times, audio_onset_times, x_vec, stim_trace, audio_trace, surpressed_contact_onset_times, surpressed_contact_trace]
            list_of_var_lists_by_bpm.append(var_list)
            
            delays_list, len_rhythm_ms = determine_delays_list(rhythm_substr, bpm, header_dict)
            
            repeat_list = header_dict['phase_repeats_list']

            surpressed_contact_trace, audio_trace = surpress()

            distances_array = emd_per_phase_calc(surpressed_contact_onset_times, surpressed_contact_trace, audio_onset_times, audio_trace, delays_list)
            dist_array = np.copy(distances_array)

            # what is this doing???
            dist_array[dist_array == -1] = max(dist_array)
        
            list_of_emd_distances_by_bpm.append(dist_array)
            # fig, ax = plt.subplots()
            # label_list = ["EMD", "VPD"]
            # normalize EMD

            distances_array = np.divide(distances_array, np.max(distances_array))
            emds_normed.append(distances_array)

            # get intervals in this rhythm
            var_lists = get_all_intervals(header_dict, audio_onset_times, delays_list, surpressed_contact_onset_times)
            list_of_WK_var_lists_by_bpm.append(var_lists)
            

        #______________________________________#
        ###### END OF PER TRIAL ANALYSIS ####### (still per rhythm)

        all_emds_normed = all_emds_normed + emds_normed
        # plot_emds(emds_normed, header_dict)

        print("done")


        # organize intervals

        intervs, unique_intervals, num_unique = count_intervals(header_dict['rhythm_strings'][rhythm_index]) 

        means = []
        sds = []
        # wing kris across bpms for each phase
        for i in range(header_dict['num_phases']): # for each phase
            # list_of_vars_for_phase = [] # first dim, num bpms long. second dim, phase, third, ground truth, user, user_error.
            interval_means_list = []
            interval_sd_list = []
            for j in range(len(header_dict['bpms'])): # grab the data for this phase at each tempo
                variables_for_this_phase_at_this_tempo = list_of_WK_var_lists_by_bpm[j][i]
                # list_of_vars_for_phase.append(variables_for_this_phase_at_this_tempo)
                ground_truth_intervs = variables_for_this_phase_at_this_tempo[0] # get all gt intervals
                # ground_truth_intervs_across_rhythms_flat = sum(ground_truth_intervs_across_rhythms, [])
                # histo_intervals(ground_truth_intervs)
                # get list of unique intervals and indices that map all intervals to their assigned unique interval
                unique_gt_intervals, indices = compile_unique_interval_list(ground_truth_intervs, ems_constants.interval_tolerance)
                # unique_gt_intervals, indices = cluster_intervals(ground_truth_intervs, num_intervals)
                user_intervals_across_rhythms = variables_for_this_phase_at_this_tempo[1] # get all user intervasl
                # user_intervals_across_rhythms_flat = sum(user_intervals_across_rhythms, [])
                list_of_user_intervals_by_target_interval = [[] for _ in range(len(unique_gt_intervals))] # make a list of lists as long as unique intervals
                for k in range(len(user_intervals_across_rhythms)): # for every user interval
                    target_interval_index = indices[k] # find its unique target 
                    # now add it to the list of intervals produced for that target
                    # within the list of unique intervals
                    list_of_user_intervals_by_target_interval[target_interval_index].append(user_intervals_across_rhythms[k])
                interval_means_list = interval_means_list + [np.nanmean(np.array(item_list)) for item_list in list_of_user_intervals_by_target_interval]
                interval_sd_list = interval_sd_list + [np.nanstd(np.array(item_list)) for item_list in list_of_user_intervals_by_target_interval]
            means.append(interval_means_list)
            sds.append(interval_sd_list)

        fig, axes = plt.subplots(2,3) # change this according to num phase?
        fig.suptitle(f"{header_dict['rhythm_strings_names'][rhythm_index]} Wing Kris", fontsize=16)

        titles = header_dict['phase_name_strs']
        slopes, intercepts, r_values, p_values = plot_w_k(axes, means, sds, titles)

            


        fig, ax = plt.subplots() # change this according to num phase?
        legend_labels = ["clock var", "motor var"]
        ax.plot(np.arange(len(slopes)), slopes/max(slopes))
        ax.plot(np.arange(len(intercepts)), intercepts/max(intercepts))
        ax.set_title(header_dict['rhythm_strings_names'][rhythm_index])
        ax.legend(legend_labels)

        ax.set_title("normalized variances for clock and motor across epochs")
        plt.ion()
        plt.show()
        plt.draw()
        plt.pause(0.01)
    plot_emds(all_emds_normed, header_dict)
    print("done")


#notes

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
