import random
from numpy.core.fromnumeric import size
import ems_constants
import numpy as np
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl import load_workbook
import glob
import quantities as pq
from elephant.spike_train_dissimilarity import victor_purpura_distance
from neo import SpikeTrain
import scipy
from ems_test_analysis import count_intervals
import sys 
import time

def bar_plot_scores(scores, title):
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(len(scores)))
    ax.set_xticklabels(ems_constants.rhythm_strings_names)
    ax.set_title(title)
    ax.set_ylabel("cross ent scores")
    plt.xticks(rotation=45, ha="right")
    ax.bar(np.arange(len(scores)), scores, align='center')
    plt.tight_layout()
    
def ioi_ent(rhyth_str): # calculated as in Milne and Herff 2020
    possible_intervals = range(0, len(rhyth_str)) # x axis of probability distribtuon
    intervs, _, _ = count_intervals(rhyth_str) # get all the intervals in the rhythm
    if len(intervs) == 0:
        return 0
    summation = 0 # summing variable
    num_onsets = len(intervs) # number of onsets to normalize by
    for i in range(len(possible_intervals)): # for every possible interval
        counter = intervs.count(possible_intervals[i]) # get the number of that interval present in the rhythm
        if not(counter == 0): # if it's not 0
            summation += -1 * counter/num_onsets * np.log(counter/num_onsets) # add to entropy
    return summation


def rhythm_cross_entropy(probabilities):
    summed = 0
    for prob in probabilities: # for each probability sum negative log
        summed += -np.log(prob)
    normed = summed #(probabilities)
    return normed

def fine_grained_met(rhyth_string):
    probabilities = [] 
    for ch_index in range(len(rhyth_string)): # for each entry in string
        beat_position = int(ch_index) % 8 # get beat position
        position_probability = ems_constants.metrical_position_probabilities[beat_position] # get metrical probability by location
        if int(rhyth_string[ch_index]): # add proibability or flip if 0
            probabilities.append(position_probability)
        else:
            probabilities.append(1-position_probability)

    normed_cross_ent = rhythm_cross_entropy(probabilities)

    return(normed_cross_ent)

def classify_anchor_status(note_level, rhythm_string, ch_index):
    # determines anchor status and fetches probability associated with that status for that level of note.
    distance_to_next_level = ems_constants.distances_to_next_level[note_level] 
    # get the distance to the next level - i.e., for halfnote, index 1, it would be 4 beats away.
    preceding = rhythm_string[ch_index-distance_to_next_level] # get the preceding next level note
    if len(rhythm_string)-1 < ch_index + distance_to_next_level: # if the distance to the next level and where we are is greater than str length
        ind = 0 # next level is always the first note of the string (cause of repeat)
    else:
        ind = ch_index + distance_to_next_level # otherwise we can add like this
    next_val = rhythm_string[ind]
    if int(next_val) and int(preceding): # if both before and after are notes then both 
        probability = ems_constants.anchor_stat['both_anchored'][note_level]
    elif int(next_val) and not(int(preceding)): # if next is but not before, post
        probability = ems_constants.anchor_stat['post_anchored'][note_level]
    elif not(int(next_val)) and not(int(preceding)): # if both not notes then not
        probability = ems_constants.anchor_stat['not_anchored'][note_level]
    elif not(int(next_val)) and int(preceding): # if after is no note but before is, pre 
        probability = ems_constants.anchor_stat['pre_anchored'][note_level]
    else:
        raise ValueError('Anchor status could not be determined')
    return probability

def hierarchical_model(rhyth_string, ioi_ent_flag):
    # for a given string, for each entry, determines note level, anchor status, then probability
    probabilities = []
    for ch_index in range(len(rhyth_string)): # for each location in rhythm string
        beat_position = ch_index % 8 # get beat position
        if beat_position == 0: # if this is the first beat position of the measure
            note_level = 0 # whole note
        elif beat_position % 4 == 0: # else if this is the middle of the measure
            note_level = 1 # half note
        elif beat_position % 2 == 0: # else if this is a quarter note place
            note_level = 2 # quarter note
        else: #other wise,
            note_level = 3 # eighthnote 


        if note_level == 0: # if we had a whole note don't look at the anchor status
            note_there_probability = ems_constants.level_zero_probability
        else: # otherwise
            note_there_probability = classify_anchor_status(note_level, rhyth_string, ch_index) 
            # given note level and context, give probability that there would be a note there

        if int(rhyth_string[ch_index]):  # if there is a note there
            probability = note_there_probability # then that's the probability
        else:
            probability = 1-note_there_probability # if there's no note there then flip probability

        probabilities.append(probability) # add to probabilities

    normed_cross_ent = rhythm_cross_entropy(probabilities) # calculate 
    if ioi_ent_flag:
        ent = ioi_ent(rhyth_string)
        if ent == 0: #not to discount these strings
            ent = 0.1
        normed_cross_ent = normed_cross_ent * ent
    return(normed_cross_ent)
        



def random_rhythm(length, num_rhyths, no_double_stroke_flag, num_onsets=0):
    max_val = 2**length
    if max_val < num_rhyths:
        raise ValueError("Asking for more unique rhythms than are possible")
    rhyth_number_out = random.sample(range(1, max_val), num_rhyths)
    bin_strs = []
    counter = 0
    if num_onsets > 0:
        for number in rhyth_number_out:
            counter += 1
            sys.stdout.write("\r" + f" sorting rhythm {counter} of {num_rhyths}")   #The \r is carriage return without a line 
                                        #feed, so it erases the previous line.
            sys.stdout.flush()
            str_bin = bin(number)
            str_bin = str_bin[2:]
            if (str_bin.count('1') == num_onsets):
                if no_double_stroke_flag: # 1000000010001
                    first_and_last_ones_bool = False
                    if (len(str_bin) > length - 2):
                        if str_bin[0] == '1' and str_bin[-1] == '1':
                            first_and_last_ones_bool = True
                        elif str_bin[0:2] == '01' and str_bin[-1] == '1':
                            first_and_last_ones_bool = True
                        elif str_bin[0] == '1' and str_bin[-2:] == '10':
                            first_and_last_ones_bool = True
                        else:
                            first_and_last_ones_bool = False
                    if (str_bin.count('1') == num_onsets) and (not('11' in str_bin) and not('101' in str_bin) and not(first_and_last_ones_bool)):
                        bin_strs.append(str_bin)
                else:
                    bin_strs.append(str_bin)
            else:
                pass

    else: 
        for number in rhyth_number_out:
            counter += 1
            sys.stdout.write("\r" + f" sorting rhythm {counter} of {num_rhyths}")   #The \r is carriage return without a line 
                                        #feed, so it erases the previous line.
            sys.stdout.flush()
            str_bin = bin(number)
            str_bin = str_bin[2:]
            if no_double_stroke_flag: # 1000000010001
                first_and_last_ones_bool = False
                if (len(str_bin) > length - 2):
                    if str_bin[0] == '1' and str_bin[-1] == '1':
                        first_and_last_ones_bool = True
                    elif str_bin[0:1] == '01' and str_bin[-1] == '1':
                        first_and_last_ones_bool = True
                    elif (str_bin[0] == '1') and (str_bin[-2:-1] == '10'):
                        first_and_last_ones_bool = True
                    else:
                        pass
                if not('11' in str_bin) and not('101' in str_bin) and not(first_and_last_ones_bool):
                    bin_strs.append(str_bin)
            else:
                bin_strs.append(str_bin)
    return bin_strs


def rhythm_generator(length, attempts, model_score_ioi_ent_flag, no_double_stroke_flag, num_onsets):
    bin_list = random_rhythm(length, attempts, no_double_stroke_flag, num_onsets)
    print("rhythms generated")
    scores = []
    ioi_ents = []
    for i in range(len(bin_list)):
        sys.stdout.write("\r" + f" scoring generated rhythm {i} of {len(bin_list)}")   #The \r is carriage return without a line 
                                        #feed, so it erases the previous line.
        sys.stdout.flush()

        if len(bin_list[i]) < length:
            bin_list[i] = '0'*(length-len(bin_list[i])) + bin_list[i]
        no_ioi_hier_model_score = hierarchical_model(bin_list[i], ioi_ent_flag=model_score_ioi_ent_flag)
        scores.append(round(no_ioi_hier_model_score, 5)) # rounding to get slightly different scores/unique rhythms/not rearranged versions of each other.
        ioi_ents.append(ioi_ent(bin_list[i]))

    return bin_list, scores, ioi_ents 

    
def histo_scores(scores):
    # q25, q75 = np.percentile(gt_intervals, [25, 75])
    # bin_width = 2 * (q75 - q25) * len(gt_intervals) ** (-1/3)
    # bins = round((np.max(gt_intervals) - np.min(gt_intervals)) / bin_width)
    fig, ax = plt.subplots()
    ax.hist(scores, bins=40)  # density=False would make counts
    ax.set_ylabel("count")
    ax.set_xlabel('Hierarch Model Score')
    ax.set_title("Score Frequency")
    return


def get_closest_n_rhythms(total_binary_list, corresponding_scores, corresponding_ents, n, target_score, unique_flag):
    if unique_flag:
        unique_scores = set(corresponding_scores)
        unique_score_rhythms = []
        unique_score_ents = []
        for i in unique_scores:
            ind = corresponding_scores.index(i)
            unique_score_rhythms.append(total_binary_list[ind])
            unique_score_ents.append(corresponding_ents[ind])
        total_binary_list = unique_score_rhythms
        corresponding_scores = unique_scores
        corresponding_ents = unique_score_ents


    distance = [abs(score - target_score) for score in corresponding_scores]
    sorted_rhythms = [x for _,x in sorted(zip(distance, total_binary_list))]
    sorted_scores = [x for _,x in sorted(zip(distance, corresponding_scores))]
    sorted_ents = [x for _,x in sorted(zip(distance, corresponding_ents))]
    return sorted_rhythms[0:n], sorted_scores[0:n], sorted_ents[0:n]

    
if __name__ == '__main__':
    hiers = []
    pos = []
    leng = []

    for rhythm_ind in range(len(ems_constants.rhythm_strings)):
        rhythm = ems_constants.rhythm_strings[rhythm_ind]
        normed_cross_ent_hier = hierarchical_model(rhythm, ioi_ent_flag=1)
        hiers.append(normed_cross_ent_hier)
        normed_cross_ent_position = fine_grained_met(rhythm)
        pos.append(normed_cross_ent_position)
        leng.append(len(rhythm))
        print(f"rhythm: {ems_constants.rhythm_strings_names[rhythm_ind]} \n hierarchical ent: {normed_cross_ent_hier} \n metrical pos ent: {normed_cross_ent_position} \n \n")
    
    
    
    bar_plot_scores(hiers, "Hierarchical model cross entropy scores by rhythm")
    bar_plot_scores(leng, "Rhythm Length")
    # bar_plot_scores(pos, "Position model cross entropy scores by rhythm")
    plt.tight_layout()
    x = 0

    length = 16
    attempts = 65500

    model_score_ioi_ent_flag = 1 # if yes, include ioi entropy into model
    bin_list, scores, ioi_ents = rhythm_generator(length, attempts, model_score_ioi_ent_flag, no_double_stroke_flag=1, num_onsets=0)
    num_onsets = [rhythm.count('1') for rhythm in bin_list]
    if model_score_ioi_ent_flag:
        for j in range(5, 17, 2):
            rhythms, scores_out, _ = get_closest_n_rhythms(bin_list, scores, ioi_ents, n=3, target_score=j, unique_flag=1)
            for i in range(len(rhythms)):
                print(f"Rhythm: {rhythms[i]}, score: {scores_out[i]}")
    else:
        for j in range(8, 20, 2):
            rhythms, scores_out, ioi_ents_out = get_closest_n_rhythms(bin_list, scores, ioi_ents, n=3, target_score=j, unique_flag=1)
            for i in range(len(rhythms)):
                print(f"Rhythm: {rhythms[i]}, score: {scores_out[i]}, ioi_ent: {ioi_ents_out[i]}")

    histo_scores(scores)

    fig, ax = plt.subplots()
    ax.set_title("Num Onsets vs Hierarchical Model Score")
    ax.set_ylabel("Num Onsets")
    ax.set_xlabel("Score")
    ax.scatter(scores, num_onsets)

    x = 2
