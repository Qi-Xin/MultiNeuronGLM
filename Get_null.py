import sys
if sys.platform == 'linux':
    sys.path.append("/home/qix/MultiNeuronGLM")
else:
    sys.path.append("D:/Github/MultiNeuronGLM")
    
import pandas as pd
import utility_functions as utils
import GLM
from DataLoader import Allen_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
sns.set_theme()
import copy

def merge_dict(d1, d2):
    # d1 is the mother, d2 is the one to add to d1
    d = {}
    for k in d1.keys():
        d[k] = d1[k] + d2[k]
    return d

def get_excursion_test(function1, function2, ROI_list):
    stats_list = []
    for i, ROI in enumerate(ROI_list):
        diff = function1 - function2
        stats_list.append( np.abs( np.sum(diff[ROI]) ) )
    return np.max(stats_list)

def get_ROI(function1, function2):
    diff = np.abs(function1 - function2)
    threshold = diff.max()/2
    idx = np.where(diff >= threshold)[0]
    return np.split(idx, np.where(np.diff(idx) != 1)[0]+1)

def get_statistics_null(V1, membership, condition_ids, probe_list, num_basis_baseline):
    statistics_null = {}   # dict to return
    ROI = {}
    # Get permutated running and stationary index
    fake_running_trial_index = copy.deepcopy(V1.running_trial_index)
    np.random.shuffle(fake_running_trial_index)  
        # don't need to assign shuffled list, it's changed automatically. 
    fake_stationary_trial_index = np.logical_not(fake_running_trial_index)
    
    running_filter_temp = {}
    stationary_filter_temp = {}
    for i, target_probe in enumerate(probe_list):
        select_trials = fake_running_trial_index
        model = GLM.PP_GLM(dataset=V1, 
                           select_trials=select_trials, 
                           membership=membership, 
                           condition_ids=condition_ids)
        model.add_effect('inhomogeneous_baseline', num=num_basis_baseline)
        for j, input_probe in enumerate(probe_list):
            if i==j:
                continue
            model.add_effect('coupling', probe_list[j], peaks_max=100, num=5, nonlinear=0.3)
        model.fit(probe_list[i], verbose=False)
        filter_list = model.get_filter(ci=True)
        running_filter_temp[i,-1] = filter_list[0]
        k = 1
        for j, input_probe in enumerate(probe_list):
            if i==j:
                continue
            running_filter_temp[i,j] = filter_list[k]
            k += 1

        select_trials = fake_stationary_trial_index
        model = GLM.PP_GLM(dataset=V1, 
                           select_trials=select_trials, 
                           membership=membership, 
                           condition_ids=condition_ids)
        model.add_effect('inhomogeneous_baseline', num=num_basis_baseline)
        for j, input_probe in enumerate(probe_list):
            if i==j:
                continue
            model.add_effect('coupling', probe_list[j], peaks_max=100, num=5, nonlinear=0.3)
        model.fit(probe_list[i], verbose=False)
        filter_list = model.get_filter(ci=True)
        stationary_filter_temp[i,-1] = filter_list[0]
        k = 1
        for j, input_probe in enumerate(probe_list):
            if i==j:
                continue
            stationary_filter_temp[i,j] = filter_list[k]
            k += 1

        filter_index = i,-1
        function1 = np.exp( running_filter_temp[filter_index][0] )
        function2 = np.exp( stationary_filter_temp[filter_index][0] )
        if filter_index not in statistics_null.keys():
            statistics_null[filter_index] = []
        ROI[filter_index] = get_ROI(function1, function2)
        statistics_null[filter_index].append( get_excursion_test(function1, function2, ROI[filter_index]) )
        
        for j, input_probe in enumerate(probe_list):
            if i==j:
                continue
            filter_index = i,j
            function1 = running_filter_temp[filter_index][0]
            function2 = stationary_filter_temp[filter_index][0]
            if filter_index not in statistics_null.keys():
                statistics_null[filter_index] = []
            ROI[filter_index] = get_ROI(function1, function2)
            statistics_null[filter_index].append( get_excursion_test(function1, function2, ROI[filter_index]) )
    return statistics_null


# Load selected group_id
import pickle
with open('group_id_all_a_c/membership.pickle', 'rb') as handle:
    membership = pickle.load(handle)
with open('group_id_all_a_c/condition_ids.pickle', 'rb') as handle:
    condition_ids = pickle.load(handle)
    
# Load LFP data
start_time = 0.0
end_time = 0.50
padding = 0.1
V1 = Allen_dataset(fps=1000,
                   start_time=start_time, 
                   end_time=end_time,
                   padding=padding,
#                    orientation=[0],
                   session_id=757216464,
                   selected_probes=['probeA', 'probeB', 'probeC', 'probeD', 'probeE', 'probeF'],
#                    temporal_frequency=[1,2,4],
                   stimulus_condition_id=[275, 277, 246, 255, 272, 248, 283, 266, 274, 276, 286, 271, 268, 270],
                   stimulus_name='drifting_gratings')

# V1.get_lfp()
# V1.remove_padding(padding)
V1.get_trial_metric_per_unit_per_trial()
# V1.get_trial_metric_per_unit_per_trial(metric_type='spike_times')
V1.get_running(method="mine")



num_basis_baseline = 30

probe_list = V1.selected_probes
running_filter = {}
stationary_filter = {}
ROI = {}
statistics = {}

for i, target_probe in enumerate(probe_list):
    select_trials = V1.running_trial_index
    model = GLM.PP_GLM(dataset=V1, 
                       select_trials=select_trials, 
                       membership=membership, 
                       condition_ids=condition_ids)
    model.add_effect('inhomogeneous_baseline', num=num_basis_baseline)
    for j, input_probe in enumerate(probe_list):
        if i==j:
            continue
        model.add_effect('coupling', probe_list[j], peaks_max=100, num=5, nonlinear=0.3)
    model.fit(probe_list[i], verbose=False)
    filter_list = model.get_filter(ci=True)
    running_filter[i,-1] = filter_list[0]
    k = 1
    for j, input_probe in enumerate(probe_list):
        if i==j:
            continue
        running_filter[i,j] = filter_list[k]
        k += 1
    
    select_trials = V1.stationary_trial_index
    model = GLM.PP_GLM(dataset=V1, 
                       select_trials=select_trials, 
                       membership=membership, 
                       condition_ids=condition_ids)
    model.add_effect('inhomogeneous_baseline', num=num_basis_baseline)
    for j, input_probe in enumerate(probe_list):
        if i==j:
            continue
        model.add_effect('coupling', probe_list[j], peaks_max=100, num=5, nonlinear=0.3)
    model.fit(probe_list[i], verbose=False)
    filter_list = model.get_filter(ci=True)
    stationary_filter[i,-1] = filter_list[0]
    k = 1
    for j, input_probe in enumerate(probe_list):
        if i==j:
            continue
        stationary_filter[i,j] = filter_list[k]
        k += 1
    
    filter_index = i,-1
    function1 = np.exp( running_filter[filter_index][0] )
    function2 = np.exp( stationary_filter[filter_index][0] )
    ROI[filter_index] = get_ROI(function1, function2)
    statistics[filter_index] = get_excursion_test(function1, function2, ROI[filter_index])
    for j, input_probe in enumerate(probe_list):
        if i==j:
            continue
        filter_index = i,j
        function1 = running_filter[filter_index][0]
        function2 = stationary_filter[filter_index][0]
        ROI[filter_index] = get_ROI(function1, function2)
        statistics[filter_index] = get_excursion_test(function1, function2, ROI[filter_index])




import multiprocessing
import os
# PROCESSES = os.cpu_count()-20
PROCESSES = 4
from GLM import get_statistics_null

p = multiprocessing.Process(target=get_statistics_null, args=(V1, membership, condition_ids, probe_list, num_basis_baseline))
# p = Process(target=get_statistics_null)
p.start()
p.join()

# with multiprocessing.Pool(processes = PROCESSES) as pool:
#     results = [pool.apply_async(get_statistics_null, (V1, membership, condition_ids, probe_list, num_basis_baseline)) 
#             for i_null in [0,1]]
#     pool.close()
    
#     statistics_null = results[0].get()
#     print("done!")
#     for result in tqdm(results[1:]):
#         statistics_null_new = result.get()
#         statistics_null = merge_dict(statistics_null, statistics_null_new)
        
# with open('null.pickle', 'wb') as handle:
#     pickle.dump( statistics_null, handle )


