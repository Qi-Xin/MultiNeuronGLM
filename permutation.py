import sys
if sys.platform == 'linux':
    sys.path.append("/home/qix/MultiNeuronGLM")
else:
    sys.path.append("D:/Github/MultiNeuronGLM")
    
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import utility_functions as utils
import GLM
from DataLoader import Allen_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
# sns.set_theme()
sns.set_theme(style="white")
# sns.set_style('whitegrid')

# Load selected group_id
import pickle
with open('group_id_all_a_c/membership.pickle', 'rb') as handle:
    membership = pickle.load(handle)
with open('group_id_all_a_c/condition_ids.pickle', 'rb') as handle:
    condition_ids = pickle.load(handle)

if __name__ == "__main__":
    # freeze_support()
    # Load data
    start_time = 0.0
    end_time = 0.50
    padding = 0.3
    V1 = Allen_dataset(fps=1000,
                    start_time=start_time, 
                    end_time=end_time,
                    padding=padding,
    #                    orientation=[0],
                    session_id=757216464,
                    selected_probes=['probeA', 'probeB', 'probeC', 'probeD', 'probeE', 'probeF'],
    #                    temporal_frequency=[1,2,4],
                    stimulus_condition_id=[275, 277, 246, 255, 272, 248, 283, 266, 274, 276, 286, 271, 268, 270],
    #                    stimulus_condition_id=[246, 247, 248, 249, 250, 251, 252, 253, 255, 256, 257, 258, 259, 260, 261, 
    #                                           262, 263, 264, 265, 266, 267, 268, 269, 271, 272, 273, 274, 275, 276, 277,
    #                                           278, 279, 280, 281, 282, 283, 284, 285, 286, 270],
                    stimulus_name='drifting_gratings')

    # V1.get_lfp()
    # V1.remove_padding(padding)
    V1.get_trial_metric_per_unit_per_trial()
    # V1.get_trial_metric_per_unit_per_trial(metric_type='spike_times')
    V1.get_running(method="mine")

    # # The following hyperparameters turned out to be the best
    # num_f_refractory = 4
    # max_iter = 5
    # tau = 15
    # coupling_filter_params = {'peaks_max':15.2, 'num':3, 'nonlinear':0.5}
    # num_basis_baseline = 20
    # penalty = 3e-1

    # ################ No need to change below
    # probe_list = V1.selected_probes
    # running_filter = {}
    # stationary_filter = {}
    # running_output = {}
    # stationary_output = {}

    # ROI_filter = {}
    # statistics_filter = {}
    # ROI_output = {}
    # statistics_output = {}

    # running_model_list = []
    # stationary_model_list = []

    # for i, target_probe in enumerate(probe_list):
    #     select_trials = V1.running_trial_index
    #     model = GLM.PP_GLM(dataset=V1, 
    #                        select_trials=select_trials, 
    #                        membership=membership, 
    #                        condition_ids=condition_ids)
    #     model.add_effect('inhomogeneous_baseline', num=num_basis_baseline, apply_no_penalty=True)
    #     for j, input_probe in enumerate(probe_list):
    #         model.add_effect('coupling', probe_list[j], apply_no_penalty=True, **coupling_filter_params)
    #     model.add_effect('refractory_additive', target_probe, tau=tau, num=num_f_refractory, apply_no_penalty=True)
    #     model.add_effect('trial_coef')
    #     model.fit_time_warping_baseline(target_probe, verbose=False, max_iter=max_iter, penalty=penalty)
    #     running_model_list.append(model)
        
    #     filter_list = model.get_filter(ci=True)
    #     for j in range(len(model.basis_list)):
    #         running_filter[i,j-1] = filter_list[j]
    #     output_list = model.get_filter_output(ci=True)
    #     for j in range(len(model.basis_list)):
    #         running_output[i,j-1] = output_list[j]
        
        
    #     select_trials = V1.stationary_trial_index
    #     model = GLM.PP_GLM(dataset=V1, 
    #                        select_trials=select_trials, 
    #                        membership=membership, 
    #                        condition_ids=condition_ids)
    #     model.add_effect('inhomogeneous_baseline', num=num_basis_baseline, apply_no_penalty=True)
    #     for j, input_probe in enumerate(probe_list):
    #         model.add_effect('coupling', probe_list[j],apply_no_penalty=True, **coupling_filter_params)
    #     model.add_effect('refractory_additive', target_probe, tau=tau, num=num_f_refractory, apply_no_penalty=True)
    #     model.add_effect('trial_coef')
    #     model.fit_time_warping_baseline(target_probe, verbose=False, max_iter=max_iter, penalty=penalty)
    #     stationary_model_list.append(model)
        
    #     filter_list = model.get_filter(ci=True)
    #     for j in range(len(model.basis_list)):
    #         stationary_filter[i,j-1] = filter_list[j]
    #     output_list = model.get_filter_output(ci=True)
    #     for j in range(len(model.basis_list)):
    #         stationary_output[i,j-1] = output_list[j]
        
    #     # for effect filter
    #     filter_index = i,-1
    #     function1 = np.exp( running_filter[filter_index][0] )
    #     function2 = np.exp( stationary_filter[filter_index][0] )
    #     ROI_filter[filter_index], statistics_filter[filter_index] = GLM.get_excursion_statistic(function1, function2)
    #     for j, input_probe in enumerate(probe_list):
    #         filter_index = i,j
    #         function1 = running_filter[filter_index][0]
    #         function2 = stationary_filter[filter_index][0]
    #         ROI_filter[filter_index], statistics_filter[filter_index] = GLM.get_excursion_statistic(function1, function2)
        
    #     # for effect output
    #     filter_index = i,-1
    #     function1 = np.exp( running_output[filter_index][0] )
    #     function2 = np.exp( stationary_output[filter_index][0] )
    #     ROI_output[filter_index], statistics_output[filter_index] = GLM.get_excursion_statistic(function1, function2)
    #     for j, input_probe in enumerate(probe_list):
    #         filter_index = i,j
    #         function1 = running_output[filter_index][0]
    #         function2 = stationary_output[filter_index][0]
    #         ROI_output[filter_index], statistics_output[filter_index] = GLM.get_excursion_statistic(function1, function2)

    # aic = np.sum([model.aic for model in stationary_model_list]+[model.aic for model in running_model_list])
    # baseline_aic = 297548.29185407807
    # print(f"AIC improvement of the model is: {2*(baseline_aic - aic)}")

    # WITHOUT fixed peak times
    from GLM import get_statistics_null_mp
    import time
    import socket

    i = 0
    while True:
        
        statistics_null_filter_new, statistics_null_output_new = get_statistics_null_mp(50, V1, membership, condition_ids, fix_peak_time=None)
        if i==0:
            # statistics_null_filter = statistics_null_filter_new
            # statistics_null_output = statistics_null_output_new
            with open('statistics_null_filter_'+socket.gethostname()[:7]+'.pickle', 'rb') as handle:
                statistics_null_filter = pickle.load(handle)
            with open('statistics_null_output_'+socket.gethostname()[:7]+'.pickle', 'rb') as handle:
                statistics_null_output = pickle.load(handle)
        else:
            statistics_null_filter = GLM.merge_dict(statistics_null_filter, statistics_null_filter_new)
            statistics_null_output = GLM.merge_dict(statistics_null_output, statistics_null_output_new)
        with open('statistics_null_filter_'+socket.gethostname()[:7]+'.pickle', 'wb') as handle:
            pickle.dump(statistics_null_filter, handle)
        with open('statistics_null_output_'+socket.gethostname()[:7]+'.pickle', 'wb') as handle:
            pickle.dump(statistics_null_output, handle)
        print(f"Finishing {(i+1)*50} permutation. (Total: {len(statistics_null_filter[0,0])})")
        
        i += 1
        # if time.localtime().tm_hour >= 8:
        if False:
            break
    