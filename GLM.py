#%% import
from curses import raw
from multiprocessing.spawn import old_main_modules
import os
from tkinter import N, Menu
from urllib.parse import ParseResultBytes

from absl import logging
import collections
from collections import defaultdict
import io
import itertools
import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import pandas as pd
from psycopg2 import paramstyle
from regex import D
# from pyrsistent import m
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
import scipy.interpolate 
import scipy.signal
from scipy import linalg
from sklearn.linear_model import PoissonRegressor
import sklearn.model_selection
from tqdm import tqdm
import sys
import numpy as np
import copy
# import numpy.matlib
from numpy.fft import fft as fft
from numpy.fft import ifft as ifft
import scipy.stats
import copy
from scipy.special import rel_entr

import statsmodels.api as sm
import statsmodels.genmod.generalized_linear_model as smm

import utility_functions as utils

#%% PP_GLM class
class PP_GLM():
    def __init__(self, 
                 dataset=None, 
                 select_trials=None, 
                 membership=None, 
                 condition_ids=None, 
                 nt=None, 
                 ntrial=None,
                 npadding=None):
        """Initialize PP_GLM 

        Args:
            dataset (Allendataset, optional): you can input a dataset for easier use. Defaults to None.
            select_trials (array of Boole, optional): if you use dataset as input, you can specify running trials to use. Defaults to None.
            nt (int, optional): number of time bins, if don't import dataset directly. Defaults to None.
            ntrial (int, optional): number of trials, if don't import dataset directly. Defaults to None.
        """
        if dataset is None:
            self.nt = nt
            self.ntrial = ntrial
            self.select_trials = np.arange(self.ntrial)
            self.npadding = npadding
        else:
            self.dataset = dataset
            self.nt = self.dataset.nt
            
            if select_trials is None:
                self.select_trials = np.full(dataset.ntrial, True)
            else:
                self.select_trials = select_trials
            if type(select_trials[0]) == np.bool_:
                self.ntrial = self.select_trials.sum()
            else:
                self.ntrial = self.select_trials.shape[0]
            self.membership = membership
            self.condition_ids = condition_ids
            self.npadding = self.dataset.npadding
        self.effect_list = []
        self.basis_list = []
        self.basis_name = []
        self.effect_type_list = []
        self.raw_input_list = []
        self.kwargs_list = []
        self.target = None
        self.no_penalty = []

    def add_effect(self, effect_type, raw_input=None, use_all=False, apply_no_penalty=False, **kwargs):
        assert effect_type in ['homogeneous_baseline', 
                        'inhomogeneous_baseline', 
                        'coupling', 
                        'twoway_coupling', 
                        'circular', 
                        'linear',
                        'history', 
                        'varying_linear',
                        'dense_coupling',
                        'refractory',
                        'refractory_box',
                        'trial_coef', 
                        'condition_coef', 
                        'interaction', 
                        'interaction2', 
                        'interaction3'],  "Not supported effect_type!"

        # record for later use
        self.effect_type_list.append(effect_type)
        self.raw_input_list.append(raw_input)
        self.kwargs_list.append(kwargs)
        
        if type(raw_input) == np.ndarray:
            if raw_input.shape[1] > self.ntrial:
                raw_input = raw_input[:, self.select_trials]
            
        if effect_type == 'homogeneous_baseline':
            X_baseline = np.ones((self.nt*self.ntrial,1))
            self.effect_list.append(X_baseline)
            self.basis_list.append(X_baseline[0:self.nt,:])
            self.basis_name.append(effect_type)
            
        elif effect_type == 'inhomogeneous_baseline':
            X_baseline = inhomo_baseline(ntrial=self.ntrial, 
                                        start=0,
                                        end=self.nt,
                                        dt=1, 
                                        **kwargs)
            self.effect_list.append(X_baseline)
            self.basis_list.append(X_baseline[0:self.nt,:])
            self.basis_name.append(effect_type)
            
        elif effect_type == 'coupling':
            if type(raw_input) == str:
                # print(f"Assuming raw inputs are spike trains from {raw_input}")
                input_to_couple = utils.pooling_pop(self.membership, self.condition_ids, 
                                                    self.dataset, raw_input, 0, use_all=use_all)
                input_to_couple = input_to_couple[:,self.select_trials]
            elif type(raw_input) == np.ndarray:
                input_to_couple = raw_input
            else:
                raise ValueError("raw input must be either str like \"probeC\" or numpy.ndarray!")
            pillow_basis = make_pillow_basis(**kwargs)
            X_coupling = conv(input_to_couple, pillow_basis, npadding=self.npadding)
            self.effect_list.append(X_coupling)
            
            self.basis_list.append(pillow_basis)
            
            if type(raw_input) == str:
                self.basis_name.append(effect_type+" from "+utils.PROBE_CORRESPONDING[raw_input])
            else:
                self.basis_name.append(effect_type)
        
        elif effect_type == 'interaction':
            # Interaction term here is an interaction term of coupling effect and current spike count in a window
            # The term can make the coupling effect of two spikes smaller than twice the coupling effect of one spike. 
            if type(raw_input) == str:
                # print(f"Assuming raw inputs are spike trains from {raw_input}")
                input_to_couple = utils.pooling_pop(self.membership, self.condition_ids, 
                                                    self.dataset, raw_input, 0, use_all=use_all)
                input_to_couple = input_to_couple[:,self.select_trials]
            elif type(raw_input) == np.ndarray:
                input_to_couple = raw_input
            else:
                raise ValueError("raw input must be either str like \"probeC\" or numpy.ndarray!")
            
            tau = kwargs.pop('tau', 100)
            order = kwargs.pop('order', 1)
            
            refractory_spikes = np.zeros_like(input_to_couple)
            temp = refractory_spikes[0, :]
            for t in range(1, input_to_couple.shape[0]):
                temp *= np.exp(-1000.0/self.dataset.fps/tau)
                refractory_spikes[t, :] = temp
                temp += input_to_couple[t, :]
            
            refractory_spikes = refractory_spikes[-self.nt:, :]
            X_refractory = refractory_spikes.flatten('F')[:, np.newaxis]
            X_refractory /= tau
            
            pillow_basis = make_pillow_basis(**kwargs)
            X_coupling = conv(input_to_couple, pillow_basis, npadding=self.npadding)
            self.basis_list.append(pillow_basis)
            
            self.effect_list.append(X_coupling*X_refractory**order)
            
            if type(raw_input) == str:
                self.basis_name.append(effect_type+" from "+utils.PROBE_CORRESPONDING[raw_input])
            else:
                self.basis_name.append(effect_type)
                
        elif effect_type == 'dense_coupling':
            if type(raw_input) == str:
                # print(f"Assuming raw inputs are spike trains from {raw_input}")
                input_to_couple = utils.pooling_pop(self.membership, self.condition_ids, 
                                                    self.dataset, raw_input, 0, use_all=use_all)
                input_to_couple = input_to_couple[:,self.select_trials]
            elif type(raw_input) == np.ndarray:
                input_to_couple = raw_input
            else:
                raise ValueError("raw input must be either str like \"probeC\" or numpy.ndarray!")
            num = kwargs.pop('num',10)
            pillow_basis = np.diag(np.ones(num))
            X_coupling = conv(input_to_couple, pillow_basis, npadding=self.npadding)
            self.effect_list.append(X_coupling)
            
            self.basis_list.append(pillow_basis)
            
            if type(raw_input) == str:
                self.basis_name.append(effect_type+" from "+utils.PROBE_CORRESPONDING[raw_input])
            else:
                self.basis_name.append(effect_type)
        
        elif effect_type == 'refractory':
            if type(raw_input) == str:
                # print(f"Assuming raw inputs are spike trains from {raw_input}")
                input_to_couple = utils.pooling_pop(self.membership, self.condition_ids, 
                                                    self.dataset, raw_input, 0, use_all=use_all)
                input_to_couple = input_to_couple[:,self.select_trials]
            elif type(raw_input) == np.ndarray:
                input_to_couple = raw_input
            else:
                raise ValueError("raw input must be either str like \"probeC\" or numpy.ndarray!")
            tau = kwargs.pop('tau',100)
            order = kwargs.pop('order', 2)
            refractory_spikes = np.zeros_like(input_to_couple)
            temp = refractory_spikes[0, :]
            for t in range(1, input_to_couple.shape[0]):
                temp *= np.exp(-1000.0/self.dataset.fps/tau)
                refractory_spikes[t, :] = temp
                temp += input_to_couple[t, :]
            
            refractory_spikes = refractory_spikes[-self.nt:, :]
            X_refractory = refractory_spikes.flatten('F')[:, np.newaxis]
            X_refractory /= tau
            self.effect_list.append(X_refractory**order)
            self.basis_list.append(refractory_spikes.mean(axis=1)[:, np.newaxis])
            self.basis_name.append(effect_type)
            
        elif effect_type == 'refractory_box':
            if type(raw_input) == str:
                # print(f"Assuming raw inputs are spike trains from {raw_input}")
                input_to_couple = utils.pooling_pop(self.membership, self.condition_ids, 
                                                    self.dataset, raw_input, 0, use_all=use_all)
                input_to_couple = input_to_couple[:,self.select_trials]
            elif type(raw_input) == np.ndarray:
                input_to_couple = raw_input
            else:
                raise ValueError("raw input must be either str like \"probeC\" or numpy.ndarray!")
            tau = kwargs.pop('tau',100)
            tau = int(np.ceil(tau))
            order = kwargs.pop('order', 2)
            refractory_spikes = np.zeros_like(input_to_couple)
            for t in range(tau, input_to_couple.shape[0]):
                refractory_spikes[t, :] = input_to_couple[t-tau:t, :].sum(axis=0)
            refractory_spikes = refractory_spikes[-self.nt:, :]
            X_refractory = refractory_spikes.flatten('F')[:, np.newaxis]
            X_refractory /= tau
            self.effect_list.append(X_refractory**order)
            self.basis_list.append(refractory_spikes.mean(axis=1)[:, np.newaxis])
            self.basis_name.append(effect_type)
            
        elif effect_type == 'trial_coef':
            X_trial_coef = np.zeros((self.nt*self.ntrial, self.ntrial))
            for itrial in range(self.ntrial):
                X_trial_coef[(itrial*self.nt):((itrial+1)*self.nt), itrial] = 1
            self.effect_list.append(X_trial_coef)
            self.basis_list.append(np.diag(np.ones(self.ntrial)))
            self.basis_name.append(effect_type)

        elif effect_type == 'condition_coef':
            condition_list = self.dataset.presentation_table['stimulus_condition_id']
            condition_ids_np = np.array(self.condition_ids)
            X_condition_coef = np.zeros((self.nt*self.ntrial, len(self.condition_ids)))
            for itrial in range(self.ntrial):
                trial = self.dataset.spike_train.columns[itrial]
                current_condition = condition_list.loc[trial]
                icondition = np.where(condition_ids_np==current_condition)[0][0]
                X_condition_coef[(itrial*self.nt):((itrial+1)*self.nt), icondition] = 1
            X_condition_coef = X_condition_coef[:, X_condition_coef.sum(axis=0)!=0]
            self.effect_list.append(X_condition_coef)
            self.basis_list.append(np.diag( np.ones(X_condition_coef.shape[1]) ))
            self.basis_name.append(effect_type)
            
        elif effect_type == 'twoway_coupling':
            if type(raw_input) == str:
                # print(f"Assuming raw inputs are spike trains from {raw_input}")
                input_to_couple = utils.pooling_pop(self.membership, self.condition_ids, 
                                                    self.dataset, raw_input, 0, use_all=use_all)
                input_to_couple = input_to_couple[:,self.select_trials]
                input_to_couple = input_to_couple[self.npadding:, :]
            elif type(raw_input) == np.ndarray:
                input_to_couple = raw_input
            else:
                raise ValueError("raw input must be either str like \"probeC\" or numpy.ndarray!")
            pillow_basis = make_pillow_basis(**kwargs)
            X_speed_pos = conv(input_to_couple, pillow_basis, enforce_causality=True, npadding=None)
            X_speed_neg = conv_flip(input_to_couple, pillow_basis, enforce_causality=False, npadding=None)
            X_speed_pos[:,0] += X_speed_neg[:,0]
            X_speed_neg = X_speed_neg[:,1:]
            self.effect_list.append(np.hstack((X_speed_pos,X_speed_neg)))
            
            lbasis = pillow_basis.shape[0]
            temp = np.zeros((2*lbasis+1))
            temp[lbasis+1] = 1
            basis_pos = conv(temp, pillow_basis, enforce_causality=True)
            basis_neg = conv_flip(temp, pillow_basis, enforce_causality=False)
            basis_pos[:,0] += basis_neg[:,0]
            basis_neg = basis_neg[:,1:]
            pillow_basis_twoway = np.hstack((basis_pos, basis_neg))
            self.basis_list.append(pillow_basis_twoway)
            
            if type(raw_input) == str:
                self.basis_name.append(effect_type+" from "+utils.PROBE_CORRESPONDING[raw_input]+" cross-pop")
            else:
                self.basis_name.append(effect_type)
        
        elif effect_type == 'circular':
            assert type(raw_input)==np.ndarray , "Circular effects should be from LFP phase!" 
            raise ValueError("Unfinish!")
        
        elif effect_type == 'linear':
            assert type(raw_input)==np.ndarray , "Linear effects should be from instantaneous speed!" 
            X_linear = raw_input.flatten('F')[:,np.newaxis]
            self.effect_list.append(X_linear)
            self.basis_list.append(np.ones((1,1)))
            self.basis_name.append(effect_type)
            
        elif effect_type == 'varying_linear':
            assert type(raw_input)==np.ndarray , "Varying_linear effects should be from instantaneous speed!" 
            coef_basis = inhomo_baseline(ntrial=1, 
                                         start=0,
                                         end=self.nt,
                                         dt=1, 
                                         **kwargs)
            X_varying_linear = np.zeros((self.ntrial*self.nt, coef_basis.shape[1]))
            for i in range(coef_basis.shape[1]):
                single_coef_basis = coef_basis[:,i]
                single_coef_basis = single_coef_basis[:,np.newaxis]
                X_varying_linear[:,i] = (single_coef_basis*raw_input).flatten('F')

            self.effect_list.append(X_varying_linear)
            self.basis_list.append(coef_basis)
            self.basis_name.append(effect_type)
    
        # if apply no penalty to this effect
        if apply_no_penalty == True:
            no_penalty_start = np.sum([self.basis_list[i].shape[1] for i in range(len(self.basis_list)-1)])
            no_penalty_end = np.sum([self.basis_list[i].shape[1] for i in range(len(self.basis_list))])
            self.no_penalty.append(np.arange(int(no_penalty_start), int(no_penalty_end)))

    def fit(self, target, use_all=False, verbose=True, penalty=1e-10, method='mine', max_spike=None):
        if self.target is None:
            self.target = target
            if type(target) == str:
                # print(f"Assuming output is spike trains from {target}")
                self.output = utils.pooling_pop(self.membership, self.condition_ids, 
                                        self.dataset, target, 0, use_all=use_all)
                self.output = self.output[:,self.select_trials]
            elif type(target) == np.ndarray:
                self.output = target
            else:
                raise ValueError("target must be either str like \"probeC\" or numpy.ndarray!")
            if self.npadding is not None:
                self.output = self.output[self.npadding:, :]
            self.response = self.output.flatten('F')
        
        self.predictors = np.hstack(self.effect_list)
        if penalty != 0 or method=='mine':
            self.results = poisson_regression(self.response, self.predictors, L2_pen=penalty, no_penalty=self.no_penalty)
        elif method=='logit':
            success_fail, max_spike = get_success_fail(self.response, max_spike=max_spike, return_max_spike=True)
            self.results = sm.GLM(success_fail, self.predictors, family=sm.families.Binomial()).fit()
        else:
            self.results = sm.GLM(self.response, self.predictors, family=sm.families.Poisson()).fit()
        self.max_spike = max_spike
        
        self.log_lmbd = (self.predictors@self.results.params).reshape((self.nt, self.ntrial), order='F')
        self.nll = spike_trains_neg_log_likelihood(self.log_lmbd, self.output, max_spike=self.max_spike)
        self.nll_trialwise = spike_trains_neg_log_likelihood(self.log_lmbd, self.output, trial_wise=True, max_spike=self.max_spike)
        self.aic = self.predictors.shape[1] + self.nll
        self.filters = self.get_filter(ci=False)
        if verbose:
            print(f"Negative log likelihood is: {self.nll :.2f}")
            print(f"aic/2 is: {self.aic :.2f}")
        return self.results
    
    def temporal_fit(self, target, use_all=False, verbose=True, penalty=1e-10, method='mine', max_spike=None):
        pass
    
    def deviance_test(self, verbose=False):
        # Not really useful with tens of thousands of column
        sat_nll = spike_trains_neg_log_likelihood( np.log(self.output+1e-6), self.output)
        dev = 2 * (self.nll - sat_nll)
        degree_freedom = self.predictors.shape[0] - self.predictors.shape[1]
        pvalue = 1 -  scipy.stats.chi2.cdf(dev, degree_freedom)
        if verbose:
            print(sat_nll, self.nll, degree_freedom)
            print(f"p-value for deviance test is:{pvalue}")
        return pvalue
    
    def get_filter(self, ci=False):
        ### say there are one inhomo baseline and three coupling filters, 
        ### result_filter[2] contains the information for the second coupling filters
        ### if ci==True, result_filter[2][0] is the filter, result_filter[2][1] is the ci
        ### if ci==False, result_filter[2] is the filter
        effect_id_list = np.arange(len(self.basis_name))
        result_filter = []
        for effect_id in effect_id_list:
            start_col = 0
            for previous_id in range(effect_id):
                start_col += (self.effect_list[previous_id]).shape[1]
            nbasis = (self.effect_list[effect_id]).shape[1]
            end_col = start_col + nbasis
            basis = self.basis_list[effect_id]
            # estimated filter
            coef = self.results.params[start_col:end_col]
            y = (basis@coef[:,np.newaxis]).squeeze()
            # ci
            se = self.results.bse[start_col:end_col]
            one_sigma_ci = (basis@se[:,np.newaxis]).squeeze()
            if ci:
                result_filter.append([y,one_sigma_ci])
            else:
                result_filter.append(y)
        return result_filter
    
    def get_filter_output(self, trial_wise=False, ci=False):
        ### say there are one inhomo baseline and three coupling filters, 
        ### result_output[2] contains the information for the second coupling filters
        ### if ci==True, result_output[2][0] is the filter, result_output[2][1] is the ci
        ### if ci==False, result_output[2] is the filter
        effect_id_list = np.arange(len(self.basis_name))
        result_output = []
        for effect_id in effect_id_list:
            start_col = 0
            for previous_id in range(effect_id):
                start_col += (self.effect_list[previous_id]).shape[1]
            nbasis = (self.effect_list[effect_id]).shape[1]
            end_col = start_col + nbasis

            # estimated effect output
            coef = self.results.params[start_col:end_col]
            if self.basis_name[effect_id] in ["inhomogeneous_baseline"]:
                predicters_temp = self.basis_list[effect_id]
                if trial_wise:
                    predicters_temp = np.tile(predicters_temp, (self.ntrial, 1))
                coef = self.results.params[start_col:end_col]
                y = (predicters_temp@coef[:,np.newaxis]).squeeze()
                se = self.results.bse[start_col:end_col]
                one_sigma_ci = (predicters_temp@se[:,np.newaxis]).squeeze()
                if trial_wise:
                    y_all_trial = y
            else:
                predicters_temp = self.effect_list[effect_id]
                coef = self.results.params[start_col:end_col]
                y_all_trial = (predicters_temp@coef[:,np.newaxis]).squeeze()
                y_mat = y_all_trial.reshape((self.nt, self.ntrial), order='F')
                y = y_mat.mean(axis=1)
                # one_sigma_ci = y_mat.std(axis=1)/np.sqrt(self.ntrial)
                one_sigma_ci = y_mat.std(axis=1)
            if ci:
                result_output.append([y,one_sigma_ci])
            else:
                if trial_wise:
                    result_output.append(y_all_trial)
                else:
                    result_output.append(y)
        return result_output
    
    def get_filter_contribution(self, time_range=None, auto_pick=None):
        if time_range is None:
            time_range = [self.dataset.start_time, self.dataset.end_time]
        time_range = [int(time*self.dataset.fps) for time in time_range]
        result_output = self.get_filter_output(ci=False)
        total_output = np.vstack((result_output)).T
        total_output = total_output.sum(axis=1)
        # print(total_output.shape)
        if auto_pick is None:
            total_info = get_three_measure_entire_length(total_output[time_range[0]:time_range[1]], exp=True)
            individual_info = []
            for i, output in enumerate(result_output):
                if self.effect_type_list[i] in ['inhomogeneous_baseline', 'coupling']:
                    minus_one_output = copy.deepcopy(total_output)
                    minus_one_output = minus_one_output - output + np.mean(output)
                    temp_info = get_three_measure_entire_length(minus_one_output[time_range[0]:time_range[1]], exp=True)
                    individual_info.append(total_info - temp_info)
            return individual_info
        else:
            measure = auto_pick
            individual_info = []
            for i, output in enumerate(result_output):
                if self.effect_type_list[i] in ['inhomogeneous_baseline', 'coupling']:
                    minus_one_output = copy.deepcopy(total_output)
                    minus_one_output = minus_one_output - output + np.mean(output)
                    best_time_range = get_best_time_range(total_output, minus_one_output, measure, time_range)
                    # print(f"{i}, {best_time_range}")
                    total_info = get_three_measure_entire_length(total_output[best_time_range[0]:best_time_range[1]], exp=True)
                    temp_info = get_three_measure_entire_length(minus_one_output[best_time_range[0]:best_time_range[1]], exp=True)
                    individual_info.append(total_info - temp_info)
            return individual_info

    def get_filter_contribution_trial_wise(self, time_range=None, auto_pick=None):
        if time_range is None:
            time_range = [self.dataset.start_time, self.dataset.end_time]
        time_range = [int(time*self.dataset.fps) for time in time_range]
        raw_result_output_list = self.get_filter_output(trial_wise=True, ci=False)
        result_output = []
        for ieffect, raw_result_output in enumerate(raw_result_output_list):
            if self.effect_type_list[ieffect]=='inhomogeneous_baseline': 
                result_output.append( raw_result_output.reshape((self.nt, self.ntrial), order='F') )
                ibaseline = ieffect
            elif self.effect_type_list[ieffect]=='coupling':
                result_output.append( raw_result_output.reshape((self.nt, self.ntrial), order='F') )
                if utils.PROBE_CORRESPONDING_INVERSE[self.basis_name[ieffect][-2:]] == self.target:
                    iselfeffect = ieffect
            elif self.effect_type_list[ieffect]=='trial_coef':
                # add refractory to itself's own effect
                result_output[ibaseline] += raw_result_output.reshape((self.nt, self.ntrial), order='F')
            elif self.effect_type_list[ieffect]=='refractory':
                result_output[iselfeffect] += raw_result_output.reshape((self.nt, self.ntrial), order='F')
            else:
                raise ValueError("only support trial_coef and refractory in addition to inhomogeneous_baseline and coupling!")
        total_output_all = np.stack(result_output, axis=2)
        total_output_all = total_output_all.sum(axis=2)
        # print(total_output.shape)
        # print(total_output_all.shape)
        individual_info_all = []
        for itrial in tqdm(range(total_output_all.shape[1])):
            total_output = total_output_all[:, itrial]
            if auto_pick is None:
                total_info = get_three_measure_entire_length(total_output[time_range[0]:time_range[1]], exp=True)
                individual_info = []
                for i, output in enumerate(result_output):
                    output = output[:, itrial]
                    if self.effect_type_list[i] in ['inhomogeneous_baseline', 'coupling']:
                        minus_one_output = copy.deepcopy(total_output)
                        minus_one_output = minus_one_output - output + np.mean(output)
                        temp_info = get_three_measure_entire_length(minus_one_output[time_range[0]:time_range[1]], exp=True)
                        individual_info.append(total_info - temp_info)
                individual_info_all.append(individual_info) 
            else:
                measure = auto_pick
                individual_info = []
                for i, output in enumerate(result_output):
                    output = output[:, itrial]
                    minus_one_output = copy.deepcopy(total_output)
                    minus_one_output = minus_one_output - output + np.mean(output)
                    best_time_range = get_best_time_range(total_output, minus_one_output, measure, time_range)
                    # print(f"{i}, {best_time_range}")
                    total_info = get_three_measure_entire_length(total_output[best_time_range[0]:best_time_range[1]], exp=True)
                    temp_info = get_three_measure_entire_length(minus_one_output[best_time_range[0]:best_time_range[1]], exp=True)
                    individual_info.append(total_info - temp_info)
                individual_info_all.append(individual_info) 
        return np.array(individual_info_all)
    
    def get_filter_output_merge(self, trial_wise=False, ci=False):
        if trial_wise:
            result_output = []
            raw_result_output_list = self.get_filter_output(trial_wise=True, ci=False)
            for ieffect, raw_result_output in enumerate(raw_result_output_list):
                if self.effect_type_list[ieffect]=='inhomogeneous_baseline': 
                    result_output.append( raw_result_output.reshape((self.nt, self.ntrial), order='F') )
                    ibaseline = ieffect
                elif self.effect_type_list[ieffect]=='coupling':
                    result_output.append( raw_result_output.reshape((self.nt, self.ntrial), order='F') )
                    if utils.PROBE_CORRESPONDING_INVERSE[self.basis_name[ieffect][-2:]] == self.target:
                        iselfeffect = ieffect
                elif self.effect_type_list[ieffect]=='trial_coef':
                    # add refractory to itself's own effect
                    result_output[ibaseline] += raw_result_output.reshape((self.nt, self.ntrial), order='F')
                elif self.effect_type_list[ieffect]=='refractory':
                    result_output[iselfeffect] += raw_result_output.reshape((self.nt, self.ntrial), order='F')
                else:
                    raise ValueError("only support trial_coef and refractory in addition to inhomogeneous_baseline and coupling!")
            return result_output
        else:
            raise ValueError("Unfinished!")
            # result_output = []
            # raw_result_output_list = self.get_filter_output(trial_wise=False, ci=False)
            # for ieffect, raw_result_output in enumerate(raw_result_output_list):
            #     if self.effect_type_list[ieffect]=='inhomogeneous_baseline': 
            #         result_output.append( raw_result_output.reshape((self.nt, self.ntrial), order='F') )
            #         ibaseline = ieffect
            #     elif self.effect_type_list[ieffect]=='coupling':
            #         result_output.append( raw_result_output.reshape((self.nt, self.ntrial), order='F') )
            #         if utils.PROBE_CORRESPONDING_INVERSE[self.basis_name[ieffect][-2:]] == self.target:
            #             iselfeffect = ieffect
            #     elif self.effect_type_list[ieffect]=='trial_coef':
            #         # add refractory to itself's own effect
            #         result_output[ibaseline] += raw_result_output.reshape((self.nt, self.ntrial), order='F')
            #     elif self.effect_type_list[ieffect]=='refractory':
            #         result_output[iselfeffect] += raw_result_output.reshape((self.nt, self.ntrial), order='F')
            #     else:
            #         raise ValueError("only support trial_coef and refractory in addition to inhomogeneous_baseline and coupling!")
            # return result_output
        
    def test(self, test_trials, use_all=False, verbose=False):
        self.test_model = PP_GLM(dataset=self.dataset, 
                           select_trials=test_trials, 
                           membership=self.membership, 
                           condition_ids=self.condition_ids)
        if type(self.target) == str:
            # print(f"Assuming output is spike trains from {target}")
            self.test_model.output = utils.pooling_pop(self.test_model.membership, self.test_model.condition_ids, 
                                                       self.test_model.dataset, self.target, 0, use_all=use_all)
            self.test_model.output = self.test_model.output[:,test_trials]
        elif type(self.target) == np.ndarray:
            self.test_model.output = self.target[:,test_trials]
        else:
            raise ValueError("target must be either str like \"probeC\" or numpy.ndarray!")
        if self.npadding is not None:
                self.test_model.output = self.test_model.output[self.npadding:, :]
        for i_effect, effect_type in enumerate(self.effect_type_list):
            raw_input = self.raw_input_list[i_effect]
            kwargs = self.kwargs_list[i_effect]
            self.test_model.add_effect(effect_type, raw_input=raw_input, **kwargs)

        self.test_model.response = self.test_model.output.flatten('F')
        self.test_model.predictors = np.hstack(self.test_model.effect_list)
        self.test_model.results = self.results
        self.test_model.log_lmbd = (self.test_model.predictors@self.test_model.results.params).\
            reshape((self.test_model.nt, self.test_model.ntrial), order='F')
        self.test_model.log_lmbd_ci = (self.test_model.predictors@self.test_model.results.bse).\
            reshape((self.test_model.nt, self.test_model.ntrial), order='F')
        self.test_model.nll = spike_trains_neg_log_likelihood(self.test_model.log_lmbd, self.test_model.output)
        self.test_model.nll_trialwise = spike_trains_neg_log_likelihood(self.test_model.log_lmbd, self.test_model.output, trial_wise=True)
        self.test_model.aic = self.test_model.predictors.shape[1] + self.test_model.nll
        return self.test_model.nll

    def fit_time_warping_baseline(self, target, use_all=False, max_iter=100, penalty=1e-10, warp_interval=[[0, 0.15], [0.15, 0.35]], 
                                  tol=1e-10, method='mine', max_spike=None, verbose=True):
        assert 'inhomogeneous_baseline' in self.effect_type_list, "You must create an inhomogeneous baseline before changing it to time-warp baseline!"
        
        ALPHA = 0.5   # to smooth the optimization process
        BETA = 0.0   # to smooth the optimization process
        THETA = 0.9   # Mean converge

        # Find the effect index that should be warpped
        i_effect = [i_effect for i_effect,effect_type in enumerate(self.effect_type_list) 
            if effect_type=='inhomogeneous_baseline'][0]
        
        # Initialization
        self.shifts = np.zeros((self.ntrial, 2*len(warp_interval)))
        nll_old = np.inf
        X_baseline_original = self.effect_list[i_effect]
        
        for iter in range(max_iter):
            # update coef (based on *warped* effect_list[i_effect])
            self.fit(target, use_all=use_all, verbose=False, penalty=penalty, method=method, max_spike=max_spike)
            
            # 'inhomo' and 'inhomo_template' are based on 'basis_list', so they are not warped
            
            inhomo_template = self.get_filter_output(trial_wise=False, ci=False)[i_effect]
            result_output = self.get_filter_output(trial_wise=True, ci=False)
            inhomo = result_output[i_effect]
            total_output = np.vstack((result_output)).T
            total_output = total_output.sum(axis=1)
            minus_one_output = total_output - inhomo
            if verbose:
                print(f"After the {iter} th iteration of fitting: {self.nll}")
                
            # Update shifts (based on non-warping inhomo baseline)
            best_shift, nll  = get_best_shift(self.dataset.time_line, inhomo_template, minus_one_output, 
                                              self.response, self.nt, max_spike=self.max_spike, warp_interval=warp_interval)
            if iter==0:
                self.shifts = best_shift
            else:
                for i_interval in range(len(warp_interval)):
                    self.shifts[:,2*i_interval+1] = BETA*self.shifts[:,2*i_interval+1] + (1-BETA)*best_shift[:,2*i_interval+1]
                    # self.shifts[:,2*i_interval] = ALPHA*self.shifts[:,2*i_interval] + (1-ALPHA)*best_shift[:,2*i_interval]
                    self.shifts[:,2*i_interval] = self.shifts[:,2*i_interval+1].mean()  
            for i_interval in range(len(warp_interval)):
                self.shifts[:,2*i_interval+1] = (self.shifts[:,2*i_interval+1] - self.shifts[:,2*i_interval])*THETA + self.shifts[:,2*i_interval]
            X_baseline_warp = apply_warping_to_predictors(self.dataset.time_line, X_baseline_original, self.shifts, self.nt, 
                                                          warp_interval=warp_interval)
            self.effect_list[i_effect] = X_baseline_warp
            
            if verbose:
                print(f"After the {iter} th iteration of warping: {nll}")
            
            # if not_updating, break
            if nll_old - nll < tol:
                # Finished fitting
                pass
                # break
            nll_old = nll
            
        self.fit(target, use_all=use_all, verbose=False, penalty=penalty, method=method, max_spike=max_spike)
        # Finished fitting
        if iter == max_iter:
            print("Maximum iteration reach!")
        self.basis_name[i_effect] = 'time_warping_inhomogeneous_baseline'
        self.inhomo_template = inhomo_template
        self.nll = nll
        
#%% Binomial GLM 'logit'
def get_link(method):
    if method=='logit':
        link = lambda x: 1/(1+np.exp(-x))
    else:
        link = np.exp
    return link

def get_success_fail(response, return_max_spike=False, max_spike=None):
    success_fail = np.zeros((*response.shape, 2))
    if response.ndim == 1:
        success_fail[:,0] = response
    else:
        success_fail[:,:,0] = response
    if max_spike is None:
        max_spike = int(response.max() * 1)
    if response.ndim == 1:
        success_fail[:,1] = max_spike - response
    else:
        success_fail[:,:,1] = max_spike - response
    if return_max_spike:
        return success_fail, max_spike
    else:
        return success_fail
    
#%% Time-warping baseline
### None MP
def get_best_shift(time_line, inhomo_template, minus_one_output, response, nt, max_spike=None, warp_interval=[[0, 0.15], [0.15, 0.35]]):
    ntrial = int(len(response)/nt)
    best_shifts = np.zeros((ntrial, 2*len(warp_interval)))
    total_nll = 0
    rcd_log_lmbd = np.zeros_like(response)
    for itrial in range(ntrial):
        best_shifts_trial, best_nll_trial = get_best_shift_single(time_line, 
                                                inhomo_template, 
                                                minus_one_output[itrial*nt:(itrial+1)*nt], 
                                                response[itrial*nt:(itrial+1)*nt], 
                                                max_spike=max_spike, 
                                                warp_interval=warp_interval)
        best_shifts[itrial, :] = best_shifts_trial
        total_nll += best_nll_trial
    return best_shifts, total_nll


### MP version (unfortunately this is slower than the non MP version)
# def get_best_shift(time_line, inhomo_template, minus_one_output, response, nt):
#     import multiprocessing
#     import os
#     ntrial = int(len(response)/nt)
#     best_shifts = np.zeros((ntrial, 4))
#     total_nll = 0
#     rcd_log_lmbd = np.zeros_like(response)
#     # PROCESSES = os.cpu_count()-2
#     PROCESSES = 3
#     PARALLEL_BATCH_SIZE = ntrial
#     nbatch = int(np.ceil(ntrial/PARALLEL_BATCH_SIZE))
    
#     if sys.platform == 'linux':
#         for ibatch in range(nbatch):
#             with multiprocessing.get_context('spawn').Pool(processes = PROCESSES) as pool:
#                 results = [pool.apply_async(get_best_shift_single, (time_line, 
#                                                                     inhomo_template, 
#                                                                     minus_one_output[itrial*nt:(itrial+1)*nt], 
#                                                                     response[itrial*nt:(itrial+1)*nt]))
#                             for itrial in (np.arange(PARALLEL_BATCH_SIZE) + ibatch*PARALLEL_BATCH_SIZE)]
#                 pool.close()
#                 for iresult, result in enumerate(results):
#                     itrial = iresult + ibatch*PARALLEL_BATCH_SIZE
#                     best_shifts_trial, best_nll_trial = result.get()
#                     best_shifts[itrial, :] = best_shifts_trial
#                     total_nll += best_nll_trial
#         return best_shifts, total_nll
#     else:
#         raise ValueError("Multiprocessing only support on Linux at the moment!")
    

def get_best_shift_single(time_line, inhomo_template, minus_one_output, response, max_spike=None, warp_interval=[[0, 0.15], [0.15, 0.35]]):
    to_return = []
    for i_interval, interval in enumerate(warp_interval):
        search_grid = np.arange(interval[0], interval[1], 0.002)
        peak = time_line[np.sum(time_line<interval[0])+np.argmax(inhomo_template[time_line>=interval[0]])]
        sources = [interval[0], peak, interval[1]]
        best_nll = np.inf
        for moved_peak in search_grid:
            targets = [interval[0], moved_peak, interval[1]]
            warped = linear_time_warping_single(time_line, inhomo_template, sources, targets, verbose=False)
            nll = spike_trains_neg_log_likelihood(warped+minus_one_output, response, max_spike=max_spike)
            if nll <= best_nll:
                best_shift_peak = moved_peak
                best_warped = warped
                best_nll = nll
        to_return.append(peak)
        to_return.append(best_shift_peak)
    return np.array(to_return), best_nll

def apply_warping_to_predictors(time_line, X_baseline_original, shifts, nt, warp_interval=[[0, 0.15], [0.15, 0.35]]):
    ntrial = int(X_baseline_original.shape[0]/nt)
    X_baseline_warp = np.zeros_like(X_baseline_original)
    for itrial in range(ntrial):
        sources = []
        targets = []
        for i_interval, interval in enumerate(warp_interval):
            sources.append([warp_interval[i_interval][0], shifts[itrial, 2*i_interval], warp_interval[i_interval][1]])
            targets.append([warp_interval[i_interval][0], shifts[itrial, 2*i_interval+1], warp_interval[i_interval][1]])
        # print(sources)
        for i_col in range(X_baseline_warp.shape[1]):
            for i_interval, interval in enumerate(warp_interval):
                if i_interval==0:
                    to_warp = X_baseline_original[itrial*nt:(itrial+1)*nt, i_col]
                    warped = linear_time_warping_single(time_line, to_warp, sources[i_interval], targets[i_interval], verbose=False)
                else:
                    to_warp = warped
                    warped = linear_time_warping_single(time_line, to_warp, sources[i_interval], targets[i_interval], verbose=False)
            X_baseline_warp[itrial*nt:(itrial+1)*nt, i_col] = warped
    return X_baseline_warp

def linear_time_warping_single(t, f, sources, targets, verbose=True):
    """Time warping function for the intensity.

    Args:
        sources: Positions of input `f` needed to be shifted.
        targets: New positions of the sources. The rest of curve will be shifted
            linearly in between sources.
    """
    sources = np.array(sources)
    targets = np.array(targets)
    t_interp = t.copy()

    for i in range(1, len(sources)):
        source_left = sources[i-1]
        source_right = sources[i]
        target_left = targets[i-1]
        target_right = targets[i]

        # Linearly stretch the source intervals to the target interverals.
        t_target_index = (t >= target_left) & (t < target_right)
        t_target = t[t_target_index]
        if len(t_target) == 0:
            continue
        t_interp[t_target_index] = ((t_target - target_left) *
            (source_right - source_left) / (target_right - target_left)
            + source_left)
    # Run the linear interporation using the sample points.
    f_warp = np.interp(t_interp, t, f)
    return f_warp
    
#%% Simulation
def simulate(model_list, probe_list=['probeA', 'probeB', 'probeC', 'probeD', 'probeE', 'probeF']):
    # Get {probe2num} dictionary
    probe2num = {}
    for iprobe, probe in enumerate(probe_list):
        probe2num[probe] = iprobe
    
    # Get three dimension matrix of coupling filters for better computing.   
    npop = len(model_list)
    max_histories = 1
    nt = model_list[0].nt
    allowed_effect_type = ['inhomogeneous_baseline', 'coupling', 'trial_coef']
    baseline_mat = np.zeros((nt, npop))
    coupling_mat = np.zeros((max_histories, npop, npop))
    
    for ineuron in range(npop):
        assert all(effect_type in allowed_effect_type for effect_type in model_list[ineuron].effect_type_list), "Only support inhomogeneous_baseline and coupling effects now!"
        model = model_list[ineuron]
        for ieffect, effect_type in enumerate(model.effect_type_list):
            
            if effect_type in ['inhomogeneous_baseline']:
                baseline_mat[:, ineuron] = model.filters[ieffect]
            elif effect_type in ['coupling']:
                nhistories = len(model.filters[ieffect])
                probe_name = utils.PROBE_CORRESPONDING_INVERSE[model.basis_name[ieffect][-2:]]
                iprobe = probe2num[probe_name]
                if nhistories > max_histories:
                    coupling_mat_old = coupling_mat
                    coupling_mat = np.zeros((nhistories, npop, npop))
                    coupling_mat[-max_histories:, :, :] = coupling_mat_old
                    max_histories = nhistories
                coupling_mat[-nhistories:, iprobe, ineuron] = np.flip(model.filters[ieffect])
    
    spikes, log_firing_rate = simulate_baseline_coupling(baseline_mat, coupling_mat)
    return spikes, log_firing_rate
    
def simulate_baseline_coupling(baseline_mat, coupling_mat):
    MAX_FIRING_RATE = np.log(10000)
    max_histories, _, npop = coupling_mat.shape
    nt = baseline_mat.shape[0]
    spikes = np.zeros((nt, npop, 1))
    log_firing_rate = copy.deepcopy(baseline_mat[:,:,np.newaxis])
    spikes[0,:,0] = np.random.poisson(np.exp(log_firing_rate[0,:,0]))
    
    for t in range(1, nt):
        nhistories = min(t, max_histories)
        temp_log_firing_rate = (coupling_mat[-nhistories:, :, :] * spikes[(t-nhistories):(t), :, :]).sum(axis=(0, 1))
        log_firing_rate[t,:,0] += temp_log_firing_rate
        # log_firing_rate[t,:,0] = np.minimum(log_firing_rate[t,:,0], MAX_FIRING_RATE)
        spikes[t,:,0] = np.random.poisson(np.exp(log_firing_rate[t,:,0]))
    
    log_firing_rate = log_firing_rate.squeeze()
    spikes = spikes.squeeze()
    return spikes, log_firing_rate

def simulate_individual_history(baseline_mat, coupling_mat, history_list, nneuron_list=None):
    MAX_FIRING_RATE = np.log(10000)
    # nneuron: number of individual neuorns
    # npop: number of populations
    temp = copy.deepcopy(history_list)
    history_list = temp
    max_histories_history = 0
    if nneuron_list is not None:
        assert len(history_list) == len(nneuron_list), "The number of populations should matach!"
        for i, history in enumerate(history_list):
            assert history_list[i].ndim == 1
            history_list[i] = np.matlib.repmat(history, nneuron_list[i], 1).T
            max_histories_history = max(max_histories_history, history.shape[0])
    else:
        for i, history in enumerate(history_list):
            nneuron_list[i] = history.shape[1]
            max_histories_history = max(max_histories_history, history.shape[0])

    max_histories_coupling, _, npop = coupling_mat.shape
    nt = baseline_mat.shape[0]
    pop_spikes = np.zeros((nt, npop, 1))
    ind_spikes = [np.zeros((nt, nneuron_list[i])) for i in range(npop)]
    log_firing_rate_pop_level = copy.deepcopy(baseline_mat[:,:,np.newaxis])  # np.newaxis doesn't create a new data array!!!
    # t=0
    log_firing_rate_ind_only_history_rcd = []
    for ipop in range(npop):
        log_firing_rate_ind_only_history_rcd.append(np.zeros((nt, nneuron_list[ipop])))
        log_firing_rate_ind = log_firing_rate_pop_level[0, ipop, 0]*np.ones(nneuron_list[ipop]) \
                            - np.log(nneuron_list[i])
        log_firing_rate_ind_only_history = 0
        log_firing_rate_ind += log_firing_rate_ind_only_history
        ind_spikes[ipop][0, :] = np.random.poisson(np.exp(log_firing_rate_ind))
        pop_spikes[0,ipop,0] = np.sum(ind_spikes[ipop][0, :])
    
    for t in range(1, nt):
        nhistories = min(t, max_histories_coupling)
        log_firing_rate_coupling = (coupling_mat[-nhistories:, :, :] * pop_spikes[(t-nhistories):(t), :, :]).sum(axis=(0, 1))
        log_firing_rate_pop_level[t,:,0] += log_firing_rate_coupling
        for ipop in range(npop):
            log_firing_rate_ind = log_firing_rate_pop_level[t, ipop, 0]*np.ones(nneuron_list[ipop]) #- np.log(nneuron_list[ipop])
            nhistories = min(t, max_histories_history)
            log_firing_rate_ind_only_history = (history_list[ipop][-nhistories:,:] * ind_spikes[ipop][(t-nhistories):(t), :]).sum(axis=0)
            log_firing_rate_ind += log_firing_rate_ind_only_history
            log_firing_rate_ind = np.minimum(log_firing_rate_ind, MAX_FIRING_RATE)
            ind_spikes[ipop][t, :] = np.random.poisson(np.exp(log_firing_rate_ind))
            pop_spikes[t,ipop,0] = np.sum(ind_spikes[ipop][t, :])
            log_firing_rate_ind_only_history_rcd[ipop][t, :] = log_firing_rate_ind_only_history

    return pop_spikes.squeeze(), log_firing_rate_pop_level.squeeze(), log_firing_rate_ind_only_history_rcd

#%% KS measurement
def get_three_measure_entire_length(f, exp=False):
    if exp==False:
        if np.any(f<=0):
            exp = True
            print("Setting f to exp(f) for nonnegativity!")
    if exp:
        f = np.exp(f)
    pdf = f/f.sum()
    cdf = np.cumsum(pdf)
    pdfUniform = 1/len(pdf) * np.ones(len(pdf))
    cdfUniform = np.cumsum(pdfUniform)
    # KL = rel_entr(pdf, pdfUniform).sum()
    KL = 0
    KS = np.max(np.abs(cdf-cdfUniform))
    # Wasser = np.sum(np.abs(cdf-cdfUniform)*1/len(pdf))
    Wasser = 0
    return np.array([KL, KS, Wasser])

def get_measure_func(measure, f):
    def get_measure(l, win, f=f, measure=measure):
        r = l + win
        # if r > 500:
        #     r = 500
        return get_three_measure_entire_length(f[int(l):int(r)], exp=True)[measure]
    return get_measure

def get_best_time_range(total_output, minus_one_output, measure, time_range):
    total_output_func = get_measure_func(measure, total_output)
    minus_one_output_func = get_measure_func(measure, minus_one_output)
    nt = time_range[1] - time_range[0]
    kl = np.full((nt, nt), -np.inf)
    for l in range(time_range[0], time_range[1]):
        for r in range(l+1, time_range[1]):
            kl[l-time_range[0], r-time_range[0]] = total_output_func(l,r-l) - minus_one_output_func(l,r-l)
    best_l, best_r = np.where(kl==kl.max())
    return [best_l[0]+time_range[0], best_r[0]+time_range[0]]

#%% Plotting GLM
def plot_GLM_one_effect(model, effect_id, results=None, title=None, label=None, color=None):
    start_col = 0
    for previous_id in range(effect_id):
        start_col += (model.effect_list[previous_id]).shape[1]
    nbasis = (model.effect_list[effect_id]).shape[1]
    end_col = start_col + nbasis
    if results is None:
        results = model.results
    if model.basis_name[effect_id] in ['inhomogeneous_baseline',
                                       'homogeneous_baseline', 
                                       'time_warping_inhomogeneous_baseline']:
        use_exp = True
    else:
        use_exp = False
    try:
        # try to get standard error from "results", if failed, just ignore standard error
        utils.plot_filter(model.basis_list[effect_id], results.params[start_col:end_col], 
                      results.bse[start_col:end_col], label=label, color=color, exp=use_exp)
    except:
        utils.plot_filter(model.basis_list[effect_id], results.params[start_col:end_col], 
                np.zeros(end_col-start_col), label=label, color=color, exp=use_exp)
    plt.title(title)
    plt.legend()
    if model.basis_name[effect_id] == 'twoway_coupling':
        length = int(model.basis_list[effect_id].shape[0]/2)
        plt.xticks([0, length, length*2], [-length, 0, length])
        
    
def plot_GLM_compare(model, effect_id_list=None,  results_list=None, title_list=None, label_list=None, color_list=['r','b'] ):
    if effect_id_list is None:
        effect_id_list = np.arange(len(model.basis_name))
    if title_list is None:
        title_list = [ model.basis_name[i] for i in effect_id_list]
    if results_list is None:
        results_list = [model.results]
    if label_list is None:
        label_list = [ ' ' for i in effect_id_list]
    i_effect = 0
    for effect_id in effect_id_list:
        for i_results in range(len(results_list)):
            plot_GLM_one_effect(model, 
                                effect_id, 
                                results=results_list[i_results], 
                                title=title_list[i_effect], 
                                label=label_list[i_results], 
                                color=color_list[i_results])
        i_effect += 1
        plt.show()

#%% Get predictors for coupling effects
def conv_flip(raw_input, kernel, npadding=None, enforce_causality=True):
    """ Causility enforced convolution. e.g. Spike trains convolve with post-spike filter; Stimulus convolve with stimulus filter. 

    Args:
        spike (1d vector): spike trains of a single trial
        kernel (2d vector): a (nt, nbasis) matrix, which contains multiple basis

    Returns:
        X [type]: [description]
    """
    raw_input = np.flipud(raw_input)
    nbasis = kernel.shape[1]
    return np.flipud(conv(raw_input, kernel, npadding=npadding, enforce_causality=enforce_causality))

def conv(raw_input, kernel, npadding=None, enforce_causality=True):
    """ Causility enforced convolution. e.g. Spike trains convolve with post-spike filter; Stimulus convolve with stimulus filter. 

    Args:
        spike (1d vector): spike trains of multiple trials
        kernel (2d vector): a (nt, nbasis) matrix, which contains multiple basis

    Returns:
        X [type]: [description]
    """
    if kernel.ndim == 1:
        kernel = kernel[:,np.newaxis]
    if raw_input.ndim == 1:
        raw_input = raw_input[:,np.newaxis]
    nbasis = kernel.shape[1]
    nt, ntrial = raw_input.shape
    if npadding is not None:
        nt = nt - npadding
    X = np.zeros((nt*ntrial,nbasis))
    for ibasis in range(nbasis):
        X[:,ibasis] = conv_multi_trial(raw_input, kernel[:,ibasis], merge_trial=True, npadding=npadding, enforce_causality=enforce_causality)
    return X

def conv_multi_trial(raw_input, kernel, merge_trial=False, npadding=None, enforce_causality=True):
    """ Causility enforced convolution. e.g. Spike trains convolve with post-spike filter; Stimulus convolve with stimulus filter. 

    Args:
        spike (1d vector): spike trains of a single trial
        kernel (1d vector): one basis

    Returns:
        [type]: [description]
    """
    if raw_input.ndim == 1:
        raw_input = raw_input[:,np.newaxis]
    nt, ntrial = raw_input.shape
    if enforce_causality:
        kernel = np.hstack((np.array([0]), kernel))
    nn = nt + len(kernel) - 1
    G = ifft(fft(raw_input,nn,axis=0)*fft(kernel,nn)[:,np.newaxis],axis=0)
    G = G[0:len(raw_input)].real
    G[np.abs(G)<1e-10] = 0
    if npadding is not None:
        G = G[npadding:,:]
    if merge_trial:
        G = G.flatten('F')
    return G

#%% Make basis for inhomo baseline, coupling filter (Pillow basis), etc.
def inhomo_baseline(ntrial=1, start=0, end=1e3, dt=1, num=10, add_constant_basis=False, apply_trial=None):
    basis = make_b_spline_basis(
        t_min=start, 
        t_max=end, 
        dt=dt, 
        num_basis=num, 
        add_constant_basis=add_constant_basis, 
        verbose=False)
    if basis.ndim == 1:
        basis = basis[:,np.newaxis]
    nt = basis.shape[0]
    if apply_trial is None:
        baseline = np.tile(basis, (ntrial, 1))
    else:
        baseline = np.tile(np.zeros(basis.shape), (ntrial, 1))
        for i in range(ntrial):
            if apply_trial[i]:
                baseline[(i*nt):(i*nt+nt)] = basis
    return baseline

def make_pillow_basis(num=10, peaks_min=0, peaks_max=100, nonlinear=0.2, dt=1, verbose=False):
    """ Generating raised cosine basis

    Args:
        num (int, optional): Number of basis. Defaults to 10.
        peaks_min (float, optional): Position of the first basis peak. Defaults to 0.
        peaks_max (float, optional): Position of the last basis peak. Defaults to 100.
        h_nonlin (float, optional): Range from 0 to 1. Determines how nonlinear these basis functions would be. Defaults to 0.2.
        dt (int/float, optional): Length of time bin. Defaults to 1.

    Returns:
        ihbasis: nt by num matrix, each column for each basis
    """    
    
    assert 0<=nonlinear<=1, "h_nonlin should be from 0 to 1"
    nonlinear = nonlinear*peaks_max
    nlin = lambda x: np.log(x+1e-10)
    invnl = lambda x: np.exp(x)-1e-10
    hpeaks = np.array([peaks_min, peaks_max])
    yrnge = nlin(hpeaks+nonlinear)

    db = np.diff(yrnge)[0]/(num-1)
    ctrs = np.linspace(yrnge[0], yrnge[1], num)[None,:]
    mxt = (invnl(yrnge[1]+2*db)-nonlinear).astype(int)
    iht = np.arange(0,mxt,dt)[:,None]
    nt = len(iht)
    ff = lambda x, c, dc: (np.cos(np.maximum(-np.pi,np.minimum(np.pi, (x-c)*np.pi/dc/2)))+1)/2
    ihbasis = ff(np.tile(nlin(iht+nonlinear), (1, num)), np.tile(ctrs, (nt, 1)), db)
    if verbose:
        plt.figure()
        plt.plot(ihbasis, '-')
        plt.show()
    return ihbasis

def make_b_spline_basis(
    num_basis=10,
    t_max=1000,
    t_min=0,
    add_constant_basis=False,
    dt=1,
    spline_order=2,
    verbose=False):
    """Constructs B-spline basis with knots equal distance.

    Args:
        t_range: [left_end, right_end].
    """
    # construct_b_spline_basis
    num_knots = num_basis-spline_order+1
    knots = np.linspace(t_min, t_max, num_knots)
    knots = np.hstack((np.ones(spline_order) * t_min, 
						knots,
						np.ones(spline_order) * t_max))
    basis_matrix = make_b_spline_basis_arbitrary_knots(
      	spline_order, knots, dt, add_constant_basis, verbose)

    return basis_matrix

def make_b_spline_basis_arbitrary_knots(
        spline_order,
        knots,
        dt,
        add_constant_basis,
        verbose):
    """Constructs B-spline basis."""

    num_basis = len(knots) - spline_order - 1
    num_rows = int(np.round((knots[-1] - knots[0]) / dt))
    t = np.linspace(knots[0], knots[-1], num_rows)
    basis_matrix = np.zeros((len(t), num_basis))
    interpolate_token=[0, 0, spline_order]
    interpolate_token[0] = np.array(knots)

    for i in range(num_basis):
        basis_coefficients = [0] * num_basis
        basis_coefficients[i] = 1.0 
        interpolate_token[1] = basis_coefficients
        y = scipy.interpolate.splev(t, interpolate_token)
        basis_matrix[:, i] = y

    if add_constant_basis:
        basis_matrix = np.hstack((np.ones((len(t), 1)), basis_matrix))
    max_scale = np.max(basis_matrix,axis=0)
    basis_matrix = basis_matrix/max_scale
    if verbose:
        plt.figure()
        plt.plot(t, basis_matrix)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    return basis_matrix

#%% Generate spike train
def generate_spike_train(lmbd, random_seed=None):
    """Generate one trial of spike train using firing rate lamdba.

    Args:
        lmbd: The firing rate.

    Returns:
        One spike train.
    """
    if random_seed:
        np.random.seed(random_seed)

    spike_train = np.zeros(len(lmbd))
    for t in range(len(lmbd)):
        num_spikes = np.random.poisson(lmbd[t])
        spike_train[t] = num_spikes
    return spike_train

#%% Calculating log likelihood
def spike_trains_neg_log_likelihood(log_lmbd, spike_trains, trial_wise=False, max_spike=None):
    """Calculates the log-likelihood of a spike train given log firing rate.

    When it calculates the log_likelihood funciton, it assumes that it is a
    function of lambda instead of spikes. So it drops out the terms that are not
    related to the lambda, which is the y! (spikes factorial) term.

    Args:
        log_lmbd: The format can be in two ways.
                timebins 1D array.
                (nt, ntrials) numpy array. Different trials have differnet intensity.
                        In this case, `spike_trains` and `log_lmbd` have matching rows.
        spike_trains: (nt, ntrials) numpy array.
    """
    # Having maximum spikes is just like binomial regression. 
    # Having inf maximum spikes is Poisson
    
    if max_spike is None:
        if spike_trains.ndim == 1:
            spike_trains = spike_trains[:, np.newaxis]
        nt, ntrial= spike_trains.shape
        # Default is Poisson
        if log_lmbd.ndim == 2:    # Trialwise intensity function.
            nll = - (spike_trains * log_lmbd)
            nll += np.exp(log_lmbd)
            if trial_wise:
                return nll.sum(axis=0)
            else:
                return nll.sum()
        elif log_lmbd.ndim == 1:    # Single intensity for all trials.
            nll = - spike_trains.sum(axis=1) @ log_lmbd
            nll += np.exp(log_lmbd).sum() * ntrial
            return nll
    else:
        # Binomial
        if spike_trains.ndim == 1:
            spike_trains = spike_trains[:, np.newaxis]
        success_fail = get_success_fail(spike_trains, max_spike=max_spike)
        link = get_link('logit')
        lmbd = link(log_lmbd)
        if log_lmbd.ndim == 2:    # Trialwise intensity function.
            nll = -lmbd*success_fail[:,:,0] - (1-lmbd)*success_fail[:,:,1]
            if trial_wise:
                return nll.sum(axis=0)
            else:
                return nll.sum()
        elif log_lmbd.ndim == 1:    # Single intensity for all trials.
            success_fail_sum = success_fail.sum(axis=1)
            nll = -lmbd*success_fail_sum[:,0] - (1-lmbd)*success_fail_sum[:,1]
            return nll.sum()


class poisson_regression_result():
    def __init__(self, params, bse):
        self.params = params
        self.bse = bse

#%% Core code for fitting Poisson GLM
def poisson_regression(
        Y,
        X,
        L2_pen=1e-6,
        max_num_iterations=100, 
        tol=1e-8,
        no_penalty=[]):
    """Fit Poisson GLM.

    The coefficients beta is fitted using Newton's method.
    Args:
        Y: (nt*ntrial, ) numpy vector
        X: (nt*ntrial, num_predictor) numpy array
    """
    Y = Y[:, np.newaxis]
    assert Y.shape[0] == X.shape[0], "Predictors X should match the shape of Y"
    num_predictor = X.shape[1]

    beta = np.zeros((num_predictor, 1))
    penalty_vec = np.ones((num_predictor, 1))
    if (X[:,0] == 1).all():
        # The first column is the constant baseline, set the constant to mean firing rate. 
        beta[0] = np.log(Y.sum()/len(Y))
        penalty_vec[0] = 0
    for no_penalty_term in no_penalty:
        penalty_vec[no_penalty_term] = 0
    penalty_matrix = np.diag(penalty_vec.squeeze())
    log_lmbda_hat = (X @ beta)

    nll = spike_trains_neg_log_likelihood(log_lmbda_hat, Y) + L2_pen * np.linalg.norm(beta*penalty_vec)**2
    nll_old = np.inf
    for iter_index in range(max_num_iterations):
        # Newton's method.
        # g: search direction
        mu = np.exp(X @ beta)
        grad = - (X.T @ Y) + (X.T @ mu) + 2*L2_pen * penalty_vec * beta
        hessian = (X.T) @ (mu * X) + 2*L2_pen * penalty_matrix
        g = np.linalg.inv(hessian) @ grad 
        lr = 1
        ALPHA = 0.4
        BETA = 0.2
        
        # Backtracking line search.
        while True:
            beta_tmp = beta - lr * g
            log_lmbd_tmp = (X @ beta_tmp)
            nll_left = spike_trains_neg_log_likelihood(log_lmbd_tmp, Y) + L2_pen * np.linalg.norm(beta_tmp*penalty_vec)**2
            nll_right = nll - ALPHA * lr * grad.T @ g

            if (nll_left > nll_right or
                    np.isnan(nll_left) or
                    np.isnan(nll_right)):
                lr *= BETA
                # print(f"update learning_rate: {lr}")
            else:
                break
        if iter_index == max_num_iterations - 1:
            print('Warning: Reaches maximum number of iterations.')
            
        # Update beta, negtive log-likelihood.
        beta = beta_tmp
        nll = nll_left
        # print(iter_index, nll)
        # Check convergence.
        if abs(nll - nll_old) < tol:
            break
        nll_old = nll
    
    # Get standard error
    mu = np.exp(X @ beta)
    hessian = X.T @ (mu * X) + 2*L2_pen * penalty_matrix
    bse = np.sqrt(np.diag(np.linalg.inv(hessian)))
    return poisson_regression_result(beta.squeeze(), bse)


#%% Excursion test
def get_excursion_test(function1, function2, ROI_list):
    stats_list = []
    for i, ROI in enumerate(ROI_list):
        diff = function1 - function2
        stats_list.append( np.sum( np.abs(diff[ROI]) ) )
    return [np.max(stats_list)]

def get_ROI(function1, function2):
    diff = np.abs(function1 - function2)
    threshold = diff.max()/2
    idx = np.where(diff >= threshold)[0]
    return np.split(idx, np.where(np.diff(idx) != 1)[0]+1)

def merge_dict(d1, d2):
    # d1 is the mother, d2 is the one to add to d1
    d = {}
    for k in d1.keys():
        d[k] = d1[k] + d2[k]
    return d

def get_statistics_null_excursion(V1, membership, condition_ids, probe_list, num_basis_baseline, coupling_filter_params):
    penalty = 3e-1
    tau = 10
    order = 2
    num_basis_baseline = 30
    max_iter = 10
    coupling_filter_params = {'peaks_max':50, 'num':6, 'nonlinear':0.3}

    ################ No need to change below
    fake_running_trial_index = copy.deepcopy(V1.running_trial_index)
    np.random.shuffle(fake_running_trial_index)  # don't need to assign shuffled list, it's changed automatically. 
    fake_stationary_trial_index = np.logical_not(fake_running_trial_index)
    
    probe_list = V1.selected_probes
    running_filter = {}
    stationary_filter = {}
    running_output = {}
    stationary_output = {}
    ROI_filter = {}
    statistics_filter = {}
    ROI_output = {}
    statistics_output = {}
    running_model_list = []
    stationary_model_list = []

    for i, target_probe in enumerate(probe_list):
        select_trials = fake_running_trial_index
        model = PP_GLM(dataset=V1, 
                        select_trials=select_trials, 
                        membership=membership, 
                        condition_ids=condition_ids)
        model.add_effect('inhomogeneous_baseline', num=num_basis_baseline, apply_no_penalty=True)
        for j, input_probe in enumerate(probe_list):
            model.add_effect('coupling', probe_list[j], apply_no_penalty=True, **coupling_filter_params)
        model.add_effect('refractory', target_probe, order=order, tau=tau, apply_no_penalty=True, **coupling_filter_params)
        model.add_effect('trial_coef')
        model.fit_time_warping_baseline(target_probe, verbose=False, max_iter=max_iter, penalty=penalty)
        running_model_list.append(model)
        filter_list = model.get_filter(ci=False)
        running_filter[i,-1] = filter_list[0]
        k = 1
        for j, input_probe in enumerate(probe_list):
    #         if i==j:
    #             continue
            running_filter[i,j] = filter_list[k]
            k += 1
        filter_list = model.get_filter_output(ci=False)
        running_output[i,-1] = filter_list[0]
        k = 1
        for j, input_probe in enumerate(probe_list):
    #         if i==j:
    #             continue
            running_output[i,j] = filter_list[k]
            k += 1
        
        select_trials = fake_stationary_trial_index
        model = PP_GLM(dataset=V1, 
                        select_trials=select_trials, 
                        membership=membership, 
                        condition_ids=condition_ids)
        model.add_effect('inhomogeneous_baseline', num=num_basis_baseline, apply_no_penalty=True)
        for j, input_probe in enumerate(probe_list):
            model.add_effect('coupling', probe_list[j],apply_no_penalty=True, **coupling_filter_params)
        model.add_effect('refractory', target_probe, order=order, tau=tau,apply_no_penalty=True, **coupling_filter_params)
        model.add_effect('trial_coef')
        model.fit_time_warping_baseline(target_probe, verbose=False, max_iter=max_iter, penalty=penalty)
        stationary_model_list.append(model)
        filter_list = model.get_filter(ci=False)
        stationary_filter[i,-1] = filter_list[0]
        k = 1
        for j, input_probe in enumerate(probe_list):
    #         if i==j:
    #             continue
            stationary_filter[i,j] = filter_list[k]
            k += 1
        filter_list = model.get_filter_output(ci=False)
        stationary_output[i,-1] = filter_list[0]
        k = 1
        for j, input_probe in enumerate(probe_list):
    #         if i==j:
    #             continue
            stationary_output[i,j] = filter_list[k]
            k += 1
        
        # for effect filter
        filter_index = i,-1
        function1 = np.exp( running_filter[filter_index] )
        function2 = np.exp( stationary_filter[filter_index] )
        ROI_filter[filter_index] = get_ROI(function1, function2)
        statistics_filter[filter_index] = get_excursion_test(function1, function2, ROI_filter[filter_index])
        for j, input_probe in enumerate(probe_list):
            filter_index = i,j
            function1 = running_filter[filter_index]
            function2 = stationary_filter[filter_index]
            ROI_filter[filter_index] = get_ROI(function1, function2)
            statistics_filter[filter_index] = get_excursion_test(function1, function2, ROI_filter[filter_index])
            
        # for effect output
        filter_index = i,-1
        function1 = np.exp( running_output[filter_index] )
        function2 = np.exp( stationary_output[filter_index] )
        ROI_output[filter_index] = get_ROI(function1, function2)
        statistics_output[filter_index] = get_excursion_test(function1, function2, ROI_output[filter_index])
        for j, input_probe in enumerate(probe_list):
            filter_index = i,j
            function1 = running_output[filter_index]
            function2 = stationary_output[filter_index]
            ROI_output[filter_index] = get_ROI(function1, function2)
            statistics_output[filter_index] = get_excursion_test(function1, function2, ROI_output[filter_index])
    return statistics_filter, statistics_output
# Multiprocess version of null distribution

def get_statistics_null_mp(n_null, V1, membership, condition_ids, probe_list, num_basis_baseline, coupling_filter_params):
    """Get the distribution of test statistics (excursion test) under the null hypothesis. 
    Null hypothesis is that trial-wise running state doesn't affect neural response. So null 
    statistics is sample from random shuffling the running state of each trial. 
    To avoid KiB Swap running of out memory issue, parallel processing has a limitation of total
    tasks at the queue. The number of tasks is PARALLEL_BATCH_SIZE. 

    Args:
        n_null (int): number of samples of the null distributin
        V1 (DataLoader.Allen_dataset): the object containing all data and experimental information
        membership (pandas frame): IPRF result
        condition_ids (list): IPRF result
        probe_list (list): a list like ['probeA', 'probeC']
        num_basis_baseline (int): number of B-spline basis for inhomogeneous baseline

    Returns:
        dict: a dict whose key-value pair denote the null statistics distributino samples of a 
        certain filter. 
    """

    import multiprocessing
    import os
    # PROCESSES = os.cpu_count()-2
    PROCESSES = 5
    PARALLEL_BATCH_SIZE = 5
    nbatch = int(np.ceil(n_null/PARALLEL_BATCH_SIZE))
    print(f"Starting multiprocessing on {sys.platform}. \nCores={PROCESSES}. \nBatch size={PARALLEL_BATCH_SIZE}")
    if sys.platform == 'linux':
        with tqdm(total=n_null) as pbar:
            for ibatch in range(nbatch):
                with multiprocessing.get_context('spawn').Pool(processes = PROCESSES) as pool:               
                    results = [pool.apply_async(get_statistics_null_excursion, (V1, membership, condition_ids, probe_list, 
                                                                                num_basis_baseline, coupling_filter_params)) 
                            for i_null in np.arange(PARALLEL_BATCH_SIZE)]
                    pool.close()
                    if ibatch == 0:
                        # The first batch the first return result will be the very first "statistics_null"
                        statistics_filter_null, statistics_output_null = results[0].get()
                        pbar.update(1)
                        for result in results[1:]:
                            statistics_filter_null_new, statistics_output_null_new = result.get()
                            statistics_filter_null = merge_dict(statistics_filter_null, statistics_filter_null_new)
                            statistics_output_null = merge_dict(statistics_output_null, statistics_output_null_new)
                            pbar.update(1)
                    else:
                        for result in results:
                            statistics_filter_null_new, statistics_output_null_new = result.get()
                            statistics_filter_null = merge_dict(statistics_filter_null, statistics_filter_null_new)
                            statistics_output_null = merge_dict(statistics_output_null, statistics_output_null_new)
                            pbar.update(1)
                            
    else:
        with multiprocessing.Pool(processes = PROCESSES) as pool:
            print(f"Starting multiprocessing on {sys.platform}. Cores={PROCESSES}")
            results = [pool.apply_async(get_statistics_null, (V1, membership, condition_ids, probe_list, num_basis_baseline)) 
                    for i_null in np.arange(n_null)]
            pool.close()
            
            statistics_null = results[0].get()
            print("done!")
            for result in tqdm(results[1:]):
                statistics_null_new = result.get()
                statistics_null = merge_dict(statistics_null, statistics_null_new)
    return statistics_filter_null, statistics_output_null



def get_statistics_null_parametric_bootstrap(V1, membership, condition_ids, probe_list, num_basis_baseline):
    pass
    # Fit six model
    # for f in all filters:
        # set f to all 0s
        # for i_bt in range(n_bt):
            # simulated a whole dataset
            # regress and get an estimate of f, f hat
            # calculate and record test statistics
        # get p-value of that filter f
        
    # To-do: simulation; 
    #        test statistics try: sum(abs(f)); excursion on abs(f); KL for positive
    

def corr(C):
    """
    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling 
    for the remaining variables in C.
    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable
    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """
    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            res_j = C[:, j]
            res_i = C[:, i]
            corr = scipy.stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr
    return P_corr


def partial_corr(C):
    """
    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling 
    for the remaining variables in C.
    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable
    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """
    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot( beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)
            
            corr = scipy.stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr
        
    return P_corr