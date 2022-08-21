"""Data models."""
from curses import raw
import os
from tkinter import Menu

from absl import logging
import collections
from collections import defaultdict
import io
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from psycopg2 import paramstyle
from regex import D
# from pyrsistent import m
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
import scipy.interpolate 
import scipy.signal
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
            self.ntrial = self.select_trials.sum()
            self.membership = membership
            self.condition_ids = condition_ids
            self.npadding = self.dataset.npadding
        self.effect_list = []
        self.basis_list = []
        self.basis_name = []
        self.effect_type_list = []
        self.raw_input_list = []
        self.kwargs_list = []

    def add_effect(self, effect_type, raw_input=None, use_all=False, **kwargs):
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
                        'trial_coef'],  "Not supported effect_type!"

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
            refractory_spikes = np.zeros_like(input_to_couple)
            temp = refractory_spikes[0, :]
            for t in range(1, input_to_couple.shape[0]):
                temp *= np.exp(-1000.0/self.dataset.fps/tau)
                refractory_spikes[t, :] = temp
                temp += input_to_couple[t, :]
            
            refractory_spikes = refractory_spikes[-self.nt:, :]
            X_refractory = refractory_spikes.flatten('F')[:, np.newaxis]
            self.effect_list.append(X_refractory)
            self.basis_list.append(refractory_spikes.mean(axis=1)[:, np.newaxis])
            self.basis_name.append(effect_type)
            
        elif effect_type == 'trial_coef':
            X_trial_coef = np.zeros((self.nt*self.ntrial, self.ntrial))
            for itrial in range(self.ntrial):
                X_trial_coef[(itrial*self.nt):((itrial+1)*self.nt), itrial] = 1
            self.effect_list.append(X_trial_coef)
            self.basis_list.append(np.diag(np.ones(self.ntrial)))
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
        
        elif effect_type == 'history':
            raise ValueError("Unfinish!")
 

    def fit(self, target, use_all=False, verbose=True, penalty=1e-10, method='mine'):
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
        # self.results = sm.GLM(self.response, self.predictors, family=sm.families.Poisson()).fit()
        if penalty != 0 or method=='mine':
            self.results = poisson_regression(self.response, self.predictors, L2_pen=penalty)
        elif method=='logit':
            success_fail = np.zeros((len(self.response), 2))
            success_fail[:,0] = self.response
            self.spikes_max = int(self.response.max() * 1)
            success_fail[:,1] = self.spikes_max - self.response
            self.results = sm.GLM(success_fail, self.predictors, family=sm.families.Binomial()).fit()
        else:
            self.results = sm.GLM(self.response, self.predictors, family=sm.families.Poisson()).fit()
        self.log_lmbd = (self.predictors@self.results.params).reshape((self.nt, self.ntrial), order='F')
        # self.log_lmbd_ci = (self.predictors@self.results.bse).reshape((self.nt, self.ntrial), order='F')
        self.nll = spike_trains_neg_log_likelihood(self.log_lmbd, self.output)
        self.nll_trialwise = spike_trains_neg_log_likelihood(self.log_lmbd, self.output, trial_wise=True)
        self.aic = self.predictors.shape[1] + self.nll
        self.filters = self.get_filter(ci=False)
        if verbose:
            print(f"Negative log likelihood is: {self.nll :.2f}")
            print(f"aic/2 is: {self.aic :.2f}")
        return self.results
    
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
    
    def get_filter_output(self, ci=False):
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
            if self.basis_name[effect_id] == "inhomogeneous_baseline":
                predicters_temp = self.basis_list[effect_id]
                coef = self.results.params[start_col:end_col]
                y = (predicters_temp@coef[:,np.newaxis]).squeeze()
                se = self.results.bse[start_col:end_col]
                one_sigma_ci = (predicters_temp@se[:,np.newaxis]).squeeze()
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
                minus_one_output = copy.deepcopy(total_output)
                minus_one_output = minus_one_output - output + np.mean(output)
                temp_info = get_three_measure_entire_length(minus_one_output[time_range[0]:time_range[1]], exp=True)
                individual_info.append(total_info - temp_info)
            return individual_info
        else:
            measure = auto_pick
            individual_info = []
            for i, output in enumerate(result_output):
                minus_one_output = copy.deepcopy(total_output)
                minus_one_output = minus_one_output - output + np.mean(output)
                best_time_range = get_best_time_range(total_output, minus_one_output, measure, time_range)
                # print(f"{i}, {best_time_range}")
                total_info = get_three_measure_entire_length(total_output[best_time_range[0]:best_time_range[1]], exp=True)
                temp_info = get_three_measure_entire_length(minus_one_output[best_time_range[0]:best_time_range[1]], exp=True)
                individual_info.append(total_info - temp_info)
            return individual_info
            

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

def simulate(model_list, probe_list=['probeA', 'probeB', 'probeC', 'probeD', 'probeE', 'probeF']):
    nneuron = len(model_list)
    # Get three dimension matrix of coupling filters for better computing. 
    # for 
    # Start simulate one by one
    
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
    KL = rel_entr(pdf, pdfUniform).sum()
    # KL = rel_entr(pdf, pdfUniform).sum()
    KS = np.max(np.abs(cdf-cdfUniform))
    Wasser = np.sum(np.abs(cdf-cdfUniform)*1/len(pdf))
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

def plot_GLM_one_effect(model, effect_id, results=None, title=None, label=None, color=None):
    start_col = 0
    for previous_id in range(effect_id):
        start_col += (model.effect_list[previous_id]).shape[1]
    nbasis = (model.effect_list[effect_id]).shape[1]
    end_col = start_col + nbasis
    if results is None:
        results = model.results
    if model.basis_name[effect_id] in ['inhomogeneous_baseline','homogeneous_baseline']:
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

def spike_trains_neg_log_likelihood(log_lmbd, spike_trains, trial_wise=False):
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
    if spike_trains.ndim == 1:
        spike_trains = spike_trains[:, np.newaxis]
    nt, ntrial= spike_trains.shape
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


class poisson_regression_result():
    def __init__(self, params, bse):
        self.params = params
        self.bse = bse

def poisson_regression(
        Y,
        X,
        L2_pen=1e-6,
        max_num_iterations=100, 
        tol=1e-8):
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

def merge_dict(d1, d2):
    # d1 is the mother, d2 is the one to add to d1
    ds = [d1, d2]
    d = {}
    for k in d1.keys():
        d[k] = d1[k] + d2[k]
    return d

def get_statistics_null_excursion(V1, membership, condition_ids, probe_list, num_basis_baseline, coupling_filter_params):
    statistics_null = {}   # dict to return
    ROI_null = {}
    # Get permutated running and stationary index
    fake_running_trial_index = copy.deepcopy(V1.running_trial_index)
    np.random.shuffle(fake_running_trial_index)  
        # don't need to assign shuffled list, it's changed automatically. 
    fake_stationary_trial_index = np.logical_not(fake_running_trial_index)
    
    running_filter_temp = {}
    stationary_filter_temp = {}

    for i, target_probe in enumerate(probe_list):
        select_trials = fake_running_trial_index
        model = PP_GLM(dataset=V1, 
                           select_trials=select_trials, 
                           membership=membership, 
                           condition_ids=condition_ids)
        model.add_effect('inhomogeneous_baseline', num=num_basis_baseline, add_constant_basis=False)
        for j, input_probe in enumerate(probe_list):
            model.add_effect('coupling', probe_list[j], **coupling_filter_params)
        model.fit(probe_list[i], verbose=False)
        filter_list = model.get_filter(ci=True)
        running_filter_temp[i,-1] = filter_list[0]
        k = 1
        for j, input_probe in enumerate(probe_list):
            running_filter_temp[i,j] = filter_list[k]
            k += 1

        select_trials = fake_stationary_trial_index
        model = PP_GLM(dataset=V1, 
                           select_trials=select_trials, 
                           membership=membership, 
                           condition_ids=condition_ids)
        model.add_effect('inhomogeneous_baseline', num=num_basis_baseline, add_constant_basis=False)
        for j, input_probe in enumerate(probe_list):
            model.add_effect('coupling', probe_list[j], **coupling_filter_params)
        model.fit(probe_list[i], verbose=False)
        filter_list = model.get_filter(ci=True)
        stationary_filter_temp[i,-1] = filter_list[0]
        k = 1
        for j, input_probe in enumerate(probe_list):
            stationary_filter_temp[i,j] = filter_list[k]
            k += 1

        filter_index = i,-1
        function1 = np.exp( running_filter_temp[filter_index][0] )
        function2 = np.exp( stationary_filter_temp[filter_index][0] )
        if filter_index not in statistics_null.keys():
            statistics_null[filter_index] = []
        ROI_null[filter_index] = get_ROI(function1, function2)
        statistics_null[filter_index].append( get_excursion_test(function1, function2, ROI_null[filter_index]) )
        
        for j, input_probe in enumerate(probe_list):
            filter_index = i,j
            function1 = running_filter_temp[filter_index][0]
            function2 = stationary_filter_temp[filter_index][0]
            if filter_index not in statistics_null.keys():
                statistics_null[filter_index] = []
            ROI_null[filter_index] = get_ROI(function1, function2)
            statistics_null[filter_index].append( get_excursion_test(function1, function2, ROI_null[filter_index]) )
    return statistics_null

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
    PARALLEL_BATCH_SIZE = 50
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
                        statistics_null = results[0].get()
                        pbar.update(1)
                        for result in results[1:]:
                            statistics_null_new = result.get()
                            statistics_null = merge_dict(statistics_null, statistics_null_new)
                            pbar.update(1)
                    else:
                        for result in results:
                            statistics_null_new = result.get()
                            statistics_null = merge_dict(statistics_null, statistics_null_new)
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
    return statistics_null



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