"""Data models."""
import os

from absl import logging
import collections
from collections import defaultdict
import io
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
import scipy.interpolate 
import sklearn.model_selection
from tqdm import tqdm

import numpy as np
import numpy.matlib
from numpy.fft import fft as fft
from numpy.fft import ifft as ifft

class PP_GLM():
    def __init__(self, dataset=None):
        self.dataset = dataset

    def add_effect(input, type):
        assert type in ['homogeneous_baseline', 
                        'inhomogeneous_baseline', 
                        'coupling', 
                        'twoway_coupling', 
                        'circular', 
                        'identical']

def conv_flip(raw_input, kernel, enforce_causality=True):
    """ Causility enforced convolution. e.g. Spike trains convolve with post-spike filter; Stimulus convolve with stimulus filter. 

    Args:
        spike (1d vector): spike trains of a single trial
        kernel (2d vector): a (nt, nbasis) matrix, which contains multiple basis

    Returns:
        X [type]: [description]
    """
    raw_input = np.flipud(raw_input)
    nbasis = kernel.shape[1]
    return np.flipud(conv(raw_input, kernel, enforce_causality=enforce_causality))

def conv(raw_input, kernel, enforce_causality=True):
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
    X = np.zeros((nt*ntrial,nbasis))
    for ibasis in range(nbasis):
        X[:,ibasis] = conv_multi_trial(raw_input, kernel[:,ibasis], merge_trial=True, enforce_causality=enforce_causality)
    return X

def conv_multi_trial(raw_input, kernel, enforce_causality=True, merge_trial=False):
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
    if merge_trial:
        G = G.flatten('F')
    return G

def inhomo_baseline(ntrial=1, start=0, end=1e3, dt=1, num=10, add_constant_basis=True, apply_trial=None):
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
        baseline = np.matlib.repmat(basis, ntrial, 1)
    else:
        baseline = np.matlib.repmat(np.zeros(basis.shape), ntrial, 1)
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
    ihbasis = ff(np.matlib.repmat(nlin(iht+nonlinear), 1, num), np.matlib.repmat(ctrs, nt, 1), db)
    if verbose:
        plt.figure()
        plt.plot(ihbasis, '-')
        plt.show()
    return ihbasis

# def make_spline(start=0, end=1e3, dt=1, num=10, verbose=False):
#     from bspline import Bspline
#     num = num-1
#     knot_vector = np.hstack((np.array([start,start]), np.linspace(start,end,num), np.array([end,end])))
#     basis = Bspline(knot_vector,2)
    
#     x_min = np.min(basis.knot_vector)
#     x_max = np.max(basis.knot_vector)
#     x = np.arange(x_min, x_max+dt, dt)
#     N = np.array([basis(i) for i in x])
#     N = N[0:-1, :]
#     max_scale = np.max(N,axis=0)
#     N = N/max_scale
#     if verbose:
#         plt.figure()
#         plt.plot(N, '-')
#         plt.show()
#     return N

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

def spike_trains_neg_log_likelihood(log_lmbd, spike_trains):
    """Calculates the log-likelihood of a spike train given log firing rate.

    When it calculates the log_likelihood funciton, it assumes that it is a
    function of lambda instead of spikes. So it drops out the terms that are not
    related to the lambda, which is the y! (spikes factorial) term.

    Args:
        log_lmbd: The format can be in two ways.
                timebins 1D array.
                trials x timebins matrix. Different trials have differnet intensity.
                        In this case, `spike_trains` and `log_lmbd` have matching rows.
        spike_trains: Trials x timebins matrix.
    """
    num_trials, num_bins = spike_trains.shape

    log_lmbd = np.array(log_lmbd)
    if len(log_lmbd.shape) == 2:    # Trialwise intensity function.
        x, num_bins_log_lmbd = log_lmbd.shape
        if x != num_trials:
            print('log_lmbda_hat.shape:', log_lmbda_hat.shape)
            print('spikes.shape:', spikes.shape)
            raise ValueError('Number of trials does not match intensity size.')
        if num_bins != num_bins_log_lmbd:
            print('log_lmbda_hat.shape:', log_lmbda_hat.shape)
            print('spikes.shape:', spikes.shape)
            raise ValueError('The length of log_lmbd should be equal to spikes.')

        # Equivalent to row wise dot product then take the sum.
        nll = - np.sum(spike_trains * log_lmbd)
        nll += np.exp(log_lmbd).sum()
        return nll

    elif len(log_lmbd.shape) == 1:    # Single intensity for all trials.
        num_bins_log_lmbd = len(log_lmbd)
        if num_bins != num_bins_log_lmbd:
            print('log_lmbda_hat.shape:', log_lmbda_hat.shape)
            print('spikes.shape:', spikes.shape)
            raise ValueError('The length of log_lmbd should be equal to spikes.')
        nll = - np.dot(spike_trains.sum(axis=0), log_lmbd)
        nll += np.exp(log_lmbd).sum() * num_trials
        return nll

class SmoothingSpline(object):

    @classmethod
    def poisson_regression(
            cls,
            spikes,
            basis,
            max_num_iterations=100):
        """Fit the inhomogeneous point process using basis fit.

        The beta is fitted using Newton's method.

        Args:
            spikes: num_trials x num_spike_bins
            basis: num_samples x num_basis
        """
        num_trials, num_spike_bins = spikes.shape
        num_samples, num_basis = basis.shape
        if num_spike_bins != num_samples:
            raise ValueError(
                    'The length of the basis should be the same as that of spikes.')

        # beta = np.random.rand(num_basis, 1) - 5
        beta = np.ones((num_basis, 1)) * 0
        log_lmbda_hat = (basis @ beta).reshape(-1)

        nll = cls.spike_trains_neg_log_likelihood(log_lmbda_hat, spikes)
        nll_old = float("inf")
        # print(nll)

        for iter_index in range(max_num_iterations):
            mu = np.exp(basis @ beta)
            gradient = - (spikes.sum(axis=0) @ basis).T.reshape(num_basis, 1)
            gradient += (mu.T @ basis).T * num_trials
            hessian = basis.T @ (mu * basis) * num_trials
            # Gradient descent.
            # beta_delta = gradient
            # Newton's method.
            beta_delta = np.linalg.inv(hessian) @ gradient

            learning_rate = 1
            ALPHA = 0.4
            BETA = 0.2
            # Backtracking line search.
            while True:
                beta_tmp = beta - learning_rate * beta_delta
                log_lmbd_tmp = (basis @ beta_tmp).reshape(-1)
                nll_left = cls.spike_trains_neg_log_likelihood(log_lmbd_tmp, spikes)
                nll_right = nll - ALPHA * learning_rate * gradient.T @ beta_delta

                if (nll_left > nll_right or
                        np.isnan(nll_left) or
                        np.isnan(nll_right)):
                    learning_rate *= BETA
                    # print('update learning_rate: ', learning_rate)
                else:
                    break

            if iter_index == max_num_iterations - 1:
                print('Warning: Reaches maximum number of iterations.')
                
            # Update beta, negtive log-likelihood.
            beta = beta_tmp
            nll = nll_left
            # print(iter_index, nll)
            # Check convergence.
            if abs(nll - nll_old) < 1e-6:
                break
            nll_old = nll

        log_lmbda_hat = (basis @ beta).reshape(-1)
        # print('beta: ', beta)
        return log_lmbda_hat, beta

    @classmethod
    def build_smoothing_spline_penalty_matrix(cls, knots):
        """Built the quadratic penalty matrix in smoothing spline.

        The algorithmis is from Rodriguez 2001 - Smoothing spline regression

        TODO: There is an error in this formula. Need to check first. Instead, I am
                using numerical methods to calculate the second derivatives.

        Args:
            knots: A list of knots poitions.
        """
        n = len(knots)
        h = np.diff(knots)
        h_inv = 1 / h

        w_diag = (h[:-1] + h[1:]) / 3
        w_off = h[1:-1] / 6 
        W = np.diag(w_diag) + np.diag(w_off, 1) + np.diag(w_off, -1)
        W_inv = np.linalg.inv(W)

        d_ii = h_inv[:-1]
        d_ii1 = - h_inv[:-1] - h_inv[1:]
        d_ii2 = h_inv[1:]
        D = np.zeros((n-2,n))
        i, j = np.indices(D.shape)
        D[i==j] = d_ii
        D[i==j-1] = d_ii1
        D[i==j-2] = d_ii2

        K = D.T @ W_inv @ D

        return K


    @classmethod
    def construct_basis_omega(
            cls,
            time_line,
            knots=100,
            verbose=False):
        """Builds spline basis and smoothing penalty matrix."""
        basis, _ = cls.bspline_basis(
                knots=knots,
                knots_range=[time_line[0], time_line[-1]],
                sample_points=time_line,
                show_plot=False)

        # Smoothing spline penalty matrix.
        dt = 0.0005
        riemann_integral_t    = np.linspace(
                time_line[0], time_line[-1],
                int((time_line[-1] - time_line[0]) / dt) + 1)
        basis_2dev, _ = cls.bspline_basis(
                knots=knots,
                knots_range=[time_line[0], time_line[-1]],
                sample_points=riemann_integral_t,
                derivative_ord=2,
                show_plot=verbose)

        # Smoothing spline 2nd derivative pentalty.
        Omega = basis_2dev.T @ basis_2dev * dt
        # Simple Identity matrix penalty.
        # Omega = np.eye(num_basis)
        if verbose:
            plt.figure()
            seaborn.heatmap(Omega)
            plt.show()
        return basis, Omega


    @classmethod
    def poisson_regression_smoothing_spline(
            cls,
            spikes,
            time_line,
            constant_fit=False,
            log_lambda_offset=0,
            lambda_tuning=1e-8,
            lambda_baseline_tuning=0,
            learning_rate=0.5,
            max_num_iterations=200,
            beta_initial=None,
            beta_baseline_initial=None,
            basis=None,
            Omega=None,
            num_knots=100,
            verbose=0,
            verbose_warning=True):
        """Fit the inhomogeneous pont process using basis fit.

        The beta is fitted using Newton's method.

        Args:
            spikes: num_trials x num_spike_bins
            basis: num_samples x num_basis
            log_lambda_offset: The offset is an additional predictor variable, but 
                    with a coefficient value fixed at 1, which means we do not optimize
                    over it. The format can be:
                    num_spike_bins: 1-D array. non-constant baseline. For every trial.
                    constant: scalar. For every trial.
                    num_trials x 1: different constant offset for every trial.
                    num_trials x num_spike_bins: different non-constant offsets for 
                            different trials.
            method: 'newton', 'gradient'
            verbose: 3 levels. 0. quite. 1. iteration info. 2. with plots.

        Returns:
            log_lambda_hat, (beta, beta_baseline, log_lambda_offset).
        """
        num_trials, num_spike_bins = spikes.shape
        spikes_cum = spikes.sum(axis=0)
        log_lambda_offset = np.array(log_lambda_offset)
        trial_wise_offset = False

        if len(log_lambda_offset.shape) == 2:
            trial_wise_offset = True
            x, y = log_lambda_offset.shape
            if x != num_trials:
                raise ValueError('Offset values wrong')

        if constant_fit:
            basis = np.zeros((num_spike_bins, 1))
            Omega = np.zeros((1, 1))
        # In this case, no smoothing penalty.
        elif basis is not None and Omega is None:
            num_samples, num_basis = basis.shape
        elif basis is None and Omega is None:
            # spikes_cum_tmp = spikes.copy()
            # spikes_cum_tmp[0], spikes_cum_tmp[-1] = 1, 1
            # dt = time_line[1] - time_line[0]
            # knots = np.where(spikes_cum_tmp !=0 )[0] * dt
            basis, Omega = cls.construct_basis_omega(
                    time_line, knots=num_knots, verbose=verbose==2)
        num_samples, num_basis = basis.shape

        if num_spike_bins != num_samples:
            raise ValueError(
                    'The length of the basis should be the same as that of spikes.')

        # beta = np.random.rand(num_basis, 1) - 5
        if beta_initial is None:
            beta = np.zeros((num_basis, 1)) * 0
        else:
            beta = beta_initial

        if beta_baseline_initial is None:
            beta_baseline = -1.9
        else:
            beta_baseline = beta_baseline_initial

        # `log_lambda_hat` is num_bins x 1 matrix.
        log_lambda_hat = basis @ beta + beta_baseline
        # `log_lambda_offset` is num_trials x 1 matrix.
        log_lambda_hat = log_lambda_hat.reshape(-1) + log_lambda_offset
        nll = cls.spike_trains_neg_log_likelihood(log_lambda_hat, spikes)
        nll_old = float("inf")

        for iter_index in range(max_num_iterations):
            eta = basis @ beta + beta_baseline
            eta = eta.reshape(-1) + log_lambda_offset
            mu = np.exp(eta).reshape(-1, num_spike_bins)
            if trial_wise_offset:
                mu = mu.sum(axis=0).reshape(-1, num_spike_bins)
            else:
                mu = mu * num_trials

            gradient = basis.T @ (- spikes_cum.reshape(-1, 1) + mu.T)
            if Omega is not None:
                gradient += 2 * num_trials * lambda_tuning * Omega @ beta

            gradient_baseline = - spikes_cum.sum() + mu.sum()
            gradient_baseline += (2 * num_trials * lambda_baseline_tuning *
                                                        beta_baseline)

            hessian = basis.T @ (mu.T * basis)
            if Omega is not None:
                hessian += 2 * num_trials * lambda_tuning * Omega

            hessian_baseline = mu.sum()
            hessian_baseline += 2 * num_trials * lambda_baseline_tuning

            if constant_fit:
                # We make hessian 1 since we know that for sure gradient is 0.
                hessian = np.ones((1, 1))
            # Gradient descent.
            # beta_delta = gradient
            # Newton's method.
            beta_delta = np.linalg.inv(hessian) @ gradient
            beta_baseline_delta = gradient_baseline / hessian_baseline

            ALPHA = 0.4
            BETA = 0.2
            loop_cnt = 0
            # Backtracking line search.
            while True:
                beta_tmp = beta - learning_rate * beta_delta
                beta_baseline_tmp = beta_baseline - learning_rate * beta_baseline_delta

                log_lambda_tmp = basis @ beta_tmp + beta_baseline_tmp
                log_lambda_tmp = log_lambda_tmp.reshape(-1) + log_lambda_offset
                nll_left = cls.spike_trains_neg_log_likelihood(log_lambda_tmp, spikes)
                nll_right = nll - ALPHA * learning_rate * (gradient.T @ beta_delta +
                        gradient_baseline * beta_baseline_delta)

                if (nll_left > nll_right or
                        np.isnan(nll_left) or
                        np.isnan(nll_right)):
                    learning_rate *= BETA
                    if verbose > 0:
                        print('Update learning_rate: ', learning_rate)
                else:
                    break

                loop_cnt += 1
                if loop_cnt >= 20:
                    break

            if iter_index == max_num_iterations - 1 and verbose_warning:
                print(f'Warning: Reaches maximum {max_num_iterations} iterations.')
            # Update beta, negtive log-likelihood.
            beta = beta_tmp
            beta_baseline = beta_baseline_tmp

            nll = nll_left
            if iter_index % 200 == 0 and verbose > 0:
                print(iter_index, nll)
            # Check convergence.
            if abs(nll - nll_old) < 1e-9:
                break
            nll_old = nll

        if verbose > 0:
            print('Total iterations:', iter_index)

        log_lambda_hat = basis @ beta + beta_baseline
        log_lambda_hat = log_lambda_hat.reshape(-1)
        return log_lambda_hat, (beta, beta_baseline, log_lambda_offset,
                                                        hessian, hessian_baseline, nll)

    @classmethod
    def poisson_regression_smoothing_spline_CV(
            cls,
            spikes,
            time_line,
            constant_fit=False,
            log_lambda_offset=0,
            lambda_tuning_list=[0],
            n_splits=5,
            learning_rate=0.5,
            max_num_iterations=2000,
            verbose=True):
        """CV for the smoothing spline fitting for selecting tuning parameter.

        We split the data into K roughly equal-sized parts; We fit the model to the 
        other K − 1 parts of the data, and calculate the prediction error of the 
        fitted model when predicting the kth part of the data. Then the 
        cross-validation estimate of prediction error is the mean of all repetition.
        Typical choices of K are 5 or 10 (see below). The case K = N is known as 
        leave-one-out cross-validation. Given a set of models f(x,α) indexed by a t
        uning parameter α, Our final chosen model is f(x,αˆ), which we then fit to 
        all the data.
    
        Hastie, Tibshirani, and Friedman 2009 - The elements of statistiacal
        learning, sec. 7.10.1.

        Args:
            spikes: num_trials x num_spike_bins
            basis: num_samples x num_basis
        """
        kf = sklearn.model_selection.KFold(
                n_splits=n_splits, shuffle=True, random_state=1)
        kf.get_n_splits(spikes)

        nll_test_array = np.zeros([len(lambda_tuning_list), n_splits])

        for lmbd_idx, lmbd_tuning in enumerate(lambda_tuning_list):
            for itr, (train_index, test_index) in enumerate(kf.split(spikes)):
                if verbose:
                    print("TRAIN:", train_index, "TEST:", test_index)

                log_lmbda_hat, beta_hat = cls.poisson_regression_smoothing_spline(
                        spikes[train_index],
                        time_line,
                        constant_fit,
                        log_lambda_offset,
                        lambda_tuning=lmbd_tuning,
                        learning_rate=learning_rate,
                        max_num_iterations=max_num_iterations,
                        verbose=verbose)

                nll_test = cls.spike_trains_neg_log_likelihood(
                        log_lmbda_hat,
                        spikes[test_index])
                nll_test_array[lmbd_idx, itr] = nll_test

        return nll_test_array

    @classmethod
    def sigmoid(cls, x):
        return 1 / (1 + np.exp(-x))

    @classmethod
    def logistic_regression_smoothing_spline(
            cls,
            spikes,
            time_line,
            constant_fit=False,
            lambda_tuning=1e-8,
            lambda_baseline_tuning=0,
            learning_rate=0.5,
            max_num_iterations=100,
            beta_initial=None,
            beta_baseline_initial=None,
            basis=None,
            Omega=None,
            num_knots=100,
            verbose=0,
            verbose_warning=True):
        """Fit the inhomogeneous point process using basis fit.

        NOTE: This is just a test function, not for strict application.

        The beta is fitted using Newton's method.

        Args:
            spikes: num_trials x num_spike_bins
            basis: num_samples x num_basis
            method: 'newton', 'gradient'
            verbose: 3 levels. 0. quite. 1. iteration info. 2. with plots.

        Returns:
            log_lambda_hat, (beta, beta_baseline).
        """
        num_trials, num_spike_bins = spikes.shape
        spikes_cum = spikes.sum(axis=0)
        print('spikes.shape', spikes.shape)

        if constant_fit:
            basis = np.zeros((num_spike_bins, 1))
            Omega = np.zeros((1, 1))
        # In this case, no smoothing penalty.
        elif basis is not None and Omega is None:
            num_samples, num_basis = basis.shape
        elif basis is None and Omega is None:
            # spikes_cum_tmp = spikes.copy()
            # spikes_cum_tmp[0], spikes_cum_tmp[-1] = 1, 1
            # dt = time_line[1] - time_line[0]
            # knots = np.where(spikes_cum_tmp !=0 )[0] * dt
            basis, Omega = cls.construct_basis_omega(
                    time_line, knots=num_knots, verbose=verbose==2)
        num_samples, num_basis = basis.shape

        if num_spike_bins != num_samples:
            raise ValueError(
                    'The length of the basis should be the same as that of spikes.')

        # beta = np.random.rand(num_basis, 1) - 5
        if beta_initial is None:
            beta = np.ones((num_basis, 1)) * 0
        else:
            beta = beta_initial

        if beta_baseline_initial is None:
            beta_baseline = -1.9
        else:
            beta_baseline = beta_baseline_initial

        logit_p_hat = basis @ beta + beta_baseline
        nll = (-spikes_cum.reshape(-1, 1).T @ logit_p_hat).sum()
        nll += np.log(1 + np.exp(logit_p_hat)).sum()
        nll_old = float("inf")
        nll_list = np.zeros(max_num_iterations)

        for iter_index in range(max_num_iterations):
            eta = basis @ beta + beta_baseline
            mu = cls.sigmoid(eta).reshape(-1, num_spike_bins)

            gradient = -basis.T @ spikes_cum.reshape(-1, 1)
            gradient += basis.T @ np.log(1 + np.exp(eta)) * num_trials
            if Omega is not None:
                gradient += 2 * num_trials * lambda_tuning * Omega @ beta

            gradient_baseline = - spikes_cum.sum() + mu.sum() * num_trials
            gradient_baseline += (2 * num_trials * lambda_baseline_tuning *
                                                        beta_baseline)

            hessian = basis.T @ (cls.sigmoid(eta) * basis)
            if Omega is not None:
                hessian += 2 * num_trials * lambda_tuning * Omega
            hessian_baseline = mu.sum() * num_trials
            hessian_baseline += 2 * num_trials * lambda_baseline_tuning

            # Gradient descent.
            # beta_delta = gradient
            # beta_baseline_delta = gradient_baseline
            # Newton's method.
            beta_delta = np.linalg.inv(hessian) @ gradient
            beta_baseline_delta = gradient_baseline / hessian_baseline
            # beta update.
            beta = beta - learning_rate * beta_delta
            beta_baseline = beta_baseline - learning_rate * beta_baseline_delta

            logit_p_hat = basis @ beta + beta_baseline
            nll = (-spikes_cum.reshape(-1, 1).T @ logit_p_hat).sum()
            nll += np.log(1 + np.exp(logit_p_hat)).sum() * num_trials
            nll_list[iter_index] = nll

            # if iter_index % 10 == 0 and verbose > 0:
            #     print(iter_index, nll)
            # # Check convergence.
            # if abs(nll - nll_old) < 1e-9:
            #     break
            # nll_old = nll

        if verbose > 0:
            print('Total iterations:', iter_index)
            plt.plot(nll_list)

        log_lambda_hat = basis @ beta + beta_baseline
        log_lambda_hat = log_lambda_hat.reshape(-1)
        return log_lambda_hat, (beta, beta_baseline, hessian, hessian_baseline, nll)


    # TODO
    @classmethod
    def least_square_regression_smoothing_spline(
            cls,
            spikes,
            time_line,
            lambda_tuning=1e-8,
            lambda_baseline_tuning=0,
            learning_rate=0.5,
            max_num_iterations=100,
            num_knots=100,
            verbose=0,
            verbose_warning=True):
        """Fit the inhomogeneous point process using basis fit.

        NOTE: This is just a test function, not for strict application.

        The beta is fitted using Newton's method.

        Args:
            spikes: num_trials x num_spike_bins
            basis: num_samples x num_basis
            method: 'newton', 'gradient'
            verbose: 3 levels. 0. quite. 1. iteration info. 2. with plots.

        Returns:
            log_lambda_hat, (beta, beta_baseline).
        """
        num_trials, num_spike_bins = spikes.shape
        spikes_cum = spikes.sum(axis=0)
        print('spikes.shape', spikes.shape)

        basis, Omega = cls.construct_basis_omega(
                time_line, knots=num_knots, verbose=verbose==2)
        num_samples, num_basis = basis.shape

        if num_spike_bins != num_samples:
            raise ValueError(
                    'The length of the basis should be the same as that of spikes.')

        beta = np.ones((num_basis, 1)) * 0
        beta_baseline = 0

        logit_p_hat = basis @ beta + beta_baseline
        nll = (-spikes_cum.reshape(-1, 1).T @ logit_p_hat).sum()
        nll += np.log(1 + np.exp(logit_p_hat)).sum()
        nll_old = float("inf")
        nll_list = np.zeros(max_num_iterations)

        for iter_index in range(max_num_iterations):
            eta = basis @ beta + beta_baseline
            mu = cls.sigmoid(eta).reshape(-1, num_spike_bins)

            gradient = -basis.T @ spikes_cum.reshape(-1, 1)
            gradient += basis.T @ np.log(1 + np.exp(eta)) * num_trials
            if Omega is not None:
                gradient += 2 * num_trials * lambda_tuning * Omega @ beta

            gradient_baseline = - spikes_cum.sum() + mu.sum() * num_trials
            gradient_baseline += (2 * num_trials * lambda_baseline_tuning *
                                                        beta_baseline)

            hessian = basis.T @ (cls.sigmoid(eta) * basis)
            if Omega is not None:
                hessian += 2 * num_trials * lambda_tuning * Omega
            hessian_baseline = mu.sum() * num_trials
            hessian_baseline += 2 * num_trials * lambda_baseline_tuning

            # Gradient descent.
            # beta_delta = gradient
            # beta_baseline_delta = gradient_baseline
            # Newton's method.
            beta_delta = np.linalg.inv(hessian) @ gradient
            beta_baseline_delta = gradient_baseline / hessian_baseline
            # beta update.
            beta = beta - learning_rate * beta_delta
            beta_baseline = beta_baseline - learning_rate * beta_baseline_delta

            logit_p_hat = basis @ beta + beta_baseline
            nll = (-spikes_cum.reshape(-1, 1).T @ logit_p_hat).sum()
            nll += np.log(1 + np.exp(logit_p_hat)).sum() * num_trials
            nll_list[iter_index] = nll


        log_lambda_hat = basis @ beta + beta_baseline
        log_lambda_hat = log_lambda_hat.reshape(-1)
        return log_lambda_hat, (beta, beta_baseline, hessian, hessian_baseline, nll)

