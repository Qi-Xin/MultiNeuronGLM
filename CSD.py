# -*- coding: utf-8 -*-

import sys
# sys.path.append('D:/Github/rCSD')
import numpy as np
import pandas as pd
import os
import time
import copy
import matplotlib.pyplot as plt
import pickle
import copy

from forward_models import b_fwd_1d, fwd_model_1d
import utility_functions as utils
from scipy.optimize import minimize


class CSD:
    
    def __init__(self, dataset, **kwargs):
        
        """ get necessary information from the input dataset for easy use """
        self.dataset = dataset
        self.x = self.dataset.x
        self.t = self.dataset.t
        self.nx, self.nt, self.ntrial, self.y = utils.check_and_get_size(self.dataset.lfp)
        self.pred_csd = {}
        self.mse = {}
        self.mse_time_point = {}
        
        if self.dataset.source == "Allen":
            self.nz = kwargs.pop('nz', self.dataset.nx)
            self.a = kwargs.pop('a', self.x.min())
            self.b = kwargs.pop('b', self.x.max())
            # CSD spatial locations , microns
            self.z = np. linspace(self.a, self.b, self.nz)[:, None]
            self.t = self.t*1e3     # in ms

            self.R = kwargs.pop('R', 200)
            self.lam_smooth = kwargs.pop('lam_smooth', 1)
            self.lam_region = kwargs.pop('lam_region', 2)
            self.lam_lasso = kwargs.pop('lam_lasso', 0.7)
            self.method = kwargs.pop('method', 'L1')
            self.intervals_csd_raw = np.round(self.dataset.intervals_lfp*self.nz/self.nx).astype(int)
            if len(self.intervals_csd_raw)!=2:
                self.intervals_csd = self.intervals_csd_raw[np.ix_([0, 2, 3, 5])]
            else:
                self.intervals_csd = self.intervals_csd_raw
            
        elif self.dataset.source == "Tobias":
            self.nz = kwargs.pop('nz', 23*5)
            self.a = kwargs.pop('a', 0+10)
            self.b = kwargs.pop('b', 2300-10)
            # CSD spatial locations , microns
            self.z = np. linspace(self.a, self.b, self.nz)[:, None]

            self.R = kwargs.pop('R', 150)
            self.lam_smooth = kwargs.pop('lam_smooth', 5)
            self.lam_region = kwargs.pop('lam_region', 5)
            self.lam_lasso = kwargs.pop('lam_lasso', 15)
            self.method = kwargs.pop('method', 'L1')
            self.intervals_csd = np.array([0, np.nonzero(self.z > 1000)[0][0],
                                      np.nonzero(self.z > 1500)[0][0], self.nz])
            
        elif self.dataset.source == "Simulation":
            self.nz = kwargs.pop('nz', 23*5)
            self.a = kwargs.pop('a', 0)
            self.b = kwargs.pop('b', 2300)
            # CSD spatial locations , microns
            self.z = np. linspace(self.a, self.b, self.nz)[:, None]

            self.R = kwargs.pop('R', 150)
            self.lam_smooth = kwargs.pop('lam_smooth', 1)
            self.lam_region = kwargs.pop('lam_region', 5)
            self.lam_lasso = kwargs.pop('lam_lasso', 15)
            self.method = kwargs.pop('method', 'L1')
            self.intervals_lfp = np.array([0, np.nonzero(self.x > 1000)[0][0],
                                  np.nonzero(self.x > 1500)[0][0], self.nx])
            self.structure_acronyms = np.array(['Superficial', 'Medium', 'Deep'])
            self.intervals_csd = np.array([0, np.nonzero(self.z > 1000)[0][0],
                                      np.nonzero(self.z > 1500)[0][0], self.nz])
        else:
            raise ValueError

        self.A = b_fwd_1d(self.z.T-self.x, self.R)
        self.change_target(using_normalize=True)
        
    def change_target(self, using_smooth=False, using_mean=False, using_normalize=False, trial=None, method=None):
        if using_smooth:
            assert hasattr(self.dataset, 'lfp_smooth'), "You need to presmooth the LFP before fitting CSD on smoothed LFP"
            if using_mean:
                self.y = self.dataset.mean_lfp_smooth
            else:
                self.y = self.dataset.lfp_smooth
        else:
            if using_mean:
                self.y = self.dataset.mean_lfp
            else:
                self.nx, self.nt, self.ntrial, self.y = utils.check_and_get_size(self.dataset.lfp)
            
        if using_normalize:
            self.y = utils.normalize_var(self.y)
            
        if trial == None:
            pass
        else:
            self.y = self.y[:,:,trial]
            if self.y.ndim == 2:
                self.y = self.y[:,:,None]
        self.y = copy.deepcopy(self.y)      # So that the data in the dataset object remains unchanged for further reference. 
        if method != None:
            self.method = method
        self.nx, self.nt, self.ntrial, self.y = utils.check_and_get_size(self.y)
        
    def update_rcsd_paras(self, **kwargs):
        self.R = kwargs.pop('R', self.R)
        self.lam_smooth = kwargs.pop('lam_smooth', self.lam_smooth)
        self.lam_region = kwargs.pop('lam_region', self.lam_region)
        self.lam_lasso = kwargs.pop('lam_lasso', self.lam_lasso)
        self.method = kwargs.pop('method', self.method)
        
    def fit_rcsd(self, mean_converge=False, verbose=False, **kwargs):
        if mean_converge==True:
            assert hasattr(self, 'mean_csd'), "To use mean converge, you must have a mean CSD! "
        self.update_rcsd_paras(**kwargs)
        start_t = time.process_time()
        if mean_converge:
            self.pred_csd['rcsd'] = rcsd_multi_trial_parallel(
                self.y, self.A, self.intervals_csd, self.lam_smooth, self.lam_region, self.lam_lasso, self.method, self.mean_csd)
        else:
            self.pred_csd['rcsd'] = rcsd_multi_trial_parallel(
                self.y, self.A, self.intervals_csd, self.lam_smooth, self.lam_region, self.lam_lasso, self.method)
        end_t = time.process_time()
        self.get_mse('rcsd')
        if verbose:
            return self.pred_csd['rcsd']
        
    def get_mse(self,method):
        pred_lfp = np.zeros((self.y.shape))
        for itrial in range(self.ntrial):
            pred_lfp[:,:,itrial] = self.A@self.pred_csd[method][:,:,itrial]
        self.mse[method] = np.sum( (pred_lfp - self.y)**2, axis=(0,1) )
        self.mse_time_point[method] = np.sum( (pred_lfp - self.y)**2, axis=(0) )
        
    def fit_tcsd(self):
        self.pred_csd['tcsd'] = tcsd(self.y)
        return self.pred_csd['tcsd']
        
    def fit_gpcsd(self, reload_model=True, reload_name='gpcsd_model_.pkl' ):
        from gpcsd.gpcsd1d import GPCSD1D
        from gpcsd.covariances import GPCSD1DSpatialCovSE
        from gpcsd.covariances import GPCSDTemporalCovMatern
        from gpcsd.covariances import GPCSDTemporalCovSE, GPCSDHalfNormalPrior
        n_restarts = 10
        spatial_cov = GPCSD1DSpatialCovSE(self.x, a=self.a-200, b=self.b+200)
        matern_cov = GPCSDTemporalCovMatern(self.t)
        matern_cov.params['ell']['prior'].set_params(1., 20.)
        SE_cov = GPCSDTemporalCovSE(self.t)
        SE_cov.params['ell']['prior'].set_params(30., 100.)
        sig2n_prior = [GPCSDHalfNormalPrior(0.1) for i in range(self.nx)]
        # gpcsd_model = GPCSD1D(self.y, self.x, self.t, 
        #                     sig2n_prior=sig2n_prior,
        #                     spatial_cov=spatial_cov, 
        #                     temporal_cov_list=[SE_cov, matern_cov], a=self.a-200, b=self.b+200)
        gpcsd_model = GPCSD1D(self.y, self.x, self.t, 
                            sig2n_prior=sig2n_prior,
                            spatial_cov=spatial_cov, 
                            temporal_cov_list=[SE_cov], a=self.a-200, b=self.b+200)
        # gpcsd_model = GPCSD1D(self.y, self.x, self.t)
        gpcsd_model.R['value'] = 200.0

        if reload_model and os.path.isfile(reload_name):
            with open(reload_name, 'rb') as f:
                params = pickle.load(f)
            gpcsd_model.restore_model_params(params)
        else:
            gpcsd_model.fit(n_restarts=n_restarts, fix_R=True, verbose=True)
            # gpcsd_model.fit(n_restarts=n_restarts, verbose=True)

        gpcsd_model.predict(self.z, self.t)
        self.gpcsd = gpcsd_model
        self.pred_csd['gpcsd'] = gpcsd_model.csd_pred
        print(gpcsd_model)

        # from gpcsd.gpcsd1d import GPCSD1D
        # start_t = time.process_time()
        # gpcsd = GPCSD1D(self.y, self.x, self.t)
        # gpcsd.fit(n_restarts=10)
        # gpcsd.predict(self.z, self.t)
        # self.pred_csd['gpcsd'] = gpcsd.csd_pred
        # end_t = time.process_time()
        # print('gpCSD took %0.2f s ' % (end_t - start_t))
        
    def fit_kcsd(self):
        from kcsd import KCSD1D
        start_t = time.process_time()
        print("Start fitting kernel CSD")
        # use first five trials concatenated for estimating parameters (for computational reasons)
        R_true = self.R
        deltaz = (self.b-self.a)/self.nz
        ntrial_cv = min(self.ntrial, 5)
        if self.nt >= 10:
            t_subset = np.arange(0, self.nt, self.nt//10)
        kcsd_model = KCSD1D(self.x, self.y[:, t_subset, :ntrial_cv].reshape((self.nx, -1)), gdx=deltaz, h=R_true)
        kcsd_model.cross_validate(Rs=np.linspace(100, 1000, 10))
        kcsd_R = kcsd_model.R       # Note that this R is not the R in GPCSD or rCSD
        kcsd_lambda = kcsd_model.lambd
        # Predict on test set
        pred_csd_temp = np.zeros((self.nz, self.nt, self.ntrial))
        for i in range(self.y.shape[2]):
            kcsd_model_tmp = KCSD1D(self.x, self.y[:,:,i].squeeze(), gdx=deltaz, h=R_true,
                                    R_init=kcsd_R, lambd=kcsd_lambda)
            pred_csd_temp[:, :, i] = kcsd_model_tmp.values()
        self.pred_csd['kcsd'] = pred_csd_temp
        end_t = time.process_time()
        print('kCSD took %0.2f s ' % (end_t - start_t))
        

        
    def cv(self, center, n_cv=None, n_range=9, trial=[0], mean_converge=False, **kwargs):
        if trial is None:
            trial = np.arange(self.ntrial)
        if mean_converge==True:
            assert hasattr(self, 'mean_csd'), "To use mean converge, you must have a mean CSD! "
            mean_csd = self.mean_csd
        else:
            mean_csd = None
        if n_cv == None:
            n_cv = self.nx
        R_list = kwargs.pop('R_list', np.logspace(np.log10(center['R'])-1.0, np.log10(center['R'])+1.0, n_range))
        lam_lasso_list = kwargs.pop('lam_lasso_list', np.logspace(np.log10(center['lam_lasso'])-2.0, np.log10(center['lam_lasso'])+2.0, n_range))
        lam_smooth_list = kwargs.pop('lam_smooth_list', np.logspace(np.log10(center['lam_smooth'])-2.0, np.log10(center['lam_smooth'])+2.0, n_range))
        lam_region_list = kwargs.pop('lam_region_list', np.logspace(np.log10(center['lam_region'])-2.0, np.log10(center['lam_region'])+2.0, n_range))
        
        cv_error_R = np.zeros((n_range, n_cv, len(trial)))
        cv_error_lasso = np.zeros((n_range, n_cv, len(trial)))
        cv_error_smooth = np.zeros((n_range, n_cv, len(trial)))
        cv_error_region = np.zeros((n_range, n_cv, len(trial)))
        
        for itrial in trial:
            Y_smo = self.y[:,:,itrial]
            my_dicts = center
            z = self.z
            x = self.x
            intervals_csd = self.intervals_csd
            
            if R_list is not None:
                cv_error_R[:,:,itrial] = cv_single_parallel(Y_smo, z,x, intervals_csd,
                                                'R',R_list, copy.deepcopy(my_dicts), 
                                                n_cv, self.method, mean_csd=mean_csd)
            if lam_lasso_list is not None:
                cv_error_lasso[:,:,itrial] = cv_single_parallel(Y_smo, z,x, intervals_csd,
                                                    'lam_lasso',lam_lasso_list, 
                                                    copy.deepcopy(my_dicts), n_cv, 
                                                    self.method, mean_csd=mean_csd)
            if lam_smooth_list is not None:
                cv_error_smooth[:,:,itrial] = cv_single_parallel(Y_smo, z,x, intervals_csd,
                                                     'lam_smooth',lam_smooth_list,
                                                     copy.deepcopy(my_dicts), n_cv, 
                                                     self.method, mean_csd=mean_csd)
            if lam_region_list is not None:
                cv_error_region[:,:,itrial] = cv_single_parallel(Y_smo, z,x, intervals_csd,
                                                     'lam_region',lam_region_list, 
                                                     copy.deepcopy(my_dicts), n_cv, 
                                                     self.method, mean_csd=mean_csd)

        ann_size = 25
        plt.figure(figsize=(13, 10))
        plt.subplot(221)
        plt.xscale('log')
        plt.yscale('log')
        if R_list is None:
            plt.text(1.0, 1.0, "NA", fontweight="bold",size=ann_size, weight='bold')
        else:
            plt.errorbar( R_list , np.mean(cv_error_R,axis=(1,2)) , yerr=np.std(cv_error_R,axis=(1,2))/np.sqrt(n_cv))
            self.cv_error_R = cv_error_R
            print("Lowest:", np.min(np.mean(cv_error_lasso,axis=(1,2))))
        plt.ylabel('CV error')
        plt.xlabel('R  (radius of the CSD cylinder)')
        # plt.title('Relationship between CV prediction error and hyperparameter R')

        plt.subplot(222)
        plt.xscale('log')
        plt.yscale('log')
        if lam_lasso_list is None:
            plt.text(1.0, 1.0, "NA", fontweight="bold",size=ann_size, weight='bold')
        else:
            plt.errorbar( lam_lasso_list , np.mean(cv_error_lasso,axis=(1,2)) , yerr=np.std(cv_error_lasso,axis=(1,2))/np.sqrt(n_cv))
            self.cv_error_lasso = cv_error_lasso
            print("Lowest:", np.min(np.mean(cv_error_lasso,axis=(1,2))))
        plt.ylabel('CV error')
        plt.xlabel(r'$\lambda$  (Lasso)')
        
        plt.subplot(223)
        plt.xscale('log')
        plt.yscale('log')
        if lam_smooth_list is None:
            plt.text(1.0, 1.0, "NA", fontweight="bold",size=ann_size, weight='bold')
        else:
            plt.errorbar( lam_smooth_list , np.mean(cv_error_smooth,axis=(1,2)) , yerr=np.std(cv_error_smooth,axis=(1,2))/np.sqrt(n_cv))
            self.cv_error_smooth = cv_error_smooth
            print("Lowest:", np.min(np.mean(cv_error_lasso,axis=(1,2))))
        plt.ylabel('CV error')
        plt.xlabel(r'$\lambda_s$  (Smoothness)')
        # plt.title('Relationship between CV prediction error and hyperparameter R')

        plt.subplot(224)
        plt.xscale('log')
        plt.yscale('log')
        if lam_region_list is None:
            plt.text(1.0, 1.0, "NA", fontweight="bold",size=ann_size, weight='bold')
        else:
            plt.errorbar( lam_region_list , np.mean(cv_error_region,axis=(1,2)) , yerr=np.std(cv_error_region,axis=(1,2))/np.sqrt(n_cv))
            self.cv_error_region = cv_error_region
            print("Lowest:", np.min(np.mean(cv_error_lasso,axis=(1,2))))
        plt.ylabel('CV error')
        plt.xlabel(r'$\lambda_c$  (Charge conservation)')
        # plt.title('Relationship between CV prediction error and hyperparameter R')

        plt.show()
        
    def fit(self, *args):
        if len(args) == 0:
            args = ['except rCSD']
        if 'except rCSD' or 'all' or 'tcsd' in args:
            self.fit_tcsd()
        if 'except rCSD' or 'all' or 'kcsd' in args:
            self.fit_kcsd()
        if 'except rCSD' or 'all' or 'gpcsd' in args:
            self.fit_gpcsd()
        if 'all' or 'rcsd' in args:
            self.fit_rcsd()
            
    
    def plot_lfp(self, trial=0):
        target1 = self.dataset.lfp[:,:,trial]
        target2 = self.dataset.lfp_smooth[:,:,trial]
        v_range = np.max(target1, target2)
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.imshow(target1, aspect='auto', cmap='bwr',vmin=-v_range,vmax=v_range)
        plt.title('LFP')
        plt.xlabel('Time (ms)')
        plt.subplot(122)
        plt.imshow(target2, aspect='auto', cmap='bwr',vmin=-v_range,vmax=v_range)
        plt.title('LFP (smooth)')
        plt.xlabel('Time (ms)')
        plt.show()
    
    def plot_result(self, results_from='rcsd', trial=0, plot_nonzero=False):
        plt_csd = self.pred_csd[results_from]
        plt_csd = plt_csd[:,:,trial]
        pred_lfp = fwd_model_1d(plt_csd, self.z, self.x, self.R)
        
        ann_size = 25
        fig = plt.figure(figsize=(15, 15))
        n_plot_tot = 3
        if plot_nonzero:
            n_plot_tot += 1
        if self.dataset.source == "Simulation":
            n_plot_tot += 1
        
        n_plot = 0
        if self.dataset.source == "Simulation":
            n_plot += 1
            ax = plt.subplot(n_plot_tot, 1, n_plot)
            plt.text(-0.2, 1.0, chr(64+n_plot), fontweight="bold",
                    transform=ax.transAxes, size=ann_size, weight='bold')
            plt.imshow(utils.normalize(self.dataset.gt_csd), vmin=-1, vmax=1, cmap='bwr')
            utils.add_label_csd("Predicted CSD", self.z)
        
        n_plot += 1
        ax = plt.subplot(n_plot_tot, 1, n_plot)
        plt.text(-0.2, 1.0, chr(64+n_plot), fontweight="bold",
                transform=ax.transAxes, size=ann_size, weight='bold')
        plt.imshow(utils.normalize(plt_csd), vmin=-1, vmax=1, cmap='bwr')
        utils.add_label_csd("Predicted CSD", self.z)

        if plot_nonzero:
            n_plot += 1
            ax = plt.subplot(n_plot_tot, 1, n_plot)
            plt.text(-0.2, 1.0, chr(64+n_plot), fontweight="bold",
                    transform=ax.transAxes, size=ann_size, weight='bold')
            plt.spy(plt_csd)
            utils.add_label_csd("Non-zero coefficients of predicted CSD", self.z)

        n_plot += 1
        ax = plt.subplot(n_plot_tot, 1, n_plot)
        plt.text(-0.2, 1.0, chr(64+n_plot), fontweight="bold",
                transform=ax.transAxes, size=ann_size, weight='bold')
        plt.imshow(utils.normalize(self.y[:,:,trial]), vmin=-1, vmax=1, cmap='bwr')
        plt.gca().set_aspect(self.nz/self.nx)
        utils.add_label_csd("Experimental LFP", self.x)

        n_plot += 1
        ax = plt.subplot(n_plot_tot, 1, n_plot)
        plt.text(-0.2, 1.0, chr(64+n_plot), fontweight="bold",
                transform=ax.transAxes, size=ann_size, weight='bold')
        plt.imshow(utils.normalize(pred_lfp), vmin=-1, vmax=1, cmap='bwr')
        plt.gca().set_aspect(self.nz/self.nx)
        utils.add_label_csd("Predicted LFP", self.x)
        # plot LFP
        # plot pred LFP
        # plot gt CSD (if available)
        # plot CSD
        # plot non-zero CSD (if rCSD)
        pass
    
    def plot_compare(self, results_from=None, trial=0):

        vmlfp = 1
        vmcsd = 1
        
        if self.dataset.source == 'Simulation':
            tot_im = 6
        else:
            tot_im = 5
            
        i = 1
        plt.rcParams.update({'font.size': 12})
        f = plt.figure(figsize=(16, 4))
        ax = plt.subplot(1, tot_im, i)
        plt.subplot(1, tot_im, i)
        plt.imshow(utils.normalize(self.y[:,:,trial]), aspect='auto', vmin=-vmlfp, vmax=vmlfp, cmap='bwr', 
                   extent=[self.t[0, 0], self.t[-1, 0], self.x[-1, 0], self.x[0, 0]])
        plt.title('Noisy LFP')
        plt.ylabel('Depth')
        plt.xlabel('Time')
        i += 1
        axtmp = plt.subplot(1, tot_im, i, sharey = ax)
        axtmp.set_yticklabels([])
        if self.dataset.source == 'Simulation':
            plt.subplot(1, tot_im, i, sharey = ax)
            plt.imshow(utils.normalize(self.dataset.gt_csd[:,:,trial]), aspect='auto', vmin=-vmcsd, vmax=vmcsd, cmap='bwr', 
                       extent=[self.t[0, 0], self.t[-1, 0], self.z[-1, 0], self.z[0, 0]])
            plt.title('Ground truth CSD')
            plt.xlabel('Time')
        i += 1
        plt.subplot(1, tot_im, i, sharey = ax)
        plt.imshow(utils.normalize(self.pred_csd['tcsd'][:,:,trial]), aspect='auto', vmin=-vmcsd, vmax=vmcsd, cmap='bwr', 
                   extent=[self.t[0, 0], self.t[-1, 0], self.z[-1, 0], self.z[0, 0]])
        plt.title('tCSD')
        plt.xlabel('Time')
        i += 1
        plt.subplot(1, tot_im, i, sharey = ax)
        plt.imshow(utils.normalize(self.pred_csd['gpcsd'][:,:,trial]), aspect='auto', vmin=-vmcsd, vmax=vmcsd, cmap='bwr', 
                   extent=[self.t[0, 0], self.t[-1, 0], self.z[-1, 0], self.z[0, 0]])
        plt.title('GPCSD')
        plt.xlabel('Time')
        i += 1
        plt.subplot(1, tot_im, i, sharey = ax)
        plt.imshow(utils.normalize(self.pred_csd['kcsd'][:,:,trial]), aspect='auto', vmin=-vmcsd, vmax=vmcsd, cmap='bwr', 
                        extent=[self.t[0, 0], self.t[-1, 0], self.z[-1, 0], self.z[0, 0]])
        plt.title('kCSD')
        plt.xlabel('Time')
        i += 1
        plt.subplot(1, tot_im, i, sharey = ax)
        im = plt.imshow(utils.normalize(self.pred_csd['rcsd'][:,:,trial]), aspect='auto', vmin=-vmcsd, vmax=vmcsd, cmap='bwr', 
                        extent=[self.t[0, 0], self.t[-1, 0], self.z[-1, 0], self.z[0, 0]])
        plt.title('rCSD')
        plt.xlabel('Time')

        clb = f.colorbar(im, ax=f.axes)
        clb.ax.set_title('a.u.')
        plt.show()

    
    def plot_compare_analysis(self, results_from=None):
        # assert there are more than one trial
        assert self.ntrial > 1, "In order to use compare analysis, "
        import scipy.interpolate
        import scipy.stats
        xshort = self.x[1:-1]
        csd_interior_electrodes = np.zeros((self.nx-2, self.nt, self.ntrials))
        for trial in range(self.ntrials):
            csdinterp = scipy.interpolate.RectBivariateSpline(self.z, self.t, self.csd[:, :, trial])
            csd_interior_electrodes[:, :, trial] = csdinterp(xshort, self.t)
            
        tcsd_meansqerr = np.nanmean(np.square(self.tcsd_pred[1:-1, :, :] - utils.normalize(csd_interior_electrodes[:, :,:])), 
                                    axis=(0, 1)) 
        gpcsd_meansqerr = np.nanmean(np.square(self.gpcsd_pred-utils.normalize(self.csd_true)), axis=(0, 1)) 
        kcsd_meansqerr = np.nanmean(np.square(self.kcsd_pred[:, :, :] - utils.normalize(self.csd_true[1:, :, :])), axis=(0, 1)) 
        rcsd_meansqerr = np.nanmean(np.square(self.csd_lasso - utils.normalize(self.csd_true[:, :, :])), axis=(0, 1)) 

        plt.figure()
        plt.boxplot([tcsd_meansqerr,gpcsd_meansqerr, kcsd_meansqerr,rcsd_meansqerr ], labels=['tCSD','GPCSD', 'kCSD','rCSD'])
        plt.ylabel('MSE')
        plt.show()

        print('tCSD average MSE across trials: %0.3g' % np.mean(tcsd_meansqerr))
        print('kCSD average MSE across trials: %0.3g' % np.mean(kcsd_meansqerr))
        print('GPCSD average MSE across trials: %0.3g' % np.mean(gpcsd_meansqerr))
        print('rCSD average MSE across trials: %0.3g' % np.mean(rcsd_meansqerr))
        
        # plot box plot
        # plot scatter plot
        pass
    
    def plot_spectrum(self, results_from=None, channel=None, trial=None, noplot=True):
        # plot gt csd if available
        # plot all available keys in {pred_csd} or all keys in {results_from}
        from scipy import signal
        raw_lfp = self.y
        if trial != None:
            raw_lfp = raw_lfp[:, :, [trial]]
        if channel != None:
            raw_lfp = raw_lfp[[channel], :, :]
        Pxx_spec_lfp_total = 0
        for ichannel in range(raw_lfp.shape[0]):
            for itrial in range(raw_lfp.shape[2]):
                f_lfp, Pxx_spec_lfp = signal.periodogram(raw_lfp[ichannel, :, itrial], self.dataset.fps, scaling='spectrum')
                Pxx_spec_lfp_total += Pxx_spec_lfp
        if noplot == False:
            plt.figure()
            plt.semilogy(f_lfp[1:], np.sqrt(Pxx_spec_lfp_total[1:]),label='LFP')    # Starting from 1 just to delete the first entry
            # plt.ylim([1e-4, 5e1])
            plt.xlim([0, 5e1])
            plt.legend()
            plt.title('Spectrum of LFP ')
            plt.xlabel('frequency [Hz]')
            plt.ylabel('Spectrum')
        else:
            return (f_lfp[1:], np.sqrt(Pxx_spec_lfp_total[1:]))

    def plot_spectrum_running(self, channel=None, speed_threshold=1):
        from scipy import signal
        raw_lfp = self.y
        if channel is not None:
            raw_lfp = raw_lfp[channel, :, :]
        self.running_trial_index = self.dataset.running_trial_index
        self.stationary_trial_index = self.dataset.stationary_trial_index
        lfp_running = raw_lfp[:,:,self.running_trial_index]
        lfp_stationary = raw_lfp[:,:,self.stationary_trial_index]

        plt.figure()
        # Check for shape
        f_lfp, Pxx_spec_lfp = signal.periodogram(raw_lfp[0, :, 0], self.dataset.fps, scaling='spectrum')
        nfreq = len(Pxx_spec_lfp)
        plot_lfp = lfp_running
        Pxx_spec_lfp_total = np.zeros((plot_lfp.shape[2],nfreq))
        for itrial in range(plot_lfp.shape[2]):
            Pxx_spec_lfp_channel = 0
            for ichannel in range(plot_lfp.shape[0]):
                f_lfp, Pxx_spec_lfp = signal.periodogram(plot_lfp[ichannel, :, itrial], self.dataset.fps, scaling='spectrum')
                Pxx_spec_lfp_channel += Pxx_spec_lfp
            Pxx_spec_lfp_total[itrial, :] = Pxx_spec_lfp_channel
        x = f_lfp[1:]
        y = np.mean(Pxx_spec_lfp_total[:,1:], axis=0)
        ci = np.std(Pxx_spec_lfp_total[:,1:], axis=0)/np.sqrt(plot_lfp.shape[2])
        plt.semilogy(x, y, label='Running', color='r')
        plt.fill_between(x, (y-ci), (y+ci), color='r', alpha=.3)

        plot_lfp = lfp_stationary
        Pxx_spec_lfp_total = np.zeros((plot_lfp.shape[2],nfreq))
        for itrial in range(plot_lfp.shape[2]):
            Pxx_spec_lfp_channel = 0
            for ichannel in range(plot_lfp.shape[0]):
                f_lfp, Pxx_spec_lfp = signal.periodogram(plot_lfp[ichannel, :, itrial], self.dataset.fps, scaling='spectrum')
                Pxx_spec_lfp_channel += Pxx_spec_lfp
            Pxx_spec_lfp_total[itrial, :] = Pxx_spec_lfp_channel
        x = f_lfp[1:]
        y = np.mean(Pxx_spec_lfp_total[:,1:], axis=0)
        ci = np.std(Pxx_spec_lfp_total[:,1:], axis=0)/np.sqrt(plot_lfp.shape[2])
        plt.semilogy(x, y,label='Stationary', color='b')
        plt.fill_between(x, (y-ci), (y+ci), color='b', alpha=.3)
        plt.legend()
        plt.title('Spectrum of LFP ')
        plt.xlabel('frequency [Hz]')
        plt.ylabel('Spectrum')

        print(x[np.argmax(y)])
# if __name__ == "__main__":
#     rCSD()
#     winsound.Beep(500, 3000)

def plot_multi(im_list, title_list, zaxis_same=False):
    raise ValueError("Unfinished function! ")
    # assert len(im_list) == len(names_list), "Number of images must equal the number of subplot titles! "
    # tot_im = len(im_list)
    # if zaxis_same == False:
    #     for i in range(tot_im):
    
    # plot gt csd if available
    # plot all available keys in {pred_csd} or all keys in {results_from}
            


def cross_validation_single(y, z,x, intervals_csd, para_name, para_range, my_dicts):
    raise ValueError("Unfinished!")

    """
    Args:
        y ([type]): [description]
        z ([type]): [description]
        x ([type]): [description]
        intervals_csd ([type]): [description]
        para_name ([type]): [description]
        para_range ([type]): [description]
        my_dicts ([type]): [description]

    Returns:
        [type]: [description]
    """    
    Cv_Rep_Trial = 24
    nx = len(x)
    delete_electrode_list = np.random.choice(range(nx), Cv_Rep_Trial, replace=False)
    cv_error = np.zeros((len(para_range),Cv_Rep_Trial))
    frames = np.arange(0,10)
    
    for i, para in enumerate(para_range):
        print("Cross validating "+para_name+": "+chr(para))
        my_dicts[para_name] = para
        A = b_fwd_1d(z.T-x, my_dicts['R'])
        
        for i_delete, delete_electrode in enumerate(delete_electrode_list):

            csd_temp = rcsd_single_trial(np.delete(y[:,frames], delete_electrode, axis=0), 
                                           np.delete(A, delete_electrode, axis=0), intervals_csd, 
                                           my_dicts['lam_smooth'], my_dicts['lam_region'], my_dicts['lam_lasso'])
            y_temp = A@csd_temp
            cv_error[i,i_delete] =  np.linalg.norm(y_temp[delete_electrode,:]-y[delete_electrode,frames])**2
    return cv_error

def cv_single_parallel(y, z,x, intervals_csd, para_name, para_range, my_dicts, n_cv, method='L1',mean_csd=None):
    if type(para_range)==type(None):
        return None
    
    nx = len(x)
    
    # delete_electrode_list = np.random.choice(range(nx), Cv_Rep_Trial, replace=False)
    if n_cv<=15:
        delete_electrode_list = np.arange(3, nx, nx//n_cv)
        delete_electrode_list = delete_electrode_list[:n_cv]
    else:
        delete_electrode_list = np.random.choice(range(nx), n_cv, replace=False)
    Cv_Rep_Trial = len(delete_electrode_list)
    cv_error = np.zeros((len(para_range),Cv_Rep_Trial))
    frames = np.arange(0,y.shape[1])
    
    for i, para in enumerate(para_range):
        my_dicts[para_name] = para
        A = b_fwd_1d(z.T-x, my_dicts['R'])
        print(para_name,para)
        Y_pool = np.zeros((y.shape[0]-1,len(frames),Cv_Rep_Trial))
        A_pool = np.zeros((A.shape[0]-1,A.shape[1],Cv_Rep_Trial))
        for i_delete, delete_electrode in enumerate(delete_electrode_list):
            Y_pool[:,:,i_delete] = np.delete(y[:,frames], delete_electrode, axis=0)
            A_pool[:,:,i_delete] = np.delete(A, delete_electrode, axis=0)

        csd_pool = rcsd_multi_trial_parallel(Y_pool, A_pool, intervals_csd, 
                                           my_dicts['lam_smooth'], my_dicts['lam_region'], my_dicts['lam_lasso'],method,mean_csd)
        for i_delete, delete_electrode in enumerate(delete_electrode_list):
            y_temp = A@csd_pool[:,:,i_delete]
            cv_error[i,i_delete] =  np.linalg.norm(y_temp[delete_electrode,:]-y[delete_electrode,frames])**2
    
    return cv_error



def rcsd_multi_trial_parallel(Y_multi, A, intervals_csd, lam_smooth, lam_region, lam_lasso,method='L1',mean_csd=None):
    import multiprocessing
    from tqdm import tqdm
    # from rCSD import rcsd_single_trial
    import os
    import numpy as np
    PROCESSES = os.cpu_count()-2

    with multiprocessing.Pool(processes = PROCESSES) as pool:
        nz = A.shape[1]
        nx, nt, ntrial = Y_multi.shape
        csd = np.zeros((nz,nt,ntrial))
        ntrial = np.arange(Y_multi.shape[2]).tolist()
        
        if np.ndim(A) == 3:
            results = [pool.apply_async(rcsd_single_trial, (Y_multi[:,:,itrial], A[:,:,itrial], intervals_csd, lam_smooth, 
                                                          lam_region, lam_lasso,method,mean_csd)) for itrial in ntrial]
        else:
            results = [pool.apply_async(rcsd_single_trial, (Y_multi[:,:,itrial], A, intervals_csd, lam_smooth, 
                                                          lam_region, lam_lasso,method,mean_csd)) for itrial in ntrial]
        pool.close()
        
        i = 0
        for result in tqdm(results):
            csd[:,:,i] = result.get()
            i += 1
            
        # [result.wait() for result in results]
        # i = 0
        # for result in results:
        #     csd[:,:,i] = result.get()
        #     i += 1
        
    return csd


def rcsd_single_trial(Y, A, intervals_csd, lam_smooth, lam_region, lam_lasso, method='L1', mean_csd=None):
    nx, nz = A.shape
    nt = Y.shape[1]
    csd_lasso = np.zeros((nz, nt))
    csd_ini = np.zeros((nz))
    nregion = intervals_csd.shape[0]-1
    mat_region = np.zeros((nz, nregion))
    # mat_region[0,0] = 1
    for iregion in range(nregion):
        mat_region[intervals_csd[iregion]:intervals_csd[iregion+1], iregion].fill(1)
    zeros_vector = np.zeros((nz-1, 1))
    iden_mat_smooth = np.eye(nz-1)
    mat_diff = np.block([iden_mat_smooth, zeros_vector]) - \
        np.block([zeros_vector, iden_mat_smooth])

    if method=='L1':
        for it in range(nt):
            csd_lasso[:, it] = rcsd_single_time(Y[:, it], A, mat_region, mat_diff,
                                                csd_ini, lam_smooth, lam_region, lam_lasso)
            # print('frame being proccessing:', it)
            csd_ini = csd_lasso[:, it]
    elif method=='L2':
        for it in range(nt):
            if mean_csd is None:
                mean_csd_single_time = None
            else:
                mean_csd_single_time = mean_csd[:,it]
            csd_lasso[:, it] = rcsd_single_time_L2(Y[:, it], A, mat_region, mat_diff,
                                                   csd_ini, lam_smooth, lam_region, lam_lasso, mean_csd_single_time)
            # print('frame being proccessing:', it)
            csd_ini = csd_lasso[:, it]
    else:
        raise(ValueError,"\'method\' has to be L1 or L2!")
    return csd_lasso

def rcsd_single_time_L2(y, A, mat_region, mat_diff, csd_ini, lam_smooth, lam_region, lam_lasso, mean_csd_single_time=None):
    if mean_csd_single_time is None:
        mean_csd_single_time = np.zeros(csd_ini.shape)

    res = minimize(Loss_L2, 
                    csd_ini, 
                    args=(y, A, mat_region, mat_diff, lam_smooth, lam_region, lam_lasso, mean_csd_single_time), 
                    method='BFGS', 
                    jac=Loss_L2_der,
                    options={'disp': False,'gtol': 1e-02})
    return res.x
    
    # ### Pytorch optimize ###
    # # Convert numpy to tensor
    # lr = 1e-2
    # csd_mean_ts = torch.tensor(csd_mean)
    # A_ts = torch.tensor(A)
    # mat_region_ts = torch.tensor(mat_region)
    # mat_diff_ts = torch.tensor(mat_diff)
    # y_ts = torch.tensor(y)
    # csd_temp_ts = torch.tensor(csd_ini, requires_grad=True)
    
    # loss = 1/2*torch.sum(torch.pow(y_ts-A_ts@csd_temp_ts,2)) \
    #     + lam_smooth/2*torch.sum(torch.pow(mat_diff_ts@csd_temp_ts,2)) \
    #     + lam_region/2*torch.sum(torch.pow(mat_region_ts.T@csd_temp_ts,2)) \
    #     + lam_lasso/2*torch.sum(torch.pow(csd_temp_ts-csd_mean_ts,2))
    # optimizer = torch.optim.SGD([csd_temp_ts], lr=lr,momentum=0.9)
    
    # last_loss = np.inf
    # for epoch in range(1000):
        
    #     optimizer.zero_grad()
    #     # Forward pass
    #     # Compute Loss
    #     loss = 1/2*torch.sum(torch.pow(y_ts-A_ts@csd_temp_ts,2)) \
    #         + lam_smooth/2*torch.sum(torch.pow(mat_diff_ts@csd_temp_ts,2)) \
    #         + lam_region/2*torch.sum(torch.pow(mat_region_ts.T@csd_temp_ts,2)) \
    #         + lam_lasso/2*torch.sum(torch.pow(csd_temp_ts-csd_mean_ts,2))
        
    #     # Backward pass
    #     loss.backward()
    #     optimizer.step()
    #     # print(epoch,':', loss.item())
    #     current_loss = loss.item()
    #     if last_loss - current_loss < 0:
    #         print("lr too large!")
    #         lr = lr/2
    #         for g in optimizer.param_groups:
    #             g['lr'] = 0.001
    #     else:
    #         if last_loss - current_loss <=1e-5:
    #             break
    #         else:
    #             last_loss = current_loss
    # if epoch>=1e3 -2:
    #     print("ran for 1e3 epoch and still not stop!")
    
    # print("loss", current_loss)
    # return csd_temp_ts.detach().numpy()
    
def Loss_L2(csd_temp, y, A, mat_region, mat_diff, lam_smooth, lam_region, lam_lasso, mean_csd_single_time):
    csd_temp_value = 1/2*np.linalg.norm(y-A@csd_temp)**2 \
        + lam_smooth/2*np.linalg.norm(mat_diff@csd_temp)**2 \
        + lam_region/2*np.linalg.norm(mat_region.T@csd_temp)**2 \
        + lam_lasso/2*np.linalg.norm(csd_temp-mean_csd_single_time)**2
    return csd_temp_value
    
def Loss_L2_der(csd_temp, y, A, mat_region, mat_diff, lam_smooth, lam_region, lam_lasso, mean_csd_single_time):
    
    der = -(y-A@csd_temp)@A \
        + lam_smooth*mat_diff@csd_temp@mat_diff \
        + lam_region*mat_region.T@csd_temp@mat_region.T \
        + lam_lasso*(csd_temp-mean_csd_single_time)

    return der




def rcsd_single_time(y, A, mat_region, mat_diff, csd_ini, lam_smooth, lam_region, lam_lasso):
    # A few key point in the optimization algorithm:
        
    # First part update all coefficients simultaneously. 
    # nloop: maximum iteration; 
    # nDichotomy: search steps = 1/2**iDichotomy (nDichotomy is maximum of iDichotomy)
    
    # Second part update all coefficients immediately (greedy)
    # non_update: if the nonzero coefficients doesn't change for some iteration, it stops
    
    nx, nz = A.shape
    nloop = 10
    # y = y[:,None]
    csd_best = copy.deepcopy(csd_ini)
    # csd_best = np.zeros((nz,1))
    csd_temp = copy.deepcopy(csd_ini)
    csd_best_value = np.inf
    csd_temp_value = np.inf
    csd_record = np.zeros((nz, nloop))
    csd_record[:, 0] = csd_ini.squeeze()
    non_update = 0
    for iloop in range(1, nloop):
        update_weight = np.zeros((nz, 1))
        csd_temp = csd_record[:, iloop-1]
        for iz in range(nz):
            rho = 0
            iregion = np.where(mat_region[iz, :])[0][0]
            csd_mask_j = copy.deepcopy(csd_temp)
            csd_mask_j[iz] = 0

            if iz == 0:
                rho += lam_smooth*(csd_temp[1])
            elif iz == nz-1:
                rho += lam_smooth*(csd_temp[nz-2])
            else:
                rho += lam_smooth*(csd_temp[iz-1] + csd_temp[iz+1])
            rho += -lam_region*(mat_region[:, iregion]*csd_mask_j).sum()
            rho += (A[:, iz]*(y-A@csd_mask_j)).sum()
            k = 2*lam_smooth + lam_region + np.linalg.norm(A[:, iz])**2

            if rho < -lam_lasso:
                new_value = (rho+lam_lasso)/k
            elif rho > lam_lasso:
                new_value = (rho-lam_lasso)/k
            else:
                new_value = 0

            update_weight[iz] = new_value - csd_temp[iz]
        
        
        # search for the best value in current direction (1/2 step, 1/4 step, 1/8 step ...)
        nDichotomy = 10
        csd_search_value = np.nan*np.zeros(nDichotomy)
        csd_temp = csd_record[:, iloop-1]
        csd_search_value[0] = np.inf
        for iDichotomy in range(1,nDichotomy):
            csd_temp = csd_record[:, iloop-1] + 1/2**iDichotomy*update_weight.squeeze()
            csd_search_value[iDichotomy] = 1/2*np.linalg.norm(y-A@csd_temp)**2 \
                + lam_smooth/2*np.linalg.norm(mat_diff@csd_temp)**2 \
                + lam_region/2*np.linalg.norm(mat_region.T@csd_temp)**2 \
                + lam_lasso*np.linalg.norm(csd_temp, 1)
            if csd_search_value[iDichotomy] > csd_search_value[iDichotomy-1]:
                csd_temp = csd_record[:, iloop-1] + 1/2**(iDichotomy-1)*update_weight.squeeze()
                break
        
        csd_record[:, iloop] = csd_temp.squeeze()
        if csd_search_value[iDichotomy] <= csd_best_value:
            csd_best_value = csd_search_value[iDichotomy-1]
            csd_best = copy.deepcopy(csd_temp)
            non_update = 0
        else:
            non_update += 1

        if non_update >= 3:
            break
    
    # print('total iteration for the first part', iloop)
    # print('best value for the first part', csd_best_value)
    
    
    nloop = 1000
    csd_record = np.zeros((nz, nloop))
    # update immediately
    for iloop in range(1, nloop):
        update_weight = np.zeros((nz, 1))
        csd_temp = copy.deepcopy(csd_best)
        for iz in range(nz):
            rho = 0
            iregion = np.where(mat_region[iz, :])[0][0]
            csd_mask_j = copy.deepcopy(csd_temp)
            csd_mask_j[iz] = 0

            if iz == 0:
                rho += lam_smooth*(csd_temp[1])
            elif iz == nz-1:
                rho += lam_smooth*(csd_temp[nz-2])
            else:
                rho += lam_smooth*(csd_temp[iz-1] + csd_temp[iz+1])
            rho += -lam_region*(mat_region[:, iregion]*csd_mask_j).sum()
            rho += (A[:, iz]*(y-A@csd_mask_j)).sum()
            k = 2*lam_smooth + lam_region + np.linalg.norm(A[:, iz])**2

            if rho < -lam_lasso:
                new_value = (rho+lam_lasso)/k
            elif rho > lam_lasso:
                new_value = (rho-lam_lasso)/k
            else:
                new_value = 0

            csd_temp[iz] = new_value

        csd_record[:, iloop] = csd_temp.squeeze()

        csd_temp_value = 1/2*np.linalg.norm(y-A@csd_temp)**2 \
            + lam_smooth/2*np.linalg.norm(mat_diff@csd_temp)**2 \
            + lam_region/2*np.linalg.norm(mat_region.T@csd_temp)**2 \
            + lam_lasso*np.linalg.norm(csd_temp, 1)
        
        if csd_temp_value <= csd_best_value:
            csd_best = copy.deepcopy(csd_temp)
            csd_best_value = csd_temp_value
            
        if all((csd_record[:, iloop]==0)==(csd_record[:, iloop-1]==0)):
            non_update += 1
        else:
            non_update = 0
            
            

    # if update_step<=1e-5:
        if non_update >= 5:
            break

    # fig, ax = plt.subplots()
    # maximumCSD = np.abs(csd_record.max())
    # intervals_csd = np.round(intervals*nz/nx).astype(int)
    # interval_midpoints_csd = [ (aa + (bb - aa) /2) *nz/nx for aa, bb in zip(intervals[:-1], intervals[1:])]
    # ax.imshow (csd_record[:,0:iloop], vmin = -maximumCSD, vmax = maximumCSD,cmap='bwr')
    # ax.set_yticks(intervals_csd)
    # ax.set_yticks(interval_midpoints_csd, minor=True)
    # ax.set_yticklabels(structure_acronyms, minor=True)
    # plt.tick_params("y", which="major", labelleft=False, length=40)

    # print('total iteration for the second part', iloop)
    # print('best value for the second part', csd_best_value)
    
    return csd_best.squeeze()


def tcsd(lfp):
    """
    Does traditional CSD estimator with no smoothing (may want to smooth LFP first)
    :param lfp: (nx, nt, ntrial)
    :return: (nx, nt, ntrial) values of CSD
    """
    nx = lfp.shape[0]
    nt = lfp.shape[1]
    ntrials = lfp.shape[2]
    csd = np.zeros((nx, nt, ntrials))
    for x in range(1,nx-1):
        csd[x, :, :] = lfp[x+1, :, :] + lfp[x-1, :, :] - 2*lfp[x, :, :]
    return -csd




def get_power_phase(data, npadding, lowcut, highcut):
    data = copy.copy(data)
    nx, nt, ntrial = data.shape
    phase = np.zeros((nx, nt-2*npadding, ntrial))
    power = np.zeros((nx, nt-2*npadding, ntrial))

    for itrial in range(ntrial):
        raw_signal = data[:,:,itrial]
        instantaneous_phase, instantaneous_power = utils.get_phase(raw_signal, 500, lowcut=lowcut, highcut=highcut, 
                                                                   npadding=npadding)
        phase[:,:,itrial] = instantaneous_phase
        power[:,:,itrial] = instantaneous_power
    return phase, power

def pooling(data, merge):
    new_data = np.zeros((data.shape[0], int(data.shape[1]/merge), data.shape[2]))
    for i in range(merge):
        new_data += data[:, i::merge, :]
    new_data = new_data/merge
    return new_data