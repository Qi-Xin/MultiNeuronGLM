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
# from rCSD import rcsd_single_trial, rcsd_single_time, rcsd_multi_trial_parallel

"""Three data loading adapters that read or generate the same standard of LFP data from Allen insititue, Prof. Teichert, 
and simulation. """

from scipy.optimize import minimize

# import torch
# from torch.autograd import Variable
# from torch.nn import functional as F

class Dataset:

    def remove_padding(self, padding_time):
        npadding = int(padding_time*self.fps)
        self.lfp = self.lfp[:,npadding:-npadding,:]

    def get_power_phase(self, padding_time, lowcut=20, highcut=35):
        self.phase, self.power = get_power_phase(self.lfp, npadding=int(self.fps*padding_time), lowcut=lowcut, highcut=highcut)


    def get_mean_lfp(self):
        self.mean_lfp = self.lfp.mean(axis=2)
        self.mean_lfp = self.mean_lfp[:,:,None]

    def pre_smooth(self, moving_size = None, pooling_size = None):
        if moving_size == None:
            default_moving_dict = {'Allen':1, 'Tobias':1, 'Simulation':1}
            moving_size = default_moving_dict[self.source]
        if pooling_size == None:
            default_pooling_dict = {'Allen':5, 'Tobias':5, 'Simulation':1}
            moving_size = default_pooling_dict[self.source]
        self.lfp_smooth = utils.moving_average(self.lfp, pooling_size, moving_size)
        self.mean_lfp_smooth = utils.moving_average(self.mean_lfp, pooling_size, moving_size)

    def align_lfp(self):
        temp = np.swapaxes(self.lfp, 1,2)    # Temporally convert (electrode, time, trial) -> (electrode, trial, time)
        self.aligned_lfp = self.lfp.reshape(temp.shape[0],-1)
        
    def show(self, trial=0):
        if self.source == 'Simulation':
            plt.figure(figsize=(8, 6))
            plt.subplot(121)
            plt.imshow(self.gt_csd[:,:,trial], aspect='auto', cmap='bwr')
            plt.title('True CSD')
            plt.ylabel('Depth (microns)')
            plt.xlabel('Time (ms)')
            plt.subplot(122)
            plt.imshow(self.lfp[:,:,trial], aspect='auto', cmap='bwr')
            plt.title('LFP (noisy)')
            plt.xlabel('Time (ms)')
            plt.show()
        else:
            plt.figure(figsize=(12, 6))
            plt.imshow(self.lfp[:,:,trial], aspect='auto', cmap='bwr')
            plt.title('LFP')
            plt.xlabel('Time (ms)')
            plt.show()


class Allen_dataset(Dataset):
    """ For drifting gratings, there are 30 unknown trials, 15*5*8=600 trials for 8 directions, 5 temporal frequencies, 
    15 iid trials each conditions. """
    def __init__(self, **kwargs):
        self.source = "Allen"
        self.session_id = kwargs.pop('session_id', 791319847)
        self.probe_id = kwargs.pop('probe_id', 805008600)
        self.stimulus_name = kwargs.pop('stimulus_name',
                                        'drifting_gratings_contrast')
        self.orientation = kwargs.pop('orientation', None)
        self.temporal_frequency = kwargs.pop('temporal_frequency', None)
        self.contrast = kwargs.pop('contrast', None)
        self.stimulus_condition_id = kwargs.pop('stimulus_condition_id', None)
        self.start_time = kwargs.pop('start_time', -0.5)
        self.end_time = kwargs.pop('end_time', 0)
        self.fps = kwargs.pop('fps', 1250)
        
        from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
        if sys.platform == 'linux':
            self.manifest_path = os.path.join('/home/qix/ecephys_cache_dir/', "manifest.json")
        elif sys.platform == 'win32' or 'darwin':
            self.manifest_path = os.path.join('D:/ecephys_cache_dir/', "manifest.json")
        else:
            raise ValueError("Undefined device!")
        self._cache = EcephysProjectCache.from_warehouse(manifest=self.manifest_path)
        self._session = self._cache.get_session_data(self.session_id)
        if self.stimulus_name == "All":
            self._presentation_table = self._session.stimulus_presentations
        else:
            if isinstance(self.stimulus_name ,str):
                idx = self._session.stimulus_presentations.stimulus_name == self.stimulus_name
            else:
                idx = self._session.stimulus_presentations.stimulus_name .isin(self.stimulus_name) 
            if self.orientation != None:
                idx = idx & (self._session.stimulus_presentations.orientation.isin(self.orientation))
            if self.temporal_frequency != None:
                idx = idx & (self._session.stimulus_presentations.temporal_frequency.isin(self.temporal_frequency))
            if self.contrast != None:
                idx = idx & (self._session.stimulus_presentations.contrast.isin(self.contrast))
            if self.stimulus_condition_id != None:
                idx = idx & (self._session.stimulus_presentations['stimulus_condition_id'].isin(self.stimulus_condition_id))
            self._presentation_table = self._session.stimulus_presentations [idx]
        self._presentation_times = self._presentation_table.start_time.values
        self._presentation_ids = self._presentation_table.index.values
        
        # self.get_lfp()

    def get_running(self, method="Pillow"):
        speed = self._session.running_speed
        speed['mean_time'] = (speed['start_time']+speed['end_time'])/2
        self.mean_speed = np.zeros(self.ntrial)
        self.min_speed = np.zeros(self.ntrial)
        self.max_speed = np.zeros(self.ntrial)
        for i in range(self.ntrial):
            speed_temp = speed[np.logical_and(speed['mean_time']<self._presentation_times[i]+self.end_time , 
                                self._presentation_times[i]+self.start_time<speed['mean_time']).values]['velocity'].values
            self.mean_speed[i] = speed_temp.mean()
            self.min_speed[i] = speed_temp.min()
            self.max_speed[i] = speed_temp.max()
        if method=="Pillow":
            self.running_trial_index = np.logical_and( self.mean_speed >= 3 , self.min_speed >= 0.5 )
            self.stationary_trial_index = np.logical_and( self.mean_speed < 0.5 , self.max_speed < 3 )
        else:
            self.running_trial_index = self.mean_speed >= 1
            self.stationary_trial_index = self.mean_speed < 1



    def get_lfp(self):
        ### May separate into two smaller functions, which also incorperate with get_Allen_spike_train

        lfp_data = self._session.get_lfp(self.probe_id)
        trial_window = np.arange(self.start_time,self.end_time, 1/self.fps)
        time_selection = np.concatenate([trial_window + t for t in self._presentation_times])
        inds = pd.MultiIndex.from_product((self._presentation_ids, trial_window), 
                                      names=('presentation_id', 'time_from_presentation_onset'))
        ds = lfp_data.sel(time = time_selection, method='nearest').to_dataset(name = 'aligned_lfp')
        ds = ds.assign(time=inds).unstack('time')
        self.lfp = ds['aligned_lfp'].values     # Three dimensions. e.g. (77, 540, 625). Channels, trials, times
        self.lfp = np.swapaxes(self.lfp,1,2)    # Swap time and trial. e.g. (77, 625, 540). Channels, times, trials
        try:
            location = self._session.channels[['probe_vertical_position','probe_horizontal_position']]
            location = location.loc[lfp_data['channel'].values].values
            self.x = location[:, 0]      # LFP spatial locations , microns
        except:
            location = np.arange(0,self.lfp.shape[0])*40.0
            self.x = location

        self.t = np.linspace(self.start_time,self.end_time,self.lfp.shape[1] )[:,None]
        self.x = self.x[:, None]
        self.channel = lfp_data["channel"].values
        try:
            self.structure_acronyms, self.intervals_lfp = self._session.channel_structure_intervals(self.channel)
        except:
            self.structure_acronyms = np.array([np.nan], dtype=object)
            self.intervals_lfp = np.array([ 0, self.lfp.shape[0]])
        self.nx, self.nt, self.ntrial, self.lfp = utils.check_and_get_size(self.lfp)
        self.get_mean_lfp()

    def get_spike_train_sparse(self):
        self._units_pd = self._session.units[(self._session.units.probe_id == self.probe_id)]
        self.unit_id_list = self._units_pd.index.values
        self._channel_index = self._units_pd.loc[self.unit_id_list].channel_local_index
        self.st_channel_id_list = []
        self.nunit = len(self.unit_id_list)
        self.spike_train_sparse = []
        for i in range(self.nunit):
            self.st_channel_id_list.append( self._session.channels[(self._session.channels.local_index == self._channel_index.values[i]) & 
                                                (self._session.channels.probe_id == self.probe_id)].index.values[0] )
            spike_times = self._session.spike_times[self.unit_id_list[i]]
            self.spike_train_sparse.append([ spike_times[ np.logical_and(spike_times>t+self.start_time, spike_times<t+self.end_time) ]-t 
                                     for t in self._presentation_times ])
            
        self.st_channel_id_list = np.array(self.st_channel_id_list)
        
    def get_spike_train(self):
        # (nunit, nt, ntrial)
        from scipy.sparse import csr_matrix
        self.get_spike_train_sparse()

        self.spike_train = np.zeros((self.nunit, self.nt, self.ntrial))
        for i in range(self.nunit):
            row = np.array([])
            col = np.array([])
            data = np.array([])
            for trial in range(self.ntrial):
                temp = self.spike_train_sparse[i][trial]
                nspike = len(temp)
                row = np.hstack( (row, np.array(temp)*self.fps) )
                col = np.hstack( (col, trial*np.ones(nspike)) )
                data = np.hstack( (data, np.ones(nspike)) )
                
            
            self.spike_train[i,:,:] = csr_matrix((data, (row, col)), shape=(self.nt, self.ntrial)).toarray()
        self.spike_count = self.spike_train.sum(axis=(1))
        
    def get_pooled_spike_train(self):
        pooled_spike_train = []
        for i in range(self.nunit):
            pooled_spike_train.append( np.concatenate(self.spike_train_sparse[i]) )
        return pooled_spike_train
    
    def get_fr(self):
        import smoothing_spline
        fit_model = smoothing_spline.SmoothingSpline()
        time_line = np.arange(self.start_time,self.end_time,1.0/self.fps)
        eta_smooth_tuning = 1e-10
        f_basis, f_Omega = fit_model.construct_basis_omega(
            time_line, knots=15, verbose=False)
        
        self.fr = np.zeros((self.nunit, self.nt, self.ntrial))
        for i in range(self.nunit):
            for trial in range(self.ntrial):
                temp_spike_train = self.spike_train[i,:,trial]
                # log_lambda_hat, beta = fit_model.poisson_regression(temp_spike_train[None,:], f_basis)
                log_lambda_hat, (beta, beta_baseline, log_lambda_offset, hessian, hessian_baseline, nll) \
                    = fit_model.poisson_regression_smoothing_spline(
                        temp_spike_train[None,:], time_line, basis=f_basis, Omega=f_Omega, constant_fit=False)
                self.fr[i, :, trial] = np.exp(log_lambda_hat)

        
    def get_kernel_fr(self, bandwidth=0.03):
        self.bandwidth = bandwidth
        from sklearn.neighbors import KernelDensity

        emp_fr_X = np.arange(self.start_time,self.end_time, 1/self.fps)
        emp_fr_X = emp_fr_X[:,None]
        emp_fr = np.zeros((self.nunit, emp_fr_X.shape[0], self.ntrial))
        for i in range(self.nunit):
            for trial in range(self.ntrial):
                points = np.array(self.spike_train_sparse[i][trial])
                
                if len(points)>0:
                    points = points[:,None]
                    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(points)
                    logdensity = kde.score_samples( emp_fr_X )
                    emp_fr[i,:, trial] = np.exp(logdensity)
                    total_spike = points.shape[0]
                    emp_fr[i,:, trial] = emp_fr[i,:, trial]*total_spike
        self.kernel_fr = emp_fr
        return emp_fr

    def get_psth(self, bandwidth=0.02):
        self.bandwidth = bandwidth
        from sklearn.neighbors import KernelDensity
        pooled_spike_train = self.get_pooled_spike_train() 
        emp_fr_X = np.arange(self.start_time,self.end_time, 1/self.fps)
        emp_fr_X = emp_fr_X[:,None]
        emp_fr = np.zeros((self.nunit, emp_fr_X.shape[0]))
        for i in range(self.nunit):
            points = np.array(pooled_spike_train[i])
            points = points[:,None]
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(points)
            logdensity = kde.score_samples( emp_fr_X )
            emp_fr[i,:] = np.exp(logdensity)
            total_spike = points.shape[0]
            emp_fr[i,:] = emp_fr[i,:]*total_spike/self.ntrial
        self.emp_fr = emp_fr
        return emp_fr
    


class Tobias_dataset(Dataset):
    def __init__(self, **kwargs):
        self.source = "Tobias"
        self.dataset_id = kwargs.pop('dataset_id', 'Walter_20160512')
        self.probe_id = kwargs.pop('probe_id', 805008600)
        self.start_time = kwargs.pop('start_time', 0)
        self.end_time = kwargs.pop('end_time', self.start_time+500)
        self.get_Tobias()
        
    def get_lfp(self):
        import scipy.io
        mat = scipy.io.loadmat("D:/LFP data/"+self.dataset_id+".mat")
        self.lfp = mat['data']
        self.lfp = self.lfp[:, self.start_time: self.end_time]
        # Set up spatial locations , temporal locations
        self.t = np. linspace(0, self.nt, self.nt)[:, None]       # time points , milliseconds
        self.x = np. linspace(0, 2300, 24)[:, None]     # LFP spatial locations , microns
        self.nx, self.nt, self.ntrial, self.lfp = utils.check_and_get_size(self.lfp)
        self.get_mean_lfp()
        self.intervals_lfp = np.array([0, np.nonzero(self.x > 1000)[0][0],
                                np.nonzero(self.x > 1500)[0][0], self.nx])
        self.structure_acronyms = np.array(['Superficial', 'Medium', 'Deep'])
    
class Simulation_dataset(Dataset):
    def __init__(self, **kwargs):
        """To be done later. Need to include: gaussian bumps; the cases from jupyter notebooks"""
        self.source = "Simulation"
        self.noise_amp = kwargs.pop('noise_amp', 0.03)
        if 'gt_csd' in kwargs:
            print("not finished! Can't allow user self design gt csd at the moment! ")
            raise ValueError("Unfinished!")
            # self.gt_csd = kwargs['gt_csd']
            # self.generate_lfp(noise_amp)
        else:
            self.gt_csd_id = kwargs.pop('gt_csd_id', 0)
            self.get_csd()
            self.get_lfp()

    def get_csd(self):
        import ground_true_csd_bank
        self.R = 150
        self.nx = 24
        self.nz = 5*23
        self.x = np. linspace(0, 2300, self.nx)[:, None]
        self.z = np. linspace(0+10, 2300-10, self.nz)[:, None]
        self.gt_csd, self.t, self.nt = ground_true_csd_bank.csd_simple_templete(self.gt_csd_id, self.z)
        self.ntrial = self.gt_csd.shape[2]
        
    def get_lfp(self):
        lfp_noiseless = fwd_model_1d(self.gt_csd, self.z, self.x, self.R)
        noise = self.noise_amp * np.abs(lfp_noiseless).max()*np.random.randn(self.nx, self.nt, self.ntrial)
        self.lfp = lfp_noiseless + noise
        self.nx, self.nt, self.ntrial, self.lfp = utils.check_and_get_size(self.lfp)
        self.get_mean_lfp()
        self.intervals_lfp = np.array([0, np.nonzero(self.x > 1000)[0][0],
                                np.nonzero(self.x > 1500)[0][0], self.nx])
        self.structure_acronyms = np.array(['Superficial', 'Medium', 'Deep'])



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