import numpy as np
import emcee 
from ligeor.models.twogaussian import TwoGaussianModel
from ligeor.models.polyfit import Polyfit
from ligeor.utils.lcutils import *
from multiprocessing import Pool
import tempfile
import subprocess
from scipy.interpolate import interp1d

class EmceeSampler(object):

    def __init__(self, filename, period_init, t0_init, n_downsample=0, nbins=1000, **kwargs):

        '''
        Initializes a sampler for the light curve stored in 'filename'
        with determined initial values for the period and t0.

        Parameters
        ----------
        filename: str
            Filename to load the raw light curve from (expected format: times, fluxes, sigmas)
        period_init: float
            Initial value of the period (from code/triage)
        t0_init: float
            Initial value of the time of superior conjunction (t0)
        n_downsample: int
            Number of data points to skip in raw light curve for downsampling
        '''
        self._filename = filename
        self._nbins = nbins
        self._period_init = period_init 
        self._t0_init = t0_init
        lc = load_lc(filename, n_downsample=n_downsample, **kwargs)
        self._times, self._fluxes, self._sigmas = lc['times'], lc['fluxes'], lc['sigmas']

    def initial_fit(self):
        # overriden by subclass
        return None
    
    def logprob(self):
        # overriden by subclass
        return None

    def run_sampler(self, nwalkers=32, niters=2000, progress=True):
        init_vals = self._initial_fit
        bestfit_vals = np.array([self._period_init, *init_vals])
        pos = bestfit_vals + 1e-4 * np.random.randn(nwalkers, len(bestfit_vals))
        nwalkers, ndim = pos.shape
        
        with Pool(nwalkers) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.logprob, pool=pool)
            sampler.run_mcmc(pos, niters, progress=progress)
        
        return sampler
        

    def compute_results(self, sampler, burnin = 1000, save_lc=True, save_file=''):

        if save_lc and len(save_file) == 0:
            raise ValueError('Please provide a file name to save the model to or set save_lc=False.')
        
        # process and log solution
        log_prob = sampler.get_log_prob(flat=True, discard=burnin)
        flat_samples = sampler.get_chain(flat=True, discard=burnin)
        flat_blobs = sampler.get_blobs(flat=True, discard=burnin)
        
        #figure out if there is branching in the solution and find the highest logp branch
        try:
            hist = np.histogram(log_prob, bins=50)
            arg_logp_max = np.argwhere(hist[0] != 0)[-1]
            logp_lim = hist[1][arg_logp_max-1]
            samples_top = flat_samples[log_prob >= logp_lim]
            blobs_top = flat_blobs[log_prob >= logp_lim]
            # log_prob_mean = np.mean(log_prob[log_prob >= logp_lim])
        except:
            samples_top = flat_samples
            blobs_top = flat_blobs
            # log_prob_mean = np.mean(log_prob)
        
        ndim = samples_top.shape[1]
        solution = []

        for j in range(ndim):
            mcmc = np.percentile(samples_top[:, j], [16, 50, 84])
            q = np.diff(mcmc)
            solution.append([mcmc[1], q[0], q[1]])

        solution = np.array(solution)

        # compute the new period and errs
        means = solution[:,0]
        sigmas_low = solution[:,1]
        sigmas_high = solution[:,2]

        period = means[0]
        period_sigma_low = sigmas_low[0]
        period_sigma_high = sigmas_high[0]
        
        self._period_mcmc = {'value': period,
                            'sigma_low': period_sigma_low, 
                            'sigma_high': period_sigma_high}

        # compute the blob parameters
        # phasemin, residuals_mean, residuals_stdev, ecl1_area, ecl2_area
        ndim_blobs = blobs_top.shape[1]

        solution_blobs = []
        for j in range(ndim_blobs):
            mcmc_blob = np.percentile(blobs_top[:, j], [16, 50, 84])
            q_blob = np.diff(mcmc_blob)
            solution_blobs.append([mcmc_blob[1], q_blob[0], q_blob[1]])

        solution_blobs = np.array(solution_blobs)
        means_blobs = solution_blobs[:,0]
        sigmas_low_blobs = solution_blobs[:,1]
        sigmas_high_blobs = solution_blobs[:,2]

        phasemin_mean = means_blobs[0]
        phasemin_sigma_low = sigmas_low_blobs[0]
        phasemin_sigma_high = sigmas_high_blobs[0]


        if np.isnan(self._t0_init):
            t0_new = 0 + period*phasemin_mean + int((self._times.min()/period)+1)*(period)
        else:
            t0_new = self._t0_init + period*phasemin_mean

        t0_sigma_low = (period**2 * phasemin_sigma_low**2 + period_sigma_low**2 * phasemin_mean**2)**0.5
        t0_sigma_high = (period**2 * phasemin_sigma_high**2 + period_sigma_high**2 * phasemin_mean**2)**0.5

        self._t0_mcmc = {'value': t0_new,
                        'sigma_low': t0_sigma_low, 
                        'sigma_high': t0_sigma_high}

        # store the rest of the model parameters and their uncertainties
        self.compute_model(means, sigmas_low, sigmas_high, save_lc = save_lc, save_file=save_file)
        self.compute_eclipse_params(means_blobs, sigmas_low_blobs, sigmas_high_blobs)


    def save_results_to_file(self, results_file, type='ephemerides', ind=''):
        if type=='ephemerides':
            with open(results_file, 'a') as f:
                f.write('{},{},{},{},{},{},{},{}\n'.format(ind,
                                                            self._period_mcmc['value'],
                                                            self._period_mcmc['sigma_low'],
                                                            self._period_mcmc['sigma_high'],
                                                            self._t0_mcmc['value'],
                                                            self._t0_mcmc['sigma_low'],
                                                            self._t0_mcmc['sigma_high'],
                                                            self._chi2))

        elif type=='model_values':
            results_str = self._func+','
            for i,mkey in enumerate(self._model_values.keys()):
                results_str += '{},{}'.format(self._model_values[mkey],self._model_values_errs[mkey])
                if i < len(self._model_values.keys())-1:
                    results_str += ','
                else:
                    results_str += '\n'

            with open(results_file, 'a') as f:
                f.write('{},{}'.format(ind,results_str))


        elif type == 'eclipse_parameters':
            with open(results_file, 'a') as f:
                # pos1, pos1_err, width1, width1_err, depth1, depth1_err, 
                # pos2, pos2_err, width2, width2_err, depth2, depth2_err, 
                # res_mean, res_mean_err, res_stdev, res_stdev_err, 
                # ecl1_area, ecl1_area_err, ecl2_area, ecl2_area_err
                ecl = self._eclipse_params
                eclerr = self._eclipse_params_errs
                f.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(ind,
                    ecl['pos1'], eclerr['pos1'], ecl['width1'], eclerr['width1'], ecl['depth1'], eclerr['depth1'],
                    ecl['pos2'], eclerr['pos2'], ecl['width2'], eclerr['width2'], ecl['depth2'], eclerr['depth2'],
                    ecl['residuals_mean'], eclerr['residuals_mean'], ecl['residuals_stdev'], eclerr['residuals_stdev'],
                    ecl['ecl1_area'], eclerr['ecl1_area'], ecl['ecl2_area'], eclerr['ecl2_area']
                ))

        else:
            raise NotImplementedError


    def compute_model(self, means, sigmas_low, sigmas_high):
        # overriden by subclass
        return None


    def compute_eclipse_params(self, means, sigmas_low, sigmas_high):
        # pos1, width1, depth1, pos2, width2, depth2, ecl1_area, ecl2_area, residuals_mean, residuals_stdev
        eclipse_params = {'pos1': np.nan, 'width1': np.nan, 'depth1': np.nan, 
                          'pos2': np.nan, 'width2': np.nan, 'depth2': np.nan,
                          'ecl1_area': np.nan, 'ecl2_area': np.nan, 
                          'residuals_mean': np.nan, 'residuals_stdev': np.nan
                        }

        eclipse_params_err = {'pos1': np.nan, 'width1': np.nan, 'depth1': np.nan, 
                          'pos2': np.nan, 'width2': np.nan, 'depth2': np.nan,
                          'ecl1_area': np.nan, 'ecl2_area': np.nan, 
                          'residuals_mean': np.nan, 'residuals_stdev': np.nan
                        }

        for ind, eclkey in enumerate(eclipse_params.keys()):
            # print(eclkey, means[ind], np.max((sigmas_low[ind],sigmas_high[ind])))
            eclipse_params[eclkey] = means[ind]
            eclipse_params_err[eclkey] = np.max((sigmas_low[ind],sigmas_high[ind]))

        
        self._eclipse_params = eclipse_params
        self._eclipse_params_errs = eclipse_params_err




class EmceeSamplerTwoGaussian(EmceeSampler):

    def __init__(self, filename, period_init, t0_init, n_downsample=0, nbins=1000, **kwargs):
        super(EmceeSamplerTwoGaussian,self).__init__(filename, period_init, t0_init, n_downsample, nbins, **kwargs)


    def initial_fit(self):
        phases, fluxes_ph, sigmas_ph = phase_fold(self._times, 
                                                        self._fluxes, 
                                                        self._sigmas, 
                                                        period=self._period_init, 
                                                        t0=self._t0_init)

        twogModel = TwoGaussianModel(phases=phases, 
                                    fluxes=fluxes_ph, 
                                    sigmas=sigmas_ph, 
                                    )
        twogModel.fit()

        # self._twogModel_init = twogModel
        self._func = twogModel.best_fit['func']
        self._model_params = twogModel.best_fit['param_names']
        self._initial_fit = twogModel.best_fit['param_vals'][0]


    def logprob(self, values):

        fmax = self._fluxes.max()
        fmin = self._fluxes.min()
        fdiff = fmax - fmin

        bounds = {'C': ((0),(fmax)),
            'CE': ((0, 1e-6, -0.5),(fmax, fdiff, 0.5)),
            'CG': ((0., -0.5, 0., 0.), (fmax, 0.5, fdiff, 0.5)),
            'CGE': ((0., -0.5, 0., 0., 1e-6, -0.5),(fmax, 0.5, fdiff, 0.5, fdiff, 0.5)),
            'CG12': ((0.,-0.5, 0., 0., -0.5, 0., 0.),(fmax, 0.5, fdiff, 0.5, 0.5, fdiff, 0.5)),
            'CG12E1': ((0.,-0.5, 0., 0., -0.5, 0., 0., 1e-6),(fmax, 0.5, fdiff, 0.5, 0.5, fdiff, 0.5, fdiff)),
            'CG12E2': ((0.,-0.5, 0., 0., -0.5, 0., 0., 1e-6),(fmax, 0.5, fdiff, 0.5, 0.5, fdiff, 0.5, fdiff))}
        
        period, *model_vals = values

        for i,param_val in enumerate(model_vals):
            if param_val < bounds[self._func][0][i] or param_val > bounds[self._func][1][i]:
                # raise Warning('out of prior', self._func, bounds[self._func][0][i], bounds[self._func][1][i], param_val)
                return -np.inf, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            
        # fold with period
        phases, fluxes_ph, sigmas_ph = phase_fold(self._times, 
                                                        self._fluxes, 
                                                        self._sigmas, 
                                                        period=period, 
                                                        t0=self._t0_init)
        # compute model with the selected twog function
        # TODO: review this part for potentially more efficient option
        twog_func = getattr(TwoGaussianModel, self._func.lower())
        model = twog_func(phases, *model_vals)

        C = model_vals[self._model_params.index('C')]
        if self._func in ['CG', 'CGE']:

            mu = model_vals[self._model_params.index('mu1')]
            sigma = model_vals[self._model_params.index('sigma1')]
            d = model_vals[self._model_params.index('d1')]
            
            # compute eclipse parameters
            pos1, pos2 = mu, np.nan
            width1, width2 = 5.6*sigma, np.nan 
            depth1, depth2 = C - fluxes_ph[np.argmin(np.abs(phases-pos1))], np.nan

            # compute eclipse area
            phi_top = mu + 2.8*sigma
            phi_bottom = mu - 2.8*sigma
            ecl1_area = TwoGaussianModel.compute_gaussian_area(mu, sigma, d, phi_bottom, phi_top)
            ecl2_area = np.nan
        
        elif self._func in ['CG12', 'CG12E1', 'CG12E2']:
            mu1_ind, mu2_ind = self._model_params.index('mu1'), self._model_params.index('mu2')
            sigma1_ind, sigma2_ind = self._model_params.index('sigma1'), self._model_params.index('sigma2')
            d1_ind, d2_ind = self._model_params.index('d1'), self._model_params.index('d2')
            
            mu1, mu2 = model_vals[mu1_ind], model_vals[mu2_ind]
            sigma1, sigma2 = model_vals[sigma1_ind], model_vals[sigma2_ind]
            d1, d2 = model_vals[d1_ind], model_vals[d2_ind]

            # compute eclipse parameters
            pos1, pos2 = mu1, mu2
            width1, width2 = 5.6*sigma1, 5.6*sigma2 
            depth1, depth2 = C - fluxes_ph[np.argmin(np.abs(phases-pos1))], C - fluxes_ph[np.argmin(np.abs(phases-pos2))]

            phi1_top, phi2_top = mu1 + 2.8*sigma1, mu2 + 2.8*sigma2
            phi1_bottom, phi2_bottom = mu2 - 2.8*sigma2, mu2-2.8*sigma2
            ecl1_area = TwoGaussianModel.compute_gaussian_area(mu1, sigma1, d1, phi1_bottom, phi1_top)
            ecl2_area = TwoGaussianModel.compute_gaussian_area(mu2, sigma2, d2, phi2_bottom, phi2_top)

        else:
            pos1, pos2, width1, width2, depth1, depth2 = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            ecl1_area, ecl2_area = np.nan, np.nan

        residuals_mean, residuals_stdev = compute_residuals_stdev(fluxes_ph, model)
        # print('residuals: ', residuals_mean, residuals_stdev)
        logprob = -0.5*(np.sum((fluxes_ph-model)**2/sigmas_ph**2))
        # print(logprob, pos1, width1, depth1, pos2, width2, depth2)#, ecl1_area, ecl2_area, residuals_mean, residuals_stdev)
        return logprob, pos1, width1, depth1, pos2, width2, depth2, ecl1_area, ecl2_area, residuals_mean, residuals_stdev

    
    def compute_model(self, means, sigmas_low, sigmas_high, save_lc = True, save_file=''):
        model_results = {'C': np.nan, 'mu1': np.nan, 'd1': np.nan, 'sigma1': np.nan, 
                        'mu2': np.nan, 'd2': np.nan, 'sigma2': np.nan, 'Aell': np.nan, 'phi0': np.nan
                        }
        model_results_err = {'C': np.nan, 'mu1': np.nan, 'd1': np.nan, 'sigma1': np.nan, 
                        'mu2': np.nan, 'd2': np.nan, 'sigma2': np.nan, 'Aell': np.nan, 'phi0': np.nan
                        }
        
        # results_str = '{}'.format(func)
        for mkey in model_results.keys():
            if mkey in self._model_params:
                pind = self._model_params.index(mkey)
                model_results[mkey] = means[pind+1]
                model_results_err[mkey] = np.max((sigmas_low[pind+1],sigmas_high[pind+1]))
                # results_str += ',{},{}'.format(model_results[mkey],model_results_err[mkey])
        
        self._model_values = model_results
        self._model_values_errs = model_results_err
        chi2 = np.nan

        if save_lc:
            phases_obs, fluxes_ph_obs, sigmas_ph_obs = phase_fold(self._times, 
                                                    self._fluxes, 
                                                    self._sigmas, 
                                                    period=self._period_mcmc['value'], 
                                                    t0=self._t0_mcmc['value'])
        
            phases_syn = np.linspace(-0.5,0.5,self._nbins)
            twog_func = getattr(TwoGaussianModel, self._func.lower())
            fluxes_syn =twog_func(phases_syn, *means[1:])
            
            np.savetxt(save_file, np.array([phases_syn, fluxes_syn]).T)

            fluxes_syn_extended = np.hstack((fluxes_syn[(phases_syn > 0)], fluxes_syn, fluxes_syn[phases_syn < 0.]))
            phases_syn_extended = np.hstack((phases_syn[(phases_syn > 0)]-1., phases_syn, phases_syn[phases_syn < 0.]+1.))
            fluxes_interp = interp1d(phases_syn_extended, fluxes_syn_extended)
            fluxes_model = fluxes_interp(phases_obs)
            chi2 = -0.5*(np.sum((fluxes_ph_obs-fluxes_model)**2/sigmas_ph_obs**2))
        
        self._chi2 = chi2


# class EmceeSamplerPolyfit(EmceeSampler):


#     def __init__(self, filename, period_init, t0_init, n_downsample=0, nbins=1000, **kwargs):
#         super(EmceeSamplerPolyfit,self).__init__(filename, period_init, t0_init, n_downsample, nbins, **kwargs)


#     def initial_fit(self):
#         phases, fluxes_ph, sigmas_ph = phase_fold(self._times, 
#                                                         self._fluxes, 
#                                                         self._sigmas, 
#                                                         period=self._period_init, 
#                                                         t0=self._t0_init,
#                                                         )

#         temp = tempfile.NamedTemporaryFile(mode='w+t', suffix=".lc")
#         np.savetxt(temp.name, np.array([phases, fluxes_ph, sigmas_ph]).T)

#         proc = subprocess.Popen(["polyfit --find-knots -n 1000 -i 1000 --summary-output %s" % temp.name], stdout=subprocess.PIPE, shell=True)
#         (out, err) = proc.communicate()
#         out = np.array((out.decode("utf-8")).split('\n'))[0]
#         init_knots = np.array(out.split('\t')[-4:]).astype(float)
#         temp.close()

#         return init_knots


#     def logprob(self, values):
#         tempfile.tempdir='temp_lcs/'
        
#         bounds = [[-0.5,-0.5,-0.5,-0.5],[0.5,0.5,0.5,0.5]]
#         period, *model_vals = values

#         for i,param_val in enumerate(model_vals):
#             if param_val < bounds[0][i] or param_val > bounds[1][i]:
#                 print('out of prior', bounds[0][i], bounds[1][i], param_val)
#                 return -np.inf

#         phases, fluxes_ph, sigmas_ph = phase_fold(self._times, 
#                                                         self._fluxes, 
#                                                         self._sigmas, 
#                                                         period=period, 
#                                                         t0=self._t0_init)
        
#         temp = tempfile.NamedTemporaryFile(mode='w+t', suffix=".lc")
#         np.savetxt(temp.name, np.array([phases, fluxes_ph, sigmas_ph]).T)

#         knots = ' '.join([str(elem) for elem in model_vals]) 

#         pfModel = Polyfit(phases=phases, fluxes=fluxes_ph, sigmas=sigmas_ph)
#         chi2, phase_min, knots = pfModel.fit(niters = 0, nbins=self._nbins)

#         return -chi2, phase_min


#     def compute_model(self, means, sigmas_low, sigmas_high, save_lc=True):

#         model_results = {'k1': np.nan, 'k2': np.nan, 'k3': np.nan, 'k4': np.nan, 
#                         'phasemin': np.nan
#                         }
#         model_results_err = {'k1': np.nan, 'k2': np.nan, 'k3': np.nan, 'k4': np.nan, 
#                         'phasemin': np.nan
#                         }

#         for pind,mkey in enumerate(model_results.keys()):
#                 model_results[mkey] = means[pind+1]
#                 model_results_err[mkey] = np.max((sigmas_low[pind+1],sigmas_high[pind+1]))

#         self._model_values = model_results
#         self._model_values_errs = model_results_err
#         chi2 = np.nan
        
#         if save_lc:
#             phases_obs, fluxes_ph_obs, sigmas_ph_obs = phase_fold(self._times, 
#                                                     self._fluxes, 
#                                                     self._sigmas, 
#                                                     period=self._period_mcmc.value, 
#                                                     t0=self._t0_mcmc.value)

#             temp = tempfile.NamedTemporaryFile(mode='w+t', suffix=".lc")
#             np.savetxt(temp.name, np.array([phases_obs, fluxes_ph_obs, sigmas_ph_obs]).T)

#             save_file = self._filename + '.pf'
#             knots = ' '.join([str(elem) for elem in means[1:]]) 
#             proc = subprocess.Popen([f'polyfit -k {knots} -n {self._nbins} -i 0 {temp.name} > {save_file}'], stdout=subprocess.PIPE, shell=True) #% (knots, nbins, nitera temp.name, save_file)]
#             temp.close()

#             lc_syn = np.loadtxt(save_file)
#             phases_syn, fluxes_syn = lc_syn[:,0], lc_syn[:,1]

#             fluxes_syn_extended = np.hstack((fluxes_syn[:,1][(phases_syn > 0)], fluxes_syn, fluxes_syn[:,1][phases_syn[:,0] < 0.]))
#             phases_syn_extended = np.hstack((phases_syn[:,1][(phases_syn > 0)]-1., phases_syn, phases_syn[:,1][phases_syn[:,0] < 0.]+1.))
#             fluxes_interp = interp1d(phases_syn_extended, fluxes_syn_extended)
#             fluxes_model = fluxes_interp(phases_obs)
#             chi2 = -0.5*(np.sum((fluxes_ph_obs-fluxes_model)**2/sigmas_ph_obs**2))
        
#         self._chi2 = chi2


