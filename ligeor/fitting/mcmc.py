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
        nbins: int
            Number of bins (applies to the computed final model).
        '''
        self._filename = filename
        self._nbins = nbins
        self._period_init = period_init 
        self._t0_init = t0_init
        lc = load_lc(filename, n_downsample=n_downsample, **kwargs)
        self._times, self._fluxes, self._sigmas = lc['times'], lc['fluxes'], lc['sigmas']

    def initial_fit(self):
        '''
        Runs an initial fit to the data with the chosen model (two-Gaussian or polyfit).
        '''
        # overriden by subclass
        return None
    
    def logprob(self, values):
        '''
        Computes the logprobability of the sample.

        Parameters
        ----------
        values: array-like
            period (for phase folding) + model values
        '''
        # overriden by subclass
        return None

    def run_sampler(self, nwalkers=32, niters=2000, progress=True):
        '''
        Initializes and runs an emcee sampler.
        
        Parameters
        ----------
        nwalkers: int
            Number of walkers for emcee.
        niters: int
            Number of iterations to run.
        progress: bool
            If True, will output the progress (requires tqdm).
        '''
        if ~hasattr(self, '_initial_fit'):
            raise ValueError('Initial fit not found. Run sampler.initial_fit() before calling run_sampler.')
        else:
            init_vals = self._initial_fit
            bestfit_vals = np.array([self._period_init, *init_vals])
            pos = bestfit_vals + 1e-4 * np.random.randn(nwalkers, len(bestfit_vals))
            nwalkers, ndim = pos.shape
            
            with Pool(1) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, self.logprob, pool=pool)
                sampler.run_mcmc(pos, niters, progress=progress)
            
            return sampler
        

    def compute_results(self, sampler, burnin = 1000, save_lc=True, save_file='', show=False, failed=False):
        '''
        Computes a summary of the results from the sampler.

        The results computed include: ephemerides, model parameters and eclipse parameters. 

        Parameters
        ----------
        sampler: object
            The emcee sampler, initialized and run with .run_sampler()
        burnin: int
            Number of initial iterations to discard.
        save_lc: bool
            If True, will save the light curve to a file.
        save_file: str
            Filename to save to, if save_lc = True.
        show: bool
            If True, will show plot of the resulting light curve.
        failed: bool
            If True, all computed values are np.nan
        '''

        if save_lc and len(save_file) == 0:
            raise ValueError('Please provide a file name to save the model to or set save_lc=False.')
        
        # store the rest of the model parameters and their uncertainties
        means, sigmas_low, sigmas_high, means_blobs, sigmas_low_blobs, sigmas_high_blobs = self.compute_ephemerides(sampler, burnin, failed=failed)
        self.compute_model(means, sigmas_low, sigmas_high, save_lc = save_lc, save_file=save_file, show=show, failed=failed)
        self.compute_eclipse_params(means_blobs, sigmas_low_blobs, sigmas_high_blobs, failed=failed)

    def compute_ephemerides(self, sampler, burnin, failed=False):
        '''
        Computes mean and standard deviation for the period and t0 from the sample.

        Parameters
        ----------
        sampler: object
            The emcee sampler, initialized and run with .run_sampler()
        burnin: int
            Number of initial iterations to discard.
        '''

        if failed:
            self._period_mcmc = {'value': np.nan,
                    'sigma_low': np.nan, 
                    'sigma_high': np.nan}

            self._t0_mcmc = {'value': np.nan,
                    'sigma_low': np.nan, 
                    'sigma_high': np.nan}

            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        else:
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
                mcmc = np.nanpercentile(samples_top[:, j], [16, 50, 84])
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
                mcmc_blob = np.nanpercentile(blobs_top[:, j], [16, 50, 84])
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

            return means, sigmas_low, sigmas_high, means_blobs, sigmas_low_blobs, sigmas_high_blobs


    def save_results_to_file(self, results_file, type='ephemerides', ind=''):
        '''
        Save the resulting ephemerides, model or eclipse parameters to a file.

        Parameters
        ----------
        results_file: str
            Filename to save to
        type: str
            Which parameters to store. Available choices: ['ephemerides', 'model_values', 'eclipse_parameters']
        ind: str
            Index of the object (if looping through a list, otherwise optional)
        '''

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


    def compute_model(self, means, sigmas_low, sigmas_high, save_lc = True, save_file='', show=False, failed=False):
        '''
        Computes the model parameter values from the sample.

        Parameters
        ----------
        means: array-like
            Mean values from the sample
        sigmas_low: array-like
            Standard deviation of samples < mean
        sigmas_high: array_like
            Standard deviation of samples > mean
        save_lc: bool
            If True, saves the model light curve to a file
        save_file: str
            File name to save light curve to, if save_lc=True.
        show: bool
            If True, will display a plot of the model light curve.
        failed: bool
            If True, all model parameters are np.nan
        '''
        # overriden by subclass
        return None


    def compute_eclipse_params(self, means, sigmas_low, sigmas_high, failed=False):
        '''
        Computes the model parameter values from the sample.

        Parameters
        ----------
        means: array-like
            Mean values from the sample
        sigmas_low: array-like
            Standard deviation of samples < mean
        sigmas_high: array_like
            Standard deviation of samples > mean
        failed: bool
            If true, all eclipse parameters are np.nan
        '''
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

        if not failed:
            for ind, eclkey in enumerate(eclipse_params.keys()):
                # print(eclkey, means[ind], np.max((sigmas_low[ind],sigmas_high[ind])))
                eclipse_params[eclkey] = means[ind]
                eclipse_params_err[eclkey] = np.max((sigmas_low[ind],sigmas_high[ind]))

        
        self._eclipse_params = eclipse_params
        self._eclipse_params_errs = eclipse_params_err


class EmceeSamplerTwoGaussian(EmceeSampler):

    def __init__(self, filename, period_init, t0_init, n_downsample=0, nbins=1000, **kwargs):
        '''
        Initializes a TwoGaussian sampler for the light curve stored in 'filename'
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
        nbins: int
            Number of bins (applies to the computed final model).
        '''
        
        super(EmceeSamplerTwoGaussian,self).__init__(filename, period_init, t0_init, n_downsample, nbins, **kwargs)


    def initial_fit(self):
        '''
        Runs an initial fit to the data with the chosen model (two-Gaussian or polyfit).
        '''
        phases, fluxes_ph, sigmas_ph = phase_fold(self._times, 
                                                        self._fluxes, 
                                                        self._sigmas, 
                                                        period=self._period_init, 
                                                        t0=self._t0_init,
                                                        interval = '05')

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
        '''
        Computes the logprobability of the sample.

        Parameters
        ----------
        values: array-like
            period (for phase folding) + model values
        '''

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
                                                        t0=self._t0_init,
                                                        interval='05')
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

    
    def compute_model(self, means, sigmas_low, sigmas_high, save_lc = True, save_file='', show=False, failed=False):
        '''
        Computes the model parameter values from the sample.

        Parameters
        ----------
        means: array-like
            Mean values from the sample
        sigmas_low: array-like
            Standard deviation of samples < mean
        sigmas_high: array_like
            Standard deviation of samples > mean
        save_lc: bool
            If True, saves the model light curve to a file
        save_file: str
            File name to save light curve to, if save_lc=True.
        show: bool
            If True, will display a plot of the model light curve.
        failed: bool
            If True, all model parameters are np.nan
        '''        
        
        model_results = {'C': np.nan, 'mu1': np.nan, 'd1': np.nan, 'sigma1': np.nan, 
                        'mu2': np.nan, 'd2': np.nan, 'sigma2': np.nan, 'Aell': np.nan, 'phi0': np.nan
                        }
        model_results_err = {'C': np.nan, 'mu1': np.nan, 'd1': np.nan, 'sigma1': np.nan, 
                        'mu2': np.nan, 'd2': np.nan, 'sigma2': np.nan, 'Aell': np.nan, 'phi0': np.nan
                        }
        
        # results_str = '{}'.format(func)
        if failed:
            self._model_values = model_results
            self._model_values_errs = model_results_err
            self._chi2 = np.nan
        
        else:
            for mkey in model_results.keys():
                if mkey in self._model_params:
                    pind = self._model_params.index(mkey)
                    model_results[mkey] = means[pind+1]
                    model_results_err[mkey] = np.max((sigmas_low[pind+1],sigmas_high[pind+1]))
                    # results_str += ',{},{}'.format(model_results[mkey],model_results_err[mkey])
                
            self._model_values = model_results
            self._model_values_errs = model_results_err
            chi2 = np.nan

            phases_obs, fluxes_ph_obs, sigmas_ph_obs = phase_fold(self._times, 
                                                    self._fluxes, 
                                                    self._sigmas, 
                                                    period=self._period_mcmc['value'], 
                                                    t0=self._t0_init, interval='05')
            twog_func = getattr(TwoGaussianModel, self._func.lower())
            fluxes_model = twog_func(phases_obs, *means[1:])
            chi2 = -0.5*(np.sum((fluxes_ph_obs-fluxes_model)**2/sigmas_ph_obs**2))
            
            if show:
                import matplotlib.pyplot as plt
                plt.plot(phases_obs, fluxes_ph_obs, 'k.')
                plt.plot(phases_obs, fluxes_model, 'r-')
                plt.show()

            if save_lc:
                np.savetxt(save_file, np.array([phases_obs, fluxes_model]).T)

            self._chi2 = chi2


class EmceeSamplerPolyfit(EmceeSampler):

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
        nbins: int
            Number of bins (applies to the computed final model).
        '''
        
        super(EmceeSamplerPolyfit,self).__init__(filename, period_init, t0_init, n_downsample, nbins, **kwargs)


    def initial_fit(self, knots = [], coeffs = []):
        '''
        Runs an initial fit to the data with the chosen model (two-Gaussian or polyfit).
        '''
        
        if len(knots) == 0:
            phases, fluxes_ph, sigmas_ph = phase_fold(self._times, 
                                                            self._fluxes, 
                                                            self._sigmas, 
                                                            period=self._period_init, 
                                                            t0=self._t0_init, interval='01')

            polyfit = Polyfit(phases=phases, 
                                        fluxes=fluxes_ph, 
                                        sigmas=sigmas_ph, 
                                        )
            polyfit.fit()

            self._initial_fit = np.hstack((np.array(polyfit.knots), np.array(polyfit.coeffs).reshape(12)))

        else:
            if len(coeffs) == 0:
                phases, fluxes_ph, sigmas_ph = phase_fold(self._times, 
                                                            self._fluxes, 
                                                            self._sigmas, 
                                                            period=self._period_init, 
                                                            t0=self._t0_init, interval='01')

                polyfit = Polyfit(phases=phases, 
                                            fluxes=fluxes_ph, 
                                            sigmas=sigmas_ph, 
                                            )
                polyfit.fit(knots = knots)
                self._initial_fit = np.hstack((np.array(knots), np.array(polyfit.coeffs).reshape(12)))
            
            else:
                self._initial_fit = np.hstack((np.array(knots), np.array(coeffs).reshape(12)))

        self._func = 'PF'
        self._model_params = ['k1', 'k2', 'k3', 'k4', 'c11', 'c12', 'c13', 'c21', 'c22', 'c23', 'c31', 'c32', 'c33', 'c41', 'c42', 'c43']
        self._bounds = []
        for i,value in enumerate(self._initial_fit):
            self._bounds.append([value-0.1,value+0.1*value])

    def logprob(self, values):
        '''
        Computes the logprobability of the sample.

        Parameters
        ----------
        values: array-like
            period (for phase folding) + model values
        '''
        # bounds = [[-1e-5,-1e-5,-1e-5,-1e-5],[1+1e-5,1+1e-5,1+1e-5,1+1e-5]]
        # bounds = [[-1,-1,-1,-1],[2,2,2,2]]
        period, *model_vals = values

        for i,param_val in enumerate(model_vals[:4]):
            if param_val < self._bounds[i][0] or param_val > self._bounds[i][1]:
                print('out of prior', self._bounds[i][0], self._bounds[i][1], param_val)
                return -np.inf, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        phases, fluxes_ph, sigmas_ph = phase_fold(self._times, 
                                                        self._fluxes, 
                                                        self._sigmas, 
                                                        period=period, 
                                                        t0=self._t0_init, interval='01')
        
        try:
            #TODO: figure out the cause for the "cannot unpack error"
            polyfit = Polyfit(phases=phases, 
                                        fluxes=fluxes_ph, 
                                        sigmas=sigmas_ph)
            polyfit.fit(knots = np.array(model_vals[:4]), coeffs = np.array(model_vals[4:]).reshape(4,3))
            
            eclipse_dir = polyfit.compute_eclipse_params()
            pos1, pos2 = eclipse_dir['primary_position'], eclipse_dir['secondary_position']
            width1, width2 = eclipse_dir['primary_width'], eclipse_dir['secondary_width']
            depth1, depth2 = eclipse_dir['primary_depth'], eclipse_dir['secondary_depth']
            ecl1_area, ecl2_area = polyfit.eclipse_area[1], polyfit.eclipse_area[2]
            residuals_mean, residuals_stdev = compute_residuals_stdev(fluxes_ph, polyfit.model)

            # print('residuals: ', residuals_mean, residuals_stdev)
            logprob = -0.5*(np.sum((fluxes_ph-polyfit.model)**2/sigmas_ph**2))
            # print(logprob, pos1, width1, depth1, pos2, width2, depth2)#, ecl1_area, ecl2_area, residuals_mean, residuals_stdev)
            return logprob, pos1, width1, depth1, pos2, width2, depth2, ecl1_area, ecl2_area, residuals_mean, residuals_stdev
        except:
            return -np.inf, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


    def compute_model(self, means, sigmas_low, sigmas_high, save_lc = True, save_file='', show=False, failed=False):
        '''
        Computes the model parameter values from the sample.

        Parameters
        ----------
        means: array-like
            Mean values from the sample
        sigmas_low: array-like
            Standard deviation of samples < mean
        sigmas_high: array_like
            Standard deviation of samples > mean
        save_lc: bool
            If True, saves the model light curve to a file
        save_file: str
            File name to save light curve to, if save_lc=True.
        show: bool
            If True, will display a plot of the model light curve.
        failed: bool
            If True, all model parameters are np.nan
        '''
                
        model_results = {'k1': np.nan, 'k2': np.nan, 'k3': np.nan, 'k4': np.nan,
                        'c11': np.nan, 'c12': np.nan, 'c13': np.nan,
                        'c21': np.nan, 'c22': np.nan, 'c23': np.nan,
                        'c31': np.nan, 'c32': np.nan, 'c33': np.nan,
                        'c41': np.nan, 'c42': np.nan, 'c43': np.nan
                        }
        model_results_err = {'k1': np.nan, 'k2': np.nan, 'k3': np.nan, 'k4': np.nan,
                            'c11': np.nan, 'c12': np.nan, 'c13': np.nan,
                            'c21': np.nan, 'c22': np.nan, 'c23': np.nan,
                            'c31': np.nan, 'c32': np.nan, 'c33': np.nan,
                            'c41': np.nan, 'c42': np.nan, 'c43': np.nan
                        }
        if failed:
            self._model_values = model_results
            self._model_values_errs = model_results_err
            self._chi2 = np.nan

        else:
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

            
            phases_obs, fluxes_ph_obs, sigmas_ph_obs = phase_fold(self._times, 
                                                    self._fluxes, 
                                                    self._sigmas, 
                                                    period=self._period_mcmc['value'], 
                                                    t0=self._t0_init,
                                                    interval='01')
        
            polyfit = Polyfit(phases=phases_obs, 
                                    fluxes=fluxes_ph_obs, 
                                    sigmas=sigmas_ph_obs)

            knots = np.array(list(self._model_values.values())[:4])
            coeffs = np.array(list(self._model_values.values())[4:]).reshape(4,3)
            polyfit.fit(knots = knots, coeffs = coeffs)
            fluxes_model = polyfit.fv(x = phases_obs)
            chi2 = -0.5*(np.sum((fluxes_ph_obs-fluxes_model)**2/sigmas_ph_obs**2))
            self._chi2 = chi2

            if show:
                import matplotlib.pyplot as plt
                plt.plot(phases_obs, fluxes_ph_obs, 'k.')
                plt.plot(phases_obs, fluxes_model, 'r-')
                plt.show()

            if save_lc:
                np.savetxt(save_file, np.array([phases_obs, fluxes_model]).T)