import numpy as np
import emcee 
from ligeor.twogaussian import TwoGaussianModel
from ligeor.polyfit import Polyfit
from ligeor.utils.lcutils import *
from multiprocessing import Pool
import tempfile
import subprocess
from scipy.interpolate import interp1d

class EmceeSampler(object):

    def __init__(self, filename, period_init, t0_init, n_downsample=0, nbins=1000):

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
        self._times, self._fluxes, self._sigmas = load_lc(filename, n_downsample=n_downsample)

    def initial_fit(self):
        # overriden by subclass
        return None
    
    def logprob(self):
        # overriden by subclass
        return None

    def run_sampler(self, nwalkers=32, niters=2000, progress=True):
        init_vals = self.initial_fit()
        bestfit_vals = np.array([self._period_init, *init_vals])
        pos = bestfit_vals + 1e-4 * np.random.randn(nwalkers, len(bestfit_vals))
        nwalkers, ndim = pos.shape
        
        with Pool(nwalkers) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.logprob, pool=pool)
            sampler.run_mcmc(pos, niters, progress=progress)
        
        return sampler
        

    def compute_results(self, sampler, burnin = 1000, save_lc=True):
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
            log_prob_mean = np.mean(log_prob[log_prob >= logp_lim])
        except:
            samples_top = flat_samples
            blobs_top = flat_blobs
            log_prob_mean = np.mean(log_prob)
        
        ndim = samples_top.shape[1]
        solution = []
        labels = ['period', *self.model_param]
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

        # compute the new t0 and errs
        mcmc_phasemin = np.percentile(blobs_top, [16, 50, 84])
        q_phasemin = np.diff(mcmc_phasemin)

        phasemin_mean = mcmc_phasemin[1]
        phasemin_sigma_low = q_phasemin[0]
        phasemin_sigma_high = q_phasemin[1]

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
        self.compute_model(means, sigmas_low, sigmas_high, save_lc = save_lc)


    def save_results_to_file(self, results_file, type='ephemerides', ind=''):
        if type=='ephemerides':
            with open(results_file, 'a') as f:
                f.write('{},{},{},{},{},{},{},{},{}\n'.format(ind,
                                                            self._period_mcmc['value'],
                                                            self._period_mcmc['sigma_low'],
                                                            self._period_mcmc['sigma_high'],
                                                            self._t0_mcmc['value'],
                                                            self._t0_mcmc['sigma_low'],
                                                            self._t0_mcmc['sigma_high'],
                                                            self._chi2))

        elif type=='model_values':
            for i,mkey in enumerate(self._model_values.keys()):
                results_str += '{},{}'.format(self._model_values[mkey],self._model_values_err[mkey])
                if i < len(self._model_values.keys()-1):
                    results_str += ','
                else:
                    results_str += '\n'

            with open(results_file, 'a') as f:
                f.write('{},{}'.format(ind,results_str))

        else:
            raise NotImplementedError

    def compute_model_values(self, means, sigmas_low, sigmas_high):
        return None


class EmceeSamplerTwoGaussian(EmceeSampler):

    def __init__(self, filename, period_init, t0_init):
        super(EmceeSamplerTwoGaussian,self).__init__(filename, period_init, t0_init)


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

        if twogModel.best_fit['func'] == 'C':
            return None
        else:
            self.func = twogModel.best_fit['func']
            self.model_params = twogModel.best_fit['param_names']
            return twogModel.best_fit['param_vals']


    def logprob(self, values):
        fmax = self._fluxes.max()
        fmin = self._fluxes.min()
        fdiff = fmax - fmin

        bounds = {'C': ((0),(fmax)),
            'CE': ((0, 1e-6, -0.5),(fmax, fdiff, 0.5)),
            'CG': ((0., -0.5, 0., 0.), (fmax, 0.5, fdiff, 0.5)),
            'CGE': ((0., -0.5, 0., 0., 1e-6),(fmax, 0.5, fdiff, 0.5, fdiff)),
            'CG12': ((0.,-0.5, 0., 0., -0.5, 0., 0.),(fmax, 0.5, fdiff, 0.5, 0.5, fdiff, 0.5)),
            'CG12E1': ((0.,-0.5, 0., 0., -0.5, 0., 0., 1e-6),(fmax, 0.5, fdiff, 0.5, 0.5, fdiff, 0.5, fdiff)),
            'CG12E2': ((0.,-0.5, 0., 0., -0.5, 0., 0., 1e-6),(fmax, 0.5, fdiff, 0.5, 0.5, fdiff, 0.5, fdiff))}
        
        period, *model_vals = values
        
        for i,param_val in enumerate(model_vals):
            if param_val < bounds[self.func][0][i] or param_val > bounds[self.func][1][i]:
                raise Warning('out of prior', self.func, bounds[self.func][0][i], bounds[self.func][1][i], param_val)
                return -np.inf
            
        # fold with period
        phases, fluxes_ph, sigmas_ph = phase_fold(self._times, 
                                                        self._fluxes, 
                                                        self._sigmas, 
                                                        period=period, 
                                                        t0=self._t0_init)
        # compute model with the selected twog function
        # TODO: review this part for potentially more efficient option
        model = TwoGaussianModel.twogfuncs[self.func](phases, *model_vals)
        return -0.5*(np.sum((fluxes_ph-model)**2/sigmas_ph**2)), model_vals[1]

    
    def compute_model(self, means, sigmas_low, sigmas_high, save_lc = True):
        model_results = {'C': np.nan, 'mu1': np.nan, 'd1': np.nan, 'sigma1': np.nan, 
                        'mu2': np.nan, 'd2': np.nan, 'sigma2': np.nan, 'Aell': np.nan
                        }
        model_results_err = {'C': np.nan, 'mu1': np.nan, 'd1': np.nan, 'sigma1': np.nan, 
                        'mu2': np.nan, 'd2': np.nan, 'sigma2': np.nan, 'Aell': np.nan
                        }
        
        # results_str = '{}'.format(func)
        for mkey in model_results.keys():
            if mkey in self.model_params:
                pind = self.model_params.index(mkey)
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
                                                    period=self._period_mcmc.value, 
                                                    t0=self._t0_mcmc.value)
        
            phases_syn = np.linspace(-0.5,0.5,self._nbins)
            fluxes_syn = TwoGaussianModel.twogfuncs[self.func](phases_syn, *means[1:])
            
            np.savetxt(self._filename+'.2g', np.array([phases_syn, fluxes_syn]).T)

            fluxes_syn_extended = np.hstack((fluxes_syn[:,1][(phases_syn > 0)], fluxes_syn, fluxes_syn[:,1][phases_syn[:,0] < 0.]))
            phases_syn_extended = np.hstack((phases_syn[:,1][(phases_syn > 0)]-1., phases_syn, phases_syn[:,1][phases_syn[:,0] < 0.]+1.))
            fluxes_interp = interp1d(phases_syn_extended, fluxes_syn_extended)
            fluxes_model = fluxes_interp(phases_obs)
            chi2 = -0.5*(np.sum((fluxes_ph_obs-fluxes_model)**2/sigmas_ph_obs**2))
        
        self._chi2 = chi2


class EmceeSamplerPolyfit(EmceeSampler):


    def __init__(self, filename, period_init, t0_init):
        super(EmceeSamplerPolyfit,self).__init__(filename, period_init, t0_init)


    def initial_fit(self):
        phases, fluxes_ph, sigmas_ph = phase_fold(self._times, 
                                                        self._fluxes, 
                                                        self._sigmas, 
                                                        period=self._period_init, 
                                                        t0=self._t0_init,
                                                        )

        temp = tempfile.NamedTemporaryFile(mode='w+t', suffix=".lc")
        np.savetxt(temp.name, np.array([phases, fluxes_ph, sigmas_ph]).T)

        proc = subprocess.Popen(["polyfit --find-knots -n 1000 -i 1000 --summary-output %s" % temp.name], stdout=subprocess.PIPE, shell=True)
        (out, err) = proc.communicate()
        out = np.array((out.decode("utf-8")).split('\n'))[0]
        init_knots = np.array(out.split('\t')[-4:]).astype(float)
        temp.close()

        return init_knots


    def logprob(self, values):
        tempfile.tempdir='temp_lcs/'
        
        bounds = [[-0.5,-0.5,-0.5,-0.5],[0.5,0.5,0.5,0.5]]
        period, *model_vals = values

        for i,param_val in enumerate(model_vals):
            if param_val < bounds[0][i] or param_val > bounds[1][i]:
                print('out of prior', bounds[0][i], bounds[1][i], param_val)
                return -np.inf

        phases, fluxes_ph, sigmas_ph = phase_fold(self._times, 
                                                        self._fluxes, 
                                                        self._sigmas, 
                                                        period=period, 
                                                        t0=self._t0_init)
        
        temp = tempfile.NamedTemporaryFile(mode='w+t', suffix=".lc")
        np.savetxt(temp.name, np.array([phases, fluxes_ph, sigmas_ph]).T)

        knots = ' '.join([str(elem) for elem in model_vals]) 

        pfModel = Polyfit(phases=phases, fluxes=fluxes_ph, sigmas=sigmas_ph)
        chi2, phase_min, knots = pfModel.fit(niters = 0, nbins=self._nbins)

        return -chi2, phase_min


    def compute_model(self, means, sigmas_low, sigmas_high, save_lc=True):

        model_results = {'k1': np.nan, 'k2': np.nan, 'k3': np.nan, 'k4': np.nan, 
                        'phasemin': np.nan
                        }
        model_results_err = {'k1': np.nan, 'k2': np.nan, 'k3': np.nan, 'k4': np.nan, 
                        'phasemin': np.nan
                        }

        for pind,mkey in enumerate(model_results.keys()):
                model_results[mkey] = means[pind+1]
                model_results_err[mkey] = np.max((sigmas_low[pind+1],sigmas_high[pind+1]))

        self._model_values = model_results
        self._model_values_errs = model_results_err
        chi2 = np.nan
        
        if save_lc:
            phases_obs, fluxes_ph_obs, sigmas_ph_obs = phase_fold(self._times, 
                                                    self._fluxes, 
                                                    self._sigmas, 
                                                    period=self._period_mcmc.value, 
                                                    t0=self._t0_mcmc.value)

            temp = tempfile.NamedTemporaryFile(mode='w+t', suffix=".lc")
            np.savetxt(temp.name, np.array([phases_obs, fluxes_ph_obs, sigmas_ph_obs]).T)

            save_file = self._filename + '.pf'
            knots = ' '.join([str(elem) for elem in means[1:]]) 
            proc = subprocess.Popen([f'polyfit -k {knots} -n {self._nbins} -i 0 {temp.name} > {save_file}'], stdout=subprocess.PIPE, shell=True) #% (knots, nbins, nitera temp.name, save_file)]
            temp.close()

            lc_syn = np.loadtxt(save_file)
            phases_syn, fluxes_syn = lc_syn[:,0], lc_syn[:,1]

            fluxes_syn_extended = np.hstack((fluxes_syn[:,1][(phases_syn > 0)], fluxes_syn, fluxes_syn[:,1][phases_syn[:,0] < 0.]))
            phases_syn_extended = np.hstack((phases_syn[:,1][(phases_syn > 0)]-1., phases_syn, phases_syn[:,1][phases_syn[:,0] < 0.]+1.))
            fluxes_interp = interp1d(phases_syn_extended, fluxes_syn_extended)
            fluxes_model = fluxes_interp(phases_obs)
            chi2 = -0.5*(np.sum((fluxes_ph_obs-fluxes_model)**2/sigmas_ph_obs**2))
        
        self._chi2 = chi2


