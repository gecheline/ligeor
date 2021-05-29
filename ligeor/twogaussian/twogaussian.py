import numpy as np
from scipy.optimize import curve_fit
from ligeor.utils.lcutils import *


class TwoGaussianModel(object):

    def __init__(self, phases=[], fluxes=[], sigmas=[], filename='', n_downsample=1, usecols=(0,1,2), delimiter=','):
        '''
        Computes the two-Gaussian model light curves of the input data.

        Parameters
        ----------
        phases: array-like
            Input orbital phases, must be on the range [-0.5,0.5]
        fluxes: array-like
            Input fluxes
        sigmas: array-like
            Input sigmas (corresponding flux uncertainities)
        filename: str
            Filename from which to load a PHASE FOLDED light curve.
        n_downsample: int
            Number of data points to skip in loaded light curve (for downsampling)
        usecols: array-like, len 2 or 3
            Indices of the phases, fluxes and sigmas columns in file.
        '''

        if len(phases) == 0 or len(fluxes) == []:
            try:
                lc = load_lc(filename, n_downsample=n_downsample, phase_folded=True, usecols=usecols, delimiter=delimiter)
                self.phases = lc['phases']
                self.fluxes = lc['fluxes']
                self.sigmas = lc['sigmas']
            except Exception as e:
                raise ValueError(f'Loading light curve failed with exception {e}')
        else:
            self.filename = filename
            self.phases = phases 
            self.fluxes = fluxes 
            self.sigmas = sigmas

        # this is just all the parameter names for each model
        self.twogfuncs = {'C': TwoGaussianModel.const, 
                    'CE': TwoGaussianModel.ce, 
                    'CG': TwoGaussianModel.cg, 
                    'CGE': TwoGaussianModel.cge, 
                    'CG12': TwoGaussianModel.cg12, 
                    'CG12E': TwoGaussianModel.cg12e
                    }

        self.params = {'C': ['C'],
                'CE': ['C', 'Aell', 'phi0'],
                'CG': ['C', 'mu1', 'd1', 'sigma1'],
                'CGE': ['C', 'mu1', 'd1', 'sigma1', 'Aell', 'phi0'],
                'CG12': ['C', 'mu1', 'd1', 'sigma1', 'mu2', 'd2', 'sigma2'],
                'CG12E': ['C', 'mu1', 'd1', 'sigma1', 'mu2', 'd2', 'sigma2', 'Aell', 'phi0']}


    # def check_fit(self):
    #     '''
    #     Checks for anomalies in the best fit:
    #     - overlapping Gaussians
    #     - a Gaussian fitted to noise
    #     - a Gaussian fitted to out-of-eclipse variability
    #     '''

    #     best_fit_func = list(self.models.keys())[np.nanargmax(list(self.bics.values()))]

    #     # CHECK IF TWO GAUSSIANS FIT THE SAME ECLIPSE
    #     if best_fit_func in ['CG12', 'CG12E1', 'CG12E2']:
    #         C, mu1, d1, sigma1, mu2, d2, sigma2 = self.fits[best_fit_func][0][:7]
    #         best_fit_func_check = self.check_overlapping_eclipses(best_fit_func, mu1, mu2, sigma1, sigma2)
    #         if best_fit_func != best_fit_func_check:
    #             # refit with secondary 0.5 away from primary
    #             new_mu2 = mu1 + 0.5
    #             new_mu2 = new_mu2 - 1 if new_mu2 > 0.5 else new_mu2
    #             self.fit_twoGaussian_models(init_pos=[mu1, new_mu2], init_widths=[sigma1, 0.01])
    #             # check one last time after refitting
    #             C, mu1, d1, sigma1, mu2, d2, sigma2 = self.fits[best_fit_func][0][:7]
    #             best_fit_func = self.check_overlapping_eclipses(best_fit_func, mu1, mu2, sigma1, sigma2)

    #     # CHECK IF ANY OF THE ECLIPSES FIT THE DATA NOISE
    #     if best_fit_func in ['CG', 'CGE']:
    #         C, mu1, d1, sigma1 = self.fits[best_fit_func][0][:4]
    #         best_fit_func = self.check_eclipse_fitting_noise(best_fit_func, self.fluxes, self.models[best_fit_func], d1, d2=np.nan)
        
    #     elif best_fit_func in ['CG12', 'CG12E1', 'CG12E2']:
    #         C, mu1, d1, sigma1, mu2, d2, sigma2= self.fits[best_fit_func][0][:7]
    #         # check if primary fits noise
    #         best_fit_func_check = self.check_eclipse_fitting_noise(best_fit_func, self.fluxes, self.models[best_fit_func], d1, d2=d2)
    #         if best_fit_func_check == 'refit':
    #             # we need to refit such that the secondary eclipse is considered the primary and remove the primary
    #             best_fit_func = 'CG' if best_fit_func == 'CG12' else 'CGE'
    #             new_mu2 = mu2 + 0.5
    #             new_mu2 = new_mu2 - 1 if new_mu2 > 0.5 else new_mu2
    #             self.fit_twoGaussian_models(init_pos=[mu2, new_mu2], init_widths=[sigma2, 0.])
    #         else:
    #             best_fit_func = best_fit_func_check

    #     # CHECK IF ANY OF THE ECLIPSES FIT THE OUT OF ECLIPSE VARIABILITY
    #     if best_fit_func in ['CG', 'CGE']:
    #         C, mu1, d1, sigma1 = self.fits[best_fit_func][0][:4]
    #         best_fit_func = self.check_eclipse_fitting_cosine(best_fit_func, sigma1)
        
    #     elif best_fit_func in ['CG12', 'CG12E1', 'CG12E2']:
    #         C, mu1, d1, sigma1, mu2, d2, sigma2 = self.fits[best_fit_func][0][:7]
    #         # check if primary fits noise
    #         best_fit_func_check = self.check_eclipse_fitting_cosine(best_fit_func, sigma1, sigma2)
    #         if best_fit_func_check == 'refit':
    #             # we need to refit such that the secondary eclipse is considered the primary and remove the primary
    #             best_fit_func = 'CGE'
    #             new_mu2 = mu2 + 0.5
    #             new_mu2 = new_mu2 - 1 if new_mu2 > 0.5 else new_mu2
    #             self.fit_twoGaussian_models(init_pos=[mu2, new_mu2], init_widths=[sigma2, 0.])
    #         else:
    #             best_fit_func = best_fit_func_check

    #     return best_fit_func


    def fit(self):
        '''
        Computes all two-gaussian models and chooses the best fit.

        The fitting is carried out in the following steps: 
            1. fit each of the 7 models,
            2. compute their model light curves
            3. compute each model's BIC value
            4. assign the model with highest BIC as the 'best fit'
        '''

        self.fit_twoGaussian_models()
        # compute all model light curves
        self.compute_twoGaussian_models()
        # compute corresponding BIC values
        self.compute_twoGaussian_models_BIC()
        
        # best_fit_func = self.check_fit()
        # choose the best fit as the one with highest BIC
        best_fit_func = list(self.models.keys())[np.nanargmax(list(self.bics.values()))]

        self.best_fit = {}
        self.best_fit['func'] = best_fit_func
        self.best_fit['model'] = self.models[best_fit_func]
        self.best_fit['param_vals'] = self.fits[best_fit_func]
        self.best_fit['param_names'] = self.params[best_fit_func]


    def save_model(self, nbins=1000, func='', param_values = [], save_file=''):
        '''
        Save the best fit model to a file.

        Parameters
        ----------
        nbins: int
            The number of phase points between -0.5 and 0.5 to compute the model in
        save_file: str
            The filename to save to.
        '''
        
        if len(save_file) == 0:
            save_file = self.filename + '.2g'
        phases = np.linspace(-0.5,0.5,nbins)
        fluxes = self.twogfuncs[func](phases, *param_values)
        np.savetxt(save_file, np.array([phases, fluxes]).T)

    # HELPER FUNCTIONS

    @staticmethod
    def ellipsoidal(phi, Aell, phi0):
        '''
        Ellipsoidal model, defined as $y = (1/2) A_{ell} \cos (4 \pi (\phi - \phi_0))$
        
        Parameters
        ----------
        phi: float or array-like
            The input phase or phases array to compute the model in
        Aell: float
            Amplitude of the elliposoidal
        phi0: float
            Phase-point to center the elliposoidal on (position of primary or secondary eclipse)

        Returns
        -------
        y: array-like
            $y = (1/2) A_{ell} \cos (4 \pi (\phi - \phi_0))$
        '''
        # just the cosine component with the amplitude and phase offset as defined in Mowlavi (2017)
        return 0.5*Aell*np.cos(4*np.pi*(phi-phi0))

    @staticmethod
    def gaussian(phi, mu, d, sigma):
        '''
        Gaussian model, defined as $y = d \exp(-(\phi-\mu)^2/(2\sigma^2))$

        Parameters
        ----------
        phi: float or array-like
            The input phase or phases array to compute the model in
        mu: float
            Position of the Gaussian
        d: float
            Amplitude of the Gaussian
        sigma: float
            Scale (FWHM) of the Gaussian

        Returns
        -------
        y: array-like
            $y = d \exp(-(\phi-\mu)^2/(2\sigma^2))$
        '''

        # one Gaussian
        return d*np.exp(-(phi-mu)**2/(2*sigma**2))

    @staticmethod
    def gsum(phi, mu, d, sigma):
        '''
        Copies the Gaussian to mu +/- 1 and mu +/- 2 to account for 
        extended phase-folded light curves that cover multiple orbits.
        
        Parameters
        ----------
        phi: float or array-like
            The input phase or phases array to compute the model in
        mu: float
            Position of the Gaussian
        d: float
            Amplitude of the Gaussian
        sigma: float
            Scale (FWHM) of the Gaussian

        Returns
        -------
        y: array-like
            Array with gaussians at mu, mu +/-1 and mu +/- 2.
        '''

        gauss_sum = np.zeros(len(phi))
        for i in range(-2,3,1):
            gauss_sum += TwoGaussianModel.gaussian(phi,mu+i,d,sigma)
        return gauss_sum

    # MODELS as defined in Mowalvi (2017)

    @staticmethod
    def const(phi, C):
        '''The constant model y = C'
        
        Parameters
        ----------
        phi: float or array-like
            The input phase or phases array to compute the model in
        C: float
            value of the constant

        Returns
        -------
        y: array-like
            $y = C$
        '''

        # constant term
        return C*np.ones(len(phi))

    @staticmethod
    def ce(phi, C, Aell, phi0):
        '''Constant + ellipsoidal model
        
        Parameters
        ----------
        phi: float or array-like
            The input phase or phases array to compute the model in
        C: float
            value of the constant
        Aell: float
            Amplitude of the elliposoidal
        phi0: float
            Phase-point to center the elliposoidal on (position of primary or secondary eclipse)

        Returns
        -------
        y: array-like
            y = const(phi, C) - ellipsoidal(phi, Aell, phi0)
        '''

        return TwoGaussianModel.const(phi, C) - TwoGaussianModel.ellipsoidal(phi, Aell, phi0)

    @staticmethod
    def cg(phi, C, mu, d,  sigma):
        '''Constant + Gaussian model
        
        Parameters
        ----------
        phi: float or array-like
            The input phase or phases array to compute the model in
        C: float
            value of the constant
        mu: float
            Position of the Gaussian
        d: float
            Amplitude of the Gaussian
        sigma: float
            Scale (FWHM) of the Gaussian

        Returns
        -------
        y: array-like
            y = const(phi, C) - gsum(phi, mu, d, sigma)
        '''
        # constant + one gaussian (just primary eclipse)
        return TwoGaussianModel.const(phi, C) - TwoGaussianModel.gsum(phi, mu, d, sigma)

    @staticmethod
    def cge(phi, C, mu, d, sigma, Aell, phi0):
        '''
        Constant + Gaussian + ellipsoidal model

        Parameters
        ----------
        phi: float or array-like
            The input phase or phases array to compute the model in
        C: float
            value of the constant
        mu: float
            Position of the Gaussian
        d: float
            Amplitude of the Gaussian
        sigma: float
            Scale (FWHM) of the Gaussian
        Aell: float
            Amplitude of the elliposoidal
        phi0: float
            Phase-point to center the elliposoidal on (position of primary or secondary eclipse)

        Returns
        -------
        y: array-like
            y = const(phi, C) - gsum(phi, mu, d, sigma) - ellipsoidal(phi, Aell, mu)
        '''
        return TwoGaussianModel.const(phi, C) - TwoGaussianModel.ellipsoidal(phi, Aell, phi0) - TwoGaussianModel.gsum(phi, mu, d, sigma)

    @staticmethod
    def cg12(phi, C, mu1, d1, sigma1, mu2, d2, sigma2):
        '''
        Constant + two Gaussians model
        
        Parameters
        ----------
        phi: float or array-like
            The input phase or phases array to compute the model in
        C: float
            value of the constant
        mu1: float
            Position of the first Gaussian
        d1: float
            Amplitude of the first Gaussian
        sigma1: float
            Scale (FWHM) of the first Gaussian
        mu2: float
            Position of the second Gaussian
        d2: float
            Amplitude of the second Gaussian
        sigma2: float
            Scale (FWHM) of the second Gaussian

        Returns
        -------
        y: array-like
            y = const(phi, C) - gsum(phi, mu1, d1, sigma1) - gsum(phi, mu2, d2, sigma2) 
        '''
        return TwoGaussianModel.const(phi, C) - TwoGaussianModel.gsum(phi, mu1, d1, sigma1) - TwoGaussianModel.gsum(phi, mu2, d2, sigma2)

    @staticmethod
    def cg12e(phi, C, mu1, d1, sigma1, mu2, d2, sigma2, Aell, phi0):
        '''
        Constant + two Gaussians + ellipsoidal centered on the primary eclipse
        
        Parameters
        ----------
        phi: float or array-like
            The input phase or phases array to compute the model in
        C: float
            value of the constant
        mu1: float
            Position of the first Gaussian
        d1: float
            Amplitude of the first Gaussian
        sigma1: float
            Scale (FWHM) of the first Gaussian
        mu2: float
            Position of the second Gaussian
        d2: float
            Amplitude of the second Gaussian
        sigma2: float
            Scale (FWHM) of the second Gaussian
        Aell: float
            Amplitude of the elliposoidal
        phi0: float
            Phase-point to center the elliposoidal on (position of primary or secondary eclipse)


        Returns
        -------
        y: array-like
            y = const(phi, C) - gsum(phi, mu1, d1, sigma1) - gsum(phi, mu2, d2, sigma2) - ellipsoidal(phi, Aell, mu1)
        '''
        return TwoGaussianModel.const(phi, C) - TwoGaussianModel.gsum(phi, mu1, d1, sigma1) - TwoGaussianModel.gsum(phi, mu2, d2, sigma2) - TwoGaussianModel.ellipsoidal(phi, Aell, phi0)

    # @staticmethod
    # def cg12e2(phi, C, mu1, d1, sigma1, mu2, d2, sigma2, Aell):
    #     '''
    #     Constant + two Gaussians + ellipsoidal centered on the secondary eclipse
        
    #     Parameters
    #     ----------
    #     phi: float or array-like
    #         The input phase or phases array to compute the model in
    #     C: float
    #         value of the constant
    #     mu1: float
    #         Position of the first Gaussian
    #     d1: float
    #         Amplitude of the first Gaussian
    #     sigma1: float
    #         Scale (FWHM) of the first Gaussian
    #     mu2: float
    #         Position of the second Gaussian
    #     d2: float
    #         Amplitude of the second Gaussian
    #     sigma2: float
    #         Scale (FWHM) of the second Gaussian
    #     Aell: float
    #         Amplitude of the elliposoidal
    #     phi0: float
    #         Phase-point to center the elliposoidal on (position of primary or secondary eclipse)


    #     Returns
    #     -------
    #     y: array-like
    #         y = const(phi, C) - gsum(phi, mu1, d1, sigma1) - gsum(phi, mu2, d2, sigma2) - ellipsoidal(phi, Aell, mu2)
    #     '''
    #     return TwoGaussianModel.const(phi, C) - TwoGaussianModel.gsum(phi, mu1, d1, sigma1) - TwoGaussianModel.gsum(phi, mu2, d2, sigma2) - TwoGaussianModel.ellipsoidal(phi, Aell, mu2)


    @staticmethod
    def lnlike(y, yerr, ymodel):
        '''
        Computes the loglikelihood of a model.

        $\log\mathrm{like} = \sum_i \log(\sqrt{2\pi} \sigma_i) + (y_i - model_i)^2/(2\sigma_i^2)
        '''
        if yerr is not None:
            return -np.sum(np.log((2*np.pi)**0.5*yerr)+(y-ymodel)**2/(2*yerr**2))
        else:
            return -np.sum((y-ymodel)**2)

    def bic(self, ymodel, nparams):
        '''
        Computes the Bayesian Information Criterion (BIC) value of a model.

        BIC = 2 lnlike - n_params \log(n_data)
        '''
        if self.sigmas is not None:
            return 2*self.lnlike(self.fluxes, self.sigmas, ymodel) - nparams*np.log(len(self.fluxes))
        else:
            return self.lnlike(self.fluxes, self.sigmas, ymodel)

    @staticmethod
    def find_eclipse(phases, fluxes):
        phase_min = phases[np.nanargmin(fluxes)]
        ph_cross = phases[fluxes - np.nanmedian(fluxes) > 0]
        # this part looks really complicated but it really only accounts for eclipses split
        # between the edges of the phase range - if a left/right edge is not found, we look for
        # it in the phases on the other end of the range
        # we then mirror the value back on the side of the eclipse position for easier width computation
        try:
            arg_edge_left = np.argmin(np.abs(phase_min - ph_cross[ph_cross<phase_min]))
            edge_left = ph_cross[ph_cross<phase_min][arg_edge_left]
        except:
            arg_edge_left = np.argmin(np.abs((phase_min+1)-ph_cross[ph_cross<(phase_min+1)]))
            edge_left = ph_cross[ph_cross<(phase_min+1)][arg_edge_left]-1
        try:
            arg_edge_right = np.argmin(np.abs(phase_min-ph_cross[ph_cross>phase_min]))
            edge_right = ph_cross[ph_cross>phase_min][arg_edge_right]
        except:
            arg_edge_right = np.argmin(np.abs((phase_min-1)-ph_cross[ph_cross>(phase_min-1)]))
            edge_right = ph_cross[ph_cross>(phase_min-1)][arg_edge_right]+1

        return phase_min, edge_left, edge_right

    @staticmethod
    def estimate_eclipse_positions_widths(phases, fluxes, diagnose_init=False):
        pos1, edge1l, edge1r = TwoGaussianModel.find_eclipse(phases, fluxes)
        fluxes_sec = fluxes.copy()
        fluxes_sec[((phases > edge1l) & (phases < edge1r)) | ((phases > edge1l+1) | (phases < edge1r-1))] = np.nan
        pos2, edge2l, edge2r = TwoGaussianModel.find_eclipse(phases, fluxes_sec)


        if diagnose_init:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10,8))
            plt.plot(phases, fluxes, '.')
            plt.axhline(y=np.median(fluxes), c='orange')
            for i,x in enumerate([pos1, edge1l, edge1r]):
                ls = '-' if i==0 else '--'
                plt.axvline(x=x, c='r', ls=ls)
            for i,x in enumerate([pos2, edge2l, edge2r]):
                ls = '-' if i==0 else '--'
                plt.axvline(x=x, c='g', ls=ls)
            plt.show()

        return {'ecl_positions': [pos1, pos2], 'ecl_widths': [edge1r-edge1l, edge2r-edge2l]}
        
    def fit_twoGaussian_models(self, init_pos=[], init_widths=[]):
        '''
        Fits all seven models to the input light curve.
        '''
        # setup the initial parameters

        C0 = self.fluxes.max()
        ecl_dict = self.estimate_eclipse_positions_widths(self.phases, self.fluxes, diagnose_init=False)
        if len(init_pos) == 0 or len(init_widths) == 0:
            self.init_positions, self.init_widths = ecl_dict['ecl_positions'], ecl_dict['ecl_widths']
            mu10, mu20 = self.init_positions
            sigma10, sigma20 = self.init_widths
        else:
            mu10, mu20 = init_pos
            sigma10, sigma20 = init_widths
        d10 = self.fluxes.max()-self.fluxes[np.argmin(np.abs(self.phases-mu10))]
        d20 = self.fluxes.max()-self.fluxes[np.argmin(np.abs(self.phases-mu20))]
        Aell0 = 0.001
        phi0 = 0.0

        init_params = {'C': [C0,],
            'CE': [C0, Aell0, phi0],
            'CG': [C0, mu10, d10, sigma10],
            'CGE': [C0, mu10, d10, sigma10, Aell0, phi0],
            'CG12': [C0, mu10, d10, sigma10, mu20, d20, sigma20],
            'CG12E': [C0, mu10, d10, sigma10, mu20, d20, sigma20, Aell0, phi0]}

        # parameters used frequently for bounds
        fmax = self.fluxes.max()
        fmin = self.fluxes.min()
        fdiff = fmax - fmin

        bounds = {'C': ((0),(fmax)),
            'CE': ((0, 1e-6, -0.5),(fmax, fdiff, 0.5)),
            'CG': ((0., -0.5, 0., 0.), (fmax, 0.5, fdiff, 0.5)),
            'CGE': ((0., -0.5, 0., 0., 1e-6, -0.5),(fmax, 0.5, fdiff, 0.5, fdiff, 0.5)),
            'CG12': ((0.,-0.5, 0., 0., -0.5, 0., 0.),(fmax, 0.5, fdiff, 0.5, 0.5, fdiff, 0.5)),
            'CG12E': ((0.,-0.5, 0., 0., -0.5, 0., 0., 1e-6, -0.5),(fmax, 0.5, fdiff, 0.5, 0.5, fdiff, 0.5, fdiff, 0.5))}

        fits = {}

        # extend light curve on phase range [-1,1]
        phases_ext, fluxes_ext, sigmas_ext = extend_phasefolded_lc(self.phases, self.fluxes, self.sigmas)

        for key in self.twogfuncs.keys():
            try:
                fits[key] = curve_fit(self.twogfuncs[key], phases_ext, fluxes_ext, p0=init_params[key], sigma=sigmas_ext, bounds=bounds[key])
            except Exception as err:
                print("2G model {} failed with error: {}".format(key, err))
                fits[key] = np.array([np.nan*np.ones(len(init_params[key]))])

        self.fits = fits


    def compute_twoGaussian_models(self):
        '''
        Computes the model light curves given the fit solutions.
        '''
        models = {}

        for fkey in self.fits.keys():
            models[fkey] = self.twogfuncs[fkey](self.phases, *self.fits[fkey][0])

        self.models = models


    def compute_twoGaussian_models_BIC(self):
        '''
        Computes the BIC value of each model light curve.
        '''
        bics = {}
        nparams = {'C':1, 'CE':3, 'CG':4, 'CGE':6, 'CG12':7, 'CG12E':9}

        for mkey in self.models.keys():
            bics[mkey] = self.bic(self.models[mkey], nparams[mkey])

        self.bics = bics


    def compute_eclipse_params(self, interactive=False):
        '''
        Compute the positions, widths and depths of the eclipses 
        based on the two-Gaussian model solution.

        The eclipse parameters are computed following the prescription
        in Mowlavi et al. (2017):
        - eclipse positions are set to the Gaussian positions
        - eclipse withs are 5.6 times the Gaussian FWHMs
        - eclipse depths are the constant factor minus the value of 
        the model at the eclipse positions

        Parameters
        ----------
        interactive: boolean
            If True, allows the user to manually adjust the positions of 
            eclipse minimum and edges. Default: False.

        Returns
        -------
        results: dict
            A dictionary of the eclipse paramter values.
        '''
        param_vals = self.best_fit['param_vals'][0]
        param_names = self.best_fit['param_names']

        # gather values from the best fit solution
        sigma1 = param_vals[param_names.index('sigma1')] if 'sigma1' in param_names else np.nan
        sigma2 = param_vals[param_names.index('sigma2')] if 'sigma2' in param_names else np.nan
        mu1 = param_vals[param_names.index('mu1')] if 'mu1' in param_names else np.nan
        mu2 = param_vals[param_names.index('mu2')] if 'mu2' in param_names else np.nan
        C = param_vals[param_names.index('C')]

        # compute and adjust all available parameters, otherwise the entire eclipse is nan
        if not np.isnan(mu1) and not np.isnan(sigma1) and np.abs(sigma1) < 0.5:
            pos1 = mu1
            width1 = min(5.6*np.abs(sigma1), 0.5)
            depth1 = C - self.fluxes[np.argmin(np.abs(self.phases-pos1))]
        else:
            pos1 = np.nan
            width1 = np.nan
            depth1 = np.nan
        if not np.isnan(mu2) and not np.isnan(sigma2) and np.abs(sigma2) < 0.5:
            pos2 = mu2
            width2 = min(5.6*np.abs(sigma2), 0.5)
            depth2 = C - self.fluxes[np.argmin(np.abs(self.phases-pos2))]
        else:
            pos2 = np.nan
            width2 = np.nan
            depth2 = np.nan

        # compute the eclipse edges using the positons and widths
        eclipse_edges = [pos1 - 0.5*width1, pos1+0.5*width1, pos2-0.5*width2, pos2+0.5*width2]


        if interactive:
            # this option allows the user to manually adjust the eclipse positions
            # and edges and recompute the eclipse parameters

            from ligeor.utils.interactive import DraggableLine
            phases_w, fluxes_w, sigmas_w = extend_phasefolded_lc(self.phases, self.fluxes, self.sigmas)
            [ecl1_l, ecl1_r, ecl2_l, ecl2_r] = eclipse_edges

            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(111)
            ax.plot(phases_w, fluxes_w, 'k.')
            plt.plot(phases_w, self.twogfuncs[self.best_fit](phases_w, *self.fits[self.best_fit][0]), '-', label=self.best_fit)
            lines = []
            lines.append(ax.axvline(x=pos1, c='#2B71B1', lw=2, label='primary'))
            lines.append(ax.axvline(x=pos2, c='#FF702F', lw=2, label='secondary'))
            lines.append(ax.axvline(x=ecl1_l, c='#2B71B1', lw=2, ls='--'))
            lines.append(ax.axvline(x=ecl1_r, c='#2B71B1', lw=2, ls='--'))
            lines.append(ax.axvline(x=ecl2_l, c='#FF702F', lw=2, ls='--'))
            lines.append(ax.axvline(x=ecl2_r, c='#FF702F', lw=2, ls='--'))
            drs = []
            for l,label in zip(lines,['pos1', 'pos2', 'ecl1_l', 'ecl1_r', 'ecl2_l', 'ecl2_r']):   
                dr = DraggableLine(l)
                dr.label = label
                dr.connect()   
                drs.append(dr) 
            ax.legend()
            plt.show(block=True)

            print('adjusting values')

            pos1 = drs[0].point.get_xdata()[0]
            pos2 = drs[1].point.get_xdata()[0]
            ecl1_l = drs[2].point.get_xdata()[0]
            ecl1_r = drs[3].point.get_xdata()[0]
            ecl2_l = drs[4].point.get_xdata()[0]
            ecl2_r = drs[5].point.get_xdata()[0]
            width1 = ecl1_r - ecl1_l
            width2 = ecl2_r - ecl2_l
            depth1 = C - self.fluxes[np.argmin(np.abs(self.phases-pos1))]
            depth2 = C - self.fluxes[np.argmin(np.abs(self.phases-pos2))]
            
            eclipse_edges = [ecl1_l, ecl1_r, ecl2_l, ecl2_r]


        return {
            'primary_width': width1,
            'secondary_width': width2,
            'primary_position': pos1,
            'secondary_position': pos2,
            'primary_depth': depth1,
            'secondary_depth': depth2,
            'eclipse_edges': eclipse_edges
        }