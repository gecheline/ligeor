import tempfile
import subprocess
import numpy as np
from ligeor.utils.lcutils import load_lc


class Polyfit(object):

    def __init__(self, filename='', phases=None, fluxes=None, sigmas=None, n_downsample=0):
        '''
        A wrapper around https://github.com/aprsa/polyfit

        Parameters
        ----------
        phases: array-like
            Input orbital phases, must be on the range [-0.5,0.5]
        fluxes: array-like
            Input fluxes
        sigmas: array-like
            Input sigmas (corresponding flux uncertainities)

        '''
        if phases == None or fluxes == None:
            try:
                lc = load_lc(filename, n_downsample=n_downsample)
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


    def fit(self, niters=1000, nbins=1000):
        '''
        Fits a polyfit model to the data through subprocess and temporarily
        storing the data to a file for polyfit to read.

        Parameters
        ----------
        niters: int
            Number of iterations for polyfit
        nbins: int
            Number of phase bins for the polyfit model to be computed in

        Returns
        -------
            chi2: float
                The chi2 value of the fitted model.
            phasemin: float
                The position of the polyfit minimum.
            knots: array-like
                The positions of the four knots.
        '''

        temp = tempfile.NamedTemporaryFile(mode='w+t', suffix=".lc")
        np.savetxt(temp.name, np.array([self.phases, self.fluxes, self.sigmas]).T)

        proc = subprocess.Popen([f'polyfit --find-knots -n {nbins} -i {niters} --summary-output {temp.name}'], stdout=subprocess.PIPE, shell=True)
        (out, err) = proc.communicate()
        out = np.array((out.decode("utf-8")).split('\n'))[0]
        chi2 = float(out.split('\t')[1])
        phasemin = float(out.split('\t')[2])
        knots = np.array(out.split('\t')[-4:]).astype(float)
        temp.close()

        return chi2, phasemin, knots


    def save_model(self, knots, nbins=1000, niters=0, save_file=''):
        '''
        Saves a model light curve, given the knot positions, to a file.
        
        Parameters
        ----------
        knots: array-like
            The knot positions for the polyfit.
        niters: int
            Number of iterations for polyfit. Default in this case is 0 to use
            the user-provided knots and avoid refitting.
        nbins: int
            Number of phase bins for the polyfit model to be computed in.
        save_model_file: str
            Filename to save the model to.
        '''

        temp = tempfile.NamedTemporaryFile(mode='w+t', suffix=".lc")
        np.savetxt(temp.name, np.array([self.phases, self.fluxes, self.sigmas]).T)

        if len(save_file) == 0:
            save_file = self.filename + '.pf'
        proc = subprocess.Popen([f'polyfit -k {knots} -n {nbins} -i {niters} {temp.name} > {save_file}'], stdout=subprocess.PIPE, shell=True) #% (knots, nbins, nitera temp.name, save_file)]
        temp.close()

    def compute_eclipse_params(self, knots):
        return NotImplementedError


    def compute_residuals_stdev(self):
        return NotImplementedError

    def compute_eclipse_area(self, ecl=1):
        return NotImplementedError
