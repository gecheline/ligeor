import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from itertools import groupby
from numpy.core.fromnumeric import mean
from scipy.optimize import minimize
from ligeor.utils.lcutils import load_lc


class Polyfit(object):

    def __init__(self, filename='', phases=None, fluxes=None, sigmas=None, n_downsample=0,
                        xmin=-0.5, xmax=0.5, polyorder=2, chain_length=4):
        '''
        Initializes a polyfit with data and fit parameters.

        Parameters
        ----------
        filename: str
            Filename of light curve to load (if phases, fluxes and sigmas not provided)
        phases: array-like
            Input orbital phases, must be on the range [-0.5,0.5]
        fluxes: array-like
            Input fluxes
        sigmas: array-like
            Input sigmas (corresponding flux uncertainities)
        n_downsample: int
            Number of data points to skip if downsampling
        xmin: float
            Minimum orbital phase of the polyfit
        xmax: float
            Maximum orbital phase of the polyfit
        polyorder: int
            Polynomial order of each chain
        chain_length: int
            Length of the chain
        '''

        if polyorder != 2:
            raise NotImplemented('polyorder != 2 is not yet implemented.')
        if chain_length != 4:
            raise NotImplemented('chain_length != 4 is not yet implemented.')


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

        # for now, build a data attribute to match the Polychain methods
        self.data = np.array([self.phases, self.fluxes, self.sigmas]).T

        self.xmin = xmin
        self.xmax = xmax
        self.polyorder = polyorder
        self.chain_length = chain_length
        self.sdata = self.data[self.data[:,0].argsort()]

        
    def _find_knots(self, min_chain_length=8, verbose=False):
        mean = self.sdata[:,1].mean()
        lt = np.where(self.sdata[:,1]<mean)[0]
        
        chains = []
        for k, g in groupby(enumerate(lt), lambda ix: ix[0]-ix[1]):
            chain = list(map(itemgetter(1), g))
            if len(chain) > min_chain_length:
                chains.append(chain)
        if verbose:
            print(f'{len(chains)} chains found.')
        if len(chains) < 2:
            self.knots = np.array((-0.4, -0.1, 0.1, 0.4))
        elif len(chains) == 2:
            self.knots = np.sort((self.sdata[chains[0][0],0], self.sdata[chains[0][-1],0], self.sdata[chains[1][0],0], self.sdata[chains[1][-1],0]))
        else:
            lengths = np.array([len(chain) in chains])
            l = np.argsort(lengths)[::-1]
            self.knots = np.sort((self.sdata[chains[l[0]][0],0], self.sdata[chains[l[0]][-1],0], self.sdata[chains[l[1]][0],0], self.sdata[chains[l[1]][-1],0]))

        return self.knots
    

    def _find_segments(self, knots):
        self.sdata = self.data.copy()
        self.sdata[:,0][self.sdata[:,0] < knots[0]] += 1
        self.sdata = self.sdata[self.sdata[:,0].argsort()]

        segs = [np.argmax(self.sdata[:,0]>knot) for knot in knots[1:]] + [len(self.sdata)]
        return segs


    def _build_A_matrix(self, knots, segs):
        self.A = np.zeros(shape=(len(self.sdata), 9))

        self.A[:segs[0],0] = self.sdata[:segs[0],0]**2
        self.A[segs[0]:,0] = knots[1]**2
        self.A[:segs[0],1] = self.sdata[:segs[0],0]
        self.A[segs[0]:,1] = knots[1]
        self.A[:,2] = 1.0

        for k in range(3):
            self.A[segs[k]:segs[k+1],3+2*k] = self.sdata[segs[k]:segs[k+1],0]**2-knots[k+1]**2
            self.A[segs[k]:segs[k+1],4+2*k] = self.sdata[segs[k]:segs[k+1],0]-knots[k+1]
            for j in range(k):
                self.A[segs[k]:segs[k+1],3+2*j] = knots[j+2]**2-knots[j+1]**2
                self.A[segs[k]:segs[k+1],4+2*j] = knots[j+2]-knots[j+1]

        return self.A
    

    def _fit_chain(self, knots, min_pts_per_segment=5, return_ck=False):
        segs = self._find_segments(knots)
        if np.ediff1d(segs).min() < min_pts_per_segment:
            return 1e10

        A = self._build_A_matrix(knots, segs)
        ck, ssr, rank, svd = np.linalg.lstsq(self.A, self.sdata[:,1], rcond=None)

        if return_ck:
            return ck, ssr
        return ssr
    

    def _chain_coeffs(self, ck, verbose=False):
        c0 = ck[2]
        c1 = (ck[0]-ck[3])*self.knots[1]**2 + (ck[1]-ck[4])*self.knots[1] + c0
        c2 = (ck[3]-ck[5])*self.knots[2]**2 + (ck[4]-ck[6])*self.knots[2] + c1
        c3 = (ck[5]-ck[7])*self.knots[3]**2 + (ck[6]-ck[8])*self.knots[3] + c2

        if verbose:
            print(f'segment 1:\n {self.knots[0]} < x <= {self.knots[1]}, a0={ck[0]} b0={ck[1]} c0={c0}')
            print(f'segment 2:\n {self.knots[1]} < x <= {self.knots[2]}, a0={ck[3]} b0={ck[4]} c0={c1}')
            print(f'segment 3:\n {self.knots[2]} < x <= {self.knots[3]}, a0={ck[5]} b0={ck[6]} c0={c2}')
            print(f'segment 4:\n {self.knots[3]} < x <= {self.knots[0]+1}, a0={ck[7]} b0={ck[8]} c0={c3}')

        self.coeffs = ((ck[0], ck[1], c0), (ck[3], ck[4], c1), (ck[5], ck[6], c2), (ck[7], ck[8], c3))
    

    def _chain_extremes(self):
        exts_x = [-c[1]/2/c[0] for c in self.coeffs]
        knots = np.concatenate((self.knots, [self.knots[0]+1]))
        exts_x = np.array(exts_x)
        exts_y = np.zeros(len(exts_x))
        for k in range(4):
            if exts_x[k] < knots[k] or exts_x[k] > knots[k+1]:
                exts_x[k] = np.nan
            c = self.coeffs[k]
            exts_y[k] = c[0]*exts_x[k]**2 + c[1]*exts_x[k] + c[2]
        self.extremes = np.array([exts_x, exts_y]).T
    

    def _remap(self, d, sort=True):
        ncols = 1 if len(d.shape) == 1 else d.shape[1]
        x = d if ncols == 1 else d[:,0]

        while len(x[x<self.xmin]) > 0:
            x[x<self.xmin] += 1
        while len(x[x>self.xmax]) > 0:
            x[x>self.xmax] -= 1

        if sort and ncols > 1:
            d = d[d[:,0].argsort()]
        else:
            d.sort()

        return d      
    
    
    def fv(self, x):
        x[x<self.knots[0]] += 1
        y = np.empty_like(x)
        for k in range(len(self.knots)-1):
            s = (x>=self.knots[k]) & (x<self.knots[k+1])
            y[s] = self.coeffs[k][0]*x[s]**2 + self.coeffs[k][1]*x[s] + self.coeffs[k][2]
        s = x>=self.knots[3]
        y[s] = self.coeffs[3][0]*x[s]**2 + self.coeffs[3][1]*x[s] + self.coeffs[3][2]

        return y


    def fit(self, min_chain_length=8, min_pts_per_segment=5, verbose=False):
        self._find_knots(min_chain_length=min_chain_length, verbose=verbose)
        solution = minimize(self._fit_chain, self.knots, args=(min_pts_per_segment,), method='Nelder-Mead')
        self.knots = solution.x
        ck, self.ssr = self._fit_chain(self.knots, min_pts_per_segment=min_pts_per_segment, return_ck=True)
        self._chain_coeffs(ck)
        self._chain_extremes()
        self.model = self.fv(self.phases)
        
        return self.knots, self.coeffs, self.extremes


    def plot(self, x):
        if x is None:
            x = self.sdata[:,0]
    
        y = self.fv(x)
        d = np.vstack((x, y)).T

        self._remap(self.sdata)
        d = self._remap(d)
        self._remap(self.extremes)

        plt.plot(self.data[:,0], self.data[:,1], 'b.')
        plt.plot(d[:,0], d[:,1], 'r-')

        for k in range(4):
            plt.axvline(self.extremes[:,0][k], ls='--')
        plt.show()

    # def save_model(self, knots, nbins=1000, niters=0, save_file=''):
    #     '''
    #     Saves a model light curve, given the knot positions, to a file.
        
    #     Parameters
    #     ----------
    #     knots: array-like
    #         The knot positions for the polyfit.
    #     niters: int
    #         Number of iterations for polyfit. Default in this case is 0 to use
    #         the user-provided knots and avoid refitting.
    #     nbins: int
    #         Number of phase bins for the polyfit model to be computed in.
    #     save_model_file: str
    #         Filename to save the model to.
    #     '''

    #     temp = tempfile.NamedTemporaryFile(mode='w+t', suffix=".lc")
    #     np.savetxt(temp.name, np.array([self.phases, self.fluxes, self.sigmas]).T)

    #     if len(save_file) == 0:
    #         save_file = self.filename + '.pf'
    #     proc = subprocess.Popen([f'polyfit -k {knots} -n {nbins} -i {niters} {temp.name} > {save_file}'], stdout=subprocess.PIPE, shell=True) #% (knots, nbins, nitera temp.name, save_file)]
    #     temp.close()

    def compute_eclipse_params(self):
        '''
        Compute the positions, widths and depths of the eclipses 
        based on the polyfit solution.

        The eclipse parameters are computed as following:
        - eclipses are first identified as the chains corresponding to the
          two deepest minima
        - eclipse positions are set to the minima positions
        - eclipse widths are the difference between the chain knots 
        - eclipse depths are the difference between the mean function values at the
          knots and function values at the minima.

        Returns
        -------
        results: dict
            A dictionary of the eclipse paramter values.
        '''

        eclipse_args = np.argsort(self.extremes[:,1])[:2]

        # let's extend the knots array left and right so it's continuous (for computing the width)
        knots_extended = np.hstack((self.knots-0.5, self.knots, self.knots+0.5))

        knots1 = np.array([knots_extended[eclipse_args[0]+4], knots_extended[eclipse_args[0]+5]])
        knots2 = np.array([knots_extended[eclipse_args[1]+4], knots_extended[eclipse_args[1]+5]])

        mean_outofecl = self.fv(np.hstack((knots1, knots2))).mean()

        self.eclipse_params = {
            'primary_width': knots1[1]-knots1[0],
            'secondary_width': knots2[1]-knots2[0],
            'primary_position': self.extremes[:,0][eclipse_args[0]],
            'secondary_position': self.extremes[:,0][eclipse_args[1]],
            'primary_depth': mean_outofecl - self.extremes[:,1][eclipse_args[0]],
            'secondary_depth': mean_outofecl - self.extremes[:,1][eclipse_args[1]],
            'eclipse_edges': np.hstack((knots1, knots2)),
            'eclipse_coeffs': [self.coeffs[eclipse_args[0]], self.coeffs[eclipse_args[1]]]
        }
        
        return self.eclipse_params
        

    def compute_residuals_stdev(self):
        '''
        Computes the residuals of the input fluxes and best fit model

        Returns
        -------
        residuals_mean: float
            The mean of the residuals
        residuals_stdev: float
            The standard deviation of the residuals
        '''

        self.residuals_mean = np.mean((self.fluxes - self.model))
        self.residuals_stdev = np.std((self.fluxes - self.model))
        return self.residuals_mean, self.residuals_stdev

    def compute_eclipse_area(self, ecl=1):
        '''
        Computes the area under an eclipse.

        An eclipse is defined as being positioned between the knots around the deepest minima.

        Parameters
        ----------
        ecl: int
            The eclipse whose area is to be computed (1 or 2)
        
        Returns
        -------
        eclipse_area: float
            The computed area under the chosen eclipse.
        '''

        if hasattr(self, 'eclipse_area'):
            pass
        else:
            self.eclipse_area = {}

        coeffs = self.eclipse_params['eclipse_coeffs'][ecl-1]
        edges = self.eclipse_params['eclipse_edsges'][ecl-1]

        self.eclipse_area[ecl] = coeffs[0]/3*(edges[1]**3-edges[0]**3) + coeffs[1]/2*(edges[1]**2-edges[0]**2) + coeffs[2]*(edges[1]-edges[0])
