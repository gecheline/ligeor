import numpy as np

def load_lc(lc_file, n_downsample=0, phase_folded=False, usecols=(0,1,2), delimiter=','):
    '''
    Loads the light curve from lc_file and returns 
    separate arrays for times, fluxes and sigmas.

    Parameters
    ----------
    lc_file: str
        Filename to load light curve from.
    n_downsample: int
        Number of data points to skip in downsampling the lc.

    Returns
    -------
    A dictionary of the times, fluxes and sigmas retrieved from the lc file.
    '''
    
    lc = np.loadtxt(lc_file, usecols=usecols, delimiter=delimiter)

    if phase_folded:
        return {'phases': lc[:,0][::n_downsample], 
        'fluxes': lc[:,1][::n_downsample], 
        'sigmas': lc[:,2][::n_downsample]}

    else:
        return {'times': lc[:,0][::n_downsample], 
                'fluxes': lc[:,1][::n_downsample], 
                'sigmas': lc[:,2][::n_downsample]}


def phase_fold(times, fluxes, sigmas, period=1, t0=0):
    '''
    Phase-folds the light curve with a given period and t0.

    Parameters
    ----------
    times: array-like
        The observation times
    fluxes: array-like
        Observed fluxes corresponding to each time in times
    sigmas: array-like
        Uncertainties corresponding to each flux in fluxes


    Returns
    -------
    phases: array-like
        The computed orbital phases on a range [-0.5,0.5], sorted in ascending order
    fluxes_ph: array_like
        The fluxes resorted to match the phases array order
    sigmas_ph: array_like
        The sigmas resorted to match the phases array order
    '''
    
    t0 = 0 if np.isnan(t0) else t0
    phases = np.mod((times-t0)/period, 1.0)

    if isinstance(phases, float):
        if phases > 0.5:
            phases -= 1
    else:
        # then should be an array
        phases[phases > 0.5] -= 1
        
    s = phases.argsort()
    phases = phases[s]
    fluxes_ph = fluxes[s]
    sigmas_ph = sigmas[s]
    
    return phases, fluxes_ph, sigmas_ph


def extend_phasefolded_lc(phases, fluxes, sigmas):
    '''
    Takes a phase-folded light curve on the range [-0.5,0.5] and extends it on range [-1,1]
    
    Parameters
    ----------
    phases: array-like
        Array of input phases spanning the range [-0.5,0.5]
    fluxes: array-like
        Corresponding fluxes, length must be equal to that of phases
    sigmas: array-like
        Corresponsing measurement uncertainties, length must be equal to that og phases
        
    Returns
    -------
    phases_extend, fluxes_extend, sigmas_extend: array-like
        Extended arrays on phase-range [-1,1]
    
    '''
    #make new arrays that would span phase range -1 to 1:
    fluxes_extend = np.hstack((fluxes[(phases > 0)], fluxes, fluxes[phases < 0.]))
    phases_extend = np.hstack((phases[phases>0]-1, phases, phases[phases<0]+1))

    if sigmas is not None:
        sigmas_extend = np.hstack((sigmas[phases > 0], sigmas, sigmas[phases < 0.]))
    else:
        sigmas_extend = None

    return phases_extend, fluxes_extend, sigmas_extend
