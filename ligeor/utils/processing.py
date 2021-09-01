import numpy as np
try:
    import distl
except:
    raise ModuleNotFoundError


def sample_skewed_gaussian(mean, sigma_low, sigma_high, size=1000):
    '''
    Samples from a skewed Gaussian distribution (hacky).
    
    Assumes different sigma left and right from the mean, samples them separately
    and combines into one sample, stitched together at the mean.
    
    Parameters
    ----------
    mean: float
        Mean of the Gaussian distribution
    sigma_low: float
        Standard deviation of the left Gaussian
    sigma_high: float
        Standard deviation of the right Gaussian
    size: int
        Number of samples to return
        
    Returns
    -------
    An array of the samples drawn from the distribution.
    '''
    
    samples_low = distl.gaussian(mean, sigma_low).sample(size)
    samples_high = distl.gaussian(mean, sigma_high).sample(size)
    
    samples_low_cut = samples_low[samples_low < mean]
    samples_high_cut = samples_high[samples_high >= mean]
    
    return np.hstack((samples_low_cut, samples_high_cut))

def combine_dists_from_hist(samples1, samples2, bins=1000, plot=False):
    '''
    Combine samples from two models and computes new mean and standard deviation.
    
    Parameters
    ----------
    samples1: array-like
        Samples drawn from first distribution.
    samples2: array-like
        Samples drawn from second distribution.
    bins: int
        Number of bins for the histogram
    plot: bool
        If True, will display the histogram.
        
    Returns
    -------
    A dictionary with the mean, sigma_low, sigma_high and confidence of the combined
    distribution.
    '''
    
    samples_full = np.hstack((samples1, samples2))
    hist_combined = distl.histogram_from_data(samples_full, bins=bins)
    
    if plot:
        hist_combined.plot()
        
    uncs = hist_combined.uncertainties()
    return {'mean': uncs[1], 
            'sigma_low': uncs[1]-uncs[0], 
            'sigma_high': uncs[2]-uncs[1],
            'confidence': 1}

def compute_combined_period(lc_ind, results, nsamples = 10000):
    '''
    Compute the combined period from two-Gaussian and polyfit MCMC results.
    
    Parameters
    ----------
    lc_ind: int
        Index of the target in the results file.
    results: dictionary or pandas.DataFrame
        Must contain the following columns: 
        ['period_2g', 'period_sigma_low_2g', 'period_sigma_high_2g', 'chi2_2g',
        'period_pf', 'period_sigma_low_pf', period_sigma_high_pf', 'chi2_pf', 
        'period_combined', 'period_combined_sigma_low', 'period_combined_sigma_high']
        
    Returns
    -------
    Updated results object. 
    '''
    # from 2g and pf to combined periods
    
    period_2g = results['period_2g'][lc_ind]
    period_2g_sigma_low = results['period_sigma_low_2g'][lc_ind]
    period_2g_sigma_high = results['period_sigma_high_2g'][lc_ind]
    
    period_pf = results['period_pf'][lc_ind]
    period_pf_sigma_low = results['period_sigma_low_pf'][lc_ind]
    period_pf_sigma_high = results['period_sigma_high_pf'][lc_ind]
    
    weight_2g = 1./np.abs(results['chi2_2g'][lc_ind])
    weight_pf = 1./np.abs(results['chi2_pf'][lc_ind])
    
    print('2g fit: P = %.5f + %.5f - %.5f' % (period_2g, period_2g_sigma_high, period_2g_sigma_low))
    print('pf fit: P = %.5f + %.5f - %.5f' % (period_pf, period_pf_sigma_high, period_pf_sigma_low))
    
    if ~np.isnan(weight_2g) and ~np.isnan(weight_pf):
        wratio = weight_2g/weight_pf

        nsamples_2g = int(wratio*nsamples/(1+wratio))
        nsamples_pf = int(nsamples/(1+wratio))

        print('values check: w_2g = %.6f, w_pf=%.6f, N_2g = %i, N_pf = %i'
         % (weight_2g, weight_pf, nsamples_2g, nsamples_pf))

        samples_2g = sample_skewed_gaussian(period_2g, period_2g_sigma_low, period_2g_sigma_high, size=nsamples_2g)
        samples_pf = sample_skewed_gaussian(period_pf, period_pf_sigma_low, period_pf_sigma_high, size=nsamples_pf)

        combined_result = combine_dists_from_hist(samples_2g, samples_pf, bins=1000, plot=False)
        print(combined_result)
        results['period_combined'][lc_ind] = combined_result['mean']
        results['period_combined_sigma_low'][lc_ind] = combined_result['sigma_low']
        results['period_combined_sigma_high'][lc_ind] = combined_result['sigma_high']

    
    elif np.isnan(weight_2g) and ~np.isnan(weight_pf):
        results['period_combined'][lc_ind] = period_pf
        results['period_combined_sigma_low'][lc_ind] = period_pf_sigma_low
        results['period_combined_sigma_high'][lc_ind] = period_pf_sigma_high
        
    elif ~np.isnan(weight_2g) and np.isnan(weight_pf):
        results['period_combined'][lc_ind] = period_2g
        results['period_combined_sigma_low'][lc_ind] = period_2g_sigma_low
        results['period_combined_sigma_high'][lc_ind] = period_2g_sigma_high
    
    else:
        results['period_combined'][lc_ind] = np.nan 
        results['period_combined_sigma_low'][lc_ind] = np.nan
        results['period_combined_sigma_high'][lc_ind] = np.nan
    
    return results
        
def compute_combined_t0(lc_ind, results, nsamples = 10000):
    '''
    Compute the combined t0 from two-Gaussian and polyfit MCMC results.
    
    Parameters
    ----------
    lc_ind: int
        Index of the target in the results file.
    results: dictionary or pandas.DataFrame
        Must contain the following columns: 
        ['t0_2g', 't0_sigma_low_2g', 't0_sigma_high_2g', 'chi2_2g',
        't0_pf', 't0_sigma_low_pf', t0_sigma_high_pf', 'chi2_pf', 
        't0_combined', 't0_combined_sigma_low', 't0_combined_sigma_high']
        
    Returns
    -------
    Updated results object. 
    '''
    # from 2g and pf to combined periods
    
    t0_2g = results['t0_2g'][lc_ind]
    t0_2g_sigma_low = results['t0_sigma_low_2g'][lc_ind]
    t0_2g_sigma_high = results['t0_sigma_high_2g'][lc_ind]
    
    t0_pf = results['t0_pf'][lc_ind]
    t0_pf_sigma_low = results['t0_sigma_low_pf'][lc_ind]
    t0_pf_sigma_high = results['t0_sigma_high_pf'][lc_ind]
    
    weight_2g = 1./np.abs(results['chi2_2g'][lc_ind])
    weight_pf = 1./np.abs(results['chi2_pf'][lc_ind])
    
    print('2g fit: t0 = %.5f + %.5f - %.5f' % (t0_2g, t0_2g_sigma_high, t0_2g_sigma_low))
    print('pf fit: t0 = %.5f + %.5f - %.5f' % (t0_pf, t0_pf_sigma_high, t0_pf_sigma_low))
    
    if ~np.isnan(weight_2g) and ~np.isnan(weight_pf):
        wratio = weight_2g/weight_pf

        nsamples_2g = int(wratio*nsamples/(1+wratio))
        nsamples_pf = int(nsamples/(1+wratio))

        print('values check: w_2g = %.6f, w_pf=%.6f, N_2g = %i, N_pf = %i'
         % (weight_2g, weight_pf, nsamples_2g, nsamples_pf))

        samples_2g = sample_skewed_gaussian(t0_2g, t0_2g_sigma_low, t0_2g_sigma_high, size=nsamples_2g)
        samples_pf = sample_skewed_gaussian(t0_pf, t0_pf_sigma_low, t0_pf_sigma_high, size=nsamples_pf)

        combined_result = combine_dists_from_hist(samples_2g, samples_pf, bins=1000, plot=False)
        print(combined_result)
        results['t0_combined'][lc_ind] = combined_result['mean']
        results['t0_combined_sigma_low'][lc_ind] = combined_result['sigma_low']
        results['t0_combined_sigma_high'][lc_ind] = combined_result['sigma_high']

    
    elif np.isnan(weight_2g) and ~np.isnan(weight_pf):
        results['t0_combined'][lc_ind] = t0_pf
        results['t0_combined_sigma_low'][lc_ind] = t0_pf_sigma_low
        results['t0_combined_sigma_high'][lc_ind] = t0_pf_sigma_high
        
    elif ~np.isnan(weight_2g) and np.isnan(weight_pf):
        results['t0_combined'][lc_ind] = t0_2g
        results['t0_combined_sigma_low'][lc_ind] = t0_2g_sigma_low
        results['t0_combined_sigma_high'][lc_ind] = t0_2g_sigma_high
    
    else:
        results['t0_combined'][lc_ind] = np.nan 
        results['t0_combined_sigma_low'][lc_ind] = np.nan
        results['t0_combined_sigma_high'][lc_ind] = np.nan
        
    return results