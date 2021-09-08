from ligeor import TwoGaussianModel as TwoG
import numpy as np

C = 1.
mu1 = 0.
sigma1 = 0.015
d1 = 0.5
mu2 = 0.42
sigma2 = 0.01
d2 = 0.35
Aell = 0.05
phi01 = 0.
phi02 = 0.42

def test_c():
    data = np.loadtxt('../data/const.csv', delimiter=',')
    model = TwoG(phases=data[:,0], fluxes=data[:,1], sigmas=data[:,2])
    result_fit = {'func': 'C', 'param_vals': [C]}

    model.fit()
    assert(model.best_fit['func'] == result_fit['func'])
    assert((np.abs(
              model.best_fit['param_vals'][0] - 
              np.array(result_fit['param_vals'])) < 1e-2*np.ones(len(model.best_fit['param_vals'][0]))).all())
    

def test_ce():
    data = np.loadtxt('../data/ce.csv', delimiter=',')
    model = TwoG(phases=data[:,0], fluxes=data[:,1], sigmas=data[:,2])
    result_fit = {'func': 'CE', 'param_vals': [C, Aell, phi01]}
    model.fit()
    assert(model.best_fit['func'] == result_fit['func'])

    # compare only C and Aell, because phi0 can be phi0+/-0.5
    assert((np.abs(
              model.best_fit['param_vals'][0][:-1] - 
              np.array(result_fit['param_vals'][:-1])) < 1e-2*np.ones(len(model.best_fit['param_vals'][0][:-1]))).all())
    
    assert((np.abs(
              model.best_fit['param_vals'][0][-1] - 
              np.array(result_fit['param_vals'][-1])) < 1e-2) | (np.abs(
              model.best_fit['param_vals'][0][-1] - 0.5 -
              np.array(result_fit['param_vals'][-1])) < 1e-2) | (np.abs(
              model.best_fit['param_vals'][0][-1] + 0.5 -
              np.array(result_fit['param_vals'][-1])) < 1e-2))
    
def test_cg():
    data = np.loadtxt('../data/cg.csv', delimiter=',')
    model = TwoG(phases=data[:,0], fluxes=data[:,1], sigmas=data[:,2])
    result_fit = {'func': 'CG', 'param_vals': [C, mu1, d1, sigma1]}
    model.fit()
    assert(model.best_fit['func'] == result_fit['func'])
    assert((np.abs(
              model.best_fit['param_vals'][0] - 
              np.array(result_fit['param_vals'])) < 1e-2*np.ones(len(model.best_fit['param_vals'][0]))).all())


def test_cge():
    data = np.loadtxt('../data/cge.csv', delimiter=',')
    model = TwoG(phases=data[:,0], fluxes=data[:,1], sigmas=data[:,2])
    result_fit = {'func': 'CGE', 'param_vals': [C, mu1, d1, sigma1, Aell, phi01]}
    model.fit()
    assert(model.best_fit['func'] == result_fit['func'])
    assert((np.abs(
              model.best_fit['param_vals'][0][:-1] - 
              np.array(result_fit['param_vals'][:-1])) < 1e-2*np.ones(len(model.best_fit['param_vals'][0][:-1]))).all())
    
    assert((np.abs(
              model.best_fit['param_vals'][0][-1] - 
              np.array(result_fit['param_vals'][-1])) < 1e-2) | (np.abs(
              model.best_fit['param_vals'][0][-1] - 0.5 -
              np.array(result_fit['param_vals'][-1])) < 1e-2) | (np.abs(
              model.best_fit['param_vals'][0][-1] + 0.5 -
              np.array(result_fit['param_vals'][-1])) < 1e-2))

def test_cg12():
    data = np.loadtxt('../data/cg12.csv', delimiter=',')
    model = TwoG(phases=data[:,0], fluxes=data[:,1], sigmas=data[:,2])
    result_fit = {'func': 'CG12', 'param_vals': [C, mu1, d1, sigma1, mu2, d2, sigma2]}
    model.fit()
    assert(model.best_fit['func'] == result_fit['func'])
    assert((np.abs(
              model.best_fit['param_vals'][0] - 
              np.array(result_fit['param_vals'])) < 1e-2*np.ones(len(model.best_fit['param_vals'][0]))).all())


def test_cg12e1():
    data = np.loadtxt('../data/cg12e1.csv', delimiter=',')
    model = TwoG(phases=data[:,0], fluxes=data[:,1], sigmas=data[:,2])
    result_fit = {'func': 'CG12E', 'param_vals': [C, mu1, d1, sigma1, mu2, d2, sigma2, Aell, phi01]}
    model.fit()
    assert(model.best_fit['func'] == result_fit['func'])
    assert((np.abs(
              model.best_fit['param_vals'][0][:-1] - 
              np.array(result_fit['param_vals'][:-1])) < 1e-2*np.ones(len(model.best_fit['param_vals'][0][:-1]))).all())

    assert((np.abs(
            model.best_fit['param_vals'][0][-1] - 
            np.array(result_fit['param_vals'][-1])) < 1e-2) | (np.abs(
            model.best_fit['param_vals'][0][-1] - 0.5 -
            np.array(result_fit['param_vals'][-1])) < 1e-2) | (np.abs(
            model.best_fit['param_vals'][0][-1] + 0.5 -
            np.array(result_fit['param_vals'][-1])) < 1e-2))
    


def test_cg12e2():
    data = np.loadtxt('../data/cg12e2.csv', delimiter=',')
    model = TwoG(phases=data[:,0], fluxes=data[:,1], sigmas=data[:,2])
    result_fit = {'func': 'CG12E', 'param_vals': [C, mu1, d1, sigma1, mu2, d2, sigma2, Aell, phi02]}
    model.fit()
    assert(model.best_fit['func'] == result_fit['func'])
    assert((np.abs(
              model.best_fit['param_vals'][0][:-1] - 
              np.array(result_fit['param_vals'][:-1])) < 1e-2*np.ones(len(model.best_fit['param_vals'][0][:-1]))).all())

    assert((np.abs(
            model.best_fit['param_vals'][0][-1] - 
            np.array(result_fit['param_vals'][-1])) < 1e-2) | (np.abs(
            model.best_fit['param_vals'][0][-1] - 0.5 -
            np.array(result_fit['param_vals'][-1])) < 1e-2) | (np.abs(
            model.best_fit['param_vals'][0][-1] + 0.5 -
            np.array(result_fit['param_vals'][-1])) < 1e-2))


    
# def test_compute_ecl_params(model, result):
    
#     eb_dict = model.compute_eclipse_params()
#     for key in eb_dict.keys():
#         if key in result.keys():
#             assert(np.abs(eb_dict[key] - result[key]) < 5e-2)
#         elif key != 'eclipse_edges':
#             assert(np.isnan(eb_dict[key]))
#         else:
#             pass
        

# test_compute_ecl_params(model_cg, {'primary_width': 5.6*sigma1,
#                               'primary_position': mu1,
#                               'primary_depth': d1})

# test_compute_ecl_params(model_cge, {'primary_width': 5.6*sigma1,
#                           'primary_position': mu1,
#                           'primary_depth': d1})

# test_compute_ecl_params(model_cg12, {'primary_width': 5.6*sigma1,
#                               'secondary_width': 5.6*sigma2,
#                               'primary_position': mu1,
#                               'secondary_position': mu2,
#                               'primary_depth': d1,
#                               'secondary_depth': d2})
# test_compute_ecl_params(model_cg12e, {'primary_width': 5.6*sigma1,
#                           'secondary_width': 5.6*sigma2,
#                           'primary_position': mu1,
#                           'secondary_position': mu2,
#                           'primary_depth': d1,
#                           'secondary_depth': d2})
    
    
    
    
    
    
    
    