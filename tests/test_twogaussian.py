from ligeor import TwoGaussianModel as TwoG
import numpy as np


def test_initialize_filename(filename, data):
    model = TwoG(filename=filename, n_downsample=1, delimiter=',')
    assert(((model.phases == data[:,0]) & (model.fluxes == data[:,1]) & (model.sigmas == data[:,2])).all())

    
def test_initialize_data(data):
    model = TwoG(phases=data[:,0], fluxes=data[:,1], sigmas=data[:,2])
    assert(((model.phases == data[:,0]) & (model.fluxes == data[:,1]) & (model.sigmas == data[:,2])).all())
    return model


def test_estimate_ecl_pos_widths(model, result):
    est_positions = model.estimate_eclipse_positions_widths(model.phases, model.fluxes)
    estimates = {}
    estimates['pos1'] = est_positions['ecl_positions'][0]
    estimates['pos2'] = est_positions['ecl_positions'][1]
    estimates['width1'] = est_positions['ecl_widths'][0]
    estimates['width2'] = est_positions['ecl_widths'][1]
    
    for key in result.keys():
        assert(np.abs(estimates[key] - result[key]) < 2e-1)
    

def test_fit(model, result):
    model.fit()
    assert(model.best_fit['func'] == result['func'])
    assert((np.abs(
              model.best_fit['param_vals'][0] - 
              np.array(result['param_vals'])) < 1e-2*np.ones(len(model.best_fit['param_vals'][0]))).all())
    return model
    
    
def test_compute_ecl_params(model, result):
    
    eb_dict = model.compute_eclipse_params()
    for key in eb_dict.keys():
        if key in result.keys():
            assert(np.abs(eb_dict[key] - result[key]) < 5e-2)
        elif key != 'eclipse_edges':
            assert(np.isnan(eb_dict[key]))
        else:
            pass
        

# if __name__=='main':
# true values of all models
C = 1.
mu1 = 0.
sigma1 = 0.015
d1 = 0.5
mu2 = 0.42
sigma2 = 0.01
d2 = 0.35
Aell = 0.05

# load data on each synthetic model
data_c = np.loadtxt('../data/const.csv', delimiter=',')
data_cg = np.loadtxt('../data/cg.csv', delimiter=',')
data_ce = np.loadtxt('../data/ce.csv', delimiter=',')
data_cge = np.loadtxt('../data/cge.csv', delimiter=',')
data_cg12 = np.loadtxt('../data/cg12.csv', delimiter=',')
data_cg12e1 = np.loadtxt('../data/cg12e1.csv', delimiter=',')
data_cg12e2 = np.loadtxt('../data/cg12e2.csv', delimiter=',')

# check if file initialization works
test_initialize_filename('../data/cg12.csv', data_cg12)

#create a twoG model for each
# model_c = test_initialize_data(data_c)
model_cg = test_initialize_data(data_cg)
# model_ce = test_initialize_data(data_ce)
model_cge = test_initialize_data(data_cge)
model_cg12 = test_initialize_data(data_cg12)
model_cg12e1 = test_initialize_data(data_cg12e1)
model_cg12e2 = test_initialize_data(data_cg12e2)

# test estimated eclipse positions
test_estimate_ecl_pos_widths(model_cg, {'pos1': 0., 'width1': 0.015})
test_estimate_ecl_pos_widths(model_cge, {'pos1': 0., 'width1': 0.015})
test_estimate_ecl_pos_widths(model_cg12, {'pos1': 0., 'width1': 0.015, 'pos2': 0.42, 'width2': 0.01})

# test fits for all models
# test_fit(model_c, {'func': 'C', 'param_vals': [C]})
test_fit(model_cg, {'func': 'CG', 'param_vals': [C,mu1,d1,sigma1]})
# test_fit(model_ce, {'func': 'CE', 'param_vals': [C, Aell, mu1]})
test_fit(model_cge, {'func': 'CGE', 'param_vals': [C, mu1, d1, sigma1, Aell]})
test_fit(model_cg12, {'func': 'CG12', 'param_vals': [C, mu1, d1, sigma1, mu2, d2, sigma2]})
test_fit(model_cg12e1, {'func': 'CG12E1', 'param_vals': [C, mu1, d1, sigma1, mu2, d2, sigma2, Aell]})
test_fit(model_cg12e2, {'func': 'CG12E2', 'param_vals': [C, mu1, d1, sigma1, mu2, d2, sigma2, Aell]})

# test eclipse parameters for all models
test_compute_ecl_params(model_cg, {'primary_width': 5.6*sigma1,
                              'primary_position': mu1,
                              'primary_depth': d1})

test_compute_ecl_params(model_cge, {'primary_width': 5.6*sigma1,
                          'primary_position': mu1,
                          'primary_depth': d1})

test_compute_ecl_params(model_cg12, {'primary_width': 5.6*sigma1,
                              'secondary_width': 5.6*sigma2,
                              'primary_position': mu1,
                              'secondary_position': mu2,
                              'primary_depth': d1,
                              'secondary_depth': d2})
test_compute_ecl_params(model_cg12e1, {'primary_width': 5.6*sigma1,
                          'secondary_width': 5.6*sigma2,
                          'primary_position': mu1,
                          'secondary_position': mu2,
                          'primary_depth': d1,
                          'secondary_depth': d2})

test_compute_ecl_params(model_cg12e2, {'primary_width': 5.6*sigma1,
                          'secondary_width': 5.6*sigma2,
                          'primary_position': mu1,
                          'secondary_position': mu2,
                          'primary_depth': d1,
                          'secondary_depth': d2})
    
    
    
    
    
    
    
    