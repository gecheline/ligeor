.. ligeor documentation master file, created by
   sphinx-quickstart on Tue Aug 31 17:33:14 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ligeor's documentation!
==================================

*ligeor* (LIght curve GEOmetRy) fits analytical models to eclipsing binary light curves, 
designed to quickly estimate the geometric properties of the eclipses: positions, widths and depths.
Additionally, you can use the built-in MCMC samplers to refine the ephemerides and provide 
posterior distributions for the periods, t0s and eclipse parameters.


Getting Started
===============
*ligeor* depends on the following packages:
   * numpy
   * scipy
   * emcee
and optionally, to run distribution math:
   * distl

Install *ligeor* from pip
   pip install ligeor

or from source
   python setup.py build
   python setup.py install # --user (if local installation, otherwise global)


Basic Usage
===========

*Fitting a model to a light curve*

   from ligeor import TwoGaussianModel

   # initialize a model from a filename containing phases, fluxes and sigmas
   model = TwoGaussianModel(filename=filename, delimiter=',', usecols=(0,1,2), phase_folded=True)
   # fit the model
   model.fit()
   # plot the model
   model.plot()
   # compute the eclipse parameters
   _ = model.compute_eclipse_params()
   print(model.eclipse_params)

For different use cases, including non-phase folded light curves, see the Examples.

*MCMC for ephemerides refinement and eclipse parameter distributions*

   from ligeor import EmceeSamplerPolyfit 

   sampler = EmceeSamplerPolyfit(filename, period, t0, delimiter=' ', usecols = (0,1,2))
   sampler.initial_fit()
   samples = sampler.run_sampler()
   sampler.compute_results(samples, burnin=1000)




.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/fit_twogaussian
   examples/fit_polyfit
   examples/sample_twogaussian
   examples/sample_polyfit
   examples/combine_results

.. toctree::
   :maxdepth: 1
   :caption: API Docs
   
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
