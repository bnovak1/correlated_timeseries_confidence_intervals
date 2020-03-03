# correlated_timeseries_confidence_intervals

## Purpose

Estimate confidence intervals in means of correlated time series with a small number of effective samples (like molecular dynamics simulations).

## Background

The origin of this method is in the Appendix of [1]. It based on computing standard error as a function of block length, fitting this, then extrapolating to infinite block length. For correlated data, the standard error will increase with increasing block length until the blocks are longer than the correlation time. Some improvements on the original method are implemented here. The first is to give the option to vary the prefactor for the fitted function which can sometimes give a better fit. The second is to give the option to perform stationary block bootstrap sampling for each block size instead of just using a single set of blocks for each block size. This significantly reduces the noise in the data and leads to better fits. Third, ...

See here for more details.

## Usage

## References

(1) Hess, B. Determining the Shear Viscosity of Model Liquids from Molecular Dynamics Simulations. J. Chem. Phys. 2002, 116, 209–217. https://doi.org/10.1063/1.1421362.
