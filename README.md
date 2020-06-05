# Correlated timeseries confidence intervals

## Purpose

Estimate confidence intervals in means of correlated time series with a small number of effective samples (like molecular dynamics simulations). If your time series is long enough that the standard error levels off completely as a function of block length, then this method is overkill and simply using a block bootstrap sampling with a sufficiently large block length is probably sufficient.

## Background

The origin of this method is in the Appendix of [1]. It based on computing standard error as a function of block length, fitting this, then extrapolating to infinite block length. For correlated data, the standard error will increase asymptotically with increasing block length. Some improvements on the original method are implemented here. The first is to give the option to vary the prefactor for the fitted function which can sometimes give a significantly better fit. The second is to give the option to perform stationary block bootstrap sampling for each block size instead of just using a single set of blocks for each block size. This significantly reduces the noise in the data and leads to better fits.

See here for more details.

## Python package requirements

* arch
* joblib
* lmfit
* numpy
* scipy

## Usage

usage:  
confidence_interval.py [-h] [-op OUTPREFIX] [-id INDIR] [-od OUTDIR] [-tu TIME_UNIT] [-eq EQTIME] [-sk SKIP] [-vp] [-sl SIG_LEVEL] [-mb MIN_BLOCKS] [-bsn BLOCK_SIZE_NUMBER] [-cf CUSTOM_FUNC] [-nb NBOOTSTRAP] [-np NPROCS] infile colnum  

positional arguments:
* infile
  * File with time in the first column and other quantities in subsequent columns.
* colnum
  * Column number in the file with the quantity to be analyzed. The first column is numbered 0.

optional arguments:  
* -h, --help
  * show this help message and exit
* -op OUTPREFIX, --outprefix OUTPREFIX
  * Prefix for output files. Default is the prefix of the input file.
* -id INDIR, --indir INDIR 
  * Directory input file is located in. Default is current directory.
* -od OUTDIR, --outdir OUTDIR 
  * Directory to write data to. Default is current directory.
* -tu TIME_UNIT, --time_unit TIME_UNIT 
  * String to specify time units. 'ns', 'ps', etc. Default is 'ps'.
* -eq EQTIME, --eqtime EQTIME
  * Equilibration time in unit of input file. Default is 0.0.
* -sk SKIP, --skip SKIP 
  * Only use every this many data points from the input file.
* -vp, --vary_prefac   
  * Vary the prefactor instead of constraining it to a constant value of 2 times the standard deviation of all data divided by the total time covered by the data. This is a flag.
* -sl SIG_LEVEL, --sig_level SIG_LEVEL
  * Significance level for computing confidence intervals. Default is 0.05.
* -mb MIN_BLOCKS, --min_blocks MIN_BLOCKS
  * Minimum number of blocks. Default is 30.
* -bsn BLOCK_SIZE_NUMBER, --block_size_number BLOCK_SIZE_NUMBER
  * Number of block sizes to consider. Default is 100.
* -cf CUSTOM_FUNC, --custom_func CUSTOM_FUNC
  * Custom lambda function taking a single argument. This function contains the definition of the quantities which you wish to obtain the uncertainties for and should return a single value or a numpy row vector. Example -- lambda x: np.hstack((np.mean(x), np.percentile(x, 90))). If not specified, only np.mean is used.
* -nb NBOOTSTRAP, --nbootstrap NBOOTSTRAP
  * Number of bootstrap samples. Default is 100.
* -np NPROCS, --nprocs NPROCS
  * Number of processors to use for calculation. Default is all available.

### Example

Analyze column 1 of the specified file. Use nanoseconds (ns) for the time unit with an equilibration time of 0.5 ns. All other options default.

```shell
python confidence_interval.py ./velocities/ads_lower_all_velocity.xvg 1 -tu ns -eq 0.5
```

## References

(1) Hess, B. Determining the Shear Viscosity of Model Liquids from Molecular Dynamics Simulations. J. Chem. Phys. 2002, 116, 209–217. https://doi.org/10.1063/1.1421362.
