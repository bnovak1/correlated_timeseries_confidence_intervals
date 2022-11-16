# Correlated timeseries confidence intervals

## Purpose

Estimate confidence intervals in means of correlated time series with a small number of effective samples (like molecular dynamics simulations). If your time series is long enough that the standard error levels off completely as a function of block length, then this method is overkill and simply using a block bootstrap sampling with a sufficiently large block length is probably sufficient.

## Background

The origin of this method is in the Appendix of [1]. It based on computing standard error as a function of block length, fitting this, then extrapolating to infinite block length. For correlated data, the standard error will increase asymptotically with increasing block length. Some improvements on the original method are implemented here. The first is to give the option to vary the prefactor for the fitted function which can sometimes give a significantly better fit. The second is to give the option to perform stationary block bootstrap sampling for each block size instead of just using a single set of blocks for each block size. This significantly reduces the noise in the data and leads to better fits.

## Installation

```shell
pip install correlated_ts_ci
```

## Usage

### Command line

```shell
python -m correlated_ts_ci.py [-h] data_file colnum input_file
```

#### positional arguments

* data_file
  * File with time in the first column and other quantities in subsequent columns.
* colnum
  * Column number in the file with the quantity to be analyzed. The first column is numbered 0.
* input_file
  * JSON file with input parameters.
  
#### JSON keys in input_file

* outfile_prefix
  * Prefix for ouput file names. Defaults to data_file w/o extension + '_'.
* indir
  * Path to data_file and input_file. Defaults to current directory.
* outdir
  * Output directory. Defaults to current directory.
* time_unit
  * Units of time in data_file. Default = 'ps'
* eqtime (float)
  * Equilibration time in time_unit. Defaults to 0.0.
* skip (int)
  * Frequency to keep data. Default = 1, keep all data.
* seed (int)
  * Seed for random number generator. Default = 42.
* nbootstrap (int)
  * Number of bootstrap samples. Default = 100.
* custom_function
  * Custom lambda function taking a single argument. This function contains the definition of the quantities which you wish to obtain the uncertainties for and should return a single value or a numpy row vector. Default function is np.mean.
  * Example -- lambda x: np.hstack((np.mean(x), np.percentile(x, 90)))
* vary_prefactor (bool)
  * Vary the prefactor instead of constraining it to a constant value of 2 times the standard deviation of all data divided by the total time covered by the data. Defaults to false.
* sig_level (float)
  * Significance level for confidence intervals. Default = 0.05.
* min_blocks (int)
  * Minimum number of blocks. Default = 30.
* block_size_number (int)
  * Number of block sizes to consider. Default is 100.
* nprocs (int)
  * Number of processes to use. Default = 1.

#### Example

Analyze column 1 of the specified file. Use nanoseconds (ns) for the time unit with an equilibration time of 0.5 ns. All other options default.

```shell
python -m correlated_ts_ci ads_lower_all_velocity.xvg 1 ads_lower_all_velocity.json
```

ads_lower_all_velocity.json:

```JSON
{
    "time_unit": "ns",
    "eqtime": 0.5
}
```

### Script

```python
from correlated_ts_ci import ConfidenceInterval

get_confidence_interval = ConfidenceInterval(data_file, colnum, input_file)
get_confidence_interval()
```

## References

(1) Hess, B. Determining the Shear Viscosity of Model Liquids from Molecular Dynamics Simulations. J. Chem. Phys. 2002, 116, 209–217. https://doi.org/10.1063/1.1421362.
