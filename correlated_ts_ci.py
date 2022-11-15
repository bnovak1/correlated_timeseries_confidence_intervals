"""
Description
----
Compute standard errors & confidence intervals as a function of block length,
fit to a function, and extrapolate to infinite number of blocks.

Reference
----
Appendix from: Hess, B., Determining the shear viscosity of model liquids from molecular
dynamics simulations. J. Chem. Phys. 2002, 116 (1), 209-217.
"""


import argparse
import json
import os

from asteval import Interpreter
import numpy as np
import scipy.optimize as opt
from scipy import stats
from arch.bootstrap import StationaryBootstrap
from joblib import Parallel, delayed
from lmfit import Parameters, minimize

aeval = Interpreter()


class ConfidenceInterval:
    """
    Description
    ----
    Compute standard errors & confidence intervals as a function of block length,
    fit to a function, and extrapolate to infinite number of blocks.
    """

    def __init__(self, data_file, colnum, input_file):
        """
        Inputs
        ----
        :data_file: Required. File with time in first column and variable to analyze in the second
                    column. The time step of the data is assumed to be constant.
        :colnum int: Required. Number of column in file with quantity to analyze with column
                     numbers starting from 0.
        :input_file: Required. Name of JSON file containing input parameters.
        """

        self.data_file = data_file
        self.colnum = colnum
        self.input_file = input_file

        self.inputs = {}
        self.random_state = None

        self.read_input_file()

        self.data = None

    def __call__(self):
        """
        Description
        ----
        Compute standard errors & confidence intervals as a function of block length,
        fit to a function, and extrapolate to infinite number of blocks.
        """

        time_unit = self.inputs["time_unit"]

        np.random.RandomState(self.inputs["seed"])
        np.random.seed(self.inputs["seed"])

        if self.inputs["outfile_prefix"] is None:
            # prefix for output files from data_file
            self.inputs["outfile_prefix"] = (
                ".".join(os.path.split(self.data_file)[1].split(".")[:-1]) + "_"
            )

        # Add on outdir
        outfile_path = os.path.join(self.inputs["outdir"], self.inputs["outfile_prefix"])

        # open log file
        logfile = open(outfile_path + "log.txt", encoding="utf-8", mode="w")

        # get data
        (time, self.data) = self.read_data()
        data_len = len(self.data)

        # total time & time step
        time_step = time[1] - time[0]
        time_total = time[-1] - time[0]
        logfile.write("Time step: " + str(time_step) + " " + time_unit + "\n")
        logfile.write("Total time: " + str(time_total) + " " + time_unit + "\n\n")

        # number of blocks
        nblocks = np.array(range(self.inputs["min_blocks"], data_len + 1), dtype=int)

        # block sizes in number of points
        block_sizes = (data_len / nblocks).astype(int)
        block_sizes = np.unique(block_sizes)
        nskip = max(int(np.floor(len(block_sizes) / self.inputs["block_size_number"])), 1)
        block_sizes = np.unique(np.hstack((block_sizes[::nskip], block_sizes[-21:])))
        nblocks = (data_len / block_sizes).astype(int)

        # block sizes in time units
        block_lengths = time_step * block_sizes

        # stationary bootstrap for different block sizes
        result = np.array(
            Parallel(n_jobs=self.inputs["nprocs"])(
                delayed(self._bootstrap)(block_size) for block_size in block_sizes
            )
        )

        nfunc = int(result.shape[1] / 2)
        mean_unc = result[:, :nfunc]
        standard_err = result[:, nfunc:]

        # standard deviation, mean for all data
        sigma_data = standard_err[0, :] * np.sqrt(data_len)
        mean_data = np.mean(mean_unc, axis=0)

        logfile.write("Mean of data: " + str(mean_data) + "\n")
        logfile.write("Standard deviation of data: " + str(sigma_data) + "\n\n")

        # fitting to standard error
        unc_factor = stats.t.isf(self.inputs["sig_level"] / 2.0, df=self.inputs["nbootstrap"] - 1)
        wghts = np.ones(len(block_lengths))  # All weights equal
        params = Parameters()

        for ifunc in range(nfunc):

            prefactor_ini = 2.0 * sigma_data[ifunc] ** 2.0 / time_total
            if self.inputs["vary_prefactor"]:
                params.add("prefactor", value=prefactor_ini)
            else:
                params.add("prefactor", value=prefactor_ini, vary=False)

            fit_cnt = 0
            while True:

                tau1_ini = np.random.rand() * block_lengths[-1]
                alpha_max = np.mean(standard_err[-3:]) ** 2.0 / (tau1_ini * prefactor_ini)
                alpha_ini = np.random.rand() * min(alpha_max, 1.0)
                term1 = np.mean(standard_err[-3:]) ** 2.0 / prefactor_ini
                tau2_ini = (term1 - alpha_ini * tau1_ini) / (1.0 - alpha_ini)

                params.add("alpha", value=alpha_ini, min=0.0, max=1.0)
                params.add("tau1", value=tau1_ini)
                params.add("tau2", value=tau2_ini)

                try:

                    fit = minimize(
                        residual,
                        params,
                        args=(block_lengths, standard_err[:, ifunc], wghts),
                        method="nelder",
                    )
                    prefactor = fit.params["prefactor"].value
                    alpha = fit.params["alpha"].value
                    tau1 = fit.params["tau1"].value
                    tau2 = fit.params["tau2"].value
                    se_extrap = extrap(prefactor, alpha, tau1, tau2)
                    unc_extrap = unc_factor * se_extrap

                    break

                except ValueError:
                    fit_cnt += 1

                if fit_cnt > 1000:
                    tau1 = -1
                    tau2 = -1
                    break

            # fit with only one term (set alpha = 1) if tau1 or tau2 are negative or
            # greater than time_total
            if tau1 < 0 or tau2 < 0 or tau1 > time_total or tau2 > time_total:

                if tau1 > time_total or tau2 > time_total:
                    logfile.write(
                        "WARNING: Time constant > total time, \
                                   possibly insufficient data\n\n"
                    )

                params["alpha"].set(value=1.0, vary=False)

                fit_cnt = 0
                while True:

                    tau1_ini = np.random.rand() * block_lengths[-1]
                    params["tau1"].set(value=tau1_ini)

                    try:

                        fit = minimize(
                            residual,
                            params,
                            args=(block_lengths, standard_err[:, ifunc], wghts),
                            method="nelder",
                        )
                        prefactor = fit.params["prefactor"].value
                        tau1 = fit.params["tau1"].value
                        se_extrap = extrap(prefactor, 1.0, tau1, 0.0)
                        unc_extrap = unc_factor * se_extrap

                        break

                    except ValueError:
                        fit_cnt += 1

                    if fit_cnt > 1000:

                        outdata = np.column_stack((block_lengths, standard_err[:, ifunc]))

                        np.savetxt(
                            outfile_path + "block_error_fit_" + str(ifunc) + ".dat",
                            outdata,
                            header="Time (" + time_unit + "), Standard error",
                        )

                        se_extrap = np.mean(standard_err[-3:])
                        unc_extrap = unc_factor * se_extrap
                        outdata = np.array([se_extrap, unc_extrap]).reshape(1, -1)
                        np.savetxt(
                            outfile_path + "block_error_extrapolation_" + str(ifunc) + ".dat",
                            outdata,
                            header="Failed fit, use means of SE, unc values for 3 "
                            + "points with longest block lengths",
                        )

                        continue

            # Number of effective samples and correlation time where standard error has reached 99%
            # of its limitng value
            se_target = 0.99 * se_extrap
            if se_target > standard_err[-1, ifunc]:
                t_0 = block_lengths[-1]
            else:
                ind = np.argmin(np.abs(se_target - standard_err[:, ifunc]))
                t_0 = block_lengths[ind]
            min_result = opt.minimize(
                se_diff,
                t_0,
                args=(
                    se_target,
                    (
                        fit.params["prefactor"],
                        fit.params["alpha"],
                        fit.params["tau1"],
                        fit.params["tau2"],
                    ),
                ),
                method="Nelder-Mead",
            )
            t_corr = min_result.x[0]
            n_eff = self.inputs["min_blocks"] * block_lengths[-1] / t_corr

            # Save residuals vs. number of blocks to file
            outdata = np.column_stack((nblocks, fit.residual))
            np.savetxt(
                outfile_path + "residuals_" + str(ifunc) + ".dat",
                outdata,
                header="No. of blocks | Residuals of standard error | "
                + "Residuals of uncertainty",
            )

            # save fit
            outdata = np.column_stack(
                (block_lengths, standard_err[:, ifunc], standard_err[:, ifunc] - fit.residual)
            )
            np.savetxt(
                outfile_path + "block_error_fit_" + str(ifunc) + ".dat",
                outdata,
                header="Time (" + time_unit + "), Standard error, Fit to standard error",
            )

            # Extrapolate to infinite time, save mean, se, uncertainty, parameters
            logfile.write("Extrapolated standard error: " + str(se_extrap) + "\n\n")

            outdata = np.column_stack(
                (
                    mean_data[ifunc],
                    se_extrap,
                    unc_extrap,
                    fit.params["prefactor"],
                    fit.params["alpha"],
                    fit.params["tau1"],
                    fit.params["tau2"],
                    t_corr,
                    n_eff,
                )
            )
            np.savetxt(
                outfile_path + "block_error_extrapolation_" + str(ifunc) + ".dat",
                outdata,
                header="Mean, Standard error, uncertainty, prefactor, alpha, "
                + "tau1 ("
                + time_unit
                + "), tau2("
                + time_unit
                + "), correlation time ("
                + time_unit
                + "), "
                + "effective number of samples",
            )

            logfile.close()

    def read_input_file(self):
        """
        Description
        ----
        Read inputs from  JSON file. Initialize parameters to defaults if not in the input file.

        Keys in input_file
        ----
        :outfile_prefix: Prefix for ouput file names. Defaults to data_file w/o extension + '_'.
        :indir: Path to data_file and input_file. Defaults to current directory.
        :outdir: Output directory. Defaults to current directory.
        :time_unit: Units of time in data_file. Default = 'ps'
        :eqtime float: Equilibration time in time_unit. Defaults to 0.0.
        :skip: Frequency to keep data. Default = 1, keep all data.
        :seed int: Optional. Seed for random number generator. Default = 42.
        :nbootstrap int: Number of bootstrap samples. Default = 100.
        :custom_function: Custom lambda function taking a single argument. This function \
            contains the definition of the quantities which you wish to obtain the \
            uncertainties for and should return a single value or a numpy row vector. \
            Example -- lambda x: np.hstack((np.mean(x), np.percentile(x, 90))). Defaults to np.mean.
        :vary_prefactor bool: Vary the prefactor instead of constraining it to a constant value \
            of 2 times the standard deviation of all data divided by the total time covered by \
            the data. Defaults to False.
        :sig_level float: Significance level for confidence intervals. Default = 0.05.
        :min_blocks int: Minimum number of blocks. Default = 30.
        :block_size_number int: Number of block sizes to consider. Default is 100.
        :nprocs int: Number of processboes. Default = 1.
        """

        # Default values
        defaults = {
            "indir": ".",
            "outfile_prefix": None,
            "outdir": ".",
            "time_unit": "ps",
            "eqtime": 0.0,
            "skip": 1,
            "seed": 42,
            "nbootstrap": 100,
            "custom_function": None,
            "vary_prefactor": False,
            "sig_level": 0.05,
            "block_size_number": 100,
            "min_blocks": 30,
            "nprocs": 1,
        }

        # Read inputs from input_file.
        with open(self.input_file, encoding="utf-8", mode="r") as json_file:
            inputs = json.load(json_file)

        # Assign to values from input_file or defaults
        for key, value in defaults.items():
            if key in inputs:
                self.inputs[key] = inputs[key]
            else:
                self.inputs[key] = value

        self.random_state = np.random.RandomState(self.inputs["seed"])

    def read_data(self):
        """
        Description
        ----
        Read in data, remove equilibration time, and split into time and data vectors.
        """

        # read file: try numpy binary format, otherwise assume text (possibly xvg)
        try:
            data = np.load(self.inputs["indir"] + "/" + self.data_file)
        except (OSError, ValueError):
            data = self._read_xvg()

        # throw away equilibration time
        data = data[data[:, 0] >= self.inputs["eqtime"], :]

        # skip some data if skip > 1
        data = data[:: self.inputs["skip"], :]

        # split columns into time and data variables
        time = data[:, 0]
        data = data[:, self.colnum]

        return (time, data)

    def _read_xvg(self):
        """
        Description
        ----
        Read xvg file or any data file with no header or a header containing comment
        characters "#" or "@", discard header.
        """

        infile = self.inputs["indir"] + "/" + self.data_file

        with open(infile, encoding="utf-8", mode="r") as fid:

            cnt = 0
            nskiprows = 0
            while nskiprows == 0:
                data = fid.readline()
                if data[0] != "#" and data[0] != "@":
                    nskiprows = cnt
                cnt = cnt + 1

        return np.loadtxt(infile, skiprows=nskiprows)

    def _bootstrap(self, block_size):
        """
        Description
        ----
        Stationary bootstrap for different block sizes

        Inputs
        ----
        :block_size: Number of points per block

        Outputs
        ----
        List containing mean, standard error
        """

        # Update random state. If the same random state is used for different block sizes,
        # the resulting curve is too smooth.
        seed = np.random.randint(0, np.iinfo(np.int32).max)
        self.random_state = np.random.RandomState(seed)
        np.random.RandomState(seed)

        # Bootstrap setup. Set random_state for reproducibility of bootstrap sampling
        bs_func = StationaryBootstrap(block_size, self.data, random_state=self.random_state)

        try:
            bs_results = eval(
                "bs_func.apply(" + self.inputs["custom_function"] + ', self.inputs["nbootstrap"])'
            )
        except TypeError:
            bs_results = bs_func.apply(np.mean, self.inputs["nbootstrap"])

        mean_unc = np.mean(bs_results, axis=0)
        standard_error = np.std(bs_results, ddof=1, axis=0)

        return list(np.hstack((mean_unc, standard_error)))


def residual(params, time, standard_err, wghts):
    """
    Description
    ----
    Compute residuals for fit.

    Inputs
    ----
    :params: Parameters for the model
    :se: Standard error as a function of block length

    Outputs
    ----
    :residuals: Residuals
    """

    prefactor = params["prefactor"].value
    alpha = params["alpha"].value
    tau1 = params["tau1"].value
    tau2 = params["tau2"].value

    model = se_func(time, prefactor, alpha, tau1, tau2)

    residuals = (standard_err - model) * wghts

    return residuals


def se_func(time, prefactor, alpha, tau1, tau2):
    """
    Description
    ----
    Function for fitting standard error as a function of block length.

    Inputs
    ----
    :prefactor: Prefactor.
    :alpha: Fraction for tau1 term.
    :tau1: Time constant for first term.
    :tau2: Time constant for second term.

    Outputs
    ----
    Value of fit for various block lengths.
    """

    term1 = alpha * tau1 * (1.0 + (tau1 / time) * (np.exp(-time / tau1) - 1.0))
    term2 = (1.0 - alpha) * tau2 * (1.0 + (tau2 / time) * (np.exp(-time / tau2) - 1.0))
    return np.sqrt(prefactor * (term1 + term2))


def extrap(prefactor, alpha, tau1, tau2):
    """
    Description
    ----
    Extrapolate fitted function to infinite number of blocks

    Inputs
    ----
    :prefactor: Prefactor.
    :alpha: Fraction for tau1 term.
    :tau1: Time constant for first term.
    :tau2: Time constant for second term.

    Outputs
    ----
    Value at infinity
    """

    return np.sqrt(prefactor * (alpha * tau1 + (1.0 - alpha) * tau2))

    # def _se_func_ssd(params, prefactor, t, se):
    #     alpha = params[0]
    #     tau1 = params[1]
    #     tau2 = params[2]
    #     return np.sum((self._se_func(prefactor, alpha, tau1, t, tau2) - se)**2.0)


def se_diff(time, se_target, params):
    """
    Description
    ----
    Difference squared between the fitted function and a target standard error.
    Used to solve for block length at the target standard error.

    Inputs
    ----
    :time: Block length in time units.
    :se_target: Target standard error.
    :params: Tuple/list containing parameters for the model: prefactor,
             alpha (fraction for the tau1 term), tau1 (time constant for the first term.
             tau2 (time constant for the second term).

    Outputs
    ----
    Difference squared between the fitted function at t and se_target.
    """

    prefactor, alpha, tau1, tau2 = params

    standard_err = np.sqrt(
        prefactor
        * (
            alpha * tau1 * (1.0 + (tau1 / time) * (np.exp(-time / tau1) - 1.0))
            + (1.0 - alpha) * tau2 * (1.0 + (tau2 / time) * (np.exp(-time / tau2) - 1.0))
        )
    )
    return (standard_err - se_target) ** 2.0


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser()

    PARSER.add_argument(
        "data_file",
        help="File with time in the first column and other quantities in \
                            subsequent columns.",
    )
    PARSER.add_argument(
        "colnum",
        type=int,
        help="Column number in the file with the quantity to be analyzed. \
                            The first column is numbered 0.",
    )
    PARSER.add_argument("input_file", help="JSON file with input parameters.")

    ARGS = PARSER.parse_args()

    get_confidence_interval = ConfidenceInterval(ARGS.data_file, ARGS.colnum, ARGS.input_file)
    get_confidence_interval()
