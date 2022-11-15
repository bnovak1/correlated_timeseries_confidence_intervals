'''
Tests for correlated_ts_ci
'''

import glob
import os
import numpy as np

import sys
sys.path.append('.')
from correlated_ts_ci import ConfidenceInterval


def test_col1_no_vary_prefactor():
    '''
    Use column 1 in test.dat, don't vary prefactor
    '''

    # Directory to save output to
    outdir = os.path.join('tests', 'no_vary_prefactor_col_1')

    # Run the confidence interval
    conf_int = ConfidenceInterval(os.path.join('tests', 'test.dat'), 1,
                                  os.path.join('tests', 'test_col1_no_vary_prefactor.json'))
    conf_int()

    # Load files from test and compare to expected
    truth = np.loadtxt(os.path.join(outdir, 'expected_block_error_extrapolation_0.dat'))
    test = np.loadtxt(os.path.join(outdir, 'test_block_error_extrapolation_0.dat'))
    assert np.allclose(truth/test, np.ones(truth.shape))

    truth = np.loadtxt(os.path.join(outdir, 'expected_block_error_fit_0.dat'))
    test = np.loadtxt(os.path.join(outdir, 'test_block_error_fit_0.dat'))
    assert np.allclose(truth/test, np.ones(truth.shape))

    truth = np.loadtxt(os.path.join(outdir, 'expected_residuals_0.dat'))
    test = np.loadtxt(os.path.join(outdir, 'test_residuals_0.dat'))
    assert np.allclose(truth/test, np.ones(truth.shape))

    # Remove test files
    cleanup(outdir)

def test_col1_vary_prefactor():
    '''
    Use column 1 in test.dat, vary prefactor
    '''

    # Directory to save output to
    outdir = os.path.join('tests', 'vary_prefactor_col_1')

    # Run the confidence interval
    conf_int = ConfidenceInterval(os.path.join('tests', 'test.dat'), 1, 
                                  os.path.join('tests', 'test_col1_vary_prefactor.json'))
    conf_int()

    # Load file from test and compare to expected
    truth = np.loadtxt(os.path.join(outdir, 'expected_block_error_extrapolation_0.dat'))
    test = np.loadtxt(os.path.join(outdir, 'test_block_error_extrapolation_0.dat'))
    assert np.allclose(truth/test, np.ones(truth.shape))

    truth = np.loadtxt(os.path.join(outdir, 'expected_block_error_fit_0.dat'))
    test = np.loadtxt(os.path.join(outdir, 'test_block_error_fit_0.dat'))
    assert np.allclose(truth/test, np.ones(truth.shape))

    truth = np.loadtxt(os.path.join(outdir, 'expected_residuals_0.dat'))
    test = np.loadtxt(os.path.join(outdir, 'test_residuals_0.dat'))
    assert np.allclose(truth/test, np.ones(truth.shape))

    # Remove test files
    cleanup(outdir)

def cleanup(outdir):
    '''
    Remove test files
    '''

    files = glob.glob(os.path.join(outdir, 'test*'))
    for file in files:
        os.remove(file)
