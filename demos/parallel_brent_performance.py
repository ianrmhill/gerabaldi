# Copyright (c) 2025 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import timeit
from functools import partial
import numpy as np
from scipy.optimize import minimize_scalar as scipy_min

import os
import sys

# Welcome to the worst parts of Python! This line adds the parent directory of this file to module search path, from
# which the Gerabaldi module can be seen and then imported. Without this line the script cannot find the module without
# installing it as a package from pip (which is undesirable because you would have to rebuild the package every time
# you changed part of the code).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gerabaldi.math.minimize import minimize as custom_min


def eval_perf():
    """
    Performance evaluations of the parallel Brent's method implementation.
    """

    ########################################################################
    ### 1. Single sample performance
    ########################################################################
    def simple_func(x, c):
        return c*((x - 4)**2) + 5*x - 3*c

    sequential_min = partial(scipy_min, fun=simple_func, method='brent', args=(4,))
    parallel_min = partial(custom_min, func=simple_func, extra_args={'c': np.array(4,)}, bounds=(-100.0, 100.0))
    print(f"Seq min: {sequential_min()}, parallel min: {parallel_min()}")

    seq_best_perf = min(timeit.Timer(sequential_min).repeat(repeat=10, number=10))
    par_best_perf = min(timeit.Timer(parallel_min).repeat(repeat=10, number=10))
    print(f"Sequential time: {seq_best_perf}, parallel time: {par_best_perf}")

    ########################################################################
    ### 1. Single sample logarithmic performance
    ########################################################################
    def log_func(x, c):
        return np.abs(x - np.exp(c) + c)
    sequential_min = partial(scipy_min, fun=log_func, method='brent', args=(15,))
    parallel_min = partial(custom_min, func=log_func, extra_args={'c': np.array(15,)}, bounds=(1e-3, 1e10), maxiter=50, log_gold=True)
    print(f"Seq min: {sequential_min()}, parallel min: {parallel_min()}")

    seq_best_perf = min(timeit.Timer(sequential_min).repeat(repeat=10, number=10))
    par_best_perf = min(timeit.Timer(parallel_min).repeat(repeat=10, number=10))
    print(f"Sequential time: {seq_best_perf}, parallel time: {par_best_perf}")

    ########################################################################
    ### 2. Array performance
    ########################################################################
    def simple_func(x, c):
        return np.abs(x - c)

    for count in [11, 51, 101]:
        c_arr = np.linspace(-25, 25, count)
        def sequential_min():
            res = np.zeros(count)
            for i, c in enumerate(c_arr):
                res[i] = scipy_min(simple_func, method='brent', args=(c,)).x
            return res
        parallel_min = partial(custom_min, func=simple_func, extra_args={'c': c_arr}, bounds=(-100.0, 100.0), maxiter=500)

        seq_best_perf = min(timeit.Timer(sequential_min).repeat(repeat=10, number=10))
        par_best_perf = min(timeit.Timer(parallel_min).repeat(repeat=10, number=10))
        print(f"For count {count}, sequential time: {seq_best_perf}, parallel time: {par_best_perf}")

    ########################################################################
    ### 3. Direct logarithmic comparison
    ########################################################################
    def log_func(x, c):
        return np.abs(x - np.exp(c) + c)

    count = 100
    c_arr = np.linspace(-3, 16, count)
    lin_min = partial(custom_min, func=log_func, extra_args={'c': c_arr}, bounds=(1e-3, 1e10), maxiter=500)
    log_min = partial(custom_min, func=log_func, extra_args={'c': c_arr}, bounds=(1e-3, 1e10), maxiter=500, log_gold=True)
    print(f"Lin min: {lin_min()}, log min: {log_min()}")

    lin_best_perf = min(timeit.Timer(lin_min).repeat(repeat=10, number=10))
    log_best_perf = min(timeit.Timer(log_min).repeat(repeat=10, number=10))
    print(f"For count {count}, lin time: {lin_best_perf}, log time: {log_best_perf}")


if __name__ == '__main__':
    eval_perf()
