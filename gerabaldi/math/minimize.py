# Copyright (c) 2024 Ian Hill
# SPDX-License-Identifier: Apache-2.0

"""Minimization algorithm implementation leveraged for the Gerabaldi package."""

import warnings
import numpy as np

from gerabaldi.helpers import logger

# Our default machine precision / machine epsilon is based on a 64-bit computer IEEE floating point representation,
# it is recommended to increase this value for lower precision hardware
MACH_32BIT_EPS = 2 ** -24
MACH_64BIT_EPS = 2 ** -53

# These values are used when segmenting golden search intervals to maximize rate of convergence
# GF1 is approximately 0.618, GF2 is approximately 0.382 (i.e., 1 - GF1)
GOLDEN_RATIO = (1 + (5**0.5)) / 2
GF1 = GOLDEN_RATIO - 1
GF2 = 2 - GOLDEN_RATIO


def minimize(func, extra_args: dict, bounds, precision=1e-5, mach_eps=MACH_64BIT_EPS, maxiter=20, log_gold=False):
    """
    Implementation of Brent minimization with vectorization for parallelization and logarithmic search capability.
    """
    f_args_order = extra_args.keys()
    f_args = list(extra_args.values())
    # We divide by three so that abs_tol directly represents the furthest that the estimated minimum can be from the
    # true minimum
    abs_tol = precision / 3.0
    rel_tol = mach_eps ** 0.5
    # Initialize the bounds a and b within which to find the function minimum, shape determined by first function arg
    shape_to_match = f_args[0]
    a = np.full_like(shape_to_match, np.log10(bounds[0]), dtype=float) if log_gold \
        else np.full_like(shape_to_match, bounds[0], dtype=float)
    b = np.full_like(shape_to_match, np.log10(bounds[1]), dtype=float) if log_gold \
        else np.full_like(shape_to_match, bounds[1], dtype=float)
    # Initialize the parabolic interpolation points
    v = w = x = a + (GF2 * (b - a))
    as_kwargs = {}
    for i, arg in enumerate(f_args_order):
        as_kwargs[arg] = f_args[i]
    fv = fw = fx = func(10 ** x, **as_kwargs) if log_gold else func(x, **as_kwargs)
    # 'e' represents the step size between the old and new minimum estimate from two iterations ago. This bookkeeping
    # is used to force golden section searches if parabolic interpolation is not proceeding faster than a bisection
    # search method. Not strictly necessary for guaranteed convergence, but adds a guarantee that the algorithm is never
    # much slower than a raw golden section search algorithm. Brent found that using 'e' from two iterations ago worked
    # better than using 'd' from one iteration ago, no hard mathematical justification for the choice.
    e = d = np.zeros_like(a)

    # Compute the midpoint of the bounded region, which will be our final estimate once the region is very small
    m = (b + a) / 2.0
    # The function will never be evaluated at two points closer together than the tolerance
    tol = (rel_tol * np.abs(x)) + abs_tol
    # Value used multiple times, do the multiplication only once here to save time
    tol2 = tol * 2.0

    completed_iters = 0
    while (not np.all(np.less(np.abs(x - m), tol2 - (b - m)))) and completed_iters < maxiter:
        # Start by identifying which problems are candidates for a parabolic interpolation step
        use_para = np.greater(np.abs(e), tol)
        xdiff1 = np.where(use_para, x - w, 0.0)
        xdiff2 = np.where(use_para, x - v, 0.0)
        # Calculate for all elements as the diffs will be zeros for golden search problems, adds little computation
        qsub1 = xdiff1 * (fx - fv)
        qsub2 = xdiff2 * (fx - fw)
        q = 2 * (qsub2 - qsub1)

        use_para = use_para & np.not_equal(q, 0.0)
        p = np.where(use_para, (xdiff2 * qsub2) - (xdiff1 * qsub1), 0.0)
        p = np.where(np.greater(q, 0.0), -p, p)
        q = np.abs(q)

        further_boundary = np.where(np.greater_equal(x, m), a, b)
        # Numpy's 'where' has to be greedy thus p / q is always evaluated, 'use_para' is always false if q is 0
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='invalid value encountered in divide')
            step = np.where(use_para, p / q, GF2 * (further_boundary - x))
        u_tent = x + step

        change_to_gold = use_para & (np.less(b, u_tent) | np.greater(a, u_tent) |
                                     np.greater(np.abs(step), (0.5 * np.abs(e))))
        # Determine the golden search step for any problems where parabolic search just failed the final two checks
        step = np.where(change_to_gold, GF2 * (further_boundary - x), step)
        u_tent = np.where(change_to_gold, x + step, u_tent)
        # Ensure all new points 'u' will be at least 'tol' away from 'x' and at least 2 'tol' from the interval bounds
        step = np.where(np.less(u_tent - a, tol2) | np.less(b - u_tent, tol2), np.where(np.less(x, m), tol, -tol), step)
        step = np.where(np.less(np.abs(step), tol), np.where(np.greater(step, 0), tol, -tol), step)
        # Compute the new points
        u = x + step

        fn_kwargs = {}
        for i, arg in enumerate(f_args_order):
            fn_kwargs[arg] = f_args[i]
        fu = func(10 ** u, **fn_kwargs) if log_gold else func(u, **fn_kwargs)

        # Now to update all the variables
        e = d
        d = step
        is_less = np.less_equal(fu, fx)

        a = np.where(is_less & np.greater_equal(u, x), x, np.where(~is_less & np.less(u, x), u, a))
        b = np.where(is_less & np.less(u, x), x, np.where(~is_less & np.greater_equal(u, x), u, b))

        v_cond_1, v_cond_2 = (is_less | np.less_equal(fu, fw) | np.equal(w, x),
                              np.less_equal(fu, fv) | np.equal(v, w) | np.equal(v, x))
        v, fv = np.where(v_cond_1, w, np.where(v_cond_2, u, v)), np.where(v_cond_1, fw, np.where(v_cond_2, fu, fv))

        w_cond = np.less_equal(fu, fw) | np.equal(w, x)
        w, fw = np.where(is_less, x, np.where(w_cond, u, w)), np.where(is_less, fx, np.where(w_cond, fu, fw))

        x, fx = np.where(is_less, u, x), np.where(is_less, fu, fx)

        # Check whether our estimates are all good enough
        m = (b + a) / 2.0
        tol = (rel_tol * np.abs(x)) + abs_tol
        tol2 = tol * 2.0

        completed_iters += 1

    if completed_iters == maxiter:
        logger.warn('Equivalent time numerical method hit max iterations, obtained results may have reduced precision.')
    # Return the value of x from the final iteration which is the minimum point found for f
    return 10 ** x if log_gold else x
