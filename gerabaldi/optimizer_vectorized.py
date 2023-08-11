import numpy as np


# SciPy's bounded scalar minimization function. I'm already using this code but I need it vectorized!
def minimize_scalar_vectorized(func, bounds, args=(), xatol=1e-5, maxiter=500, disp=0,
                               **unknown_options):
    """

    Parameters
    ----------
    func
    bounds
    args
    xatol
    maxiter
    disp: int, optional
        If non-zero, print messages.
            0 : no message printing.
            1 : non-convergence notification messages only.
            2 : print a message on convergence too.
            3 : print iteration results.p
    unknown_options

    Returns
    -------

    """
    # Test bounds are of correct form
    if len(bounds) != 2:
        raise ValueError('Bounds must have two elements.')
    x1, x2 = bounds
    if x1 > x2:
        raise ValueError("The lower bound exceeds the upper bound.")

    return result