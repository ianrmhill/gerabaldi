"""
Custom classes for simulation results reporting and state persistence.
"""

from datetime import timedelta
from copy import deepcopy
import numpy as np

__all__ = ['TestSimState']


class ArrayTree:
    """
    Class to try to speed up the whole copy and modify process for Gerabaldi state data
    """
    value: np.array


class TestSimState:
    """
    Class to fully track the physical state of a reliability test, such as last executed step, degradation values of
    different parameters, and effective accumulated stress.
    """

    def __init__(self, init_vals: dict, mech_vals: dict, latent_vals: dict,
                 elapsed_time: timedelta | int | float = timedelta()):
        self.elapsed = elapsed_time
        self.curr_prm_vals = deepcopy(init_vals)
        self.init_prm_vals = deepcopy(init_vals)
        self.curr_deg_mech_vals = deepcopy(mech_vals)
        self.init_deg_mech_vals = deepcopy(mech_vals)
        self.latent_var_vals = deepcopy(latent_vals)
