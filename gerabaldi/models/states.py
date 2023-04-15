# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

"""Custom classes for simulation state persistence"""

from __future__ import annotations

from datetime import timedelta
from copy import deepcopy

__all__ = ['TestSimState']


class TestSimState:
    """
    Class to fully track the physical state of a reliability test, such as last executed step, degradation values of
    different parameters, and effective accumulated stress. Intended to be mutated as the test progresses.

    Attributes
    ----------
    elapsed: timedelta or int or float
        The accumulated amount of stress time up to the current point
    latent_var_vals: dict
        A nested dictionary structure containing all the sampled values for various latent variables
    init_deg_mech_vals: dict
        A nested dictionary structure containing all the initial values for various wear-out mechanisms
    init_prm_vals: dict
        A nested dictionary structure containing all the initial values for various device parameters
    init_deg_mech_vals: dict
        A nested dictionary structure containing all the current point values for various wear-out mechanisms
    init_prm_vals: dict
        A nested dictionary structure containing all the current point values for various device parameters
    """
    def __init__(self, init_prm_vals: dict, init_mech_vals: dict, latent_vals: dict,
                 elapsed_time: timedelta | int | float = timedelta()):
        """

        Parameters
        ----------
        init_prm_vals: dict
            Nested dictionary with all the initial values for various device parameters
        init_mech_vals: dict
            Nested dictionary with all the initial values for various mechanisms
        latent_vals: dict
            Nested dictionary with all the sampled values for various latent variables
        elapsed_time: timedelta or float or int, optional
            The amount of accumulated stress time thus far, typically initialized to 0 (default 0)
        """
        self.elapsed = elapsed_time
        self.curr_prm_vals = deepcopy(init_prm_vals)
        self.init_prm_vals = deepcopy(init_prm_vals)
        self.curr_deg_mech_vals = deepcopy(init_mech_vals)
        self.init_deg_mech_vals = deepcopy(init_mech_vals)
        self.latent_var_vals = deepcopy(latent_vals)
