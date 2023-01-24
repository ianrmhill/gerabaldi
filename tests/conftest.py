"""Helper classes and functions to improve testing of stochastic methods."""

import pytest
import numpy as np

from gerabaldi.models.randomvars import RandomVar
from gerabaldi.exceptions import MissingParamError


class Sequential(RandomVar):
    """Non-stochastic generator that always generates a sequence with length equal to the number of samples."""
    def __init__(self, start: int | float = 0, increment: int | float = 1, **super_opts):
        self.start = start
        self.inc = increment
        # To let a generator be initialized, we need to pass a viable distribution to numpy RNG
        self.dist_type = 'uniform'
        super().__init__(**super_opts)
        # Now can set the true dist_type
        self.dist_type = 'sequential'

    # Need to override the sample method
    def sample(self, quantity: int = 1):
        """Returns a numpy array of size 'quantity' filled with an incrementing sequence."""
        return np.linspace(self.start, self.start + (self.inc * (quantity - 1)), num=quantity)

    def _params_for_pymc(self):
        raise NotImplementedError('PyMC does not support using non-random variables as tracked variables.')

    def _params_for_numpy(self):
        raise NotImplementedError('Numpy RNG does not support non-random distributions.')

    def _params_for_pyro(self):
        raise NotImplementedError('Pyro does not support using non-random variables as tracked variables.')

    def get_centre(self):
        raise NotImplementedError('Sequential distribution has no defined centrepoint')

    def set_centre(self, value, operation=None):
        raise NotImplementedError('Sequential distribution has no defined centrepoint')

    def set_dist_params(self, **vals):
        """Sets the deterministic sample value to the argument passed as keyword 'value'."""
        changed = False
        if 'start' in vals:
            self.start = vals['start']
            changed = True
        if 'increment' in vals:
            self.inc = vals['increment']
            changed = True
        if not changed:
            raise MissingParamError('At least one of start or increment must be passed to modify normal dist params.')


@pytest.fixture
def sequential_var():
    return Sequential
