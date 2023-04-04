# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

"""Helper classes and functions to improve testing of stochastic methods."""

import pytest
import numpy as np

from gerabaldi.models.random_vars import RandomVar


class Sequential(RandomVar): # noqa: ImplementAbstract
    """Non-stochastic generator that always generates a sequence with length equal to the number of samples."""
    def __init__(self, start: int | float = 0, increment: int | float = 1, **super_opts):
        self.start = start
        self.inc = increment
        # To let a generator be initialized, we need to pass a viable distribution to numpy RNG
        self._dist_type = 'uniform'
        super().__init__(**super_opts)
        # Now can set the true dist_type
        self._dist_type = 'sequential'

    # Need to override the sample method
    def sample(self, quantity: int = 1):
        """Returns a numpy array of size 'quantity' filled with an incrementing sequence."""
        return np.linspace(self.start, self.start + (self.inc * (quantity - 1)), num=quantity)


@pytest.fixture
def sequential_var():
    return Sequential
