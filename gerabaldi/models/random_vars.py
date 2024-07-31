# Copyright (c) 2024 Ian Hill
# SPDX-License-Identifier: Apache-2.0

"""Custom generators for random variables used for experiment simulation and model specification. Necessary as the Numpy
and PyMC RVs either cannot be persisted or futz about with tensor types."""

from __future__ import annotations

import numpy as np

from gerabaldi.helpers import _on_demand_import

# Optional imports are loaded using a helper function that suppresses import errors until attempted use
pymc = _on_demand_import('pymc')
pyro = _on_demand_import('pyro', 'pyro-ppl')
dist = _on_demand_import('pyro.distributions', 'pyro-ppl')

__all__ = ['RandomVar', 'Deterministic', 'Normal', 'Gamma', 'Uniform']


class RandomVar:
    """
    Custom class for random variables due to limitations of Numpy and PyMC random variable implementations.

    This class is only exposed for typing annotations, do not instantiate directly.
    """

    def __init__(self, dist_type: str = None, name: str = None, test_seed: int = None):
        self.name = name
        if dist_type:
            self._dist_type = dist_type
        # Test seed is used to make the random variable sampling reproducible for testing purposes only
        if test_seed:
            self._generator = getattr(np.random.default_rng(seed=test_seed), self._dist_type)
        else:
            self._generator = getattr(np.random.default_rng(), self._dist_type)

    def _get_dist_params(self, target: str = 'pymc'):
        if target == 'pymc':
            return self._params_for_pymc()
        elif target == 'numpy':
            return self._params_for_numpy()
        elif target == 'pyro':
            return self._params_for_pyro()
        else:
            raise NotImplementedError(f'Invalid target library {target} requested.')

    def sample(self, quantity: int = 1) -> np.ndarray:
        """
        Return a quantity of random generated samples from the random variable distribution

        Parameters
        ----------
        quantity: int, optional
            The number of samples to obtain (default 1)

        Returns
        -------
        numpy.ndarray
            A 1D array containing the sampled values
        """
        return self._generator(size=quantity, **self._get_dist_params('numpy'))

    def get_cbi_form(self, target_framework: str = 'pymc', observed: np.ndarray = None):
        """
        This method constructs a transpiled equivalent random variable definition for the target CBI framework

        Parameters
        ----------
        target_framework: str, optional
            The CBI framework to transpile the random variable to (default 'pymc')
        observed: np.ndarray, optional
            Any observed data points for the random variable

        Returns
        -------
        various
            The transpiled equivalent random variable distribution in the CBI framework model-context form
        """
        if target_framework == 'pymc':
            try:
                return getattr(pymc, self._dist_type.capitalize())(
                    name=self.name, observed=observed, **self._get_dist_params(),
                )
            except AttributeError as e:
                raise NotImplementedError(
                    f'Distribution type {self._dist_type} is not implemented in PyMC. Please\n'
                    'change the distribution type or create a custom PyMC implementation.',
                ) from e
        elif target_framework == 'pyro':
            try:
                return pyro.sample(
                    self.name, getattr(dist, self._dist_type.capitalize())(**self._get_dist_params('pyro')),
                )
            except AttributeError as e:
                raise NotImplementedError(
                    f'Distribution type {self._dist_type} is not implemented in pyro. Please\n'
                    'change the distribution type or create a custom pyro implementation.',
                ) from e
        else:
            raise NotImplementedError(f'Target framework {target_framework} is not yet a supported CBI framework')

    # From here on down are abstract methods that inheriting distributions should implement
    def _params_for_pymc(self):
        raise NotImplementedError(f'Distribution {self._dist_type} does not define a mapping to a PyMC form.')

    def _params_for_numpy(self):
        raise NotImplementedError(f'Distribution {self._dist_type} does not define a mapping to a numpy form.')

    def _params_for_pyro(self):
        raise NotImplementedError(f'Distribution {self._dist_type} does not define a mapping to a Pyro form.')

    def get_centre(self):
        raise NotImplementedError(f'Distribution {self._dist_type} does not have a defined centre value.')

    def set_centre(self, value, operation=None):
        raise NotImplementedError(f'Distribution {self._dist_type} does not have a defined centre value.')


class Deterministic(RandomVar):
    """
    A workaround class for non-stochastic variables, always sampling the same target value

    Attributes
    ----------
    name: str
    value: float or int
    """

    def __init__(self, value: float = 0, **super_opts):
        """
        Parameters
        ----------
        name: str, optional
            An identifying name for the variable
        value: float or int, optional
            The deterministic value to always sample (default 0)
        """
        # To let a generator be initialized, we need to pass a viable distribution to numpy RNG, not actually uniform
        self._dist_type = 'uniform'
        self.value = value
        super().__init__(**super_opts)
        # Now that the object's RNG is initialized, set the true distribution type
        self._dist_type = 'deterministic'

    # Override the sample method as we are not using a Numpy distribution here
    def sample(self, quantity: int = 1) -> np.ndarray:
        """
        Returns an array of size 'quantity' filled with the specified deterministic value

        Parameters
        ----------
        quantity: int, optional
            The number of samples to obtain (default 1)

        Returns
        -------
        numpy.ndarray
            A 1D array containing the sampled values
        """
        return np.full(quantity, self.value)


class Normal(RandomVar):
    """
    Normal distribution random variable custom class

    Attributes
    ----------
    mu: float or int
    sigma: float or int
    name: str
    """

    def __init__(self, mu: float = 0, sigma: float = 1, **super_opts):
        """
        Parameters
        ----------
        mu: float or int, optional
            The centre value of the distribution (default 0)
        sigma: float or int, optional
            The standard deviation of the distribution (default 1)
        name: str, optional
            An identifying name for the variable
        test_seed: int, optional
            An RNG override to obtain reproducible sampling behaviour (for testing only)
        """
        self._dist_type = 'normal'
        self.mu = mu
        self.sigma = sigma
        super().__init__(**super_opts)

    def _params_for_pymc(self):
        return {'mu': self.mu, 'sigma': self.sigma}

    def _params_for_numpy(self):
        return {'loc': self.mu, 'scale': self.sigma}

    def _params_for_pyro(self):
        return {'loc': self.mu, 'scale': self.sigma}

    def get_centre(self) -> float | int:
        """
        Get the centre value of the distribution, i.e. mu for a normal distribution

        Returns
        -------
        float or int
        """
        return self.mu

    def set_centre(self, value, operation=None):
        """
        Set the centre value of the distribution, i.e. mu for a normal distribution

        Parameters
        ----------
        value: float or int
            The value to set or use as the first operand
        operation: Callable, optional
            A function that takes the passed value and current distribution centre value to produce a modified value
        """
        if operation:
            self.mu = operation(value, self.mu)
        else:
            self.mu = value


class Gamma(RandomVar):
    """
    Gamma distribution alpha-beta implementation

    Attributes
    ----------
    alpha: float or int
    beta: float or int
    name: str
    """

    def __init__(self, alpha: float = 2, beta: float = 2, **super_opts):
        """
        Parameters
        ----------
        alpha: float or int, optional
            The shape parameter of the distribution (default 2)
        beta: float or int, optional
            The rate parameter of the distribution (default 2)
        name: str, optional
            An identifying name for the variable
        test_seed: int, optional
            An RNG override to obtain reproducible sampling behaviour (for testing only)
        """
        self._dist_type = 'gamma'
        self.alpha = alpha
        self.beta = beta
        super().__init__(**super_opts)

    def _params_for_pymc(self):
        return {'alpha': self.alpha, 'beta': self.beta}

    def _params_for_numpy(self):
        return {'shape': self.alpha, 'scale': 1 / self.beta}

    def _params_for_pyro(self):
        return {'concentration': self.alpha, 'rate': self.beta}


class Uniform(RandomVar):
    """
    Continuous uniform distribution implementation

    Attributes
    ----------
    start: float or int
    stop: float or int
    name: str
    """

    def __init__(self, start: float = 0, stop: float = 1, **super_opts):
        """
        Parameters
        ----------
        start: float or int, optional
            The lowest value within the non-zero probability range of the distribution (default 0)
        stop: float or int, optional
            The highest value within the non-zero probability range of the distribution (default 1)
        name: str, optional
            An identifying name for the variable
        test_seed: int, optional
            An RNG override to obtain reproducible sampling behaviour (for testing only)
        """
        self._dist_type = 'uniform'
        self.start = start
        self.stop = stop
        super().__init__(**super_opts)

    def _params_for_pymc(self):
        return {'lower': self.start, 'upper': self.stop}

    def _params_for_numpy(self):
        return {'low': self.start, 'high': self.stop}

    def _params_for_pyro(self):
        return {'low': self.start, 'high': self.stop}
