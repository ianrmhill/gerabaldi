"""Custom generators for random variables used for experiment simulation and model specification. Necessary as the Numpy
and PyMC RVs are either not storable or futz about with tensor variables."""

import numpy as np

from gerabaldi.exceptions import MissingParamError
from gerabaldi.helpers import _on_demand_import

# Optional imports are loaded using a helper function that suppresses import errors until attempted use
pymc = _on_demand_import('pymc')
pyro = _on_demand_import('pyro', 'pyro-ppl')
dist = _on_demand_import('pyro.distributions', 'pyro-ppl')

__all__ = ['RandomVar', 'Deterministic', 'Normal', 'Gamma', 'Uniform']


class RandomVar:
    """
    Custom class for random variables due to limitations of Numpy and PyMC random variable implementations.
    """
    def __init__(self, dist_type: str = None, var_name: str = None, test_seed: int = None):
        self.name = var_name
        if dist_type:
            self.dist_type = dist_type
        # Test seed is used to make the random variable reproducible for testing purposes only
        if test_seed:
            self.generator = getattr(np.random.default_rng(seed=test_seed), self.dist_type)
        else:
            self.generator = getattr(np.random.default_rng(), self.dist_type)

    def _get_dist_params(self, target: str = 'pymc'):
        match target:
            case 'pymc':
                return self._params_for_pymc()
            case 'numpy':
                return self._params_for_numpy()
            case 'pyro':
                return self._params_for_pyro()
            case _:
                raise NotImplementedError(f"Invalid target library {target} requested.")

    def _params_for_pymc(self):
        return {}

    def _params_for_numpy(self):
        return {}

    def _params_for_pyro(self):
        return {}

    def set_dist_params(self, **vals):
        """Set the parameterization of the distribution using the passed keyword arguments."""
        raise NotImplementedError(f"Used distribution {self.dist_type} does not implement this method, please add an\n"
                                  "implementation or use a different distribution.")

    def sample(self, quantity: int = 1):
        """Return a quantity of random generated samples from the distribution object. Returns an array"""
        return self.generator(size=quantity, **self._get_dist_params('numpy'))

    def get_cbi_form(self, target_framework: str = 'pymc', observed=None):
        """This method constructs a PyMC random variable corresponding to the custom class distribution for inference.

        Parameters
        ----------
        observed : Any observed data points for the parameter, optional

        Returns
        -------
        cbi_dist : The same random variable distribution, but in PyMC model-context form
        """
        # Have to catch if the distribution type is not implemented in PyMC
        if target_framework == 'pymc':
            try:
                return getattr(pymc, self.dist_type.capitalize())(name=self.name, observed=observed, **self._get_dist_params())
            except AttributeError:
                raise NotImplementedError(f"Distribution type {self.dist_type} is not implemented in PyMC. Please change\n"
                                          "the distribution type or create a custom PyMC implementation.")
        elif target_framework == 'pyro':
            try:
                return pyro.sample(self.name, getattr(dist, self.dist_type.capitalize())(**self._get_dist_params('pyro')))
            except AttributeError:
                raise NotImplementedError(f"Distribution type {self.dist_type} is not implemented in pyro. Please change\n"
                                          "the distribution type or create a custom pyro implementation.")
        else:
            raise NotImplementedError(f"Target framework {target_framework} is not yet a supported CBI framework")

    def get_centre(self):
        raise NotImplementedError(f"Distribution {self.dist_type} does not have a defined centre value.")

    def set_centre(self, value, operation=None):
        raise NotImplementedError(f"Distribution {self.dist_type} does not have a defined centre value.")


class Deterministic(RandomVar):
    """A workaround class that represents a deterministic distribution. Shadows the PyMC distribution methods."""
    def __init__(self, value: int | float = 0, **super_opts):
        # To let a generator be initialized, we need to pass a viable distribution to numpy RNG
        self.dist_type = 'uniform'
        self.value = value
        super().__init__(**super_opts)
        # Now that the object's RNG is initialized, set the true distribution type
        self.dist_type = 'exact'

    # Override the sample method as we are not using a Numpy distribution here
    def sample(self, quantity: int = 1):
        """Returns an array of size 'quantity' filled with the specified deterministic value."""
        return np.full(quantity, self.value)

    def _params_for_pymc(self):
        raise NotImplementedError('PyMC does not support using deterministic variables as tracked variables.')

    def _params_for_numpy(self):
        raise NotImplementedError('Numpy RNG does not support non-random distributions.')

    def set_dist_params(self, **vals):
        """Sets the deterministic sample value to the argument passed as keyword 'value'."""
        if 'value' not in vals:
            raise MissingParamError('Argument \'value\' is required to set deterministic distribution value.')
        self.value = vals['value']


class Normal(RandomVar):
    """
    Normal distribution random variable custom class.
    """
    def __init__(self, mu: int | float = 0, sigma: int | float = 1, **super_opts):
        self.dist_type = 'normal'
        self.mu = mu
        self.sigma = sigma
        super().__init__(**super_opts)

    def _params_for_pymc(self):
        return {'mu': self.mu, 'sigma': self.sigma}

    def _params_for_numpy(self):
        return {'loc': self.mu, 'scale': self.sigma}

    def _params_for_pyro(self):
        return {'loc': self.mu, 'scale': self.sigma}

    def set_dist_params(self, **vals):
        """Sets the mean and deviation of the distribution to the keyword arguments 'mu' and 'sigma' respectively."""
        changed = False
        if 'mu' in vals:
            self.mu = vals['mu']
            changed = True
        if 'sigma' in vals:
            self.sigma = vals['sigma']
            changed = True
        if not changed:
            raise MissingParamError('At least one of mu or sigma must be passed to modify normal dist params.')

    def get_centre(self):
        return self.mu

    def set_centre(self, value, operation=None):
        if operation:
            self.mu = operation(value, self.mu)
        else:
            self.mu = value


class Gamma(RandomVar):
    """
    Gamma distribution alpha-beta implementation.
    """
    def __init__(self, alpha: int | float = 0, beta: int | float = 1, **super_opts):
        self.dist_type = 'gamma'
        self.alpha = alpha
        self.beta = beta
        super().__init__(**super_opts)

    def _params_for_pymc(self):
        return {'alpha': self.alpha, 'beta': self.beta}

    def _params_for_numpy(self):
        return {'shape': self.alpha, 'scale': self.beta}

    def _params_for_pyro(self):
        return {'concentration': self.alpha, 'rate': self.beta}

    def set_dist_params(self, **vals):
        """Sets the distribution parameters to the passed keyword arguments 'alpha' and 'beta'."""
        changed = False
        if 'alpha' in vals:
            self.alpha = vals['alpha']
            changed = True
        if 'beta' in vals:
            self.beta = vals['beta']
            changed = True
        if not changed:
            raise MissingParamError('At least one of alpha or beta must be passed to modify gamma dist params.')


class Uniform(RandomVar):
    """
    Continuous uniform distribution implementation
    """
    def __init__(self, low: int | float = 0, high: int | float = 1, **super_opts):
        self.dist_type = 'uniform'
        self.start = low
        self.stop = high
        super().__init__(**super_opts)

    def _params_for_pymc(self):
        return {'lower': self.start, 'upper': self.stop}

    def _params_for_numpy(self):
        return {'low': self.start, 'high': self.stop}

    def _params_for_pyro(self):
        return {'low': self.start, 'high': self.stop}

    def set_dist_params(self, **vals):
        if 'low' in vals:
            self.start = vals['low']
        if 'high' in vals:
            self.stop = vals['high']
