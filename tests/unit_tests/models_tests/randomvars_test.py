"""Test set ensuring correct functionality of the custom random variable classes."""

import pytest
import numpy as np
import pymc

from gerabaldi.models.randomvars import *


def test_deterministic_dist():
    det = Deterministic(var_name='test', value=10)
    # Test basic sampling behaviour and properties
    samples = det.sample(10)
    assert np.allclose(samples, np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10]))
    assert (det.name, det.value, det.dist_type) == ('test', 10, 'exact')
    # Test the errors that should get raised
    with pytest.raises(NotImplementedError):
        det._get_dist_params()
        det._get_dist_params('numpy')
        det._get_dist_params('not_an_option')
        det.get_cbi_form()
    # Test changing the RV params
    det.set_dist_params(value=5)
    samples = det.sample(2)
    assert np.allclose(samples, np.array([5, 5]))
    # Check default value behaviour
    det = Deterministic()
    samples = det.sample(2)
    assert np.allclose(samples, np.array([0, 0]))


def test_normal_dist():
    dist = Normal(mu=4, sigma=0.1, test_seed=2022)
    assert (dist.name, dist.mu, dist.sigma, dist.dist_type) == (None, 4, 0.1, 'normal')
    # Check basic sampling behaviour, first reproducibility
    samples = dist.sample(4)
    assert np.allclose(samples, np.array([4.26764, 3.91572, 4.20782, 3.84723]))
    # Check that the correct parameter names are returned
    assert dist._get_dist_params() == {'mu': 4, 'sigma': 0.1}
    assert dist._get_dist_params('numpy') == {'loc': 4, 'scale': 0.1}
    # Test default constructor arguments
    dist = Normal(test_seed=2023)
    assert (dist.mu, dist.sigma) == (0, 1)
    dist.set_dist_params(mu=-2)
    assert (dist.mu, dist.sigma) == (-2, 1)
    samples = dist.sample(3)
    assert np.allclose(samples, np.array([-1.39828, -0.84838, -3.35946]))
    # Test the errors that should get raised
    with pytest.raises(NotImplementedError):
        dist._get_dist_params('uh_oh')
    # Test that the correct pymc distribution is obtained
    with pymc.Model():
        dist2 = pymc.Normal('test2')
        dist.name = 'test_name'
        assert type(dist.get_cbi_form()) == type(dist2)


def test_gamma_dist():
    dist = Gamma(alpha=4, beta=3, test_seed=2022)
    assert (dist.name, dist.alpha, dist.beta, dist.dist_type) == (None, 4, 3, 'gamma')
    # Check basic sampling behaviour, first reproducibility
    samples = dist.sample(4)
    assert np.allclose(samples, np.array([34.65049, 27.77787, 13.43644, 2.02734]))
    # Check that the correct parameter names are returned
    assert dist._get_dist_params() == {'alpha': 4, 'beta': 3}
    assert dist._get_dist_params('numpy') == {'shape': 4, 'scale': 3}
    # Test default constructor arguments
    dist = Gamma(test_seed=2023)
    assert (dist.alpha, dist.beta) == (0, 1)
    dist.set_dist_params(alpha=2)
    assert (dist.alpha, dist.beta) == (2, 1)
    samples = dist.sample()
    assert np.allclose(samples, np.array([2.57043]))
    # Test the errors that should get raised
    with pytest.raises(NotImplementedError):
        dist._get_dist_params('uh_oh')
    # Test that the correct pymc distribution is obtained
    with pymc.Model():
        dist3 = pymc.Gamma('test3', alpha=2, beta=1)
        dist.name = 'another_name'
        assert type(dist.get_cbi_form()) == type(dist3)
