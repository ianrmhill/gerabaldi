# Copyright (c) 2024 Ian Hill
# SPDX-License-Identifier: Apache-2.0

"""Test set ensuring correct functionality of the custom random variable classes"""

import pytest
import numpy as np

from gerabaldi.models.random_vars import *
from gerabaldi.helpers import _on_demand_import

pymc = _on_demand_import('pymc')


def test_deterministic_dist():
    det = Deterministic(name='test', value=10)
    # Test basic sampling behaviour and properties
    samples = det.sample(10)
    assert np.allclose(samples, np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10]))
    assert (det.name, det.value) == ('test', 10)
    # Test the errors that should get raised
    with pytest.raises(NotImplementedError):
        det._get_dist_params()
    with pytest.raises(NotImplementedError):
        det._get_dist_params('numpy')
    with pytest.raises(NotImplementedError):
        det._get_dist_params('not_an_option')
    with pytest.raises(NotImplementedError):
        det.get_cbi_form()
    # Test changing the RV params
    det.value = 5
    samples = det.sample(2)
    assert np.allclose(samples, np.array([5, 5]))
    # Check default value behaviour
    det = Deterministic()
    samples = det.sample(2)
    assert np.allclose(samples, np.array([0, 0]))


def test_normal_dist():
    dist = Normal(mu=4, sigma=0.1, test_seed=2022)
    assert (dist.name, dist.mu, dist.sigma) == (None, 4, 0.1)
    # Check basic sampling behaviour, first reproducibility
    samples = dist.sample(4)
    assert np.allclose(samples, np.array([4.26764, 3.91572, 4.20782, 3.84723]))
    # Check that the correct parameter names are returned
    assert dist._get_dist_params() == {'mu': 4, 'sigma': 0.1}
    assert dist._get_dist_params('numpy') == {'loc': 4, 'scale': 0.1}
    # Test default constructor arguments
    dist = Normal(test_seed=2023)
    assert (dist.mu, dist.sigma) == (0, 1)
    dist.mu = -2
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
        assert type(dist.get_cbi_form()) is type(dist2)


def test_gamma_dist():
    dist = Gamma(alpha=4, beta=1 / 3, test_seed=2022)
    assert (dist.name, dist.alpha, dist.beta) == (None, 4, 1 / 3)
    # Check basic sampling behaviour, first reproducibility
    samples = dist.sample(4)
    assert np.allclose(samples, np.array([34.65049, 27.77787, 13.43644, 2.02734]))
    # Check that the correct parameter names are returned
    assert dist._get_dist_params()['alpha'] == 4
    assert dist._get_dist_params('numpy') == {'shape': 4, 'scale': 3}
    # Test default constructor arguments
    dist = Gamma(test_seed=2023)
    assert (dist.alpha, dist.beta) == (2, 2)
    dist.alpha = 3
    assert (dist.alpha, dist.beta) == (3, 2)
    samples = dist.sample()
    assert np.allclose(samples, np.array([1.88745]))
    # Test the errors that should get raised
    with pytest.raises(NotImplementedError):
        dist._get_dist_params('uh_oh')
    # Test that the correct pymc distribution is obtained
    with pymc.Model():
        dist3 = pymc.Gamma('test3', alpha=2, beta=1)
        dist.name = 'another_name'
        assert type(dist.get_cbi_form()) is type(dist3)
