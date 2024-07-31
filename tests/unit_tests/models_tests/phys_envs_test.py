# Copyright (c) 2024 Ian Hill
# SPDX-License-Identifier: Apache-2.0

"""Tests for the classes that provide the effects of the physical test environment and measuring devices used"""

import pytest
import numpy as np
import pandas as pd

from gerabaldi.models import *
from gerabaldi.exceptions import UserConfigError


def test_measurement_device():
    # First test default cases
    dev = MeasInstrument()
    assert dev.name == 'generic'
    # Test that the dynamic return typing works correctly
    assert dev.measure(3) == 3
    assert dev.measure(-1.56) == -1.56
    assert np.array_equal(dev.measure(np.array([4, 6.3])), np.array([4, 6.3]))
    assert dev.measure(pd.Series([-1, 5.2])).equals(pd.Series([-1, 5.2]))
    # Now test the unideal aspects
    dev = MeasInstrument('For Testing', 3, Gamma(0.5, 0.5, test_seed=3985), (0, 4))
    assert dev.name == 'For Testing'
    assert np.array_equal(dev.measure(np.array([-2.4, 1, 1.2345, 6, 3])), np.array([0, 2.58, 1.93, 4, 3.13]))
    # Finally, test the miserable values to handle in the fancy significant figures math
    dev = MeasInstrument('Bad Vals', 4, meas_lims=(-2, 1000))
    assert dev.measure(0) == 0
    assert dev.measure(np.inf) == 1000


def test_env_vrtn_model(sequential_var):
    # First test default model setup
    mdl = EnvVrtnMdl()
    assert mdl.name is None
    assert mdl.vrtn_type == 'offset'
    assert mdl.batch_vrtn_mdl.sample() == 0
    assert np.allclose(mdl.gen_env_vrtns(15, 2, 2), np.array([[[15, 15], [15, 15]]]))

    # Now test all the attribute setting logic
    mdl.name = 'new_name'
    assert mdl.name == 'new_name'
    mdl.vrtn_type = 'scaling'
    assert mdl.vrtn_type == 'scaling'
    assert mdl.batch_vrtn_mdl.sample() == 1
    assert np.allclose(mdl.gen_env_vrtns(15, 2, 2), np.array([[[15, 15], [15, 15]]]))
    mdl.chp_vrtn_mdl = Normal(0, 0.5, test_seed=675)
    mdl.vrtn_type = 'offset'
    assert mdl.batch_vrtn_mdl.sample() == 0
    assert np.round(mdl.chp_vrtn_mdl.sample(), 2) == -0.31
    assert np.allclose(np.round(mdl.gen_env_vrtns(15, 2, 2), 2), np.array([[[14.34, 14.34], [14.72, 14.72]]]))
    mdl.chp_vrtn_mdl = None
    assert mdl.chp_vrtn_mdl.sample() == 0
    with pytest.raises(AttributeError):
        mdl.new_attribute = 'some_val'

    # Test all the variation value generation methods
    mdl = EnvVrtnMdl(
        dev_vrtn_mdl=sequential_var(0, 0.01), chp_vrtn_mdl=sequential_var(0, 0.1), batch_vrtn_mdl=sequential_var(1, 1)
    )
    mdl.vrtn_type = 'scaling'
    mdl.dev_vrtn_mdl = Deterministic(3)
    mdl.chp_vrtn_mdl = Deterministic(2)
    assert mdl.vrtn_type == 'scaling'
    assert mdl.gen_env_vrtns(3)[0][0][0] == 18
    assert mdl.gen_env_vrtns(3, 3, 2, 3)[0][0][0] == 18
    mdl.vrtn_type = 'offset'
    mdl.dev_vrtn_mdl = sequential_var(1, 1)
    mdl.chp_vrtn_mdl = sequential_var(1, 1)
    assert mdl.gen_batch_vrtn(3) == 4
    assert mdl.gen_chp_vrtns(3) == 4
    assert np.allclose(mdl.gen_chp_vrtns(-1, 2, 3), np.array([[0, 1], [2, 3], [4, 5]]))
    assert mdl.gen_dev_vrtns(3) == 4
    assert np.allclose(mdl.gen_dev_vrtns(np.array([[-1, -2]]), 3), np.array([[[0, 1, 2], [2, 3, 4]]]))
    assert mdl.gen_env_vrtns(-3, 2, 2, 2)[1][0][1] == 7


def test_physical_test_environment(sequential_var):
    # Test default creation first
    env = PhysTestEnv()
    assert env.name == 'unspecified'
    assert type(env.vrtn_mdl('some_prm')) is EnvVrtnMdl
    assert type(env.meas_instm('another_prm')) is MeasInstrument
    assert env.gen_env_cond_vals({'p1': 125, 'prm2': 3.1415}, {'p1': 1, 'prm2': 1})['prm2']['prm2'] == 3.1415
    assert np.allclose(
        env.gen_env_cond_vals({'p1': 125}, {'p1': 2}, num_chps=2)['p1']['p1'], np.array([[[125, 125], [125, 125]]])
    )

    # Now test non-defaults model and instrument use
    env = PhysTestEnv(
        {
            'a': EnvVrtnMdl(dev_vrtn_mdl=Normal(0, 2, test_seed=5746)),
            'b': EnvVrtnMdl(dev_vrtn_mdl=Gamma(1, 1 / 3, test_seed=4635)),
        },
        {'b': MeasInstrument(precision=2)},
        'Test Env',
    )
    assert env.name == 'Test Env'
    assert env.meas_instm('b').measure(4.3674) == 4.4
    assert np.allclose(
        np.round(env.vrtn_mdl('b').gen_env_vrtns(-1, 3, 2, 1), 2),
        np.array([[[2.62, -0.69, 1.45], [-0.33, 1.84, 0.34]]]),
    )

    # Now test the generation of sets of environmental conditions
    assert env.gen_env_cond_vals({'a': 80, 'b': -1}, {'a': 1, 'b': 1})['a']['a'].round(3)[0][0][0] == 83.197
    with pytest.raises(UserConfigError):
        env.gen_env_cond_vals({'a': 3, 'b': 4}, ['a'])
    env.vrtn_mdl('a').dev_vrtn_mdl = sequential_var(0.1, 0.1)
    env.vrtn_mdl('a').chp_vrtn_mdl = sequential_var(1, 1)
    env.vrtn_mdl('a').batch_vrtn_mdl = sequential_var(0.5, 0.5)

    def bti_func(time, a, b, x):
        return (time * a) + (b * x)

    def cond_func(base, b, k):
        return base + (b * k)

    def hci_func(time, a, y):
        return (time / a) * y

    dev_mdl = DeviceMdl(
        {
            'bti': DegPrmMdl(DegMechMdl(bti_func, mdl_name='deg'), cond_shift_mdl=CondShiftMdl(cond_func)),
            'hci': DegPrmMdl(DegMechMdl(hci_func, mdl_name='deg2')),
        }
    )
    test_spec = TestSpec([MeasSpec({'bti': 2, 'hci': 3, 'b': 2}, {'a': 13, 'b': 7})], num_chps=2, num_lots=1)
    report = SimReport(test_spec)
    env_conds = env.gen_env_cond_vals({'a': 4, 'b': 10}, ['bti', 'hci'], report, dev_mdl)
    assert np.allclose(env_conds['bti']['a'], np.array([[[5.6, 5.7], [6.8, 6.9]]]))
    assert np.allclose(env_conds['hci']['a'], np.array([[[5.6, 5.7, 5.8], [6.9, 7.0, 7.1]]]))
    assert 'b' not in env_conds['hci']
    env.vrtn_mdl('b').batch_vrtn_mdl = sequential_var(2, 1)
    env.vrtn_mdl('b').chp_vrtn_mdl = sequential_var(0.5, 0.5)
    env.vrtn_mdl('b').dev_vrtn_mdl = sequential_var(0.1, 0.1)
    env_conds = env.gen_env_cond_vals({'a': 2, 'b': 10}, ['bti', 'hci', 'b'], report, dev_mdl, 'measure')
    assert np.allclose(env_conds['bti']['b'], np.array([[[12.6, 12.7], [13.3, 13.4]]]))
    assert np.allclose(env_conds['b']['b'], np.array([[[12.6], [13.2]]]))
