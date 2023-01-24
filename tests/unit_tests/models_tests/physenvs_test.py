"""Tests for the classes that provide the effects of the physical test environment and measuring devices used."""

import numpy as np
import pandas as pd

from gerabaldi.models.physenvs import *
from gerabaldi.models.randomvars import *


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
    dev = MeasInstrument('For Testing', 3, Gamma(0.5, 2, test_seed=3985), (0, 4))
    assert dev.name == 'For Testing'
    assert np.array_equal(dev.measure(np.array([-2.4, 1, 1.2345, 6, 3])), np.array([0, 2.58, 1.93, 4, 3.13]))
    # Finally, test the miserable values to handle in the fancy significant figures math
    dev = MeasInstrument('Bad Vals', 4, meas_lims=(-2, 1000))
    assert dev.measure(0) == 0
    assert dev.measure(np.inf) == 1000


def test_physical_test_environment():
    # Test default creation first
    env = PhysTestEnv()
    assert env.name == 'unspecified'
    assert type(env.get_vrtn_mdl('some_prm')) == EnvVrtnMdl
    assert type(env.get_meas_instm('another_prm')) == MeasInstrument
    assert env.gen_env_cond_vals({'p1': 125, 'prm2': 3.1415})[0] == {'p1': np.array(125), 'prm2': np.array(3.1415)}
    assert np.allclose(env.gen_env_cond_vals({'p1': 125}, num_vals=(1, 2, 2))[0]['p1'],
                       np.array([[[125, 125], [125, 125]]]))
    # Now test non-defaults
    env = PhysTestEnv({'a_prm': EnvVrtnMdl(dev_vrtn_mdl=Normal(0, 2, test_seed=5746)),
                       'b_prm': EnvVrtnMdl(dev_vrtn_mdl=Gamma(1, 3, test_seed=4635))},
                      {'b_prm': MeasInstrument(precision=2)}, 'Very Special Env')
    assert env.name == 'Very Special Env'
    assert env.get_meas_instm('b_prm').measure(4.3674) == 4.4
    assert env.gen_env_cond_vals({'a_prm': 80, 'b_prm': -1})[0]['a_prm'].round(3)[0][0][0] == 83.197
    # Test that the sensor value generation is working correctly
    assert np.allclose(env.gen_env_cond_vals({'a_prm': 80, 'b_prm': -1},
                                             sensor_counts={'b_prm': 3})[1]['b_prm'].round(3), [1.449, -0.333, 1.841])


def test_env_vrtn_model(sequential_var):
    mdl = EnvVrtnMdl()
    assert mdl.name is None
    assert mdl.vrtn_op == 'offset'
    assert mdl.batch_vrtn_mdl.sample() == 0
    mdl = EnvVrtnMdl(dev_vrtn_mdl=sequential_var(0, 0.01), chp_vrtn_mdl=sequential_var(0, 0.1),
                     batch_vrtn_mdl=sequential_var(1, 1))
    env = PhysTestEnv({'temp': mdl})
    assert np.allclose(env.gen_env_cond_vals({'temp': 45}, (2, 2, 2))[0]['temp'],
                       np.array([[[46, 46.01], [46.12, 46.13]], [[46.24, 46.25], [46.36, 46.37]]]))
    assert mdl.name == 'temp'
    mdl.vrtn_op = 'scaling'
    mdl.dev_vrtn_mdl = Deterministic(3)
    mdl.chp_vrtn_mdl = Deterministic(2)
    assert mdl.vrtn_op == 'scaling'
    assert mdl.gen_env_vrtns(3, 0, 1, 1, 1)[0][0][0] == 18
    # Test that the sensor value generation is working correctly
    assert mdl.gen_env_vrtns(3, 2, 1, 1, 1)[1][1] == 18
