# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from gerabaldi.cookbook.test_envs import ideal_env, volt_and_temp_env


def test_ideal_env():
    new_env = ideal_env()
    assert new_env.meas_instm('temp').measure(0.6475) == 0.6475


def test_volt_temp_env():
    new_env = volt_and_temp_env()
    some_vals = new_env.vrtn_mdl('vdd').batch_vrtn_mdl.sample(5)
    assert type(some_vals) == np.ndarray
    assert np.allclose(some_vals, np.array([0, 0, 0, 0, 0]))
