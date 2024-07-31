# Copyright (c) 2024 Ian Hill
# SPDX-License-Identifier: Apache-2.0

"""Tests for the state classes that are used to track program execution variables"""

from datetime import timedelta

from gerabaldi.models.states import *


def test_test_sim_state():
    # Currently the SimState class only bundles a couple of persisted values together, and thus there's not much
    # to test. Methods may be added in the future as feature set increases.
    state = SimState(
        {'val1': 1, 'val2': 2},
        {'prm1': {'mech1': 0.1, 'mech2': 0.2}, 'prm2': {'mech3': 4}},
        {'latent1': 3, 'latent2': 4},
    )
    assert state.elapsed == timedelta()
    assert state.curr_prm_vals['val1'] == 1
    assert state.init_prm_vals['val2'] == 2
    assert state.curr_deg_mech_vals['prm1']['mech2'] == 0.2
    assert state.latent_var_vals['latent2'] == 4
