# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

"""Test set ensuring correct functionality of the classes used to model physical degradation processes"""

import pytest # noqa: PackageNotInRequirements
import numpy as np

from gerabaldi.models.devices import *
from gerabaldi.models.devices import LatentMdl # noqa: ImportedFunctionNotInAll
from gerabaldi.models.random_vars import Deterministic
from gerabaldi.exceptions import UserConfigError, InvalidTypeError


def test_latent_var(sequential_var):
    # First test error cases
    with pytest.raises(UserConfigError):
        _ = LatentVar()
    with pytest.raises(InvalidTypeError):
        _ = LatentVar(Deterministic(2), vrtn_type='invalid')

    # Now test basic probabilistic definition behaviour
    ltnt = LatentVar(sequential_var(0.2, 0.2))
    assert (type(ltnt.chp_vrtn_mdl), type(ltnt.lot_vrtn_mdl), ltnt.name, ltnt.deter_val, ltnt.vrtn_type) == \
           (Deterministic, Deterministic, None, None, 'scaling')
    assert ltnt.gen_latent_vals() == np.full((1, 1, 1), 0.2)
    assert np.allclose(ltnt.gen_latent_vals(2, 2, 2), np.array([[[0.2, 0.4], [0.6, 0.8]], [[1.0, 1.2], [1.4, 1.6]]]))
    ltnt = LatentVar(sequential_var(1, 1), sequential_var(1, 1), sequential_var(1, 1))
    assert np.allclose(ltnt.gen_latent_vals(2, 2, 2), np.array([[[1, 2], [6, 8]], [[30, 36], [56, 64]]]))

    # Next test basic deterministic behaviour
    ltnt = LatentVar(deter_val=2)
    assert (type(ltnt.dev_vrtn_mdl), type(ltnt.chp_vrtn_mdl), type(ltnt.lot_vrtn_mdl),
            ltnt.name, ltnt.deter_val, ltnt.vrtn_type) == \
           (Deterministic, Deterministic, Deterministic, None, 2, 'scaling')
    assert ltnt.gen_latent_vals() == np.full((1, 1, 1), 2)
    ltnt = LatentVar(deter_val=0, chp_vrtn_mdl=sequential_var(1, 1), vrtn_type='offset')
    assert np.allclose(ltnt.gen_latent_vals(1, 2, 1), np.array([[[1], [2]]]))

    # Now test combined probabilistic + deterministic combinations
    ltnt = LatentVar(sequential_var(), sequential_var(), sequential_var(), 0.2, 'uno', 'offset')
    assert (ltnt.name, ltnt.vrtn_type) == ('uno', 'offset')
    assert np.allclose(ltnt.gen_latent_vals(2, 2, 2), np.array([[[0.2, 1.2], [3.2, 4.2]],
                                                                [[7.2, 8.2], [10.2, 11.2]]]))
    ltnt.vrtn_type = 'scaling'
    assert np.allclose(ltnt.gen_latent_vals(2, 2, 2), np.array([[[0.0, 0.0], [0.0, 0.0]],
                                                                [[1.6, 2.0], [3.6, 4.2]]]))


def test_latent_mdl(sequential_var):
    # First test the defaults
    mdl = LatentMdl()
    assert mdl.name is None
    assert mdl.compute(3) == 3
    assert mdl.gen_latent_vals() == {}
    def sample_eqn(x, y): return x - y
    mdl = LatentMdl(sample_eqn, 'TestModel', y=LatentVar(deter_val=4))
    assert mdl.name == 'TestModel'
    assert mdl.compute(2, 3) == -1
    assert np.allclose(mdl.gen_latent_vals(2, 1, 2)['y'], np.array([[[4, 4]], [[4, 4]]]))


def test_init_val_mdl_basic(sequential_var):
    mdl = InitValMdl(init_val=LatentVar(deter_val=3))
    assert (type(mdl.latent_var('init_val').dev_vrtn_mdl), type(mdl.latent_var('init_val').chp_vrtn_mdl)) == \
           (Deterministic, Deterministic)
    assert mdl.name is None
    assert np.allclose(np.full(8, 3).reshape((2, 2, 2)), mdl.gen_init_vals(2, 2, 2))

    def weird_one(base, auto_fail): return np.where(auto_fail > 0, base, 0)
    mdl = InitValMdl(init_val_eqn=weird_one, mdl_name='Very Unusual',
                     base=LatentVar(deter_val=0.6, chp_vrtn_mdl=sequential_var(0.1, 0.1), vrtn_type='offset'),
                     auto_fail=LatentVar(deter_val=0, dev_vrtn_mdl=sequential_var(-2, 1), vrtn_type='offset'))
    assert mdl.name == 'Very Unusual'
    assert np.allclose(mdl.gen_init_vals(2, 2, 2),
                       np.array([[[0, 0], [0, 0.8]], [[0.9, 0.9], [1, 1]]]))


def test_conditional_shift_model(sequential_var):
    # First check all the default arguments and behaviour
    mdl = CondShiftMdl()
    assert mdl.gen_latent_vals() == {}
    assert mdl.name is None
    assert mdl.compute() == 0

    # Now test a more interesting model
    def some_shifting(temp, vdd, a, b):
        return (a * temp) + (b * (-vdd))
    mdl = CondShiftMdl(some_shifting, 'TestShift',
                       a=LatentVar(deter_val=0.1, dev_vrtn_mdl=sequential_var(1, 0.1)),
                       b=LatentVar(deter_val=0.5, dev_vrtn_mdl=sequential_var(1, 0.1)))
    assert mdl.name == 'TestShift'
    latents = mdl.gen_latent_vals(3)
    assert np.allclose(latents['a'], [[[0.1, 0.11, 0.12]]])
    assert np.allclose(latents['b'], [[[0.5, 0.55, 0.6]]])
    assert np.allclose(mdl.compute(25, 0.9, **latents), [[[2.05, 2.255, 2.46]]])

    # TODO: Add calc_cond_vals testing


def test_deg_mech_mdl():
    """Bare-bones tests for the MechModel class."""
    def basic_eqn(time, y, z): return time * y * z
    basic_model = DegMechMdl(basic_eqn, x=LatentVar(deter_val=3), y=LatentVar(deter_val=2))

    assert basic_model.name is None
    assert basic_model.unitary == 0
    assert basic_model.compute(-1, 1, z=4) == -4
    basic_model = DegMechMdl(basic_eqn, unitary_val=1, mdl_name='hello')
    assert basic_model.name == 'hello'
    assert basic_model.unitary == 1

    # Next test that the equivalent time back-calculation is working correctly
    latents = {'z': 0.5}
    assert round(basic_model.calc_equiv_strs_time(8, 0, {'y': 4, 'unused': 3}, latents, (1, 1, 1)), 4) == 4


def test_fail_mech_mdl():
    def fail_eqn(time, temp, a): return 1 if time * temp * a > 10 else 0
    mdl = FailMechMdl(fail_eqn, a=LatentVar(Deterministic(1)))
    conds = {'temp': np.array([[[4, 2, 3]]])}
    ltnts = {'a': np.array([[[1, 0.2, 0.1]]])}

    assert mdl.calc_equiv_strs_time(1, 2, {'temp': 3}, {'a': 4}, (1, 1, 3)) == 0.0
    assert np.allclose(mdl.calc_deg_vals(np.array([[[3, 4, 5]]]), np.array([[[0, 1, 0]]]), conds, ltnts, (1, 1, 3)),
                       np.array([[[1, 1, 0]]]))


def test_deg_prm_mdl(sequential_var):
    # Test a basic parameter setup
    def degradation(time, temp, a, b): return (time**a) * (temp**b)
    def shift(temp, vdd, a, b): return a * (temp**b) * vdd
    mdl = DegPrmMdl(DegMechMdl(degradation, mdl_name='sample_mech',
                               a=LatentVar(deter_val=0.3, dev_vrtn_mdl=sequential_var(0, 0.1), vrtn_type='offset'),
                               b=LatentVar(deter_val=0.8, dev_vrtn_mdl=sequential_var(0, 0.1), vrtn_type='offset')),
                    InitValMdl(init_val=LatentVar(deter_val=2, dev_vrtn_mdl=sequential_var(), vrtn_type='offset')),
                    CondShiftMdl(shift,
                                 a=LatentVar(deter_val=0.5, dev_vrtn_mdl=sequential_var(0, 0.1), vrtn_type='offset'),
                                 b=LatentVar(deter_val=1.2, dev_vrtn_mdl=sequential_var(0, 0.1), vrtn_type='offset')))
    latents = mdl.gen_latent_vals(3)
    init_prm_vals = mdl.init_mdl.gen_init_vals(3)
    init_mech_vals = {'sample_mech': np.zeros((1, 1, 3))}
    conditions = {'temp': [[[25, 26, 30]]], 'vdd': [[[0.5, 0.55, 0.55]]]}
    prms, mechs = mdl.calc_deg_vals((1, 1, 3), {'sample_mech': 10}, conditions, init_prm_vals, latents, init_mech_vals)
    assert np.allclose(prms.round(4), [[[28.2031, 50.1494, 98.8683]]])
    prms = mdl.calc_cond_shifted_vals((1, 1, 3), {'temp': 125, 'vdd': 0.6},
                                      np.array([[[28.2031, 50.1494, 98.8683]]]), latents)
    assert np.allclose(prms.round(4), [[[126.6979, 241.7009, 461.0473]]])

    # Test that the individual vs. array computation is equivalent
    mdl.array_compute = False
    prms, mechs = mdl.calc_deg_vals((1, 1, 3), {'sample_mech': 10}, conditions, init_prm_vals, latents, init_mech_vals)
    assert np.allclose(prms.round(4), [[[28.2031, 50.1494, 98.8683]]])
    prms = mdl.calc_cond_shifted_vals((1, 1, 3), {'temp': 125, 'vdd': 0.6},
                                      np.array([[[28.2031, 50.1494, 98.8683]]]), latents)
    assert np.allclose(prms.round(4), [[[126.6979, 241.7009, 461.0473]]])

    # Test equivalent time calculation
    equiv_times = mdl.calc_equiv_strs_times((1, 1, 3), {'sample_mech': [[[26.2031, 47.1494, 94.8683]]]},
                                            conditions, init_mech_vals, latents)
    assert np.allclose(equiv_times['sample_mech'].round(4), [[[10, 10, 10]]])
    conditions = {'temp': 26, 'vdd': 0.55}
    equiv_times = mdl.calc_equiv_strs_times((1, 1, 3), {'sample_mech': [[[26.2031, 47.1494, 94.8683]]]},
                                            conditions, init_mech_vals, latents)
    assert np.allclose(equiv_times['sample_mech'].round(4), [[[9.007, 10, 13.3136]]])
