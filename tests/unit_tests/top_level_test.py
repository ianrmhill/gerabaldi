# Copyright (c) 2024 Ian Hill
# SPDX-License-Identifier: Apache-2.0

"""Tests to ensure the degradation simulator module works according to the intended simulation flow and feature set"""

import pytest
from datetime import timedelta
import numpy as np
import pandas as pd

import gerabaldi
from gerabaldi.sim import _sim_meas_step, _sim_stress_step
from gerabaldi.models import *

CELSIUS_TO_KELVIN = 273.15
BOLTZMANN_CONST_EV = 8.617e-5
SECONDS_PER_HOUR = 3600


def test_init_state_basic():
    # Start with bare minimum specification to test that mainly defaults results in the expected initial state
    def silly_eqn(time, rhyme):
        return rhyme * time

    deg_model = DeviceMdl(
        DegPrmMdl(DegMechMdl(silly_eqn, mdl_name='some mech', rhyme=LatentVar(deter_val=2)), prm_name='some param'),
    )
    init_state = gerabaldi.gen_init_state(deg_model, dev_counts={'some param': 10})
    # Check all the defaults
    assert type(init_state) is SimState
    assert init_state.latent_var_vals['some param']['some mech']['rhyme'][0][0][0] == 2
    assert init_state.curr_prm_vals['some param'][0][0][9] == 0
    assert init_state.init_prm_vals['some param'][0][0][8] == 0
    assert init_state.elapsed == timedelta()

    # Now for testing general argument options
    def silly_2(base):
        return base

    deg_model = DeviceMdl(
        {
            'freq': DegPrmMdl(
                DegMechMdl(silly_eqn, mdl_name='some mech'),
                InitValMdl(
                    init_val=LatentVar(
                        deter_val=1,
                        dev_vrtn_mdl=Normal(1, 0.01, test_seed=2222),
                        lot_vrtn_mdl=Normal(1, 0.01, test_seed=3333),
                    ),
                ),
            ),
            'gain': DegPrmMdl(
                DegMechMdl(silly_eqn, mdl_name='some other mech'),
                cond_shift_mdl=CondShiftMdl(
                    silly_2,
                    base=LatentVar(deter_val=0, vrtn_type='offset', dev_vrtn_mdl=Normal(1, 0.02, test_seed=4444)),
                ),
                init_val_mdl=InitValMdl(
                    init_val=LatentVar(deter_val=0, vrtn_type='offset', chp_vrtn_mdl=Normal(-0.2, 2, test_seed=5555)),
                ),
            ),
        },
    )
    # Run the initial state generator
    init_state = gerabaldi.gen_init_state(deg_model, dev_counts={'freq': 10, 'gain': 5}, num_chps=3, num_lots=4)
    assert round(init_state.latent_var_vals['gain']['cond']['base'][3][2][2], 5) == 1.01015
    assert round(init_state.init_prm_vals['freq'][0][0][5], 5) == 1.03063
    assert round(init_state.curr_prm_vals['gain'][1][2][2], 5) == -1.99403


def test_sim_stress_step_basic():
    # Basic execution and type checking first
    stress_step = StrsSpec({'temp': 125}, timedelta(hours=20))
    meas_step = MeasSpec({'current': 3}, {'temp': 25})
    test_spec = TestSpec([meas_step, stress_step, meas_step], 4, 2)

    # Setup initial test state
    def linear_deg(scale_factor, temp, time):
        return scale_factor * np.log(temp * time)

    deg_model = DeviceMdl(
        {
            'current': DegPrmMdl(
                DegMechMdl(linear_deg, scale_factor=LatentVar(deter_val=2e-4), mdl_name='mech1'),
                InitValMdl(init_val=LatentVar(deter_val=4e-3)),
            ),
        },
    )
    phys_state = gerabaldi.gen_init_state(
        deg_model, test_spec.calc_samples_needed(), test_spec.num_chps, test_spec.num_lots,
    )
    test_env = PhysTestEnv()
    report = SimReport(test_spec)

    # Run the function being tested
    _sim_stress_step(stress_step, phys_state, deg_model, test_env, report, 1)

    # Check types and returned values
    assert type(phys_state) is SimState
    assert round(phys_state.elapsed.total_seconds() / SECONDS_PER_HOUR, 2) == 20.00
    assert round(phys_state.curr_prm_vals['current'][0][1][2], 5) == 0.00556
    assert report.test_summary['step name'][0] == 'unspecified'
    assert report.test_summary['step type'][0] == 'stress'
    assert report.test_summary['step number'][0] == 1
    assert report.test_summary['temp'][0] == 125

    # Run again to check that the equivalent time back-calculations are working fine
    _sim_stress_step(stress_step, phys_state, deg_model, test_env, report, 2)
    assert report.test_summary['duration'][1] == timedelta(hours=20)
    assert round(phys_state.curr_prm_vals['current'][0][1][2], 5) == 0.00570


def test_sim_meas_step_basic(sequential_var):
    # Basic execution and type checking first
    meas_step = MeasSpec({'temp': 1, 'current': 8}, {'temp': 125}, 'test_meas_spec')
    test_spec = TestSpec([meas_step], num_chps=2)

    def dummy_eqn(temp):
        return temp * 2

    def cond_eqn(temp, c):
        return c - (1e-4 * (temp - 25))

    sim_model = DeviceMdl(
        DegPrmMdl(
            DegMechMdl(dummy_eqn, mdl_name='dummy'),
            InitValMdl(init_val=LatentVar(deter_val=4e-2, vrtn_type='offset', dev_vrtn_mdl=sequential_var(1, 1e-2))),
            CondShiftMdl(
                cond_eqn, c=LatentVar(deter_val=1e-2, vrtn_type='offset', dev_vrtn_mdl=sequential_var(0, 1e-3)),
            ),
            prm_name='current',
        ),
    )
    test_env = PhysTestEnv(
        meas_instms={'temp': MeasInstrument('temp', precision=2), 'current': MeasInstrument('current', precision=5)},
    )

    deg_state = gerabaldi.gen_init_state(sim_model, dev_counts={'current': 10}, elapsed_time=10, num_chps=2)
    report = SimReport(test_spec)

    _sim_meas_step(meas_step, deg_state, sim_model, test_env, report, 3)
    assert type(report.measurements) is pd.DataFrame
    assert np.round(report.measurements['measured'][0], 5) == 120
    assert report.measurements['param'][0] == 'temp'
    assert np.round(report.measurements['measured'][2], 5) == 1.051
    assert report.measurements['chip #'][16] == 1
    assert report.measurements['step #'][16] == 3
    assert np.round(report.measurements['measured'][12], 5) == 1.183
    assert report.test_summary['step name'][0] == 'test_meas_spec'
    assert report.test_summary['step type'][0] == 'measure'
    assert report.test_summary['step number'][0] == 3


def test_wearout_test_sim_basic():
    # Test a basic case first to ensure flow is working correctly
    meas = MeasSpec({'temp': 1, 'current': 3}, {'temp': 40}, name='cold but not too cold')
    strs = StrsSpec({'temp': 125}, timedelta(hours=10), name='Gimme 10')
    test = TestSpec(
        [meas, strs, meas, strs, meas], description='Test test test test', name='Test!', num_chps=3, num_lots=2,
    )

    def log_deg(scale_factor, temp, time):
        return scale_factor * np.log(temp * time)

    deg_model = DeviceMdl(
        DegPrmMdl(
            DegMechMdl(
                log_deg,
                scale_factor=LatentVar(deter_val=1e-4, chp_vrtn_mdl=Normal(1, 2e-4, test_seed=777)),
                mdl_name='test_mech',
            ),
            InitValMdl(
                init_val=LatentVar(deter_val=4e-3, dev_vrtn_mdl=Normal(0, 2e-3, test_seed=666), vrtn_type='offset'),
            ),
            prm_name='current',
        ),
    )

    test_env = PhysTestEnv()
    init_state = gerabaldi.gen_init_state(deg_model, dev_counts={'current': 3}, num_chps=3, num_lots=2)

    report = gerabaldi.simulate(test, deg_model, test_env, init_state)
    assert type(report) is SimReport
    assert len(report.measurements['param']) == 57
    assert len(report.test_summary['step number']) == 5
    assert round(report.measurements['measured'][40], 5) == 0.00560
    assert round(report.measurements['measured'][26], 5) == 0.00783


def test_vts_paper_example_1():
    # Test that the simulator is still behaving as specified in the VTS paper
    num_devices = 2
    num_chips = 2
    num_lots = 2

    ### 1. Define the qualification test procedures ###########
    full_meas = MeasSpec({'amp_gain': num_devices}, {'temp': 30 + CELSIUS_TO_KELVIN, 'vdd': 0.86}, 'All')
    # First is an HTOL test
    htol_stress = StrsSpec({'temp': 125 + CELSIUS_TO_KELVIN, 'vdd': 0.92}, 50, 'HTOL Stress')
    htol_test = TestSpec([full_meas], num_chips, num_lots, name='HTOL Similar Test')
    htol_test.append_steps([htol_stress, full_meas], 1000)
    # Next is an LTOL test
    ltol_stress = StrsSpec({'temp': -10 + CELSIUS_TO_KELVIN, 'vdd': 0.92}, 50, 'Low Temperature Stress')
    ltol_test = TestSpec([full_meas], num_chips, num_lots, name='LTOL Similar Test')
    ltol_test.append_steps([ltol_stress, full_meas], 1000)
    # Third is a more complex test to showcase the rich test support of the simulator
    ramp_cycle_relax_interval = StrsSpec({'temp': 30 + CELSIUS_TO_KELVIN, 'vdd': 0.86}, 10, 'Ramp Cycle Relax')
    ramp_cycle_stress_1 = StrsSpec({'temp': 100 + CELSIUS_TO_KELVIN, 'vdd': 0.88}, 90, 'Ramp Cycle Stress 1')
    ramp_cycle_stress_2 = StrsSpec({'temp': 110 + CELSIUS_TO_KELVIN, 'vdd': 0.90}, 90, 'Ramp Cycle Stress 2')
    ramp_cycle_stress_3 = StrsSpec({'temp': 120 + CELSIUS_TO_KELVIN, 'vdd': 0.92}, 90, 'Ramp Cycle Stress 3')
    ramp_cycle_stress_4 = StrsSpec({'temp': 130 + CELSIUS_TO_KELVIN, 'vdd': 0.94}, 90, 'Ramp Cycle Stress 4')
    ramp_cycle_stress_5 = StrsSpec({'temp': 140 + CELSIUS_TO_KELVIN, 'vdd': 0.96}, 90, 'Ramp Cycle Stress 5')
    ramp_cycle_stress_6 = StrsSpec({'temp': 150 + CELSIUS_TO_KELVIN, 'vdd': 0.98}, 90, 'Ramp Cycle Stress 6')
    ramp_cycle_stress_7 = StrsSpec({'temp': 160 + CELSIUS_TO_KELVIN, 'vdd': 1.00}, 90, 'Ramp Cycle Stress 7')
    ramp_cycle_stress_8 = StrsSpec({'temp': 170 + CELSIUS_TO_KELVIN, 'vdd': 1.02}, 90, 'Ramp Cycle Stress 8')
    ramp_cycle_stress_9 = StrsSpec({'temp': 180 + CELSIUS_TO_KELVIN, 'vdd': 1.04}, 90, 'Ramp Cycle Stress 9')
    ramp_cycle_stress_10 = StrsSpec({'temp': 190 + CELSIUS_TO_KELVIN, 'vdd': 1.06}, 90, 'Ramp Cycle Stress 10')
    ramp_failure_test = TestSpec(
        [
            full_meas,
            ramp_cycle_stress_1,
            full_meas,
            ramp_cycle_relax_interval,
            full_meas,
            ramp_cycle_stress_2,
            full_meas,
            ramp_cycle_relax_interval,
            full_meas,
            ramp_cycle_stress_3,
            full_meas,
            ramp_cycle_relax_interval,
            full_meas,
            ramp_cycle_stress_4,
            full_meas,
            ramp_cycle_relax_interval,
            full_meas,
            ramp_cycle_stress_5,
            full_meas,
            ramp_cycle_relax_interval,
            full_meas,
            ramp_cycle_stress_6,
            full_meas,
            ramp_cycle_relax_interval,
            full_meas,
            ramp_cycle_stress_7,
            full_meas,
            ramp_cycle_relax_interval,
            full_meas,
            ramp_cycle_stress_8,
            full_meas,
            ramp_cycle_relax_interval,
            full_meas,
            ramp_cycle_stress_9,
            full_meas,
            ramp_cycle_relax_interval,
            full_meas,
            ramp_cycle_stress_10,
            full_meas,
            ramp_cycle_relax_interval,
            full_meas,
        ],
        num_chips,
        num_lots,
        name='Ramp to Failure Test',
    )

    ### 2. Define the test/field environment ###
    test_env = PhysTestEnv(
        env_vrtns={
            'temp': EnvVrtnMdl(dev_vrtn_mdl=Normal(0, 0.05, test_seed=123), chp_vrtn_mdl=Normal(0, 0.2, test_seed=234)),
            'vdd': EnvVrtnMdl(
                dev_vrtn_mdl=Normal(0, 0.0003, test_seed=345), chp_vrtn_mdl=Normal(0, 0.0005, test_seed=456),
            ),
        },
        meas_instms={'temp': MeasInstrument(), 'vdd': MeasInstrument(), 'amp_gain': MeasInstrument()},
    )

    ### 3. Define the device vth degradation model ###
    # Model provided in JEP122H as generally used NBTI degradation model, equation 5.3.1
    def bti_vth_shift_empirical(a_0, e_aa, temp, vdd, alpha, time, n):
        return a_0 * np.exp(e_aa / (BOLTZMANN_CONST_EV * temp)) * (vdd**alpha) * (time**n)

    # HCI model from Takeda and Suzuki with temperature dependence added to enrich the demo, DOI: 10.1109/EDL.1983.25667

    def hci_vth_shift_empirical(time, vdd, temp, a_0, n, t_0, beta, alpha):
        return a_0 * np.exp(-alpha / vdd) * (t_0 * (temp**-beta)) * (time**n)

    # Just use a simple gm*ro gain equation to relate our threshold voltage to an analog circuit parameter

    def v_th_eqn(init, bti, hci, cond):
        return init + bti + hci + cond

    def amp_gain_eqn(v_th, u_n, c_ox, w, l, v_e, vdd):
        v_gs_1 = 0.54 * vdd
        v_gs_2 = 0.69 * vdd
        i_d_1 = 0.5 * (u_n * c_ox * (w / l)) * ((v_gs_1 - v_th) ** 2)
        i_d_2 = 0.5 * (u_n * c_ox * (w / l)) * ((v_gs_2 - v_th) ** 2)
        g_m = (2 * i_d_1) / (v_gs_1 - v_th)
        r_o = (v_e * l) / i_d_2
        return g_m * r_o

    dev_mdl = DeviceMdl(
        {
            'v_th': DegPrmMdl(
                deg_mech_mdls={
                    'bti': DegMechMdl(
                        bti_vth_shift_empirical,
                        a_0=LatentVar(deter_val=0.006, dev_vrtn_mdl=Normal(1, 0.0005, test_seed=678)),
                        e_aa=LatentVar(
                            deter_val=-0.05,
                            dev_vrtn_mdl=Normal(1, 0.0002, test_seed=789),
                            chp_vrtn_mdl=Normal(1, 0.0003, test_seed=890),
                            lot_vrtn_mdl=Normal(1, 0.0001, test_seed=901),
                        ),
                        alpha=LatentVar(
                            deter_val=9.5,
                            dev_vrtn_mdl=Normal(1, 0.002, test_seed=987),
                            chp_vrtn_mdl=Normal(1, 0.005, test_seed=876),
                        ),
                        n=LatentVar(deter_val=0.4, dev_vrtn_mdl=Normal(1, 0.0005, test_seed=765)),
                    ),
                    'hci': DegMechMdl(
                        hci_vth_shift_empirical,
                        a_0=LatentVar(deter_val=0.1, dev_vrtn_mdl=Normal(1, 0.004, test_seed=654)),
                        n=LatentVar(deter_val=0.62, dev_vrtn_mdl=Normal(1, 0.003, test_seed=543)),
                        alpha=LatentVar(
                            deter_val=7.2,
                            dev_vrtn_mdl=Normal(1, 0.03, test_seed=432),
                            chp_vrtn_mdl=Normal(1, 0.04, test_seed=321),
                        ),
                        beta=LatentVar(
                            deter_val=1.1,
                            dev_vrtn_mdl=Normal(1, 0.002, test_seed=135),
                            chp_vrtn_mdl=Normal(1, 0.001, test_seed=246),
                            lot_vrtn_mdl=Normal(1, 0.01, test_seed=357),
                        ),
                        t_0=LatentVar(deter_val=500, dev_vrtn_mdl=Normal(1, 0.001, test_seed=468)),
                    ),
                },
                init_val_mdl=InitValMdl(
                    init_val=LatentVar(
                        deter_val=0.42,
                        dev_vrtn_mdl=Normal(0, 0.0001, test_seed=579),
                        chp_vrtn_mdl=Normal(0, 0.0002, test_seed=680),
                        lot_vrtn_mdl=Normal(0, 0.0003, test_seed=791),
                        vrtn_type='offset',
                    ),
                ),
                compute_eqn=v_th_eqn,
            ),
            'amp_gain': CircPrmMdl(
                amp_gain_eqn,
                u_n=LatentVar(deter_val=1),
                c_ox=LatentVar(deter_val=1),
                w=LatentVar(deter_val=8),
                l=LatentVar(deter_val=2),
                v_e=LatentVar(deter_val=5),
            ),
        },
    )

    ### 4. Simulate the stress tests at different temperatures ###
    test_list = ['CP:RC', 'CP:HTOL', 'CP:LTOL', 'NP:RC', 'NP:HTOL', 'NP:LTOL']
    # First simulate the expected results for the current generation of products
    results = {
        test_list[0]: gerabaldi.simulate(ramp_failure_test, dev_mdl, test_env),
        test_list[1]: gerabaldi.simulate(htol_test, dev_mdl, test_env),
        test_list[2]: gerabaldi.simulate(ltol_test, dev_mdl, test_env),
    }
    # Now simulate for the upcoming process, need to adjust the initial value lot variability and 'alpha' in each model
    dev_mdl.prm_mdl('v_th').init_mdl.latent_var('init_val').lot_vrtn_mdl = Normal(0, 0.0006, test_seed=787)
    dev_mdl.prm_mdl('v_th').mech_mdl('bti').latent_var('alpha').chp_vrtn_mdl = Normal(1, 0.01, test_seed=777)
    dev_mdl.prm_mdl('v_th').mech_mdl('hci').latent_var('alpha').chp_vrtn_mdl = Normal(1, 0.08, test_seed=767)
    results[test_list[3]] = gerabaldi.simulate(ramp_failure_test, dev_mdl, test_env)
    results[test_list[4]] = gerabaldi.simulate(htol_test, dev_mdl, test_env)
    results[test_list[5]] = gerabaldi.simulate(ltol_test, dev_mdl, test_env)

    # Now validate a few of the results to ensure reproducibility
    assert round(results['CP:HTOL'].measurements.loc[11]['measured'], 2) == 28.24
    assert round(results['CP:HTOL'].measurements.loc[147]['measured'], 2) == 24.72
    assert results['NP:RC'].measurements.loc[7]['chip #'] == 1
    assert results['NP:RC'].measurements.loc[7]['device #'] == 1
    assert round(results['NP:RC'].measurements.loc[12]['measured'], 2) == 28.40
    assert round(results['NP:RC'].measurements.loc[147]['measured'], 2) == 17.36


def test_vts_paper_example_2():
    # Test that the simulator is still behaving as specified in the VTS paper
    num_samples = 10
    test_len = 24 * 7 * 52 * 0.125
    sampler = Uniform(test_seed=808)

    def defect_generator_demo_model(time, temp, v_g, t_diel, c, bond_strength, thermal_dist):
        defect_gen_prob = c * (v_g / t_diel) * ((temp**thermal_dist) / bond_strength)
        return 1 if sampler.sample() < (time * defect_gen_prob) else 0

    def oxide_failed(
        init,
        cond,
        threshold,
        dfct0,
        dfct1,
        dfct2,
        dfct3,
        dfct4,
        dfct5,
        dfct6,
        dfct7,
        dfct8,
        dfct9,
        dfct10,
        dfct11,
    ):
        # We physically model a transistor oxide layer with 12 possible defect locations
        layout = np.array(
            [dfct0, dfct1, dfct2, dfct3, dfct4, dfct5, dfct6, dfct7, dfct8, dfct9, dfct10, dfct11],
        ).reshape((3, 4))
        oxide = pd.DataFrame(layout).applymap(lambda deg: 1 if deg > threshold else 0)
        conductive_col = False
        for i in range(len(oxide.columns)):
            if (oxide[i] == 1).all():
                conductive_col = True
                break
        return 0 if conductive_col or init == 0 else 1

    def single_test(step_val, test, env):
        defect_mdl = FailMechMdl(
            defect_generator_demo_model,
            c=LatentVar(deter_val=1e-3),
            t_diel=LatentVar(deter_val=3),
            bond_strength=LatentVar(deter_val=200),
            thermal_dist=LatentVar(deter_val=step_val),
        )
        defect_dict = {'dfct' + str(i): defect_mdl for i in range(12)}
        tddb_model = DeviceMdl(
            DegPrmMdl(
                prm_name='tddb',
                init_val_mdl=InitValMdl(init_val=LatentVar(deter_val=1)),
                deg_mech_mdls=defect_dict,
                compute_eqn=oxide_failed,
                array_computable=False,
                threshold=LatentVar(deter_val=0.5),
            ),
        )
        return gerabaldi.simulate(test, tddb_model, env)

    #### 1. Define the test procedure ###########
    wkly_moderate_use = StrsSpec({'temp': 65 + CELSIUS_TO_KELVIN, 'v_g': 0.9}, 30, 'Moderate Loading')
    wkly_intensive_use = StrsSpec({'temp': 88 + CELSIUS_TO_KELVIN, 'v_g': 0.92}, 16, 'Heavy Loading')
    wkly_idle_use = StrsSpec({'temp': 30 + CELSIUS_TO_KELVIN, 'v_g': 0.87}, 122, 'Idle State')
    check_fail_states = MeasSpec({'tddb': num_samples}, {'temp': 35 + CELSIUS_TO_KELVIN, 'v_g': 0.9})

    field_use_sim = TestSpec(
        [check_fail_states],
        name='Field Use Sim',
        description='Test spec designed to represent a real-world use scenario for a consumer'
        'device with downtime and higher stress periods.',
    )
    with pytest.raises(UserWarning):
        field_use_sim.append_steps([wkly_moderate_use, wkly_intensive_use, wkly_idle_use, check_fail_states], test_len)

    #### 2. Define the test/field environment ###
    field_env = PhysTestEnv(
        env_vrtns={
            'temp': EnvVrtnMdl(dev_vrtn_mdl=Gamma(2, 1, test_seed=909), chp_vrtn_mdl=Normal(0, 0.3, test_seed=707)),
            'v_g': EnvVrtnMdl(
                dev_vrtn_mdl=Normal(0, 0.008, test_seed=505), chp_vrtn_mdl=Normal(0, 0.003, test_seed=606),
            ),
        },
        meas_instms={'temp': MeasInstrument(error=Normal(0, 1, test_seed=401), precision=4, meas_lims=(-40, 150))},
    )

    ### 4. Repeatedly simulate with slight shifts to the latent parameter '' to examine sensitivity ###
    schmoo_list = [1.05, 1.075, 1.1, 1.125, 1.15]
    results = [single_test(step, field_use_sim, field_env) for step in schmoo_list]

    # Now check a few values to ensure reproducibility
    assert results[1].measurements.loc[65]['measured'] == 1.0
    assert results[1].measurements.loc[75]['measured'] == 0.0
    assert results[4].measurements.loc[50]['measured'] == 1.0
    assert results[4].measurements.loc[60]['measured'] == 0.0
    assert results[3].measurements.loc[33]['measured'] == 1.0
    assert results[3].measurements.loc[43]['measured'] == 0.0
