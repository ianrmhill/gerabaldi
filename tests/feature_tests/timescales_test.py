import pytest
from datetime import timedelta

import gerabaldi
from gerabaldi.models import *
from gerabaldi.exceptions import UserConfigError

HOURS_PER_YEAR = 8760
SECONDS_PER_HOUR = 3600
SECONDS_PER_MILLISECOND = 0.001
SECONDS_PER_YEAR = 31536000
CELSIUS_TO_KELVIN = 273.15
NUM_DEVICES = 5


def test_stress():
    with pytest.raises(UserConfigError):
        StrsSpec(conditions={'temp': 1}, duration=99, time_unit='test_invalid')

    stress = StrsSpec(conditions={'temp': 1}, duration=99, time_unit='h')
    assert stress.duration == timedelta(hours=99)
    stress = StrsSpec(conditions={'temp': 1}, duration=99, time_unit='hours')
    assert stress.duration == timedelta(hours=99)

    stress = StrsSpec(conditions={'temp': 1}, duration=99, time_unit='s')
    assert stress.duration == timedelta(seconds=99)
    stress = StrsSpec(conditions={'temp': 1}, duration=99, time_unit='seconds')
    assert stress.duration == timedelta(seconds=99)

    stress = StrsSpec(conditions={'temp': 1}, duration=99, time_unit='ms')
    assert stress.duration == timedelta(milliseconds=99)
    stress = StrsSpec(conditions={'temp': 1}, duration=99, time_unit='milliseconds')
    assert stress.duration == timedelta(milliseconds=99)

    stress = StrsSpec(conditions={'temp': 1}, duration=99, time_unit='us')
    assert stress.duration == timedelta(microseconds=99)
    stress = StrsSpec(conditions={'temp': 1}, duration=99, time_unit='microseconds')
    assert stress.duration == timedelta(microseconds=99)

    stress = StrsSpec(conditions={'temp': 1}, duration=timedelta(hours=99))
    assert stress.duration == timedelta(hours=99)


def test_append_steps():
    test = TestSpec()
    stress = StrsSpec(conditions={'temp': 1}, duration=1, time_unit='h')
    with pytest.raises(UserConfigError):
        test.append_steps(steps=stress, loop_for_duration=3, time_unit='test_invalid')

    test.append_steps(steps=stress, loop_for_duration=3, time_unit='h')
    assert len(test.steps) == 3 / 1

    test = TestSpec()
    stress = StrsSpec(conditions={'temp': 1}, duration=1, time_unit='s')
    test.append_steps(steps=stress, loop_for_duration=3, time_unit='h')
    assert len(test.steps) == 3 / 1 * SECONDS_PER_HOUR

    test = TestSpec()
    stress = StrsSpec(conditions={'temp': 1}, duration=1, time_unit='h')
    test.append_steps(steps=stress, loop_for_duration=timedelta(hours=3))
    assert len(test.steps) == 3 / 1


def test_export_to_json():
    meas_spec = MeasSpec({'example_prm': NUM_DEVICES}, {'temp': 25 + CELSIUS_TO_KELVIN})
    strs_spec = StrsSpec({'temp': 125 + CELSIUS_TO_KELVIN}, 100)
    test_spec = TestSpec([meas_spec, strs_spec, meas_spec])
    test_env = PhysTestEnv()

    def ex_eqn(time, temp, a):
        return time * -a * temp

    dev_mdl = DeviceMdl({'example_prm': DegPrmMdl({'linear': DegMechMdl(ex_eqn, a=LatentVar(Normal(1e-3, 2e-4)))})})
    report = gerabaldi.simulate(test_spec, dev_mdl, test_env)

    with pytest.raises(UserConfigError):
        report.export_to_json(time_unit='test_invalid')

    report_json = report.export_to_json(time_unit='h')
    assert report_json['Time Units'] == 'hours'

    report_json = report.export_to_json(time_unit='s')
    assert report_json['Time Units'] == 'seconds'

    report_json = report.export_to_json(time_unit='ms')
    assert report_json['Time Units'] == 'milliseconds'

    report_json = report.export_to_json(time_unit='us')
    assert report_json['Time Units'] == 'microseconds'


def test_time_unit_propagation():
    deg_mech_0 = DegMechMdl(mdl_name='deg_mech_0', time_unit='h')
    deg_mech_1 = DegMechMdl(mdl_name='deg_mech_1', time_unit='s')
    deg_mech_2 = DegMechMdl(mdl_name='deg_mech_2')
    fail_mech_0 = FailMechMdl(mdl_name='fail_mech_0', time_unit='ms')
    fail_mech_1 = FailMechMdl(mdl_name='fail_mech_1')

    deg_mech_mdls = {
        'deg_mech_0': deg_mech_0,
        'deg_mech_1': deg_mech_1,
        'deg_mech_2': deg_mech_2,
        'fail_mech_0': fail_mech_0,
        'fail_mech_1': fail_mech_1,
    }
    prm_mdl_0 = DegPrmMdl(deg_mech_mdls=deg_mech_mdls, time_unit='us')

    deg_mech_3 = DegMechMdl(mdl_name='deg_mech_3')
    prm_mdl_1 = DegPrmMdl(deg_mech_mdls=deg_mech_3)

    dev_mdl_0 = DeviceMdl(prm_mdls={'prm_mdl_0': prm_mdl_0, 'prm_mdl_1': prm_mdl_1},
                          time_unit='s', name='dev_mdl_0')

    deg_mech_0_new = DegMechMdl(mdl_name='deg_mech_0', time_unit='h')
    deg_mech_1_new = DegMechMdl(mdl_name='deg_mech_1', time_unit='s')
    deg_mech_2_new = DegMechMdl(mdl_name='deg_mech_2')
    fail_mech_0_new = FailMechMdl(mdl_name='fail_mech_0', time_unit='ms')
    fail_mech_1_new = FailMechMdl(mdl_name='fail_mech_1')

    deg_mech_mdls_new = {
        'deg_mech_0': deg_mech_0_new,
        'deg_mech_1': deg_mech_1_new,
        'deg_mech_2': deg_mech_2_new,
        'fail_mech_0': fail_mech_0_new,
        'fail_mech_1': fail_mech_1_new,
    }
    prm_mdl_0_new = DegPrmMdl(deg_mech_mdls=deg_mech_mdls_new, time_unit='d')

    deg_mech_3_new = DegMechMdl(mdl_name='deg_mech_3')
    prm_mdl_1_new = DegPrmMdl(deg_mech_mdls=deg_mech_3_new)

    dev_mdl_1 = DeviceMdl(prm_mdls={'prm_mdl_0': prm_mdl_0_new, 'prm_mdl_1': prm_mdl_1_new}, name='dev_mdl_1')

    assert dev_mdl_0.prm_mdl('prm_mdl_0').mech_mdl('deg_mech_0').time_unit == 'h'
    assert dev_mdl_0.prm_mdl('prm_mdl_0').mech_mdl('deg_mech_1').time_unit == 's'
    assert dev_mdl_0.prm_mdl('prm_mdl_0').mech_mdl('deg_mech_2').time_unit == 'us'
    assert dev_mdl_0.prm_mdl('prm_mdl_0').mech_mdl('fail_mech_0').time_unit == 'ms'
    assert dev_mdl_0.prm_mdl('prm_mdl_0').mech_mdl('fail_mech_1').time_unit == 'us'
    assert dev_mdl_0.prm_mdl('prm_mdl_1').mech_mdl('deg_mech_3').time_unit == 's'

    assert dev_mdl_1.prm_mdl('prm_mdl_0').mech_mdl('deg_mech_0').time_unit == 'h'
    assert dev_mdl_1.prm_mdl('prm_mdl_0').mech_mdl('deg_mech_1').time_unit == 's'
    assert dev_mdl_1.prm_mdl('prm_mdl_0').mech_mdl('deg_mech_2').time_unit == 'd'
    assert dev_mdl_1.prm_mdl('prm_mdl_0').mech_mdl('fail_mech_0').time_unit == 'ms'
    assert dev_mdl_1.prm_mdl('prm_mdl_0').mech_mdl('fail_mech_1').time_unit == 'd'
    assert dev_mdl_1.prm_mdl('prm_mdl_1').mech_mdl('deg_mech_3').time_unit == 'h'


def test_time_unit():
    def eqn(a, temp, time):
        return a * time * temp

    dev_mdl = DeviceMdl(
        DegPrmMdl(
            DegMechMdl(mech_eqn=eqn, mdl_name='test_mdl', time_unit='s', a=LatentVar(deter_val=1)), prm_name='current',
        ),
    )
    meas = MeasSpec({'current': 1}, {'temp': 25})
    strs1 = StrsSpec({'temp': 1}, timedelta(hours=1))
    strs2 = StrsSpec({'temp': 2}, timedelta(hours=1))
    test = TestSpec([strs1, meas, strs2, meas])
    test_env = PhysTestEnv()
    report = gerabaldi.simulate(test, dev_mdl, test_env)
    assert round(report.measurements['measured'][0]) == 3600
    assert round(report.measurements['measured'][1]) == 10800

    dev_mdl = DeviceMdl(
        DegPrmMdl(
            DegMechMdl(mech_eqn=eqn, mdl_name='test_mdl', a=LatentVar(deter_val=1)), prm_name='current', time_unit='ms',
        ),
    )
    meas = MeasSpec({'current': 1}, {'temp': 25})
    strs1 = StrsSpec({'temp': 1}, 1, time_unit='h')
    strs2 = StrsSpec({'temp': 2}, 1000000, time_unit='ms')
    test = TestSpec([strs1, meas, strs2, meas])
    test_env = PhysTestEnv()
    report = gerabaldi.simulate(test, dev_mdl, test_env)
    assert round(report.measurements['measured'][0]) == 3600000
    assert round(report.measurements['measured'][1]) == 5600000

    dev_mdl = DeviceMdl(
        DegPrmMdl(DegMechMdl(mech_eqn=eqn, mdl_name='test_mdl', a=LatentVar(deter_val=1)), prm_name='current'),
        time_unit='d',
    )
    meas = MeasSpec({'current': 1}, {'temp': 25})
    strs1 = StrsSpec({'temp': 1}, 365, time_unit='d')
    strs2 = StrsSpec({'temp': 2}, timedelta(days=356))
    test = TestSpec([strs1, meas, strs2, meas])
    test_env = PhysTestEnv()
    report = gerabaldi.simulate(test, dev_mdl, test_env)
    assert round(report.measurements['measured'][0]) == 365
    assert round(report.measurements['measured'][1]) == 1077

    dev_mdl = DeviceMdl(DegPrmMdl(DegMechMdl(mech_eqn=eqn, mdl_name='test', a=LatentVar(deter_val=1)), prm_name='i'))
    meas = MeasSpec({'i': 1}, {'temp': 25})
    strs1 = StrsSpec({'temp': 1}, 3600, time_unit='s')
    strs2 = StrsSpec({'temp': 2}, timedelta(hours=1))
    test = TestSpec([strs1, meas, strs2, meas])
    test_env = PhysTestEnv()
    report = gerabaldi.simulate(test, dev_mdl, test_env)
    assert round(report.measurements['measured'][0]) == 1
    assert round(report.measurements['measured'][1]) == 3
