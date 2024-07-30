import gerabaldi
import pytest
from datetime import timedelta


from gerabaldi.models.devices import *
from gerabaldi.models import *
from gerabaldi.exceptions import UserConfigError, ArgOverwriteWarning
#from gerabaldi.models.test_specs import StrsSpec
from gerabaldi.helpers import _time_transformer

HOURS_PER_YEAR = 8760
SECONDS_PER_HOUR = 3600
SECONDS_PER_MILLISECOND = 0.001
SECONDS_PER_YEAR = 31536000
SECONDS_PER_HOUR = 3600
SECONDS_PER_MILLISECOND = 0.001
SECONDS_PER_YEAR = 31536000
CELSIUS_TO_KELVIN = 273.15
NUM_DEVICES = 5

def test_stress():
    with pytest.raises(UserConfigError):
        StrsSpec(conditions={"temp": 1}, duration=99, time_unit='test_invalid')

    stress = StrsSpec(conditions={"temp": 1}, duration=99, time_unit='h')
    assert stress.duration == timedelta(hours=99)
    stress = StrsSpec(conditions={"temp": 1}, duration=99, time_unit='hours')
    assert stress.duration == timedelta(hours=99)

    stress = StrsSpec(conditions={"temp": 1}, duration=99, time_unit='s')
    assert stress.duration == timedelta(seconds=99)
    stress = StrsSpec(conditions={"temp": 1}, duration=99, time_unit='seconds')
    assert stress.duration == timedelta(seconds=99)

    stress = StrsSpec(conditions={"temp": 1}, duration=99, time_unit='ms')
    assert stress.duration == timedelta(milliseconds=99)
    stress = StrsSpec(conditions={"temp": 1}, duration=99, time_unit='milliseconds')
    assert stress.duration == timedelta(milliseconds=99)

    stress = StrsSpec(conditions={"temp": 1}, duration=99, time_unit='y')
    assert stress.duration == timedelta(hours=99 * HOURS_PER_YEAR)
    stress = StrsSpec(conditions={"temp": 1}, duration=99, time_unit='years')
    assert stress.duration == timedelta(hours=99 * HOURS_PER_YEAR)

    stress = StrsSpec(conditions={"temp": 1}, duration=timedelta(hours=99))
    assert stress.duration == timedelta(hours=99)

def test_append_steps():
    test = TestSpec()
    stress = StrsSpec(conditions={"temp": 1}, duration=1, time_unit='h')
    with pytest.raises(UserConfigError):
        test.append_steps(steps=stress, loop_for_duration=3, time_unit='test_invalid')

    test.append_steps(steps=stress, loop_for_duration=3, time_unit='h')
    assert len(test.steps) == 3/1

    test = TestSpec()
    stress = StrsSpec(conditions={"temp": 1}, duration=1, time_unit='s')
    test.append_steps(steps=stress, loop_for_duration=3, time_unit='h')
    assert len(test.steps) == 3/1*SECONDS_PER_HOUR

    test = TestSpec()
    stress = StrsSpec(conditions={"temp": 1}, duration=1, time_unit='h')
    test.append_steps(steps=stress, loop_for_duration=timedelta(hours=3))
    assert len(test.steps) == 3/1

def test_export_to_json():
    meas_spec = MeasSpec({'example_prm': NUM_DEVICES}, {'temp': 25 + CELSIUS_TO_KELVIN})
    strs_spec = StrsSpec({'temp': 125 + CELSIUS_TO_KELVIN}, 100)
    test_spec = TestSpec([meas_spec, strs_spec, meas_spec])
    test_env = PhysTestEnv()
    def ex_eqn(time, temp, a):
        return time * -a * temp
    dev_mdl = DeviceMdl(
        {'example_prm': DegPrmMdl(
            {'linear': DegMechMdl(ex_eqn, a=LatentVar(Normal(1e-3, 2e-4)))})})
    report = gerabaldi.simulate(test_spec, dev_mdl, test_env)
    
    with pytest.raises(ArgOverwriteWarning):
        report.export_to_json(time_unit='test_invalid')

    report_json = report.export_to_json(time_unit='h')
    assert report_json['Time Units'] == 'Hours'

    report_json = report.export_to_json(time_unit='s')
    assert report_json['Time Units'] == 'Seconds'

    report_json = report.export_to_json(time_unit='ms')
    assert report_json['Time Units'] == 'Milliseconds'

    report_json = report.export_to_json(time_unit='y')
    assert report_json['Time Units'] == 'Years'

def test_time_unit_propagation():
    degMech0 = DegMechMdl(mdl_name="degMech0", time_unit="h")
    degMech1 = DegMechMdl(mdl_name="degMech1", time_unit="s")
    degMech2 = DegMechMdl(mdl_name="degMech2")
    failMech0 = FailMechMdl(mdl_name="failMech0", time_unit='ms')
    failMech1 = FailMechMdl(mdl_name="failMech1")

    deg_mech_mdls = {"degMech0": degMech0, "degMech1": degMech1, "degMech2": degMech2, "failMech0": failMech0, "failMech1": failMech1}
    prmMdl0 = DegPrmMdl(deg_mech_mdls=deg_mech_mdls, time_unit="y")

    degMech3 = DegMechMdl(mdl_name="degMech3")
    prmMdl1 = DegPrmMdl(deg_mech_mdls=degMech3)

    devMdl0 = DeviceMdl(prm_mdls={"prmMdl0": prmMdl0, "prmMdl1": prmMdl1}, time_unit="s", name="devMdl0")
    # ------------------------------------------------------------------------------------------------------------------------------------
    degMech0_new = DegMechMdl(mdl_name="degMech0", time_unit="h")
    degMech1_new = DegMechMdl(mdl_name="degMech1", time_unit="s")
    degMech2_new = DegMechMdl(mdl_name="degMech2")
    failMech0_new = FailMechMdl(mdl_name="failMech0", time_unit='ms')
    failMech1_new = FailMechMdl(mdl_name="failMech1")

    deg_mech_mdls_new = {"degMech0": degMech0_new, "degMech1": degMech1_new, "degMech2": degMech2_new, "failMech0": failMech0_new, "failMech1": failMech1_new}
    prmMdl0_new = DegPrmMdl(deg_mech_mdls=deg_mech_mdls_new, time_unit="y")

    degMech3_new = DegMechMdl(mdl_name="degMech3")
    prmMdl1_new = DegPrmMdl(deg_mech_mdls=degMech3_new)

    devMdl1 = DeviceMdl(prm_mdls={"prmMdl0": prmMdl0_new, "prmMdl1": prmMdl1_new}, name="devMdl1")

    assert devMdl0.prm_mdl("prmMdl0").mech_mdl("degMech0").time_unit == "h"
    assert devMdl0.prm_mdl("prmMdl0").mech_mdl("degMech1").time_unit == "s"
    assert devMdl0.prm_mdl("prmMdl0").mech_mdl("degMech2").time_unit == "y"
    assert devMdl0.prm_mdl("prmMdl0").mech_mdl("failMech0").time_unit == "ms"
    assert devMdl0.prm_mdl("prmMdl0").mech_mdl("failMech1").time_unit == "y"
    assert devMdl0.prm_mdl("prmMdl1").mech_mdl("degMech3").time_unit == "s"

    assert devMdl1.prm_mdl("prmMdl0").mech_mdl("degMech0").time_unit == "h"
    assert devMdl1.prm_mdl("prmMdl0").mech_mdl("degMech1").time_unit == "s"
    assert devMdl1.prm_mdl("prmMdl0").mech_mdl("degMech2").time_unit == "y"
    assert devMdl1.prm_mdl("prmMdl0").mech_mdl("failMech0").time_unit == "ms"
    assert devMdl1.prm_mdl("prmMdl0").mech_mdl("failMech1").time_unit == "y"
    assert devMdl1.prm_mdl("prmMdl1").mech_mdl("degMech3").time_unit == "h"


def test_time_unit():
    def eqn(a, temp, time): return a * time * temp
    dev_mdl = DeviceMdl(DegPrmMdl(DegMechMdl(mech_eqn=eqn, mdl_name="test_mdl", time_unit="s", a=LatentVar(deter_val=1)), prm_name="current"))
    meas = MeasSpec({"current": 1}, {"temp": 25})
    strs1 = StrsSpec({"temp": 1}, timedelta(hours=1))
    strs2 = StrsSpec({"temp": 2}, timedelta(hours=1))
    test = TestSpec([strs1, meas, strs2, meas])

    test_env = PhysTestEnv()
    init_state = gerabaldi.gen_init_state(dev_mdl, test.calc_samples_needed(), test.num_chps, test.num_lots)

    report = gerabaldi.simulate(test, dev_mdl, test_env, init_state)

    assert round(report.measurements["measured"][0]) == 3600
    assert round(report.measurements["measured"][1]) == 10800
    # ------------------------------------------------------------------------------------------------------------------------------------
    dev_mdl = DeviceMdl(DegPrmMdl(DegMechMdl(mech_eqn=eqn, mdl_name="test_mdl", a=LatentVar(deter_val=1)), prm_name="current", time_unit="ms"))
    meas = MeasSpec({"current": 1}, {"temp": 25})
    strs1 = StrsSpec({"temp": 1}, 1, time_unit="h")
    strs2 = StrsSpec({"temp": 2}, 1000000, time_unit="ms")
    test = TestSpec([strs1, meas, strs2, meas])

    test_env = PhysTestEnv()
    init_state = gerabaldi.gen_init_state(dev_mdl, test.calc_samples_needed(), test.num_chps, test.num_lots)

    report = gerabaldi.simulate(test, dev_mdl, test_env, init_state)

    assert round(report.measurements["measured"][0]) == 3600000
    assert round(report.measurements["measured"][1]) == 5600000
    # ------------------------------------------------------------------------------------------------------------------------------------
    dev_mdl = DeviceMdl(DegPrmMdl(DegMechMdl(mech_eqn=eqn, mdl_name="test_mdl", a=LatentVar(deter_val=1)), prm_name="current"), time_unit="y")
    meas = MeasSpec({"current": 1}, {"temp": 25})
    strs1 = StrsSpec({"temp": 1}, 1, time_unit="y")
    strs2 = StrsSpec({"temp": 2}, timedelta(days=356))
    test = TestSpec([strs1, meas, strs2, meas])

    test_env = PhysTestEnv()
    init_state = gerabaldi.gen_init_state(dev_mdl, test.calc_samples_needed(), test.num_chps, test.num_lots)

    report = gerabaldi.simulate(test, dev_mdl, test_env, init_state)

    assert round(report.measurements["measured"][0]) == 1
    assert round(report.measurements["measured"][1]) == 3
    # ------------------------------------------------------------------------------------------------------------------------------------
    dev_mdl = DeviceMdl(DegPrmMdl(DegMechMdl(mech_eqn=eqn, mdl_name="test_mdl", a=LatentVar(deter_val=1)), prm_name="current"))  # Should be defaulted to hours
    meas = MeasSpec({"current": 1}, {"temp": 25})
    strs1 = StrsSpec({"temp": 1}, 3600, time_unit="s")
    strs2 = StrsSpec({"temp": 2}, timedelta(hours=1))
    test = TestSpec([strs1, meas, strs2, meas])

    test_env = PhysTestEnv()
    init_state = gerabaldi.gen_init_state(dev_mdl, test.calc_samples_needed(), test.num_chps, test.num_lots)

    report = gerabaldi.simulate(test, dev_mdl, test_env, init_state)

    assert round(report.measurements["measured"][0]) == 1
    assert round(report.measurements["measured"][1]) == 3
