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


class Test_TestSpec:
    def __init__(self):
        pass

    def append_steps(self, loop_for_duration: timedelta | int | float = None, time_unit: str = 'hours'):
        if loop_for_duration is None:
            duration = 1984  # an arbitrary number
        else:
            if not isinstance(loop_for_duration, timedelta):
                if time_unit not in ['hours', 'seconds', 'milliseconds', 'years', 'h', 's', 'ms', 'y']:
                    raise UserConfigError("Incorrect time unit. The valid options are "
                                          "'hours' ('h'), 'seconds' ('s'), 'milliseconds' ('ms'), and 'years' ('y').")
                duration = _time_transformer(loop_for_duration, time_unit)
            else:
                duration = loop_for_duration

        return duration


class Test_SimReport:
    def __init__(self):
        pass

    def export_to_json(self, time_unit: str = 'seconds'):
        report_json = {}
        if time_unit in ['hours', 'h']:
            div_time = SECONDS_PER_HOUR
            report_json['Time Units'] = 'Hours'
        elif time_unit in ['seconds', 's']:
            div_time = 1
            report_json['Time Units'] = 'Seconds'
        elif time_unit in ['milliseconds', 'ms']:
            div_time = SECONDS_PER_MILLISECOND
            report_json['Time Units'] = 'Milliseconds'
        elif time_unit in ['years', 'y']:
            div_time = SECONDS_PER_YEAR
            report_json['Time Units'] = 'Years'
        else:
            div_time = 1
            report_json['Time Units'] = 'Seconds'
            raise ArgOverwriteWarning(f"Could not understand requested time units of {time_unit},"
                                      "defaulting to seconds.")

        return report_json['Time Units'], div_time


def test_StrsSpec_timescale():
    with pytest.raises(UserConfigError):
        StrsSpec(conditions={'tau': 4}, duration=99, time_unit='test_invalid')
    obj_a = StrsSpec(conditions={'tau': 4}, duration=99)
    obj_b = StrsSpec(conditions={'tau': 4}, duration=99, time_unit='hours')
    obj_c = StrsSpec(conditions={'tau': 4}, duration=99.9, time_unit='h')
    obj_d = StrsSpec(conditions={'tau': 4}, duration=99, time_unit='seconds')
    obj_e = StrsSpec(conditions={'tau': 4}, duration=99, time_unit='s')
    obj_f = StrsSpec(conditions={'tau': 4}, duration=99, time_unit='milliseconds')
    obj_g = StrsSpec(conditions={'tau': 4}, duration=99, time_unit='ms')
    obj_h = StrsSpec(conditions={'tau': 4}, duration=99, time_unit='years')
    obj_i = StrsSpec(conditions={'tau': 4}, duration=99, time_unit='y')
    obj_j = StrsSpec(conditions={'tau': 4}, duration=timedelta(hours=99))

    assert obj_a.duration == timedelta(hours=99)
    assert obj_b.duration == timedelta(hours=99)
    assert obj_c.duration == timedelta(hours=99.9)
    assert obj_d.duration == timedelta(seconds=99)
    assert obj_e.duration == timedelta(seconds=99)
    assert obj_f.duration == timedelta(milliseconds=99)
    assert obj_g.duration == timedelta(milliseconds=99)
    assert obj_h.duration == timedelta(hours=99 * HOURS_PER_YEAR)
    assert obj_i.duration == timedelta(hours=99 * HOURS_PER_YEAR)
    assert obj_j.duration == timedelta(hours=99)


def test_TestSpec_timescale():
    test = Test_TestSpec()
    with pytest.raises(UserConfigError):
        test.append_steps(loop_for_duration=99, time_unit='test_invalid')
    a = test.append_steps()
    b = test.append_steps(loop_for_duration=99)
    c = test.append_steps(loop_for_duration=99, time_unit='hours')
    d = test.append_steps(loop_for_duration=99.9, time_unit='h')
    e = test.append_steps(loop_for_duration=99, time_unit='seconds')
    f = test.append_steps(loop_for_duration=99, time_unit='s')
    g = test.append_steps(loop_for_duration=99, time_unit='milliseconds')
    h = test.append_steps(loop_for_duration=99, time_unit='ms')
    i = test.append_steps(loop_for_duration=99, time_unit='years')
    j = test.append_steps(loop_for_duration=99, time_unit='y')
    k = test.append_steps(loop_for_duration=timedelta(hours=99))

    assert a == 1984
    assert b == timedelta(hours=99)
    assert c == timedelta(hours=99)
    assert d == timedelta(hours=99.9)
    assert e == timedelta(seconds=99)
    assert f == timedelta(seconds=99)
    assert g == timedelta(milliseconds=99)
    assert h == timedelta(milliseconds=99)
    assert i == timedelta(hours=99 * HOURS_PER_YEAR)
    assert j == timedelta(hours=99 * HOURS_PER_YEAR)
    assert k == timedelta(hours=99)


def test_SimReport_timescale():
    test = Test_SimReport()
    with pytest.raises(ArgOverwriteWarning):
        test.export_to_json('test_invalid')
    a1, a2 = test.export_to_json()
    b1, b2 = test.export_to_json('hours')
    c1, c2 = test.export_to_json('s')
    d1, d2 = test.export_to_json('ms')
    e1, e2 = test.export_to_json('years')

    assert (a1, a2) == ('Seconds', 1)
    assert (b1, b2) == ('Hours', SECONDS_PER_HOUR)
    assert (c1, c2) == ('Seconds', 1)
    assert (d1, d2) == ('Milliseconds', SECONDS_PER_MILLISECOND)
    assert (e1, e2) == ('Years', SECONDS_PER_YEAR)


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
