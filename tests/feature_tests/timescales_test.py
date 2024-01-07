import pytest
from datetime import timedelta

from gerabaldi.exceptions import UserConfigError, ArgOverwriteWarning
from gerabaldi.models.test_specs import StrsSpec
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
            if type(loop_for_duration) != timedelta:
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
