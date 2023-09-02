from datetime import timedelta

from gerabaldi.models.test_specs import StrsSpec, TestSpec
from gerabaldi.helpers import _time_transformer
HOURS_PER_YEAR = 8760


class Test_TestSpec(TestSpec):
    def __init__(self):
        pass

    def append_steps(self, loop_for_duration: timedelta | int | float = None, time_unit: str = 'hour'):
        if loop_for_duration is None:
            duration = 1984  # an arbitrary number
        else:
            if type(loop_for_duration) != timedelta:
                duration = _time_transformer(loop_for_duration, time_unit)
            else:
                duration = loop_for_duration

        return duration


def test_StrsSpec_timescale():
    obj_a = StrsSpec(conditions={'tau': 4}, duration=99)
    obj_b = StrsSpec(conditions={'tau': 4}, duration=99, time_unit='hour')
    obj_c = StrsSpec(conditions={'tau': 4}, duration=99.9, time_unit='h')
    obj_d = StrsSpec(conditions={'tau': 4}, duration=99, time_unit='second')
    obj_e = StrsSpec(conditions={'tau': 4}, duration=99, time_unit='s')
    obj_f = StrsSpec(conditions={'tau': 4}, duration=99, time_unit='millisecond')
    obj_g = StrsSpec(conditions={'tau': 4}, duration=99, time_unit='ms')
    obj_h = StrsSpec(conditions={'tau': 4}, duration=99, time_unit='year')
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
    a = test.append_steps()
    b = test.append_steps(loop_for_duration=99)
    c = test.append_steps(loop_for_duration=99, time_unit='hour')
    d = test.append_steps(loop_for_duration=99.9, time_unit='h')
    e = test.append_steps(loop_for_duration=99, time_unit='second')
    f = test.append_steps(loop_for_duration=99, time_unit='s')
    g = test.append_steps(loop_for_duration=99, time_unit='millisecond')
    h = test.append_steps(loop_for_duration=99, time_unit='ms')
    i = test.append_steps(loop_for_duration=99, time_unit='year')
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
