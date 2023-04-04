"""Tests that ensure the effective description and execution of test specifications are correctly implemented."""

import pytest
from datetime import timedelta

from gerabaldi.models import MeasSpec, StrsSpec, TestSpec
from gerabaldi.exceptions import UserConfigError


def test_meas_spec():
    spec = MeasSpec({'voltage': 1, 'delay': 4}, {'voltage': 1.2}, name='Basic')
    assert spec.measurements == {'voltage': 1, 'delay': 4}
    assert spec.conditions == {'voltage': 1.2}
    assert spec.name == 'Basic'
    spec = MeasSpec({'voltage': 100}, {'voltage': 1.1})
    assert spec.name == 'unspecified'


def test_strs_spec():
    spec = StrsSpec({'temp': 200}, timedelta(hours=50))
    assert spec.conditions == {'temp': 200}
    assert spec.duration == timedelta(hours=50)
    assert spec.name == 'unspecified'
    spec.name = 'NewName'
    assert spec.name == 'NewName'
    # Test argument type adjustments
    spec = StrsSpec({'temp': 100}, duration=40)
    assert spec.duration == timedelta(hours=40)
    # Test 0 duration error handling
    with pytest.raises(UserConfigError):
        spec = StrsSpec({'temp': 50}, duration=0)


def test_test_spec():
    # Test the default setup first
    meas = MeasSpec({'delay': 5, 'current': 10}, {'temp': 110}, 'GenericMeas')
    stress = StrsSpec({'temp': 125, 'voltage': 1.2}, timedelta(hours=100), 'Simple Stuff')
    test = TestSpec()
    assert test.description == 'none provided'
    assert test.name == 'unspecified'
    assert (test.num_lots, test.num_chps) == (1, 1)
    assert len(test.steps) == 0
    test.append_steps([meas, stress], 300)
    assert len(test.steps) == 6

    # Test error handling for bad input combinations for appending steps in a loop
    with pytest.raises(UserConfigError):
        test.append_steps(stress, 0)
    with pytest.raises(UserConfigError):
        test.append_steps(meas, 10)
    with pytest.raises(UserConfigError):
        test.append_steps([meas, meas, meas], 10)
    with pytest.raises(UserWarning):
        test.append_steps([stress, meas], 310)
    assert len(test.steps) == 14

    # Now add non-default values
    meas2 = MeasSpec({'delay': 15, 'current': 2}, {'temp': 100}, 'GenericMeas2')
    test = TestSpec([meas, stress, meas], 2, 3, 'Good Fun', 'This test has non-default arguments')
    test.append_steps(meas2)
    assert (test.num_chps, test.num_lots) == (2, 3)
    assert test.description == 'This test has non-default arguments'
    assert test.name == 'Good Fun'
    assert len(test.steps) == 4
    assert test.calc_samples_needed() == {'delay': 15, 'current': 10}
