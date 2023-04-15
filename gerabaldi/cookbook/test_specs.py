# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import timedelta

from gerabaldi.models.test_specs import MeasSpec, StrsSpec, TestSpec
from gerabaldi.exceptions import UserConfigError

__all__ = ['htol']

CELSIUS_TO_KELVIN = 273.15


def htol(to_meas: dict[str, int], vdd_nom: float = 1.0, vdd_strs_mult: float = 1.2, strs_meas_intrvl: int = 1000,
         duration_override: int = 1000, num_chps_override: int = 77, num_lots_override: int = 3,
         temp_override: int = 125) -> TestSpec:
    """
    Build an HTOL test in a single function call for use in Gerabaldi simulations

    Parameters
    ----------
    to_meas: dict[str, int]
        Mapping from parameters that should be measured to the quantity of samples of that parameter to measure
    vdd_nom: float, optional
        The nominal/unstressed internal chip voltage (default 1.0)
    vdd_strs_mult: float, optional
        The stress factor that 'vdd_nom' is multiplied by to get the HTOL stress voltage (default 1.2)
    strs_meas_intrvl: int, optional
        If provided, measurements under stress conditions will be made every 'strs_meas_intrvl' hours during the test
    duration_override: int, optional
        If provided, use the number of hours specified as the total test length instead of the HTOL default of 1000
    num_chps_override: int, optional
        If provided, use the number of chips specified instead of the HTOL default of 77
    num_lots_override: int, optional
        If provided, use the number of lots specified instead of the HTOL default of 3
    temp_override: int, optional
        If provided, stress at the passed temperature (in Celsius) instead of the HTOL default of 125

    Returns
    -------
    TestSpec
        The configured HTOL test specification model
    """
    # Define some HTOL test constants
    htol_duration, htol_num_chips, htol_num_lots = duration_override, num_chps_override, num_lots_override
    room_temp_c, strs_temp_c = 25, temp_override
    room_vdd, strs_vdd = vdd_nom, vdd_nom * vdd_strs_mult

    meas_strs = MeasSpec(to_meas, {'temp': strs_temp_c + CELSIUS_TO_KELVIN, 'vdd': strs_vdd}, 'Stress Temp Measurement')
    meas_room = MeasSpec(to_meas, {'temp': room_temp_c + CELSIUS_TO_KELVIN, 'vdd': room_vdd}, 'Room Temp Measurement')
    htol_strs = StrsSpec({'temp': strs_temp_c + CELSIUS_TO_KELVIN, 'vdd': strs_vdd},
                         timedelta(hours=htol_duration), 'HTOL Stress')
    htol_test = TestSpec([meas_room], htol_num_chips, htol_num_lots, name='Standard HTOL',
                         description='A typical HTOL test with configurable measurements during stress.')

    # Include the option to take periodic measurements at the elevated stress conditions during the test
    if strs_meas_intrvl >= htol_duration:
        htol_test.append_steps(htol_strs)
    else:
        if htol_duration % strs_meas_intrvl != 0:
            raise UserConfigError(f"The stress measure interval of {strs_meas_intrvl} does not fit into {htol_duration}"
                                  "hours a whole number of times. Please use an interval that is a factor of the test"
                                  "duration.")
        htol_strs.duration = timedelta(hours=strs_meas_intrvl)
        htol_test.append_steps(meas_strs)
        htol_test.append_steps([htol_strs, meas_strs], loop_for_duration=htol_duration)

    # Add the end of test room temperature measurement
    htol_test.append_steps(meas_room)
    return htol_test
