"""
Custom classes for reporting results of different operations such as simulations, inferences, or test optimizations.
"""

import json
import pandas as pd
import numpy as np
from datetime import timedelta

from gerabaldi.exceptions import ArgOverwriteWarning

__all__ = ['TestSimReport']

SECONDS_PER_HOUR = 3600


def convert_time(time, **kwargs):
    return time.total_seconds() / kwargs['unit_factor']


class TestSimReport:
    """
    Class for structuring the results of simulated tests for reporting, including test info, measurements, and so on.
    """
    def __init__(self, name: str = 'generic', description: str = ""):
        self.test_name = name
        self.test_description = description
        self.measurements = pd.DataFrame()
        self.stress_summary = pd.DataFrame()

    @staticmethod
    def format_measurements(measured_vals: list | pd.Series, prm_name: str, meas_time: timedelta,
                            prm_type: str = 'parameter'):
        """
        Take a set of measured values of a parameter and condition and create a formatted dataframe to report the
        measured values in.
        """
        circ_num, dev_num, lot_num = None, None, None
        if prm_type == 'parameter':
            measured_vals = np.array(measured_vals)
            num_lots, num_devs, num_meas = measured_vals.shape
            measured_vals = measured_vals.reshape(num_meas * num_devs * num_lots)
            circ_num = np.tile(np.linspace(0, num_meas - 1, num_meas, dtype=int), num_devs * num_lots)
            dev_num = np.tile(np.repeat(np.linspace(0, num_devs - 1, num_devs, dtype=int), num_meas), num_lots)
            lot_num = np.repeat(np.linspace(0, num_lots - 1, num_lots, dtype=int), num_meas * num_devs)
        formatted = pd.DataFrame({'param': prm_name,
                                  'device #': circ_num,
                                  'chip #': dev_num,
                                  'lot #': lot_num,
                                  'time': meas_time,
                                  'measured': measured_vals})
        return formatted

    def add_measurements(self, measured: pd.DataFrame):
        """Add measurement rows to the full dataframe containing all conducted measurements."""
        self.measurements = pd.concat([self.measurements, measured], ignore_index=True)

    def add_stress_report(self, strs_conds: pd.DataFrame):
        """Add stress reporting rows to the dataframe of all reported stress phases."""
        self.stress_summary = pd.concat([self.stress_summary, strs_conds], ignore_index=True)

    def export_to_json(self, file: str = None, time_units: str = 'seconds'):
        """Formats a test report as a json string so that it can be saved as a file and passed around."""
        report_json = {'Test Name': self.test_name, 'Description': self.test_description}
        if time_units == 'hours':
            div_time = SECONDS_PER_HOUR
            report_json['Time Units'] = 'Hours'
        else:
            # Default time unit is in seconds
            if time_units != 'seconds':
                raise ArgOverwriteWarning(f"Could not understand requested time units of {time_units}, defaulting to seconds.")
            div_time = 1
            report_json['Time Units'] = 'Seconds'

        meas_cpy = self.measurements.copy()
        meas_cpy['time'] = meas_cpy['time'].apply(convert_time, unit_factor=div_time, axis=1)
        report_json['Measurements'] = meas_cpy.to_json()
        strs_cpy = self.stress_summary.copy()
        strs_cpy['duration'] = strs_cpy['duration'].apply(convert_time, unit_factor=div_time, axis=1)
        strs_cpy['start time'] = strs_cpy['start time'].apply(convert_time, unit_factor=div_time, axis=1)
        strs_cpy['end time'] = strs_cpy['end time'].apply(convert_time, unit_factor=div_time, axis=1)
        report_json['Stress Summary'] = strs_cpy.to_json()

        if file:
            with open(file, 'w') as f:
                json.dump(report_json, f)
        else:
            return report_json
