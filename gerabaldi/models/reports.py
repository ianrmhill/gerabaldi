"""
Custom classes for reporting results of different operations such as simulations, inferences, or test optimizations.
"""

import json
import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path

from gerabaldi.models.testspecs import TestSpec
from gerabaldi.exceptions import ArgOverwriteWarning

__all__ = ['TestSimReport']

SECONDS_PER_HOUR = 3600


def convert_time(time, **kwargs):
    if type(time) in [timedelta, pd.Timedelta]:
        return time.total_seconds() / kwargs['units']
    else:
        return timedelta(**{kwargs['units']: time})


class TestSimReport:
    """
    Class for structuring the results of simulated tests for reporting, including test info, measurements, and so on.
    """
    def __init__(self, test_spec: TestSpec = None, name: str = None, description: str = None, file: str = None):
        # Standard construction is using a test specification to determine all the basic test information
        if test_spec:
            self.test_name = name if name else test_spec.name
            self.test_description = description if description else test_spec.description
            self.num_chps, self.num_lots = test_spec.num_chps, test_spec.num_lots
            self.dev_counts = test_spec.calc_samples_needed()
            self.measurements = pd.DataFrame()
            self.stress_summary = pd.DataFrame()
        # Can also load existing test data from a file
        elif file:
            if file:
                try:
                    with open(file, 'r') as f:
                        report_json = json.load(f)
                except FileNotFoundError as e:
                    msg = f"Could not find the requested data file {file}, the file does not appear to exist."
                    raise FileNotFoundError(msg)
            self.test_name = report_json['Test Name']
            self.test_description = report_json['Description']
            self.measurements = pd.read_json(report_json['Measurements'])
            self.stress_summary = pd.read_json(report_json['Stress Summary'])
            # Convert the times back to timedeltas
            units = report_json['Time Units'].lower()
            self.measurements['time'] = self.measurements['time'].apply(convert_time, units=units, axis=1)
            self.stress_summary['start time'] = self.stress_summary['start time'].apply(convert_time, units=units, axis=1)
            self.stress_summary['end time'] = self.stress_summary['end time'].apply(convert_time, units=units, axis=1)
            self.stress_summary['duration'] = self.stress_summary['duration'].apply(convert_time, units=units, axis=1)
        # Allow for empty report initializations, though is not an expected use case
        else:
            self.test_name, self.test_description, self.measurements, self.stress_summary = name, description, None, None
            self.num_chps, self.num_lots, self.dev_counts = None, None, None

    @staticmethod
    def format_measurements(measured_vals: list | pd.Series | np.ndarray, prm_name: str, meas_time: timedelta,
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
        elif prm_type == 'condition':
            num_lots, num_devs, num_meas = measured_vals.shape
            measured_vals = measured_vals.reshape(num_meas * num_devs * num_lots)
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
        meas_cpy['time'] = meas_cpy['time'].apply(convert_time, units=div_time, axis=1)
        report_json['Measurements'] = meas_cpy.to_json()
        strs_cpy = self.stress_summary.copy()
        strs_cpy['duration'] = strs_cpy['duration'].apply(convert_time, units=div_time, axis=1)
        strs_cpy['start time'] = strs_cpy['start time'].apply(convert_time, units=div_time, axis=1)
        strs_cpy['end time'] = strs_cpy['end time'].apply(convert_time, units=div_time, axis=1)
        report_json['Stress Summary'] = strs_cpy.to_json()

        if file:
            # Make the directory if it doesn't yet exist, otherwise the file open will fail
            dir = file.rpartition('/')[0]
            Path(dir).mkdir(exist_ok=True)
            with open(file, 'w') as f:
                json.dump(report_json, f)
        else:
            return report_json
