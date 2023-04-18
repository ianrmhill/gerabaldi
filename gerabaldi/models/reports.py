# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

"""Custom classes for reporting results of Gerabaldi simulations"""

from __future__ import annotations

import json
import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path

from gerabaldi.models.test_specs import TestSpec
from gerabaldi.exceptions import ArgOverwriteWarning
from gerabaldi.helpers import _convert_time

__all__ = ['SimReport']

SECONDS_PER_HOUR = 3600


class SimReport:
    """
    Class for structuring the results of simulated tests for reporting, such as test info, measurements, and so forth

    Attributes
    ----------
    test_name: str
        A descriptive name for the test being simulated
    test_description: str
        A longer explanation of the test purpose and characteristics to provide context for the simulation data
    dev_counts: dict of int
        A mapping from names of device parameters being measured to the quantities of each measured per chip
    num_chps: int
        The quantity of chips in the simulation per production lot
    num_lots: int
        The quantity of lots in the simulation
    measurements: pandas.DataFrame
        A tabular data structure with all the simulated measurements of device parameters conducted during the test
    test_summary: pandas.DataFrame
        A tabular data structure that explains the stress and measure conditions used during the test
    """

    def __init__(self, test_spec: TestSpec = None, name: str = None, description: str = None, file: str = None):
        """
        Parameters
        ----------
        test_spec: TestSpec, optional
            The test specification that will be or was simulated
        name: str, optional
            A descriptive test name, can be determined from the test_spec parameter if not provided
        description: str, optional
            Description of the test purpose and characteristics, can be determined from the test_spec parameter
        file: str, optional
            Path and filename (absolute or relative to CWD) to a JSON containing a test report to load
        """
        # Standard construction is using a test specification to determine all the basic test information
        if test_spec:
            self.test_name = name if name else test_spec.name
            self.test_description = description if description else test_spec.description
            self.num_chps, self.num_lots = test_spec.num_chps, test_spec.num_lots
            self.dev_counts = test_spec.calc_samples_needed()
            self.measurements = pd.DataFrame()
            self.test_summary = pd.DataFrame()
        # Can also load existing test data from a file
        elif file:
            if file:
                try:
                    with open(file, 'r') as f:
                        report_json = json.load(f)
                except FileNotFoundError:
                    msg = f"Could not find the requested data file {file}, the file does not appear to exist."
                    raise FileNotFoundError(msg)
            self.test_name = report_json['Test Name']
            self.test_description = report_json['Description']
            self.measurements = pd.read_json(report_json['Measurements'])
            self.test_summary = pd.read_json(report_json['Test Summary'])
            # Convert the times back to time deltas
            units = report_json['Time Units'].lower()
            self.measurements['time'] = self.measurements['time'].apply(_convert_time, units=units, axis=1)
            self.test_summary['start time'] = self.test_summary['start time'].apply(_convert_time, units=units,
                                                                                    axis=1)
            self.test_summary['end time'] = self.test_summary['end time'].apply(_convert_time, units=units, axis=1)
            self.test_summary['duration'] = self.test_summary['duration'].apply(_convert_time, units=units, axis=1)
        # Allow for empty report initializations, though is not an expected use case
        else:
            self.test_name, self.test_description = name, description
            self.measurements, self.test_summary = None, None
            self.num_chps, self.num_lots, self.dev_counts = None, None, None

    @staticmethod
    def format_measurements(measured_vals: list | pd.Series | np.ndarray, prm_name: str,
                            meas_time: timedelta, meas_step: int, prm_type: str = 'parameter') -> pd.DataFrame:
        """
        Take a set of measured values of a parameter and condition and create a formatted dataframe to report the
        measured values in.

        Parameters
        ----------
        measured_vals: list or pandas.Series or numpy.ndarray
            The unformatted (i.e. simulator internal) set of measurement data to send to a tabular format
        prm_name: str
            The name of the parameter that the measured_vals measurements correspond to
        meas_time: timedelta
            The relative time at which the measurements were taken within the test duration
        meas_step: int
            The unique number of the test step that this measurement data is associated with
        prm_type: str, optional
            Whether the measurements are for a device 'parameter' or a stress 'condition' (default 'parameter')

        Returns
        -------
        pandas.DataFrame
            A formatted tabular structure that can be concatenated/appended to other sets of formatted measurements
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
                                  'step #': meas_step,
                                  'device #': circ_num,
                                  'chip #': dev_num,
                                  'lot #': lot_num,
                                  'time': meas_time,
                                  'measured': measured_vals})
        return formatted

    def add_measurements(self, measured: pd.DataFrame):
        """
        Add measurement rows to the full dataframe containing all conducted measurements

        Parameters
        ----------
        measured: pandas.DataFrame
            The formatted measurements to concatenate/append to the existing full set of measurements
        """
        self.measurements = pd.concat([self.measurements, measured], ignore_index=True)

    def add_summary_report(self, summary_info: pd.DataFrame):
        """
        Add summary reporting rows to the dataframe of all existing test step summaries

        Parameters
        ----------
        summary_info: pandas.DataFrame
            The formatted report of test step information to add to the full test summary
        """
        self.test_summary = pd.concat([self.test_summary, summary_info], ignore_index=True)

    def export_to_json(self, file: str = None, time_units: str = 'seconds') -> None | dict:
        """
        Formats a test report as a json string so that it can be saved as a file for storage or sharing

        Parameters
        ----------
        file: str, optional
            The path and file (absolute or relative to CWD) to save the report to, if not provided the JSON is returned
        time_units: str, optional
            The format to save test time instants as within the JSON (default 'seconds')

        Returns
        -------
        None or dict
            No return value if saving to file, otherwise the JSON dictionary format for the report
        """
        report_json = {'Test Name': self.test_name, 'Description': self.test_description}
        if time_units == 'hours':
            div_time = SECONDS_PER_HOUR
            report_json['Time Units'] = 'Hours'
        else:
            # Default time unit is in seconds
            if time_units != 'seconds':
                raise ArgOverwriteWarning(f"Could not understand requested time units of {time_units},"
                                          "defaulting to seconds.")
            div_time = 1
            report_json['Time Units'] = 'Seconds'

        meas_cpy = self.measurements.copy()
        meas_cpy['time'] = meas_cpy['time'].apply(_convert_time, units=div_time, axis=1)
        report_json['Measurements'] = meas_cpy.to_json()
        strs_cpy = self.test_summary.copy()
        strs_cpy['duration'] = strs_cpy['duration'].apply(_convert_time, units=div_time, axis=1)
        strs_cpy['start time'] = strs_cpy['start time'].apply(_convert_time, units=div_time, axis=1)
        strs_cpy['end time'] = strs_cpy['end time'].apply(_convert_time, units=div_time, axis=1)
        report_json['Test Summary'] = strs_cpy.to_json()

        if file:
            # Make the directory if it doesn't yet exist, otherwise the file open will fail
            data_dir = file.rpartition('/')[0]
            Path(data_dir).mkdir(exist_ok=True)
            with open(file, 'w') as f:
                json.dump(report_json, f)
        else:
            return report_json
