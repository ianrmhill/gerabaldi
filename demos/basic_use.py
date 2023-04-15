# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import os
import sys
# Welcome to the worst parts of Python! This line adds the parent directory of this file to module search path, from
# which the Gerabaldi module can be seen and then imported. Without this line the script cannot find the module without
# installing it as a package from pip (which is undesirable because you would have to rebuild the package every time
# you changed part of the code).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gerabaldi # noqa: ImportNotAtTopOfFile
from gerabaldi.models import * # noqa: ImportNotAtTopOfFile
from gerabaldi.helpers import _on_demand_import # noqa: ImportNotAtTopOfFile

click = _on_demand_import('click')
plt = _on_demand_import('matplotlib.pyplot', 'matplotlib')
sb = _on_demand_import('seaborn')

DATA_FILE_NAME = 'basic_report'

CELSIUS_TO_KELVIN = 273.15
SECONDS_PER_HOUR = 3600

NUM_DEVICES = 5


def run_simulation(save_file: str = None):
    """
    Demonstration of the simplest use of Gerabaldi, useful as a template and starting point for building your own sims.
    """

    ########################################################################
    ### 1. Define the test to simulate                                   ###
    ########################################################################
    meas_spec = MeasSpec({'example_prm': NUM_DEVICES}, {'temp': 25 + CELSIUS_TO_KELVIN})
    strs_spec = StrsSpec({'temp': 125 + CELSIUS_TO_KELVIN}, 100)
    test_spec = TestSpec([meas_spec, strs_spec, meas_spec])

    ########################################################################
    ### 2. Define the test/field environment                             ###
    ########################################################################
    test_env = PhysTestEnv()

    ########################################################################
    ### 3. Define the physical device model                              ###
    ########################################################################
    def ex_eqn(time, temp, a):
        return time * -a * temp
    dev_mdl = DeviceMdl(
        {'example_prm': DegPrmMdl(
            {'linear': DegMechMdl(ex_eqn, a=LatentVar(Normal(1e-3, 2e-4)))})})

    ########################################################################
    ### 4. Simulate the test                                             ###
    ########################################################################
    report = gerabaldi.simulate(test_spec, dev_mdl, test_env)

    # Save the simulated results to a JSON file for reuse if desired
    if save_file:
        report.export_to_json(save_file, 'hours')
    return report


def visualize(report):
    measured = report.measurements
    # Change time deltas to hours for processing
    measured['time'] = measured['time'].apply(lambda time, **kwargs: time.total_seconds() / SECONDS_PER_HOUR, axis=1)

    # Set up the figure area
    colours = ['mediumpurple', 'green', 'cornflowerblue', 'aqua', 'limegreen']
    sb.set_theme(style='whitegrid', font='Times New Roman')
    sb.set_context('talk')
    f1, p1 = plt.subplots(figsize=(8, 6))

    # Plot the degradation curves, one curve per device
    measured = measured.set_index(['param', 'device #'])
    measured = measured.sort_index()
    for smpl in range(NUM_DEVICES):
        meas = measured.loc[('example_prm', smpl)]
        p1.plot(meas['time'], meas['measured'], color=colours[smpl])

    # Set some plot properties for a clean look
    p1.set(ylabel='Linear Degradation', xlabel='Elapsed Time (hours)', title='Example of Simple Degradation')
    sb.despine()
    plt.show()


@click.command
@click.option('--data-file', default=None, help='Use existing simulated data from a JSON file.')
@click.option('--save-data', is_flag=True, default=False, help='If provided, simulated data will be saved to a JSON.')
def entry(data_file, save_data):
    if data_file is not None:
        report = TestSimReport(file=data_file)
    else:
        data_file = os.path.join(os.path.dirname(__file__), f"data/{DATA_FILE_NAME}.json") if save_data else None
        report = run_simulation(save_file=data_file)
    visualize(report)


if __name__ == '__main__':
    entry()
