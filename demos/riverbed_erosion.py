# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import click
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sb

import os
import sys
# Welcome to the worst parts of Python! This line adds the parent directory of this file to module search path, from
# which the Gerabaldi module can be seen and then imported. Without this line the script cannot find the module without
# installing it as a package from pip (which is undesirable because you would have to rebuild the package every time
# you changed part of the code).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gerabaldi
from gerabaldi.models import *

DATA_FILE_NAME = 'erosion_report'
SECONDS_PER_HOUR = 3600
HOURS_PER_YEAR = 8760

SAMPLES_PER_RIVER = 10
RIVERS_SAMPLED = 5


def simulate(save_file: str = None):
    """
    Demonstration of using Gerabaldi's advanced stochastic model and degradation model agnosticism to simulate
    degradation of other processes beyond integrated circuits; in this case fluvial erosion in rivers.
    """

    ########################################################################
    ### 1. Define the fluvial erosion stress and monitoring process/test ###
    ########################################################################
    # Note that 'tau' is the shear stress due to the flow of river water over the soil bed
    erosion_meas = MeasSpec({'riverbed_level': SAMPLES_PER_RIVER}, {'tau': 3}, 'Bed Height Sampling')
    # Each stress phase lasts a quarter of a year
    summer_strs = StrsSpec({'tau': 4}, HOURS_PER_YEAR / 4, 'Summer Season Flow')
    autumn_strs = StrsSpec({'tau': 2}, HOURS_PER_YEAR / 4, 'Autumn Season Flow')
    winter_strs = StrsSpec({'tau': 2}, HOURS_PER_YEAR / 4, 'Winter Season Flow')
    spring_strs = StrsSpec({'tau': 7}, HOURS_PER_YEAR / 4, 'Spring Season Flow')
    # We will measure 20 individual locations in 5 different rivers for 10 years
    ten_year_test = TestSpec([erosion_meas], RIVERS_SAMPLED, 1, name='Riverbed Erosion Process')
    ten_year_test.append_steps([spring_strs, summer_strs, autumn_strs, winter_strs, erosion_meas], HOURS_PER_YEAR * 10)

    ########################################################################
    ### 2. Define the test/field environment                             ###
    ########################################################################
    river_env = PhysTestEnv(env_vrtns={
        # The shear stress varies between riverbed locations a bit, but much more so between distinct rivers
        'tau': EnvVrtnMdl(dev_vrtn_mdl=Normal(0, 0.2), chp_vrtn_mdl=Normal(0, 0.8), batch_vrtn_mdl=Normal(0, 1.2)),
    }, meas_instms={
        # We can only measure to the nearest centimetre, and our measurements are typically off by half a centimetre
        'riverbed_level': MeasInstrument(precision=3, error=Normal(0, 0.005))
    })

    ########################################################################
    ### 3. Define the physical riverbed erosion model                    ###
    ########################################################################
    # Model provided in K. Klavon et al. BSTEM paper, Equation 3, DOI: https://doi.org/10.1002/esp.4073
    # Tau represents the amount of shear stress, tau_c is the critical stress threshold for erosion to occur
    def fluvial_erosion_riverbed(time, k_d, tau_c, alpha, tau):
        e_r = np.maximum(k_d * ((alpha * tau) - tau_c), 0)
        return time * e_r

    def riverbed_level(init, erosion, cond):
        return init - erosion + cond

    river_mdl = DeviceMdl(DegPrmMdl(
        prm_name='riverbed_level',
        deg_mech_mdls={
            'erosion': DegMechMdl(
                fluvial_erosion_riverbed, k_d=LatentVar(deter_val=1e-5),
                tau_c=LatentVar(Normal(2.4, 0.02), Normal(1, 0.2)), alpha=LatentVar(Normal(0.9, 0.01), Normal(1, 0.04)),
            )},
        init_val_mdl=InitValMdl(init_val=LatentVar(dev_vrtn_mdl=Normal(-2, 0.01), chp_vrtn_mdl=Normal(1, 0.02))),
        compute_eqn=riverbed_level,
        array_computable=False
    ))

    ########################################################################
    ### 4. Simulate the riverbed erosion over the specified time period  ###
    ########################################################################
    erosion_report = gerabaldi.simulate(ten_year_test, river_mdl, river_env)

    # Save the simulated results to a JSON file for reuse if desired
    if save_file:
        erosion_report.export_to_json(save_file, 'hours')
    return erosion_report


def visualize(report):
    measured = report.measurements

    # Reformat dataframe to get ready for plotting
    measured = measured.set_index(['param', 'device #', 'chip #'])
    measured = measured.sort_index()
    measured = measured.drop('lot #', axis=1)
    # Change timedeltas to hours then years for processing
    measured['time'] = measured['time'].apply(lambda time, **kwargs: time.total_seconds() / SECONDS_PER_HOUR, axis=1)
    measured['time'] = measured['time'] / HOURS_PER_YEAR

    # Set up the plots
    sb.set_theme(style='whitegrid', font='Times New Roman')
    sb.set_context('talk')
    f1, p1 = plt.subplots(figsize=(8, 6))
    f2, p2 = plt.subplots(figsize=(8, 6))

    # Now extract each individual series of measurements in turn
    colour_map = ['mediumpurple', 'green', 'cornflowerblue', 'aqua', 'limegreen']
    for rvr in range(RIVERS_SAMPLED):
        avg = np.zeros(11)
        max = np.full(11, -1000)
        min = np.full(11, 1000)
        times = measured.loc['riverbed_level', 0, 0]['time']

        for smpl in range(SAMPLES_PER_RIVER):
            # First plot the individual series
            meas = measured.loc[('riverbed_level', smpl, rvr)]
            p1.plot(meas['time'], meas['measured'], color=colour_map[rvr])
            # Also plot the average level for each time in each river, along with the spread of values
            vals = meas.reset_index()['measured']
            avg += vals
            max = np.maximum(max, vals)
            min = np.minimum(min, vals)

        avg = avg / SAMPLES_PER_RIVER
        p2.plot(times, avg, color=colour_map[rvr])
        p2.fill_between(times, min, max, color=colour_map[rvr], alpha=0.2)

    p1.set(ylabel='Riverbed Level Below Ref (metres)', xlabel='Elapsed Time (years)',
           title='Riverbed Erosion in Five Glacier-Fed Streams')
    p2.set(ylabel='Riverbed Level Below Reference (metres)', xlabel='Elapsed Time (years)', xlim=(0, 10),
           title='Riverbed Erosion in Glacial Runoff Streams')
    p2.grid(alpha=0.2)
    p2.xaxis.set_major_locator(ticker.MultipleLocator(1))
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
        report = simulate(save_file=data_file)
    visualize(report)


if __name__ == '__main__':
    entry()
