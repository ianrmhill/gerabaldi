# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

"""Gerabaldi top-level simulation execution functions"""

from __future__ import annotations

import pandas as pd
from datetime import timedelta
from copy import deepcopy

from gerabaldi.models import *
from gerabaldi.exceptions import MissingParamError, UserConfigError

__all__ = ['simulate', 'gen_init_state']

SECONDS_PER_HOUR = 3600


def gen_init_state(dev_mdl: DeviceMdl, dev_counts: dict = None, num_chps: int = 1, num_lots: int = 1,
                   quantity_info: SimReport = None,
                   elapsed_time: timedelta | int | float = timedelta()) -> SimState:
    """
    Prepare a test state object for a specified device model prior to execution/simulation of a wear-out test

    Parameters
    ----------
    dev_mdl: DeviceMdl
        The physical model of a device with measurable parameters that requires a state to be generated for
    dev_counts: dict of int, optional
        Mapping from names of different parameters to measure to the quantity of samples for each
    num_chps: int, optional
        The number of chips per lot to generate state for
    num_lots: int, optional
        The number of production lots to generate state for
    quantity_info: SimReport, optional
        Simulation report that can be used to automatically determine the number of devices, chips, and lots involved
    elapsed_time: timedelta or int or float, optional
        Initial state should be at time 0, override exists to allow for manual setup of intermediate state

    Returns
    -------
    SimState
        The constructed initial state for the test containing initial parameter and sampled latent variable values
    """
    # First determine the source that will determine how many instances/devices for each parameter to initialize
    if quantity_info and not dev_counts:
        devs, chps, lots = quantity_info.dev_counts, quantity_info.num_chps, quantity_info.num_lots
    elif dev_counts:
        devs, chps, lots = dev_counts, num_chps, num_lots
    else:
        raise UserConfigError('The number of devices for each parameter must be specified either manually or by report.'
                              'Cannot provide neither or both.')

    # Check the parameter dependencies for each circuit parameter that could depend on one or more degraded parameters
    # Have to copy the dict of measured parameters as we will modify the original copy within the below loop
    measured_devs = devs.copy()
    for prm in measured_devs:
        # First check if the parameter is a device parameter or a stress condition, skip if a stress condition
        try:
            prm_type = type(dev_mdl.prm_mdl(prm))
        except AttributeError:
            continue
        # Now get dependencies of the circuit parameter, i.e. which other device parameters are needed to compute
        if prm_type == CircPrmMdl:
            for required_prm in dev_mdl.prm_mdl(prm).get_required_prms(dev_mdl.prm_mdl_list):
                # Ensure that enough of the required degraded parameters are specified to support the circuit parameters
                if required_prm not in devs.keys() or devs[required_prm] < devs[prm]:
                    devs[required_prm] = devs[prm]
                    # If a report was used to specify device counts, update the report to include the changes
                    if quantity_info:
                        quantity_info.dev_counts[required_prm] = devs[prm]

    # Now initialize the physical state and override the elapsed time if specified
    init_state = dev_mdl.gen_init_state(devs, chps, lots)
    if elapsed_time != 0:
        init_state.elapsed = elapsed_time
    return init_state


def _sim_stress_step(step: StrsSpec, sim_state: SimState, dev_mdl: DeviceMdl,
                     test_env: PhysTestEnv, report: SimReport, step_id: int):
    # 1. Build the stochastically-adjusted set of stress conditions
    # Only need to generate stress conditions for device parameters that degrade, get the list of those parameters
    deg_prm_list = [prm for prm in dev_mdl.prm_mdl_list if type(dev_mdl.prm_mdl(prm)) == DegPrmMdl]
    strs_conds = test_env.gen_env_cond_vals(step.conditions, deg_prm_list, report, dev_mdl, 'stress')

    for prm in deg_prm_list:
        # 2. Calculate the equivalent stress times that would have been needed under the generated stress conditions to
        # obtain the prior degradation values. First we calculate the equivalent time to reach the current degradation
        equiv_times = dev_mdl.prm_mdl(prm).calc_equiv_strs_times(
            (report.num_lots, report.num_chps, report.dev_counts[prm]),
            sim_state.curr_deg_mech_vals[prm], strs_conds[prm],
            sim_state.init_deg_mech_vals[prm], sim_state.latent_var_vals[prm])
        # Now add on the time for the current stress phase
        for mech in equiv_times:
            equiv_times[mech] += step.duration.total_seconds() / SECONDS_PER_HOUR

        # 3. Simulate the degradation for each device after adding the equivalent prior stress time
        sim_state.curr_prm_vals[prm], sim_state.curr_deg_mech_vals[prm] = dev_mdl.prm_mdl(prm).calc_deg_vals(
            (report.num_lots, report.num_chps, report.dev_counts[prm]),
            equiv_times, strs_conds[prm], sim_state.init_prm_vals[prm],
            sim_state.latent_var_vals[prm], sim_state.curr_deg_mech_vals[prm]
        )

    # Update the elapsed real-world test time
    sim_state.elapsed += step.duration

    # 4. Report the stress conditions used during this step
    merged = {'step name': step.name, 'step type': 'stress', 'step number': step_id, 'duration': step.duration,
              'start time': sim_state.elapsed - step.duration, 'end time': sim_state.elapsed}
    merged.update(step.conditions)
    # Use a DataFrame instead of a Series to simplify the process of merging the reports of different stress steps
    stress_report = pd.DataFrame(merged, index=[0])
    report.add_summary_report(stress_report)


def _sim_meas_step(step: MeasSpec, sim_state: SimState, dev_mdl: DeviceMdl,
                   test_env: PhysTestEnv, report: SimReport, step_id: int):
    """
    Given a test measurement specification, generate the set of observed values. The baseline/true parameter values
    are provided within the step and deg_data arguments, the measurement process adjusts these values according to test
    conditions, stochastic variations, and imperfect measurement instruments.
    """
    # Loop through the parameters to measure and determine if they are degraded parameters or independent conditions
    meas_results = pd.DataFrame()
    # Generate 'true' values for environmental conditions specified by the step (using their variability models)
    conds = test_env.gen_env_cond_vals(step.conditions, step.measurements, report, dev_mdl, 'measure')

    for prm in step.measurements:
        # There are three types of parameters: environmental conditions, degraded parameters and derived parameters
        if prm in sim_state.curr_prm_vals:
            # Adjust the param values based on environmental conditions during measurement and the instrument used
            measured = dev_mdl.prm_mdl(prm).calc_cond_shifted_vals(
                (report.num_lots, report.num_chps, step.measurements[prm]),
                conds[prm], sim_state.curr_prm_vals[prm], sim_state.latent_var_vals[prm])
            measured = SimReport.format_measurements(measured, prm, sim_state.elapsed, step_id, 'parameter')

        elif prm in dev_mdl.prm_mdl_list:
            # Derived parameter values that can depend on multiple degraded parameters, i.e. circuit models
            measured = dev_mdl.prm_mdl(prm).calc_circ_vals(
                step.measurements[prm], conds[prm], sim_state.curr_prm_vals, sim_state.latent_var_vals[prm])
            measured = SimReport.format_measurements(measured, prm, sim_state.elapsed, step_id, 'parameter')

        elif prm in step.conditions:
            # These parameters are not degraded parameters of the device but instead environmental parameters
            measured = SimReport.format_measurements(conds[prm][prm], prm, sim_state.elapsed, step_id, 'condition')

        else:
            raise MissingParamError(f"Requested measurement of param '{prm}' failed, param is not defined within the \
            simulated test.")

        # With the measurements completed, configure the dataframe for reporting and add to the merged dataframe
        measured['measured'] = test_env.meas_instm(prm).measure(measured['measured'])
        meas_results = pd.concat((meas_results, measured), ignore_index=True)

    if step.verbose:
        print(f"Conducted measurement {step.name} at simulation time {sim_state.elapsed}.")
    report.add_measurements(meas_results)

    # Report the stress conditions used during this step
    merged = {'step name': step.name, 'step type': 'measure', 'step number': step_id, 'duration': timedelta(),
              'start time': sim_state.elapsed, 'end time': sim_state.elapsed}
    merged.update(step.conditions)
    # Use a DataFrame instead of a Series to simplify the process of merging the reports of different stress steps
    meas_report = pd.DataFrame(merged, index=[0])
    report.add_summary_report(meas_report)


def simulate(test_def: TestSpec, dev_mdl: DeviceMdl, test_env: PhysTestEnv,
             init_state: SimState = None) -> SimReport:
    """
    Simulate a given wear-out test using a given underlying model

    Parameters
    ----------
    test_def:
        A complete test description that the defines the stress conditions, durations, and data to collect
    dev_mdl:
        The underlying exact degradation model(s) to use to generate simulated test results
    test_env:
        Definition of the test environment that determines how imprecision is injected into the test results
    init_state: SimState, optional
        Starting values for device model parameters, optional as normally this will be generated automatically

    Returns
    -------
    test_report: A TestReport object containing all relevant information on the test structure, execution, and results
    """
    # The test report object assembles all the collected test data into one data structure and tracks configuration info
    test_report = SimReport(test_def)

    # Prepare the simulation data persistence structure
    if not init_state:
        sim_state = gen_init_state(dev_mdl, quantity_info=test_report)
    else:
        sim_state = deepcopy(init_state)

    # We now execute the test step by step, sequentially performing measurements and stress intervals in order
    # Note that sim_state and test_report are mutated as the test progresses, the other data structures are untouched
    for i, step in enumerate(test_def):
        # Check whether the next step is a measurement or period of stress
        if type(step) is StrsSpec:
            _sim_stress_step(step, sim_state, dev_mdl, test_env, test_report, i)
        elif type(step) is MeasSpec:
            _sim_meas_step(step, sim_state, dev_mdl, test_env, test_report, i)
    return test_report
