"""
Gerabaldi top-level simulation flow functions
"""

import pandas as pd
import numpy as np
from datetime import timedelta

from gerabaldi.models import *
from gerabaldi.exceptions import MissingParamError, UserConfigError

__all__ = ['simulate', 'gen_init_state']

SECONDS_PER_HOUR = 3600


def gen_init_state(dev_mdl: DeviceMdl, dev_counts: dict = None, num_chps: int = 1, num_lots: int = 1,
                   quantity_info: TestSimReport = None, elapsed_time: timedelta | int | float = timedelta()) -> TestSimState:
    """
    This function prepares a test state object for a specified set of test circuits and their states for use in a
    reliability test.

    Parameters
    ----------
    dev_mdl
    dev_counts: A dict of with keys of the different parameters to measure and the quantity of samples for each
    num_chps: The number of chips per lot to simulate
    num_lots: The number of lots of devices to simulate
    elapsed_time: Initial state should be at time 0, override exists to allow for manual setup of intermediate state

    Returns
    -------
    init_state
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
    for circ_prm in [prm for prm in devs if type(dev_mdl.prm_mdl(prm)) == CircPrmMdl]:
        for required_prm in dev_mdl.prm_mdl(circ_prm).get_required_prms(dev_mdl.prm_mdl_list):
            # Ensure that enough of the required degraded parameters are specified to support the circuit parameters
            if required_prm not in devs.keys() or devs[required_prm] < devs[circ_prm]:
                devs[required_prm] = devs[circ_prm]
                # If a report was used to specify device counts, update the report to include the changes
                if quantity_info:
                    quantity_info.dev_counts[required_prm] = devs[circ_prm]

    # Now initialize the physical state and override the elapsed time if specified
    init_state = dev_mdl.gen_init_state(devs, chps, lots)
    if elapsed_time != 0:
        init_state.elapsed = elapsed_time
    return init_state


def _sim_stress_step(step: StrsSpec, sim_state: TestSimState, dev_mdl: DeviceMdl,
                     test_env: PhysTestEnv, report: TestSimReport):
    # 1. Build the stochastically-adjusted set of stress conditions
    # Only need to generate stress conditions for device parameters that degrade, get the list of those parameters
    deg_prm_list = [prm for prm in dev_mdl.prm_mdl_list if type(dev_mdl.prm_mdl(prm)) == DegPrmMdl]
    cond_vals = test_env.gen_env_cond_vals(step.conditions, deg_prm_list, report, dev_mdl, 'stress')

    for prm in deg_prm_list:
        # 2. Calculate the equivalent stress times that would have been needed under the generated stress conditions to
        # obtain the prior degradation values. First we calculate the equivalent time to reach the current value of degradation
        equiv_times = dev_mdl.prm_mdl(prm).calc_equiv_strs_times(
            (report.num_lots, report.num_chps, report.dev_counts[prm]),
            sim_state.curr_deg_mech_vals[prm], cond_vals[prm], sim_state.init_deg_mech_vals[prm], sim_state.latent_var_vals[prm])
        # Now add on the time for the current stress phase
        for mech in equiv_times:
            equiv_times[mech] += step.duration.total_seconds() / SECONDS_PER_HOUR

        # 3. Simulate the degradation for each device after adding the equivalent prior stress time
        sim_state.curr_prm_vals[prm], sim_state.curr_deg_mech_vals[prm] = dev_mdl.prm_mdl(prm).calc_deg_vals(
            (report.num_lots, report.num_chps, report.dev_counts[prm]),
            equiv_times, cond_vals[prm], sim_state.init_prm_vals[prm], sim_state.latent_var_vals[prm], sim_state.curr_deg_mech_vals[prm]
        )

    # Update the elapsed real-world test time
    sim_state.elapsed += step.duration

    # 4. Report the stress conditions used during this step
    merged = {'stress step': step.name, 'duration': step.duration,
              'start time': sim_state.elapsed - step.duration, 'end time': sim_state.elapsed} | step.conditions
    # Use a DataFrame instead of a Series to simplify the process of merging the reports of different stress steps
    stress_report = pd.DataFrame(merged, index=[0])
    report.add_stress_report(stress_report)


def _sim_meas_step(step: MeasSpec, sim_state: TestSimState, dev_mdl: DeviceMdl,
                   test_env: PhysTestEnv, report: TestSimReport):
    """
    Given a test measurement specification, generate the set of observed values. The baseline/true parameter values
    are provided within the step and deg_data arguments, the measurement process adjusts these values according to test
    conditions, stochastic variations, and imperfect measurement instruments.
    """
    # Loop through the parameters to measure and determine if they are degraded parameters or independent conditions
    meas_results = pd.DataFrame()
    # Generate 'true' values for environmental conditions specified by the step (using their variability models)
    env_conds = test_env.gen_env_cond_vals(step.conditions, step.measurements, report, dev_mdl, 'measure')

    for prm in step.measurements:
        # There are three types of parameters: environmental conditions, degraded parameters and derived parameters
        if prm in sim_state.curr_prm_vals:
            # Adjust the param values based on environmental conditions during measurement and the instrument used
            measured = dev_mdl.prm_mdl(prm).calc_cond_shifted_vals(
                (report.num_lots, report.num_chps, step.measurements[prm]),
                env_conds[prm], sim_state.curr_prm_vals[prm], sim_state.latent_var_vals[prm])
            measured = TestSimReport.format_measurements(measured, prm, sim_state.elapsed, 'parameter')

        elif prm in dev_mdl.prm_mdl_list:
            # Derived parameter values that can depend on multiple degraded parameters, i.e. circuit models
            measured = dev_mdl.prm_mdl(prm).calc_circ_vals(
                step.measurements[prm], env_conds[prm], sim_state.curr_prm_vals, sim_state.latent_var_vals[prm])
            measured = TestSimReport.format_measurements(measured, prm, sim_state.elapsed, 'parameter')

        elif prm in step.conditions:
            # These parameters are not degraded parameters of the device but instead environmental parameters
            measured = TestSimReport.format_measurements(env_conds[prm][prm], prm, sim_state.elapsed, 'condition')

        else:
            raise MissingParamError(f"Requested measurement of param '{prm}' failed, param is not defined within the \
            simulated test.")

        # With the measurements completed, configure the dataframe for reporting and add to the merged dataframe
        measured['measured'] = test_env.meas_instm(prm).measure(measured['measured'])
        meas_results = pd.concat((meas_results, measured), ignore_index=True)

    if step.verbose:
        print(f"Conducted measurement {step.name} at simulation time {sim_state.elapsed}.")
    report.add_measurements(meas_results)


def simulate(test_def: TestSpec, dev_mdl: DeviceMdl, test_env: PhysTestEnv, init_state: TestSimState = None):
    """
    Simulate a given wear-out test using a given underlying model

    Parameters
    ----------
    test_def : A complete test description that the defines the stress conditions, durations, and data to collect
    dev_mdl : The underlying exact degradation model(s) to use to generate simulated test results
    test_env : Definition of the test environment that determines how imprecision is injected into the test results
    init_state: The starting values for the different device parameters that will degrade as the test proceeds

    Returns
    -------
    test_report: A TestReport object containing all relevant information on the test structure, execution, and results
    """
    # The test report object assembles all the collected test data into one data structure and tracks configuration info
    test_report = TestSimReport(test_def)

    # Prepare the simulation data persistence structure
    if not init_state:
        sim_state = gen_init_state(dev_mdl, quantity_info=test_report)
    else:
        # TODO: Ensure simulate doesn't modify the arguments (likely only init_state at risk of modification)
        #       this may be best achieved by giving TestSimState a copy method
        sim_state = init_state

    # We now execute the test step by step, sequentially performing measurements and stress intervals in order
    # Note that sim_state and test_report are mutated as the test progresses, the other arguments are left untouched
    for step in test_def:
        # Check whether the next step is a measurement or period of stress
        if type(step) is StrsSpec:
            _sim_stress_step(step, sim_state, dev_mdl, test_env, test_report)
        elif type(step) is MeasSpec:
            _sim_meas_step(step, sim_state, dev_mdl, test_env, test_report)
    return test_report
