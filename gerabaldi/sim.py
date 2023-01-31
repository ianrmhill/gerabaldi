"""
Gerabaldi top-level simulation flow functions
"""

import copy
import pandas as pd
import numpy as np
from datetime import timedelta

from gerabaldi.models import *
from gerabaldi.exceptions import MissingParamError, UserConfigError

__all__ = ['simulate', 'gen_init_state']

SECONDS_PER_HOUR = 3600


def gen_init_state(dev_mdl: DeviceModel, dev_counts: dict, num_chps: int = 1, num_lots: int = 1,
                   elapsed_time: timedelta | int | float = timedelta()):
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
    # The number of samples of different parameters, chip, and lot counts can either be specified directly or
    # inferred based on the target test that will be conducted on the generated state
    num_devs, num_chps, num_lots = dev_counts, num_chps, num_lots

    # Generate the initial state for the test, all values that need to be persisted are kept in the simulation state
    init_state = TestSimState(dev_mdl.gen_init_vals(num_devs, num_chps, num_lots),
                              dev_mdl.gen_init_mech_vals(num_devs, num_chps, num_lots),
                              dev_mdl.gen_latent_vals(num_devs, num_chps, num_lots), elapsed_time)
    return init_state


def _sim_stress_step(step: StrsSpec, prior_state: TestSimState, sim_model: DeviceModel,
                     test_env: PhysTestEnv, report: TestSimReport):
    sim_state = copy.deepcopy(prior_state)

    # 1. Build the stochastically-adjusted set of stress conditions
    # Only need to generate stress conditions for device parameters that degrade, get the list of those parameters
    deg_prm_list = [prm for prm in sim_model.prm_mdl_list if type(sim_model.prm_mdl(prm)) == DegradedParamModel]
    cond_vals = test_env.gen_env_cond_vals(step.conditions, deg_prm_list, report, sim_model, 'stress')

    # 2. Calculate the equivalent stress times that would have been needed under the generated stress conditions to
    # obtain the prior degradation values.
    equiv_times = sim_model.calc_equiv_times(prior_state.curr_deg_mech_vals, cond_vals, prior_state.init_deg_mech_vals,
                                             prior_state.latent_var_vals)
    for prm in equiv_times:
        for mech in equiv_times[prm]:
            equiv_times[prm][mech] = np.add(equiv_times[prm][mech], step.duration.total_seconds() / SECONDS_PER_HOUR)

    # 3. Simulate the degradation for each device after adding the equivalent prior stress time
    sim_state.curr_prm_vals, sim_state.curr_deg_mech_vals =\
        sim_model.calc_dev_degradation(equiv_times, cond_vals, prior_state.init_prm_vals, prior_state.latent_var_vals,
                                       prior_state.curr_deg_mech_vals)
    # Update the elapsed real-world test time
    sim_state.elapsed = prior_state.elapsed + step.duration

    # 4. Report the stress conditions used during this step
    merged = {'stress step': step.name, 'duration': step.duration,
              'start time': sim_state.elapsed - step.duration, 'end time': sim_state.elapsed} | step.conditions
    # Use a DataFrame instead of a Series to simplify the process of merging the reports of different stress steps
    stress_report = pd.DataFrame(merged, index=[0])
    report.add_stress_report(stress_report)

    return sim_state


def _sim_meas_step(step: MeasSpec, deg_data: TestSimState, sim_model: DeviceModel,
                   test_env: PhysTestEnv, report: TestSimReport):
    """
    Given a test measurement specification, generate the set of observed values. The baseline/true parameter values
    are provided within the step and deg_data arguments, the measurement process adjusts these values according to test
    conditions, stochastic variations, and imperfect measurement instruments.
    """
    # Loop through the parameters to measure and determine if they are degraded parameters or independent conditions
    meas_results = pd.DataFrame()
    # Generate 'true' values for environmental conditions specified by the step (using their variability models)
    env_conds = test_env.gen_env_cond_vals(step.conditions, step.measurements, report, sim_model, 'measure')

    for prm in step.measurements:
        # There are three types of parameters: environmental conditions, degraded parameters and derived parameters
        if prm in deg_data.curr_prm_vals:
            # Adjust the param values based on environmental conditions during measurement and the instrument used
            measured = sim_model.prm_mdl(prm).calc_cond_shifted_vals(
                step.measurements[prm], env_conds[prm], deg_data.curr_prm_vals[prm], deg_data.latent_var_vals[prm])
            measured = TestSimReport.format_measurements(measured, prm, deg_data.elapsed, 'parameter')

        elif prm in sim_model.prm_mdl_list:
            # Derived parameter values that can depend on multiple degraded parameters, i.e. circuit models
            measured = sim_model.prm_mdl(prm).calc_circ_vals(
                step.measurements[prm], env_conds[prm], deg_data.curr_prm_vals, deg_data.latent_var_vals[prm])
            measured = TestSimReport.format_measurements(measured, prm, deg_data.elapsed, 'parameter')

        elif prm in step.conditions:
            # These parameters are not degraded parameters of the device but instead environmental parameters
            measured = TestSimReport.format_measurements(env_conds[prm][prm], prm, deg_data.elapsed, 'condition')

        else:
            raise MissingParamError(f"Requested measurement of param '{prm}' failed, param is not defined within the \
            simulated test.")

        # With the measurements completed, configure the dataframe for reporting and add to the merged dataframe
        measured['measured'] = test_env.meas_instm(prm).measure(measured['measured'])
        meas_results = pd.concat((meas_results, measured), ignore_index=True)

    if step.verbose:
        print(f"Conducted measurement {step.name} at simulation time {deg_data.elapsed}.")
    report.add_measurements(meas_results)


def simulate(test_def: TestSpec, sim_model: DeviceModel, test_env: PhysTestEnv, init_state: TestSimState = None):
    """
    Simulate a given wear-out test using a given underlying model

    Parameters
    ----------
    test_def : A complete test description that the defines the stress conditions, durations, and data to collect
    sim_model : The underlying exact degradation model(s) to use to generate simulated test results
    test_env : Definition of the test environment that determines how imprecision is injected into the test results
    init_state: The starting values for the different device parameters that will degrade as the test proceeds

    Returns
    -------
    test_report: A TestReport object containing all relevant information on the test structure, execution, and results
    """
    # The test report object assembles all the collected test data into one data structure and tracks configuration info
    test_report = TestSimReport(test_def)

    if not init_state:
        init_state = gen_init_state(sim_model, test_report.dev_counts, test_report.num_chps, test_report.num_lots)
    # Prepare the simulation data persistence structure, need a copy as we must also persist the initial values
    deg_state = copy.deepcopy(init_state)

    # We now begin to execute the test step by step, sequentially performing measurements and stress intervals in order
    for step in test_def:
        # Check whether the next step is a measurement or period of stress
        if type(step) is StrsSpec:
            deg_state = _sim_stress_step(step, deg_state, sim_model, test_env, test_report)
        elif type(step) is MeasSpec:
            _sim_meas_step(step, deg_state, sim_model, test_env, test_report)

    return test_report
