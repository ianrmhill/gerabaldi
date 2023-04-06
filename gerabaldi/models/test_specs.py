# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

"""Classes for defining wear-out tests in terms of conditions, durations, execution steps, and collected data"""

from datetime import timedelta

from gerabaldi.exceptions import UserConfigError

__all__ = ['MeasSpec', 'StrsSpec', 'TestSpec']


class MeasSpec:
    """
    Definition for a set of measurements to take at some instant in time

    Attributes
    ----------
    measurements: dict of int
        Mapping from measurable device parameters to quantities of each to measure as part of the specification
    conditions: dict of float or int
        Mapping from environmental stress conditions to target values for the conditions to maintain during measurements
    name: str
        Optional descriptive name for the measurement specification
    verbose: bool
        DEPRECATED. Flag to print out notification when measurement is conducted, will be removed from class
    """
    def __init__(self, sample_counts: dict, conditions: dict, name: str = 'unspecified', print_action: bool = False):
        """
        Parameters
        ----------
        sample_counts: dict of int
            Mapping from measurable device parameters to quantities of each to measure as part of the specification
        conditions: dict of float or int
            Mapping from environmental stress conditions to target values for the conditions
        name: str, optional
            Optional descriptive name for the measurement specification (default 'unspecified')
        print_action: bool, optional
            DEPRECATED. Flag to print out notification when measurement is conducted, will be removed from class
        """
        self.measurements = sample_counts
        self.conditions = conditions
        self.name = name
        self.verbose = print_action


class StrsSpec:
    """
    Definition for a set of environmental stress conditions to maintain for a given stress duration

    Attributes
    ----------
    conditions: dict of float or int
        Mapping from environmental stress condition names to target values for those conditions
    duration: timedelta or float or int
        Duration of the stress phase
    name: str
        Optional descriptive name for the stress specification
    """
    def __init__(self, conditions: dict, duration: timedelta | int | float, name: str = 'unspecified'):
        """
        Parameters
        ----------
        conditions: dict of float or int
            Mapping from environmental stress condition names to target values for those conditions
        duration: timedelta or float or int
            Duration of the stress phase
        name: str, optional
            Descriptive name for the stress specification (default 'unspecified')
        """
        self.conditions = conditions
        # Currently, test lengths use units of hours, but are provided as timedelta objects
        if type(duration) in [int, float]:
            duration = timedelta(hours=duration)
        # Ensure the duration of the stress phase/cell is not 0
        if duration == timedelta():
            raise UserConfigError(f"Stress Specification '{name}' cannot have a time duration of 0.")
        self.duration = duration
        self.name = name


class TestSpec:
    """
    Definition of a test specification that can be executed/simulated using the Gerabaldi simulator.

    This class defines a wear-out test specification that can be executed experimentally. The process of executing a
    test involves conducting a series of stress and measurement tasks sequentially, thus this class is implemented to be
    iterable over the test steps. Every call to 'next' will return the next test action, either a stress interval to
    simulate or a set of measurements to take. Notably, however, this class does NOT track simulated degradation, it
    only specifies the test. The simulator itself persists the degradation and measurements between steps.

    Attributes
    ----------
    steps: list of StrsSpec or MeasSpec
        The sequential ordered list of stress phases and measurements to conduct that compose the test
    num_chps: int
        Quantity of chips per lot required by the test specification
    num_lots: int
        Quantity of production lots required by the test specification
    name: str
        Optional name for the test specification
    description: str
        Optional information about the test purpose, motivation, and execution details
    """
    def __init__(self, sequential_steps: list = None, num_chps: int = 1, num_lots: int = 1,
                 name: str = 'unspecified', description: str = 'none provided'):
        """
        Parameters
        ----------
        sequential_steps: list of StrsSpec or MeasSpec
            Ordered list of stress phases and measurements that compose the test
        num_chps: int
            Quantity of chips per lot required by the test specification
        num_lots: int
            Quantity of production lots required by the test specification
        name: str
            Optional name for the test specification
        description: str
            Optional information about the test purpose, motivation, and execution details
        """
        if sequential_steps is None:
            sequential_steps = []
        self.steps = sequential_steps
        self.num_chps = num_chps
        self.num_lots = num_lots
        self.description = description
        self.name = name

    def append_steps(self, steps: MeasSpec | StrsSpec | list, loop_for_duration: timedelta | int | float = None):
        """
        Append one or more test instruction steps to the end of the existing list of test steps

        Parameters
        ----------
        steps: MeasSpec or StrsSpec or list of MeasSpec or StrsSpec
            The test steps to append to the ordered list of steps already included in the test
        loop_for_duration: timedelta or int or float, optional
            If provided, the steps will be appended repeatedly until the added stress time surpasses this duration
        """
        # If not looping, simply add the steps onto the test list
        if loop_for_duration is None:
            if type(steps) is list:
                self.steps.extend(steps)
            else:
                self.steps.append(steps)
        else:
            # If we are looping the steps until some amount of elapsed time, we first convert the time for comparison
            if type(loop_for_duration) != timedelta:
                duration = timedelta(hours=loop_for_duration)
            else:
                duration = loop_for_duration

            # Ensure duration is not 0
            t = timedelta()
            if duration == t:
                raise UserConfigError('Cannot loop append test steps with a target stress duration of 0.')
            # Ensure steps to add are not purely measurement steps, as that would result in infinite steps being added
            if type(steps) is MeasSpec or (type(steps) is list and all(isinstance(step, MeasSpec) for step in steps)):
                raise UserConfigError(f"Cannot add steps to test in a loop if no stress phase present, infinite"
                                      f"test steps will result.")

            # Loop, appending the set of steps until the stress duration sums to greater than the total requested
            while t < duration:
                # Add a set of steps
                if type(steps) is list:
                    self.steps.extend(steps)
                else:
                    self.steps.append(steps)
                # Count the total time required for one loop of the steps
                if type(steps) is list:
                    for step in steps:
                        if type(step) == StrsSpec:
                            t += step.duration
                else:
                    t += steps.duration

            # Check if the duration was an integer multiple of the duration of the set of steps, warn if not
            if not (t / duration).is_integer():
                raise UserWarning(f"Appended steps did not result in an integer multiple of the duration, test "
                                  f"may be longer than intended.")

    def calc_samples_needed(self):
        """
        Determines the maximum number of samples of each parameter needed per test device by checking across the
        measurement specifications. Used to initialize the minimum number of samples needed for each simulation.

        Returns
        -------
        dict of int
            Mapping from measurable device parameters to the quantity that must be simulated to execute this test
        """
        samples_needed = {}
        for step in self.steps:
            if type(step) == MeasSpec:
                for prm in step.measurements:
                    # Add the parameter if not yet encountered
                    if prm not in samples_needed:
                        samples_needed[prm] = step.measurements[prm]
                    # Increase the sample size needed if the measurement requires more than the current amount
                    elif step.measurements[prm] > samples_needed[prm]:
                        samples_needed[prm] = step.measurements[prm]
        return samples_needed

    def __iter__(self):
        # This class is iterable to allow for test execution to proceed by simply looping through all the test steps
        self._curr_step = 0
        return self

    def __next__(self):
        if not self._curr_step >= len(self.steps):
            spec = self.steps[self._curr_step]
            self._curr_step += 1
        else:
            del self._curr_step
            raise StopIteration
        return spec
