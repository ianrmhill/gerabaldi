"""
Classes for defining wear-out tests in terms of conditions, durations, execution steps, and collected data.
"""

from datetime import timedelta

__all__ = ['MeasSpec', 'StrsSpec', 'TestSpec']


class MeasSpec:
    """Defines a set of measurements to take at some instant in time."""
    def __init__(self, sample_counts: dict, conditions: dict, name: str = 'unspecified', print_action=False):
        self.measurements = sample_counts
        self.conditions = conditions
        self.name = name
        self.verbose = print_action


class StrsSpec:
    """Defines a set of test conditions to simulate wear-out under for some duration."""
    def __init__(self, conditions: dict, duration: timedelta | int, name: str = 'unspecified'):
        self.conditions = conditions
        # Currently, test lengths use units of hours, but are provided as timedelta objects
        if type(duration) == int:
            duration = timedelta(hours=duration)
        self.duration = duration
        self.name = name


class TestSpec:
    """
    Class defining a wear-out test specification that can be executed experimentally. The process of executing a test
    involves conducting a series of stress and measurement tasks sequentially, thus this class is implemented to be
    iterable over the test steps. Every call to 'next' will return the next test action, either a stress interval to
    simulate or a set of measurements to take. Notably, however, this class does NOT track simulated degradation, it
    only specifies the test. The simulator persists the degradation and measurements between steps.
    """
    def __init__(self, sequential_steps: list = None, chp_count: int = 1, lot_count: int = 1,
                 description: str = 'none provided', name: str = 'unspecified'):
        # The test steps are a series of stress spec and measurement spec objects in sequential order
        if sequential_steps is None:
            sequential_steps = []
        self.steps = sequential_steps
        self.num_chps = chp_count
        self.num_lots = lot_count
        self.description = description
        self.name = name

    def add_steps(self, steps: MeasSpec | StrsSpec | list):
        """Append one or more test instruction steps to the end of the existing list of test steps."""
        if type(steps) is list:
            self.steps.extend(steps)
        else:
            self.steps.append(steps)

    def add_looped_steps(self, meas: MeasSpec | list, strs: StrsSpec | list,
                         duration: timedelta | int | float, fresh_meas: bool = True):
        """Convert and append a repeated stress-measure specification as the appropriate series of sequential steps."""
        if type(duration) != timedelta:
            duration = timedelta(hours=duration)
        # Loop, adding stress and measurement steps until the stress duration sums to greater than the total requested
        t = timedelta()
        while t < duration:
            # Add a measurement prior to any stress only if a fresh measurement is requested
            if t == timedelta() and fresh_meas:
                self.add_steps(meas)
            self.add_steps(strs)
            if type(strs) == list:
                for phase in strs:
                    t += phase.duration
            else:
                t += strs.duration
            self.add_steps(meas)

    def calc_samples_needed(self):
        """
        Determines the maximum number of samples of each parameter needed per test device by checking across the
        measurement specifications. Used to initialize the minimum number of samples needed for each simulation.
        """
        samples_needed = {}
        for step in self.steps:
            if type(step) == MeasSpec:
                for prm in step.measurements:
                    # Ignore measurements of stress conditions, we only care about samples of device parameters
                    if prm not in step.conditions:
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
