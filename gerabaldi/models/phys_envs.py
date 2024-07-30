# Copyright (c) 2024 Ian Hill
# SPDX-License-Identifier: Apache-2.0

"""Classes for defining physical noise sources and measurement capabilities in a test environment."""

from __future__ import annotations

from math import floor, log10
import numpy as np
import pandas as pd

# For some reason using pd.Series doesn't play well with type annotations
from pandas import Series

from gerabaldi.models.devices import DeviceMdl
from gerabaldi.models.reports import SimReport
from gerabaldi.models.random_vars import Deterministic, RandomVar
from gerabaldi.exceptions import InvalidTypeError, UserConfigError
from gerabaldi.helpers import _on_demand_import

# Optional imports are loaded using a helper function that suppresses import errors until attempted use
pymc = _on_demand_import('pymc')
pyro = _on_demand_import('pyro', 'pyro-ppl')
tc = _on_demand_import('torch')

__all__ = ['MeasInstrument', 'EnvVrtnMdl', 'PhysTestEnv']


class MeasInstrument:
    """
    Defines the capabilities for measuring a given parameter. Defaults to an ideal device/instrument.

    Attributes
    ----------
    name : str, optional
        The name of the measurement instrument, usually the param it measures (default is 'generic')
    """

    def __init__(self, name: str = 'generic', precision: int = None, error: RandomVar = None, meas_lims: tuple = None):
        """
        Parameters
        ----------
        name : str, optional
            The name of the measurement instrument, usually the param it measures (default is 'generic')
        precision : int, optional
            The number of significant figures that the measurement device can provide (default is None)
        error : RandomVar, optional
            A statistical distribution representing noise and inherent error effects (default is None)
        meas_lims : tuple of int or float, optional
            The maximum and minimum values that the measurement device can report (default is None)
        """
        self.name = name
        self._precision = precision
        self._error = error
        self._range = meas_lims

    def measure(self, true_vals: Series | float | np.ndarray):
        """
        Takes in exact parameter values and simulates the process of measuring them, returning the 'measured' values.

        Parameters
        ----------
        true_vals : int, float, ndarray, Series
            The set of underlying values to measure.

        Returns
        -------
        meas_vals : int, float, ndarray, Series
            The measured values formatted as the same type passed in.
        """

        # Identify the passed type, the return type will be formatted to match
        rtrn_type = type(true_vals)
        meas_vals = true_vals
        # Convert the input to a pandas Series if a different type
        if rtrn_type != pd.Series:
            meas_vals = pd.Series(meas_vals)

        if self._error:
            # Add offsets randomly sampled from the error statistical distribution
            meas_vals = meas_vals.add(self._error.sample(len(meas_vals)))
        if self._precision:
            # This line allows for rounding to significant figures instead of to a decimal place
            # The calculation for determining the sig figs -> decimal places is pretty standard and widely explained
            # Ternary operator is used to handle special cases that cause the rounding to fail otherwise
            meas_vals = meas_vals.apply(
                lambda x: x if x == 0 or np.isinf(x) else np.round(x, self._precision - int(floor(log10(abs(x)))) - 1),
            )
        if self._range:
            # If the measured value exceeds the measurement range then force it to the limit
            meas_vals = meas_vals.clip(lower=self._range[0], upper=self._range[1])

        # Convert the measured values back into the passed data type
        if rtrn_type == np.ndarray:
            meas_vals = np.array(meas_vals)
        elif rtrn_type is int or rtrn_type is float:
            meas_vals = meas_vals[0]
        return meas_vals


class EnvVrtnMdl:
    """
    Specifies the full stochastic model used for generating variations in environmental conditions. This model is quite
    similar to the LatentVar class, with a few key differences related to batches vs. lots and unspecified base vals.

    Attributes
    ----------
    name: str
        A useful name for the variable
    dev_vrtn_mdl: RandomVar
        The statistical distribution defining how the condition value varies between individual devices on a chip
    chp_vrtn_mdl: RandomVar
        The statistical distribution defining how the condition value varies between different chips within the test
    batch_vrtn_mdl: RandomVar
        The statistical distribution defining how the condition value varies for all chips under test in a batch
    vrtn_type: str
        How the distributions at different stochastic layers are combined to produce a final value, sum or product
    """

    # Don't let users add new attributes to the class, helps protect from typos causing bugs
    __slots__ = [
        'name',
        'vrtn_type',
        '_unitary',
        '_op',
        'batch_vrtn_mdl',
        '_batch_vrtns',
        'chp_vrtn_mdl',
        '_chp_vrtns',
        'dev_vrtn_mdl',
        '_dev_vrtns',
    ]
    name: str
    vrtn_type: str
    batch_vrtn_mdl: RandomVar
    chp_vrtn_mdl: RandomVar
    dev_vrtn_mdl: RandomVar

    def __init__(
        self,
        dev_vrtn_mdl: RandomVar = None,
        chp_vrtn_mdl: RandomVar = None,
        batch_vrtn_mdl: RandomVar = None,
        vrtn_type: str = 'offset',
        name: str = None,
    ):
        """
        Parameters
        ----------
        dev_vrtn_mdl: RandomVar, optional
            The distribution defining how the condition value varies between devices on a chip
        chp_vrtn_mdl: RandomVar, optional
            The distribution defining how the condition value varies across different chips
        batch_vrtn_mdl: RandomVar, optional
            The distribution defining how the condition value varies across production lots
        name: str, optional
            Name for the variable, used when transpiling the variable for use in CBI frameworks
        vrtn_type: str, optional
            The operation used to add the effects of the chip and lot variations, offset or scaling (default 'scaling')
        """
        self.name = name
        # Set this first to avoid a circular dependency problem between vrtn_type and <layer>_vrtn_mdl assignment logic
        self._unitary = 0
        # Note that there are no lot variations here as the test environment is completely ignorant of what lot a device
        # comes from. The analog is variations between test batches, however the simulator treats these as
        # separate tests instead, and so the test would simply be run twice. The env_vrtn_mdl effectively provides a
        # batch variation model, with the number of batches enforced to be one per test.
        self.batch_vrtn_mdl = batch_vrtn_mdl
        self.dev_vrtn_mdl = dev_vrtn_mdl
        self.chp_vrtn_mdl = chp_vrtn_mdl
        # To understand why 'offset' is default, consider temperature variation. If it was scaling, a 2% difference is
        # much greater at high temperatures, leading to variability coupled to the actual value. Although this works for
        # latent variable values due to their near-fixed base value, this is unlikely to be the intended behaviour for
        # the vast majority of environmental conditions.
        self.vrtn_type = vrtn_type

    def __setattr__(self, name, value):
        if name == 'vrtn_type':
            # If changing the variation type we need to also update all the attributes that are used to compute values
            if value not in ['scaling', 'offset']:
                raise InvalidTypeError(f"Latent {self.name} variation type can only be one of 'scaling', 'offset'.")
            # Only need to update all the attributes if the variation type is actually changing
            if not hasattr(self, 'vrtn_type') or self.vrtn_type != value:
                self._op = np.multiply if value == 'scaling' else np.add
                self._unitary = 0 if value == 'offset' else 1
                # Each variation model that is not user defined is reset to ensure the new unitary value is used
                if not self._batch_vrtns:
                    self.batch_vrtn_mdl = None
                if not self._chp_vrtns:
                    self.chp_vrtn_mdl = None
                if not self._dev_vrtns:
                    self.dev_vrtn_mdl = None
        # If changing a distribution we need to update its 'is defined' flag and set to generate the unitary if removed
        elif name == 'batch_vrtn_mdl':
            self._batch_vrtns = value is not None
            if not self._batch_vrtns:
                value = Deterministic(self._unitary)
        elif name == 'chp_vrtn_mdl':
            self._chp_vrtns = value is not None
            if not self._chp_vrtns:
                value = Deterministic(self._unitary)
        elif name == 'dev_vrtn_mdl':
            self._dev_vrtns = value is not None
            if not self._dev_vrtns:
                value = Deterministic(self._unitary)
        super().__setattr__(name, value)

    def gen_batch_vrtn(self, base_val: float) -> np.ndarray:
        """
        Generate a stochastic variation value for the environmental condition at the batch stochastic layer

        Parameters
        ----------
        base_val: numpy.ndarray or int or float
            The target value for the environmental condition that the variation will be applied to

        Returns
        -------
        numpy.ndarray
            A 3-dimensional (1, 1, 1) array with the varied value for the condition, indexing style is lot->chp->dev
        """
        return self._op(base_val, self.batch_vrtn_mdl.sample())

    def gen_chp_vrtns(self, base_vals: float | np.ndarray, num_chps: int = 1, num_lots: int = 1) -> np.ndarray:
        """
        Generate variational samples of the environmental condition at the between chips stochastic layer. Note that
        this method should be called AFTER any batch-level variations have been generated and applied, thus
        base_vals should already be a stochastic set of values at this point.

        Parameters
        ----------
        base_vals: numpy.ndarray or int or float
            The values for the environmental condition that the variations will be applied to
        num_chps: int, optional
            Quantity of devices/instances influenced by the condition on each chip (default 1)
        num_lots: int, optional
            Quantity of production lots of chips in the test batch (default 1)

        Returns
        -------
        numpy.ndarray
            A 3-dimensional array with the varied values for the condition, indexing style is lot->chp->dev
        """
        # First create an array of the correct size filled with the base value
        vals = np.full((num_lots, num_chps), base_vals)
        # Now include the effect of chip-to-chip variations
        return self._op(vals, self.chp_vrtn_mdl.sample(num_chps * num_lots).reshape((num_lots, num_chps)))

    def gen_dev_vrtns(self, base_vals: float | np.ndarray, num_devs: int = 1) -> np.ndarray:
        """
        Generate variational samples of the environmental condition at the between device stochastic layer. Note that
        this method should be called AFTER any batch and chip-level variations have been generated and applied, thus
        base_vals should already be a stochastic set of values at this point.

        Parameters
        ----------
        base_vals: numpy.ndarray or int or float
            The values for the environmental condition that the variations will be applied to
        num_devs: int, optional
            Quantity of devices/instances influenced by the condition on each chip (default 1)

        Returns
        -------
        numpy.ndarray
            A 3-dimensional array with the varied values for the condition, indexing style is lot->chp->dev
        """
        # Format the base value into a 2D numpy array if not already in that form to represent 1 chip and 1 batch
        if type(base_vals) is not np.ndarray:
            base_vals = np.array([[base_vals]])
        # Note that the operation broadcasting here needs to be done carefully, adding the device dimension to base_vals
        return self._op(
            np.expand_dims(base_vals, axis=-1),
            self.dev_vrtn_mdl.sample(num_devs * base_vals.size).reshape(
                (base_vals.shape[0], base_vals.shape[1], num_devs),
            ),
        )

    def gen_env_vrtns(
        self, base_val: float, num_devs: int = 1, num_chps: int = 1, num_lots: int = 1,
    ) -> np.ndarray:
        """
        Generate stochastic samples of the environmental condition for some number of samples, devices, and lots

        Parameters
        ----------
        base_val: int or float
            The target value for the environmental condition that the variations will be applied to
        num_devs: int, optional
            Quantity of devices/instances influenced by the condition on each chip (default 1)
        num_chps: int, optional
            Quantity of chips with latent variable devices/instances on them (default 1)
        num_lots: int, optional
            Quantity of production lots of chips (default 1)

        Returns
        -------
        numpy.ndarray
            A 3-dimensional array with the varied values for the condition, indexing style is lot->chp->dev
        """
        return self.gen_dev_vrtns(self.gen_chp_vrtns(self.gen_batch_vrtn(base_val), num_chps, num_lots), num_devs)


class PhysTestEnv:
    """
    Basic test environment in terms of measurement precisions and noise sources.

    Attributes
    ----------
    name : str
        The name of the test environment
    <prm_name>_var : RandomVar
        Stochastic distributions representing the variability for the environmental parameter in the attribute name
    <prm_name>_instm : MeasInstrument
        Measurement devices/instruments that will be used to measure the named parameter in the attribute name
    """

    def __init__(self, env_vrtns: dict | list = None, meas_instms: dict | list = None, env_name: str = 'unspecified'):
        """
        Parameters
        ----------
        env_name : str, optional
            The name of the test environment, defaults to 'unspecified'
        env_vrtns : dict or list of EnvVrtnModel, optional
            Stochastic models representing the variability of the parameter names in the dict keys or dist names
        meas_instms : dict or list of MeasInstrument, optional
            Measurement devices used to measure values for the parameters named in the dict keys or dist names
        """
        self.name = env_name
        # Non-mutable default argument setup
        if env_vrtns is None:
            env_vrtns = []
        elif type(env_vrtns) is dict:
            # If passed as a dictionary, set the object names and transform to list
            for prm in env_vrtns:
                env_vrtns[prm].name = prm
            env_vrtns = [env_vrtns[prm] for prm in env_vrtns]
        if meas_instms is None:
            meas_instms = []
        elif type(meas_instms) is dict:
            for prm in meas_instms:
                meas_instms[prm].name = prm
            meas_instms = [meas_instms[prm] for prm in meas_instms]

        # Create attributes for each parameter
        for prm in env_vrtns:
            setattr(self, prm.name + '_var', prm)
        for prm in meas_instms:
            setattr(self, prm.name + '_instm', prm)

    def meas_instm(self, prm: str) -> MeasInstrument:
        """
        Get the associated measurement instrument for the param, if none exists then return an ideal one.

        Parameters
        ----------
        prm : str
            The parameter to be measured

        Returns
        -------
        MeasDevice
            A device to measure the parameter, either one previously defined for it or an ideal one if none exists
        """
        return getattr(self, prm + '_instm', MeasInstrument(prm))

    def vrtn_mdl(self, prm: str) -> EnvVrtnMdl:
        """
        Get the associated variability model for the passed parameter.

        Parameters
        ----------
        prm : str
            The parameter to find the corresponding variation model for

        Returns
        -------
        EnvVrtnMdl
            A stochastic model representing the parameter variability, if none associated returns an exact distribution
        """
        return getattr(self, prm + '_var', EnvVrtnMdl())

    def gen_env_cond_vals(
        self,
        base_vals: dict,
        prms: list | dict,
        test_info: SimReport = None,
        dev_mdl: DeviceMdl = None,
        target: str = 'stress',
        num_chps: int = 1,
        num_lots: int = 1,
    ) -> dict:
        """
        Generate/sample stochastic values for environmental conditions/parameters

        Parameters
        ----------
        base_vals: dict
            Mapping of environmental conditions/parameters to their target base values before stochastic influences
        prms: list or dict
            The list of parameters to generate values for, if a dict values are the number of samples to generate
        test_info: SimReport, optional
            Report that can be used to determine the number of samples for each param and number of chips and lots
        dev_mdl: DeviceMdl, optional
            The full device model, used to determine which conditions are required to be generated to save work
        target: str, optional
            Either 'stress' or 'measure', used to determine which conditions are required to be generated to save work
        num_chps: int, optional
            How many chips per lot to generate sampled values for, only needed if test_info not provided
        num_lots: int, optional
            How many lots to generate sampled values for, only needed if test_info not provided

        Returns
        -------
        dict of numpy.ndarray
            A dictionary of 3D generated value arrays for each condition/parameter, indexing style lot->chp->dev
        """
        # First generate the chip variations for all conditions, as these are shared across parameters whereas device
        # specific conditions are unique and environmental conditions are oblivious to what lot a chip is from
        num_chps = test_info.num_chps if test_info else num_chps
        num_lots = test_info.num_lots if test_info else num_lots
        batch, chip = {}, {}
        for cond in base_vals:
            batch[cond] = self.vrtn_mdl(cond).gen_batch_vrtn(base_vals[cond])
            chip[cond] = self.vrtn_mdl(cond).gen_chp_vrtns(batch[cond], num_chps, num_lots)

        # If prms is a dictionary the values will be the number of devices to measure, otherwise we use the test info
        # dev counts as the reference for how many devices to generate condition values for
        if type(prms) is dict:
            prm_list = list(prms.keys())
            dev_counts = prms
        else:
            prm_list = prms
            if not test_info:
                raise UserConfigError(
                    'Generating environmental condition values requires device counts to be specified'
                    'either via the test report or through the "prms" argument.',
                )
            dev_counts = test_info.dev_counts

        # Now generate the device variations for each necessary condition for each parameter
        cond_vals = {}
        for prm in prm_list:
            # Determine which environmental conditions are relevant to the current parameter to avoid doing extra work
            if dev_mdl and prm in dev_mdl.prm_mdl_list:
                depend_conds = dev_mdl.prm_mdl(prm).get_dependencies(list(base_vals.keys()), target)
                cond_vals[prm] = {}
                for cond in depend_conds:
                    cond_vals[prm][cond] = self.vrtn_mdl(cond).gen_dev_vrtns(chip[cond], dev_counts[prm])
            else:
                # If the parameter isn't a device parameter then it must be one of the environmental conditions itself,
                # which since is not specified as part of the device model is assumed to be measured by a sensor(s) in
                # the test environment. We therefore treat each condition measurement sensor as a separate chip+device
                # in terms of stochastic variations, but shares the same test batch variation.
                # This assignment is left for readability purposes to make the following line less confusing
                cond = prm
                sensor_vals = self.vrtn_mdl(cond).gen_chp_vrtns(batch[cond], dev_counts[prm])
                # We treat each sensor as a separate chip with only 1 device for measurement each
                cond_vals[prm] = {cond: self.vrtn_mdl(cond).gen_dev_vrtns(sensor_vals, 1)}
        return cond_vals
