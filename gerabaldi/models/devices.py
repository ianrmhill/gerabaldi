# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

"""Classes for defining degradation models for tuning or use in simulating wear-out tests"""

from __future__ import annotations

import inspect
import numpy as np
from scipy.optimize import minimize_scalar
from typing import Callable

from gerabaldi.math import minimize
from gerabaldi.models.random_vars import RandomVar, Deterministic
from gerabaldi.models.states import SimState
from gerabaldi.exceptions import InvalidTypeError, UserConfigError
from gerabaldi.helpers import _on_demand_import, _loop_compute

# Optional imports are loaded using a helper function that suppresses import errors until attempted use
pymc = _on_demand_import('pymc')
pyro = _on_demand_import('pyro', 'pyro-ppl')
dist = _on_demand_import('pyro.distributions', 'pyro-ppl')
tc = _on_demand_import('torch')


__all__ = ['LatentVar', 'InitValMdl', 'CondShiftMdl', 'DegMechMdl', 'FailMechMdl',
           'DegPrmMdl', 'CircPrmMdl', 'DeviceMdl']


class LatentVar:
    """
    Modelling framework for latent variables within wear-out models.

    Representation of a probabilistic latent/hidden variable/parameter within Gerabaldi. Consists of a probability
    distribution that defines the value of the latent variable along with optional chip and lot-level distributions that
    are either multiplied or summed with the base device-level distribution to allow for sampling of values for the
    variable. Can optionally override the probabilistic modelling by providing a deterministic value, resulting in a
    more conventional style of simulation.

    Attributes
    ----------
    name: str
        A useful name for the variable
    dev_vrtn_mdl: RandomVar
        The statistical distribution defining the base value of the variable
    chp_vrtn_mdl: RandomVar
        The statistical distribution defining how the base value of the variable varies between different chips
    lot_vrtn_mdl: RandomVar
        The statistical distribution defining how the base value of the variable varies between production lots
    vrtn_type: str
        How the distributions at different stochastic layers are combined to produce a final value, sum or product
    deter_val: float or int
        Deterministic value override to obtain conventional simulation behaviour if no stochastic behaviour is desired
    """
    __slots__ = ['name', 'vrtn_type', 'deter_val', '_unitary', '_op',
                 'dev_vrtn_mdl', '_dev_vrtns', 'chp_vrtn_mdl', '_chp_vrtns', 'lot_vrtn_mdl', '_lot_vrtns']
    name: str
    vrtn_type: str
    dev_vrtn_mdl: RandomVar
    chp_vrtn_mdl: RandomVar
    lot_vrtn_mdl: RandomVar
    deter_val: int | float

    def __init__(self, dev_vrtn_mdl: RandomVar = None, chp_vrtn_mdl: RandomVar = None, lot_vrtn_mdl: RandomVar = None,
                 deter_val: float | int = None, name: str = None, vrtn_type: str = 'scaling'):
        """
        Parameters
        ----------
        dev_vrtn_mdl: RandomVar, optional
            The distribution defining the base value of the variable
        chp_vrtn_mdl: RandomVar, optional
            The distribution defining how the variable value varies across different chips
        lot_vrtn_mdl: RandomVar, optional
            The distribution defining how the variable value varies across production lots
        deter_val: float or int, optional
            Deterministic base value for the variable
        name: str, optional
            Name for the variable, used when transpiling the variable for use in CBI frameworks
        vrtn_type: str, optional
            The operation used to add the effects of the chip and lot variations, offset or scaling (default 'scaling')
        """
        # Set unitary first to avoid circular dependency problem between vrtn_type and <layer>_vrtn_mdl setattr logic
        self._unitary = 0
        self.dev_vrtn_mdl = dev_vrtn_mdl
        self.chp_vrtn_mdl = chp_vrtn_mdl
        self.lot_vrtn_mdl = lot_vrtn_mdl
        self.deter_val = deter_val
        # These must be set after so that the distributions can have their names and unitary values updated if needed
        self.name = name
        self.vrtn_type = vrtn_type
        if not dev_vrtn_mdl and deter_val is None:
            raise UserConfigError(f"Latent {self.name} definition must include either a device distribution or"
                                  "deterministic value argument.")

    def __setattr__(self, name, value):
        if name == 'name':
            # Need to also update the naming of the probabilistic distributions to have CBI export work correctly
            if value is not None:
                if self._dev_vrtns:
                    # Set the distribution name to be the variable name + '_dev' suffix, unless the device variation
                    # model fully specifies the variable value, in which case don't add the suffix
                    if not self._chp_vrtns and not self._lot_vrtns:
                        self.dev_vrtn_mdl.name = value
                    else:
                        self.dev_vrtn_mdl.name = value + '_dev'
                if self._chp_vrtns:
                    self.chp_vrtn_mdl.name = value + '_chp'
                if self._lot_vrtns:
                    self.lot_vrtn_mdl.name = value + '_lot'
        if name == 'vrtn_type':
            # If changing the variation type we need to also update all the attributes that are used to compute values
            if value not in ['scaling', 'offset']:
                raise InvalidTypeError(f"Latent {self.name} variation type can only be one of 'scaling', 'offset'.")
            # Only need to update all the attributes if the variation type is actually changing
            if not hasattr(self, 'vrtn_type') or self.vrtn_type != value:
                self._op = np.multiply if value == 'scaling' else np.add
                self._unitary = 0 if value == 'offset' else 1
                # Each variation model that is not user defined is reset to ensure the new unitary value is used
                if not self._lot_vrtns:
                    self.lot_vrtn_mdl = None # noqa: PyTypeChecker
                if not self._chp_vrtns:
                    self.chp_vrtn_mdl = None # noqa: PyTypeChecker
                if not self._dev_vrtns:
                    self.dev_vrtn_mdl = None # noqa: PyTypeChecker
        # If changing a distribution we need to update its 'is defined' flag and set to generate the unitary if removed
        if name == 'dev_vrtn_mdl':
            self._dev_vrtns = False if value is None else True
            if not self._dev_vrtns:
                value = Deterministic(self._unitary)
        if name == 'chp_vrtn_mdl':
            self._chp_vrtns = False if value is None else True
            if not self._chp_vrtns:
                value = Deterministic(self._unitary)
        if name == 'lot_vrtn_mdl':
            self._lot_vrtns = False if value is None else True
            if not self._lot_vrtns:
                value = Deterministic(self._unitary)
        super(LatentVar, self).__setattr__(name, value)

    def gen_latent_vals(self, num_devs: int = 1, num_chps: int = 1, num_lots: int = 1) -> np.ndarray:
        """
        Generate stochastic samples of the variable for the specified number of individual samples, devices, and lots.

        Parameters
        ----------
        num_devs: int, optional
            Quantity of devices/instances of this variable on each chip (default 1)
        num_chps: int, optional
            Quantity of chips with latent variable devices/instances on them (default 1)
        num_lots: int, optional
            Quantity of production lots of chips (default 1)

        Returns
        -------
        numpy.ndarray
            A 3-dimensional array with the sampled values for the variable, indexing style is lot->chp->dev
        """

        # If any/all of the distributions are not user-defined, the samples will all be unitary values (i.e. no-ops)
        lot = self.lot_vrtn_mdl.sample(num_lots).reshape((num_lots, 1, 1))
        chp = self.chp_vrtn_mdl.sample(num_lots * num_chps).reshape((num_lots, num_chps, 1))
        dev = self.dev_vrtn_mdl.sample(num_lots * num_chps * num_devs).reshape((num_lots, num_chps, num_devs))

        # Only include base deterministic value if defined
        if self.deter_val is not None:
            vals = self._op(np.full((num_lots, num_chps, num_devs), self.deter_val), dev)
        else:
            vals = dev
        return self._op(self._op(vals, chp), lot)

    def cbi_export(self, target_framework: str = 'pymc'):
        """
        Transpile the latent variable model to the equivalent definition within a Python CBI framework

        Parameters
        ----------
        target_framework: str, optional
            The name of the CBI framework to transpile the model to (default 'pymc')

        Returns
        -------
        various
            The transpiled version of the variable for the requested CBI framework
        """
        # A probabilistic distribution must be defined for the device variation model. If not, raise an error
        if self._dev_vrtns:
            if target_framework == 'pymc':
                op = np.multiply if self.vrtn_type == 'scaling' else np.add
            elif target_framework == 'pyro':
                op = tc.multiply if self.vrtn_type == 'scaling' else tc.add
            else:
                raise NotImplementedError(f"Requested target CBI framework {target_framework} is not yet supported.")

            # If a deterministic value was used we may still be able to convert the variable to a probabilistic form
            old_centre = None
            if self.deter_val:
                # If the dev_vrtn_mdl distribution has a "centre" we can set the centre to the deterministic value and
                # proceed, however if the distribution is bimodal, uniform, etc. an error must be raised.
                old_centre = self.dev_vrtn_mdl.get_centre()
                self.dev_vrtn_mdl.set_centre(self.deter_val, op)

            # Normal probabilistic model export, only incorporate the stochastic layers if used
            if self._chp_vrtns and self._lot_vrtns:
                dev = self.dev_vrtn_mdl.get_cbi_form(target_framework=target_framework)
                chp = self.chp_vrtn_mdl.get_cbi_form(target_framework=target_framework)
                lot = self.lot_vrtn_mdl.get_cbi_form(target_framework=target_framework)
                if target_framework == 'pymc':
                    cbi_dist = pymc.Deterministic(self.name, op(op(dev, chp), lot))
                else:
                    cbi_dist = op(op(dev, chp), lot)
            elif self._chp_vrtns and not self._lot_vrtns:
                dev = self.dev_vrtn_mdl.get_cbi_form(target_framework=target_framework)
                chp = self.chp_vrtn_mdl.get_cbi_form(target_framework=target_framework)
                if target_framework == 'pymc':
                    cbi_dist = pymc.Deterministic(self.name, op(dev, chp))
                else:
                    cbi_dist = op(dev, chp)
            elif not self._chp_vrtns and self._lot_vrtns:
                dev = self.dev_vrtn_mdl.get_cbi_form(target_framework=target_framework)
                lot = self.lot_vrtn_mdl.get_cbi_form(target_framework=target_framework)
                if target_framework == 'pymc':
                    cbi_dist = pymc.Deterministic(self.name, op(dev, lot))
                else:
                    cbi_dist = op(dev, lot)
            else:
                cbi_dist = self.dev_vrtn_mdl.get_cbi_form(target_framework=target_framework)
            # Return the dev_vrtn_mdl centre value to previous if it was changed to avoid any method side effects
            if old_centre:
                self.dev_vrtn_mdl.set_centre(old_centre)
            return cbi_dist
        else:
            raise Exception(f"Cannot export deterministic latent variable {self.name} for CBI use.")


class LatentMdl:
    """
    Base class for modelling physical phenomenon with hidden/uncertain underlying parameters.

    This class should not be instantiated directly, use an inheriting class.
    """
    def __init__(self, mdl_eqn: Callable = None, mdl_name: str = None, **latent_vars: LatentVar):
        self.name = mdl_name
        # Add default to ensure the object attribute can always be called
        if not mdl_eqn:
            def mdl_eqn(x): return x
        self.compute = mdl_eqn
        self.compute_args = inspect.signature(self.compute).parameters.keys()
        # Define 'true'/underlying values for equation inputs used to simulate degradation
        # No latent variable values should be specified when performing model inference
        self.latent_vars = []
        for var in latent_vars:
            # If no name was specified for the latent variable, nice to set it to the same name as used in the model
            if latent_vars[var].name == '' or not latent_vars[var].name:
                latent_vars[var].name = var
            # Indicate the latent variable attributes as private since they're supposed to be uncertain
            setattr(self, f"_latent_{var}", latent_vars[var])
            self.latent_vars.append(var)

    def latent_var(self, var: str):
        """
        Get a latent variable within the model by name

        Parameters
        ----------
        var: str
            The name of the variable to retrieve the LatentVar object for

        Returns
        -------
        LatentVar
            The Gerabaldi representation of the requested latent variable
        """
        return getattr(self, f"_latent_{var}")

    def gen_latent_vals(self, num_devs: int = 1, num_chps: int = 1, num_lots: int = 1) -> dict:
        """
        Generate 'true'/underlying sampled values for all the model's latent variables for each device, chip, and lot

        Parameters
        ----------
        num_devs: int, optional
            Quantity of devices per chip to generate samples for (default 1)
        num_chps: int, optional
            Quantity of chips per lot to generate samples for (default 1)
        num_lots: int, optional
            Quantity of production lots to generate samples for (default 1)

        Returns
        -------
        dict of numpy.ndarray
            Mapping from latent variable name to the 3D array of sampled values, indexing style lot->chp->dev
        """
        vals = {}
        for var in self.latent_vars:
            vals[var] = self.latent_var(var).gen_latent_vals(num_devs, num_chps, num_lots)
        return vals

    def export_to_cbi(self, target_framework: str = 'pymc', observed: dict = None):
        """
        Transpile the latent model for a target CBI framework. Method not yet stable, likely to change.

        Parameters
        ----------
        target_framework: str, optional
            The target CBI framework (default 'pymc')
        observed: dict, optional
            Mapping from stress conditions to measured/observed values, used for inference

        Returns
        -------
        various
            The transpiled equivalent latent model for the target CBI framework
        """
        # First create latent variables for the hidden model parameters
        vals = {}
        for var in self.latent_vars:
            vals[var] = self.latent_var(var).cbi_export(target_framework)

        # Now add the observed stress conditions to the compute dictionary
        for strs in observed:
            if strs != self.name:
                vals[strs] = observed[strs]

        # Now define the model's compute function
        computed = self.compute(**vals)

        # Finally, define the output as a variable as well
        if target_framework == 'pymc':
            return pymc.Normal(name=self.name, mu=computed, sigma=1, observed=observed[self.name])
        elif target_framework == 'pyro':
            return pyro.sample(self.name, dist.Normal(computed, 0.1), obs=observed[self.name]).to_event(1)


class MechMdl(LatentMdl):
    """
    Class for modelling the effect of a wear-out reliability mechanism on some physical parameter.

    This class should not be instantiated directly, use an inheriting class.
    """
    def __init__(self, mech_eqn: Callable = None, mdl_name: str = None, unitary_val: int = 0, **latent_vars: LatentVar):
        if not mech_eqn:
            # Default is a non-degrading model, parameter stays the same (i.e. 'fresh') regardless of stress and time
            def no_wear_out(): return unitary_val
            mech_eqn = no_wear_out
        self.unitary = unitary_val
        super().__init__(mech_eqn, mdl_name, **latent_vars)

    def gen_init_vals(self, num_devs, num_chps, num_lots):
        """
        Initialize a 3D array of unitary values for the initial degradation of the mechanism. This method assumes zero
        degradation at initial conditions, though initial values can be varied using InitValMdl or by incorporating
        non-zero initial degradation into the mechanism equation/function itself.

        Parameters
        ----------
        num_devs: int
            Quantity of devices to initialize
        num_chps: int
            Quantity of chips to initialize
        num_lots: int
            Quantity of lots to initialize

        Returns
        -------
        numpy.ndarray
            A 3D array of unitary values, indexing style lot->chp->dev
        """
        return np.full((num_lots, num_chps, num_devs), self.unitary)

    def calc_equiv_strs_time(self, deg_val, init_val, strs_conds, latents, dims):
        raise NotImplementedError('Mechanism does not define how to compute value changes under time-varying stress')

    def calc_deg_vals(self, times: np.ndarray, pre_deg_vals: np.ndarray,
                      strs_conds: dict, latents: dict, dims: tuple) -> np.ndarray:
        """
        Calculate the underlying degraded values for the parameter given a set of stress conditions and stress duration.
        Note that this method does not calculate a measured value, only the 'true' underlying degraded parameter value.

        Parameters
        ----------
        times: numpy.ndarray
            Stress duration for each sample of the mechanism
        pre_deg_vals: numpy.ndarray
            Pre-stress degradation values for each sample of the mechanism
        strs_conds: dict of numpy.ndarray
            Mapping from stress conditions to stress values for each sample of the mechanism
        latents: dict of numpy.ndarray
            Mapping from latent variables to their sampled values for each instance of the mechanism
        dims: tuple of int
            The number of lots, chips per lot, and devices per chip that compose the set of all samples of the mechanism

        Returns
        -------
        numpy.ndarray
            Post-stress degradation values for each sample of the mechanism
        """
        # To support degradation mechanisms that exhibit threshold behaviours, we first identify whether some samples
        # won't degrade under the current stress conditions based on the failure to find a valid equivalent stress time
        no_deg_mask = np.where(times > 9e9, pre_deg_vals, np.inf)

        # Add the stress time to the argument dict
        arg_vals = {'time': times}
        # Add all stress conditions, latent variables, and potentially initial values to the argument list
        for arg in self.compute_args:
            if arg in latents:
                arg_vals[arg] = latents[arg]
            elif arg in strs_conds:
                arg_vals[arg] = strs_conds[arg]
        # Calculate the degradation for all samples
        try:
            deg_vals = self.compute(**arg_vals)
        except ValueError:
            deg_vals = _loop_compute(self.compute, arg_vals, dims)

        # Overwrite the calculated degradation for the samples that we knew wouldn't degrade to the previous values
        # Note that we do a little extra work here as deg_vals is still computed for the non-degrading samples, but this
        # allows for array computation.
        return np.where(no_deg_mask == np.inf, deg_vals, no_deg_mask)


class DegMechMdl(MechMdl):
    """
    Class for modelling the effect of a degradation mechanism on some physical parameter

    Attributes
    ----------
    name: str
        A descriptive name for the mechanism model
    compute: Callable
        A callable Python function that computes the mechanism output value as a function of required inputs
    compute_args: list of str
        A listing of the model's compute equation function signature for easy identification of required model inputs
    latent_vars: list of str
        A listing of the latent variables that the model's compute equation takes as inputs
    unitary: int or float
        The value for the mechanism output that results in no effect to the parent parameter's output value
    """
    def __init__(self, mech_eqn: Callable = None, mdl_name: str = None, unitary_val: int = 0, **latent_vars: LatentVar):
        """
        Parameters
        ----------
        mech_eqn: Callable, optional
            Callable Python function computing the mechanism output value using required inputs (default no degradation)
        mdl_name: str, optional
            Descriptive name for the mechanism model
        unitary_val: int
            The value for the mechanism output that results in no effect to the parent parameter's value (default 0)
        **latent_vars: dict of LatentVar, optional
            Latent variable models used as part of the mechanism's compute equation
        """
        super().__init__(mech_eqn, mdl_name, unitary_val, **latent_vars)

    def calc_equiv_strs_time(self, deg_val: int | float, init_val: int | float,
                             strs_conds: dict, latents: dict, dims: tuple) -> float:
        """
        This method back-calculates the length of time required to reach the current level of degradation under a given
        set of stress conditions for all the different samples in the test. See the Gerabaldi VTS 2023 paper or
        documentation for an explanation of the conceptual idea and procedure used to calculate these values.

        Parameters
        ----------
        deg_val: int or float
            The current degradation for the instance of the mechanism
        init_val: int or float
            The initial degradation for the instance of the mechanism (typically 0)
        strs_conds: dict
            Mapping from named stress conditions to their stress values for this mechanism instance
        latents: dict
            Mapping from latent variables to their sampled values for this mechanism instance
        dims: tuple of int
            The number of lots, chips, and devices of form (lot, chp, dev)

        Returns
        -------
        float
            The amount of time required to reach the current value of degradation under the specified stress conditions
        """
        # Computes the absolute difference between the target value and the calculated value using a given time
        def residue(time, curr_deg_val, conds, ltnts):
            arg_vals = {'time': time}
            for arg in self.compute_args:
                if arg in ltnts:
                    arg_vals[arg] = ltnts[arg]
                elif arg in conds:
                    arg_vals[arg] = conds[arg]
            # Calculate the degradation for the sample
            return abs(curr_deg_val - self.compute(**arg_vals))

        # Minimize the difference between the output and the target/observed value using scipy optimizer
        # Since we are minimizing time we use the bounded method to ensure our time doesn't go negative
        return minimize(residue, extra_args={'curr_deg_val': deg_val, 'conds': strs_conds, 'ltnts': latents},
                        bounds=(1e-3, 1e10), maxiter=50, log_gold=True)


class FailMechMdl(MechMdl):
    """
    Class for modelling the effect of a time-independent hard failure mechanism on some physical parameter

    Attributes
    ----------
    name: str
        A descriptive name for the mechanism model
    compute: Callable
        A callable Python function that computes the mechanism output value as a function of required inputs
    compute_args: list of str
        A listing of the model's compute equation function signature for easy identification of required model inputs
    latent_vars: list of str
        A listing of the latent variables that the model's compute equation takes as inputs
    unitary: int or float
        The value for the mechanism output that results in no effect to the parent parameter's output value
    """
    def __init__(self, mech_eqn: Callable = None, mdl_name: str = None, unitary_val: int = 0, **latent_vars: LatentVar):
        """
        Parameters
        ----------
        mech_eqn: Callable, optional
            Callable Python function computing the mechanism output value using required inputs (default no degradation)
        mdl_name: str, optional
            Descriptive name for the mechanism model
        unitary_val: int
            The value for the mechanism output that results in no effect to the parent parameter's value (default 0)
        **latent_vars: dict of LatentVar, optional
            Latent variable models used as part of the mechanism's compute equation
        """
        super().__init__(mech_eqn, mdl_name, unitary_val, **latent_vars)

    def calc_equiv_strs_time(self, deg_val: int | float, init_val: int | float,
                             strs_conds: dict, latents: dict, dims: tuple) -> float:
        """
        This method matches the signature of the DegMechMdl implementation but always returns zero. This is because a
        hard-failure mechanism has no degradation component, only instantaneous failures, thus the concept of an
        equivalent stress time is meaningless. Either the failure has occurred or it hasn't, and this failure
        probability is time independent.

        Parameters
        ----------
        deg_val: int or float
            Unused
        init_val: int or float
            Unused
        strs_conds: dict
            Unused
        latents: dict
            Unused
        dims: tuple of int
            Unused

        Returns
        -------
        float
            Always 0
        """
        return 0.0

    def calc_deg_vals(self, times: np.ndarray, pre_deg_vals: np.ndarray,
                      strs_conds: dict, latents: dict, dims: tuple) -> np.ndarray:
        """
        Calculate the failure state for the mechanism given a set of stress conditions and stress duration

        Parameters
        ----------
        times: numpy.ndarray
            Stress duration for each sample of the mechanism
        pre_deg_vals: numpy.ndarray
            Pre-stress failure states for each sample of the mechanism
        strs_conds: dict of numpy.ndarray
            Mapping from stress conditions to stress values for each sample of the mechanism
        latents: dict of numpy.ndarray
            Mapping from latent variables to their sampled values for each instance of the mechanism
        dims: tuple of int
            The number of lots, chips per lot, and devices per chip that compose the set of all samples of the mechanism

        Returns
        -------
        numpy.ndarray
            Post-stress failure states for each sample of the mechanism
        """
        # Once the state has failed (i.e., unitary is the initial/unfailed state) the mechanism remains failed
        return np.where(pre_deg_vals != self.unitary, pre_deg_vals,
                        super().calc_deg_vals(times, pre_deg_vals, strs_conds, latents, dims))


class CondShiftMdl(LatentMdl):
    """
    Class for modelling the instantaneous dependency of the value of some physical parameter on stress conditions

    Attributes
    ----------
    name: str
        A descriptive name for the shift model
    compute: Callable
        A callable Python function that computes the conditional output value as a function of required inputs
    compute_args: list of str
        A listing of the model's compute equation function signature for easy identification of required model inputs
    latent_vars: list of str
        A listing of the latent variables that the model's compute equation takes as inputs
    unitary: int or float
        The value for the conditional output that results in no effect to the parent parameter's output value
    """
    def __init__(self, cond_shift_eqn: Callable = None, mdl_name: str = None,
                 unitary_val: int = 0, **latent_vars: LatentVar):
        """
        Parameters
        ----------
        cond_shift_eqn: Callable, optional
            Callable Python function computing the conditional output value using required inputs (default no shift)
        mdl_name: str, optional
            Descriptive name for the shift model
        unitary_val: int
            The value for the mechanism output that results in no effect to the parent parameter's value (default 0)
        **latent_vars: dict of LatentVar, optional
            Latent variable models used as part of the mechanism's compute equation
        """
        if not cond_shift_eqn:
            def no_cond_shift(): return unitary_val
            cond_shift_eqn = no_cond_shift
        self.unitary = unitary_val
        super().__init__(cond_shift_eqn, mdl_name, **latent_vars)

    def calc_cond_vals(self, strs_conds: dict, latents: dict, dims: tuple) -> np.ndarray:
        """
        Compute the conditional output values for the model as a function of stress conditions

        Parameters
        ----------
        strs_conds: dict of numpy.ndarray
            Mapping from stress conditions to stress values for each sample of the parent parameter
        latents: dict of numpy.ndarray
            Mapping from latent variables to their sampled values for each instance of the parent parameter
        dims: tuple of int
            The number of lots, chips per lot, and devices per chip that compose the set of all samples

        Returns
        -------
        numpy.ndarray
            The conditional output values of the model for each sample, indexing style lot->chp->dev
        """
        # Add all stress condition values to the already well formatted latents dict
        arg_vals = {}
        for arg in self.compute_args:
            if arg in latents:
                arg_vals[arg] = latents[arg]
            elif arg in strs_conds:
                arg_vals[arg] = strs_conds[arg]
        try:
            return self.compute(**arg_vals)
        except ValueError:
            return _loop_compute(self.compute, arg_vals, dims)


class InitValMdl(LatentMdl):
    """
    Class for modelling the initial value of some physical parameter that can degrade or fail

    Attributes
    ----------
    name: str
        A descriptive name for the shift model
    compute: Callable
        A callable Python function that computes the conditional output value as a function of required inputs
    compute_args: list of str
        A listing of the model's compute equation function signature for easy identification of required model inputs
    latent_vars: list of str
        A listing of the latent variables that the model's compute equation takes as inputs
    """
    def __init__(self, init_val_eqn: Callable = None, mdl_name: str = None, **latent_vars: LatentVar):
        """
        Parameters
        ----------
        cond_shift_eqn: Callable, optional
            Callable Python function computing the conditional output value using required inputs (default no shift)
        mdl_name: str, optional
            Descriptive name for the shift model
        **latent_vars: dict of LatentVar, optional
            Latent variable models used as part of the mechanism's compute equation
        """
        # The initial value will likely just be a specified value for most use cases, so default to that behaviour
        if not init_val_eqn:
            def basic(init_val): return init_val
            init_val_eqn = basic
        super().__init__(init_val_eqn, mdl_name, **latent_vars)

    def gen_init_vals(self, num_devs: int = 1, num_chps: int = 1, num_lots: int = 1):
        """
        Generate initial values for the parent parameter based on the initial value compute equation

        Parameters
        ----------
        num_devs: int, optional
            The number of device/parameter instances per chip to generate initial state for (default 1)
        num_chps: int, optional
            The number of chips per lot to generate initial state for (default 1)
        num_lots: int, optional
            The number of production lots to generate initial state for (default 1)

        Returns
        -------
        numpy.ndarray
            The initial state of all parameter instances across devices, chips, and lots
        """
        # Can either generate the dev and lot variations here, or they could be provided as method arguments
        latents = self.gen_latent_vals(num_devs, num_chps, num_lots)
        # Compute the initial values using the model's specified equation
        vals = self.compute(**latents)
        return vals


class DegPrmMdl(LatentMdl):
    """
    Class for modelling parameters that degrade over time as a function of some underlying degradation mechanisms

    Attributes
    ----------
    name: str
        Descriptive name for the degrading parameter
    mech_mdl_list: list of DegMechMdl or FailMechMdl
        A list of degradation and failure models that affect this parameter, can be retrieved via the mech_mdl method
    init_mdl: InitValModel
        The model used to determine the initial value of the parameter at t=0
    cond_mdl: CondShiftModel
        The model that specifies how the parameter shifts as an instantaneous function of conditions such as temperature
    compute: Callable
        An algebraic function that specifies how the parameter value is calculated from the composing models
    compute_args: list of str
        A listing of the model's compute equation function signature for easy identification of required model inputs
    latent_vars: list of str
        A listing of the latent variables that the model's compute equation takes as inputs
    array_compute: bool
        Flag to indicate whether the parameter's compute equation can accept numpy arrays
    """
    def __init__(self, deg_mech_mdls: DegMechMdl | FailMechMdl | dict, init_val_mdl: InitValMdl = None,
                 cond_shift_mdl: CondShiftMdl = None, compute_eqn: Callable = None,
                 prm_name: str = None, array_computable: bool = True, **latent_vars: LatentVar):
        """
        Parameters
        ----------
        deg_mech_mdls: DegMechModel or FailMechModel or dict of DegMechModel and/or FailMechModel
            The degradation mechanism models involved in the calculation of the degraded parameter
        init_val_mdl: InitValModel, optional
            The model used to generate the initial value for the parameter (defaults to a constant initial value of 0)
        cond_shift_mdl: CondDependModel, optional
            The model used to shift the parameter value according to applied conditions (defaults to no shifts)
        compute_eqn: Callable, optional
            The function used to calculate the parameter value from the sub-models (defaults to a simple summation)
        prm_name: str, optional
            The name of the degraded parameter (default None)
        array_computable: bool, optional
            Flag to indicate whether the parameter's compute equation can accept numpy arrays (default True)
        **latent_vars: dict of LatentVar, optional
            Any latent variables used within the degraded parameter's compute equation
        """
        self.init_mdl = init_val_mdl if init_val_mdl else InitValMdl(init_val=LatentVar(deter_val=0))
        self.cond_mdl = cond_shift_mdl if cond_shift_mdl else CondShiftMdl()
        # Create attributes for mechanism models based on the model names
        self.mech_mdl_list = []
        if type(deg_mech_mdls) == dict:
            for mech in deg_mech_mdls:
                deg_mech_mdls[mech].name = mech
                setattr(self, f"_{mech}_mdl", deg_mech_mdls[mech])
                self.mech_mdl_list.append(mech)
        else:
            if not deg_mech_mdls.name:
                raise UserConfigError('Please specify a name for the degradation mechanism.')
            setattr(self, f"_{deg_mech_mdls.name}_mdl", deg_mech_mdls)
            self.mech_mdl_list.append(deg_mech_mdls.name)
        # The computed parameter value is assumed to be just a simple sum of the different components, but
        # allow the user to provide a custom equation just in case
        if not compute_eqn:
            def basic_sum(init, cond, **mechs):
                return init + cond + sum(mechs.values())
            compute_eqn = basic_sum
        self.array_compute = array_computable
        super().__init__(compute_eqn, prm_name, **latent_vars)

    def mech_mdl(self, mdl):
        """
        Retrieve a DegMechMdl or FailMechMdl within the parameter model by name

        Parameters
        ----------
        mdl: str
            The name of the mechanism model to retrieve

        Returns
        -------
        DegMechMdl or FailMechMdl
        """
        return getattr(self, f"_{mdl}_mdl")

    def gen_latent_vals(self, num_devs: int = 1, num_chps: int = 1, num_lots: int = 1) -> dict:
        """
        Generates the random variations for the composite parameter model that are constant through time and thus need
        to be persisted for repeated use.

        Parameters
        ----------
        num_devs: int, optional
            The number of device instances of this parameter to generate variational values for per chip (default 1)
        num_chps: int, optional
            The number of physical chips to generate values for per lot (default 1)
        num_lots: int, optional
            The number of lots of chips to generate values for (default 1)

        Returns
        -------
        dict
            A nested dictionary of sampled values for the parameter, mechanism, conditional, and initial value latents
        """
        # Overload the gen_latent_vals method to add generation of latents for sub-models
        # Note that initial value variations are not generated because they don't need to be persisted, only used once
        latents = {'cond': self.cond_mdl.gen_latent_vals(num_devs, num_chps, num_lots)}
        for mech in self.mech_mdl_list:
            latents[mech] = self.mech_mdl(mech).gen_latent_vals(num_devs, num_chps, num_lots)
        # The degraded parameter compute equation can have its own latent variables if desired, merge those if so
        latents.update(super().gen_latent_vals(num_devs, num_chps, num_lots))
        return latents

    def gen_init_mech_vals(self, num_devs: int = 1, num_chps: int = 1, num_lots: int = 1) -> dict:
        """
        Generates initial values for the mechanisms that affect this parameter's value, typically unitary values

        Parameters
        ----------
        num_devs: int, optional
            The number of device instances of this parameter to generate variational values for per chip (defaults 1)
        num_chps: int, optional
            The number of physical chips to generate values for per lot (defaults 1)
        num_lots: int, optional
            The number of lots of chips to generate values for (defaults 1)

        Returns
        -------
        dict
            A nested dictionary of sampled values for the mechanism initial values
        """
        init_mech_vals = {}
        for mech in self.mech_mdl_list:
            init_mech_vals[mech] = self.mech_mdl(mech).gen_init_vals(num_devs, num_chps, num_lots)
        return init_mech_vals

    def _reduce_dev_dim_size(self, vals: dict, reduced_size: int) -> dict:
        new = {}
        for key, val in vals.items():
            if type(val) in [np.ndarray, list]:
                new[key] = val[:, :, :reduced_size]
            elif type(val) == dict:
                new[key] = self._reduce_dev_dim_size(vals[key], reduced_size)
        return new

    def calc_equiv_strs_times(self, strs_dims: tuple, mech_deg_vals: dict, strs_conds: dict,
                              init_vals: dict, latents: dict) -> dict[str, np.ndarray]:
        """
        This method back-calculates the length of time required to reach the passed output value given the other test
        conditions and latent parameters. See the Gerabaldi VTS 2023 paper or documentation for a conceptual explanation
        and detailed procedure information on how these values are computed.

        Parameters
        ----------
        strs_dims: tuple of int
            The number of individual parameter instances to measure across devices, chips, and lots
        mech_deg_vals: dict[str, numpy.ndarray]
            The current mechanism degradation values to determine the equivalent time to reach under the new strs_conds
        strs_conds: dict[str, float | numpy.ndarray]
            The specific stress conditions to determine the equivalent time with respect to
        init_vals: numpy.ndarray
            The initial value for each individual instance of the parameter
        latents: dict[str, numpy.ndarray | dict]
            All the hidden values for the different variables within the parameter and mechanism models

        Returns
        -------
        equiv_times: dict[str, numpy.ndarray]
            The times required to degrade from the init_vals to the mech_deg_vals under the given stress conditions
        """
        equiv_times = {}
        for mech in self.mech_mdl_list:
            args_dict = {'deg_val': mech_deg_vals[mech], 'init_val': init_vals[mech],
                         'strs_conds': strs_conds, 'latents': latents[mech], 'dims': strs_dims}
            # Currently we always loop compute because the numerical method used to calculate equivalent stress time
            # from the SciPy library is not array computable
            equiv_times[mech] = self.mech_mdl(mech).calc_equiv_strs_time(**args_dict)
        return equiv_times

    def calc_deg_vals(self, strs_dims: tuple, times: dict, strs_conds: dict,
                      init_vals: np.ndarray, latents: dict, deg_vals: dict) -> (np.ndarray, dict):
        """
        Calculate the underlying degraded values for the parameter given a set of stress conditions and stress duration.
        Note that this method does not calculate a measured value, only the 'true' underlying degraded parameter value.

        Parameters
        ----------
        strs_dims: tuple of int
            The number of individual parameter instances to measure across devices, chips, and lots
        times: dict
            Mapping from mechanisms to stress durations for each sample of the parameter
        strs_conds: dict
            Mapping from mechanisms to stress values for each sample of the parameter
        init_vals: numpy.ndarray
            The initial value for each individual instance of the parameter
        latents: dict
            Mapping from mechanisms and/or latent variables names to latent variables for each sample of the parameter
        deg_vals: dict
            Mapping from mechanisms to degraded values for each parameter sample prior to the current stress phase

        Returns
        -------
        tuple of numpy.ndarray
            The resulting degraded values for parameters and mechanisms after stress, indexing style lot->chp->dev
        """
        # The conditional model is set to its unitary value, i.e. won't affect our output, as the 'true'
        # underlying degradation is referenced to the standard value, specifically the conditions where the conditional
        # shift model leaves the base value unchanged
        arg_vals = {'init': init_vals, 'cond': self.cond_mdl.unitary}

        # Calculate the degradation for the mechanism models
        mech_vals = {}
        for mech in self.mech_mdl_list:
            mech_vals[mech] = self.mech_mdl(mech).calc_deg_vals(times[mech], deg_vals[mech],
                                                                strs_conds, latents[mech], strs_dims)
            # Also copy the mechanism degraded vals into the arguments data structure for computing the parameters
            arg_vals[mech] = mech_vals[mech]

        # Finally add the parameter equation's latent values
        for arg in self.compute_args:
            if arg not in arg_vals.keys() and arg in latents:
                arg_vals[arg] = latents[arg]

        # Now compute the parameter values using all the calculated mechanism and conditional shift values
        try:
            prm_vals = self.compute(**arg_vals)
        except ValueError:
            prm_vals = _loop_compute(self.compute, arg_vals, strs_dims)
        return prm_vals, mech_vals

    def calc_cond_shifted_vals(self, meas_dims: tuple, strs_conds: dict,
                               deg_vals: np.ndarray, latents: dict) -> np.ndarray:
        """
        Calculation of the degraded value adjusted according to the conditional shift model

        Parameters
        ----------
        meas_dims: tuple of int
            The number of individual parameter instances to measure across devices, chips, and lots
        strs_conds: dict
            Mapping from mechanisms to stress values for each sample of the parameter
        deg_vals: numpy.ndarray
            The unshifted value for each individual instance of the parameter
        latents: dict
            Mapping from models and/or latent variables names to latent variables for each sample of the parameter

        Returns
        -------
        numpy.ndarray
            The shifted parameter values according to the conditional shift model, formatted the same as the init_vals
        """
        arg_vals = {'init': deg_vals, 'cond': {}}

        # First check if we are measuring all the devices, if not we need to truncate some values
        if meas_dims[2] < arg_vals['init'].shape[2]:
            arg_vals = self._reduce_dev_dim_size(arg_vals, meas_dims[2])
            latents = self._reduce_dev_dim_size(latents, meas_dims[2])

        # Calculate the conditional shift model value
        arg_vals['cond'] = self.cond_mdl.calc_cond_vals(strs_conds, latents['cond'], meas_dims)

        # Now set the initial values to the degraded values since they fill the same role in the parameter compute
        # equation, and set the mechanism degradation models to their unitary values
        for mech in self.mech_mdl_list:
            arg_vals[mech] = self.mech_mdl(mech).unitary

        # Next are the parameter's own latent values
        for arg in self.compute_args:
            if arg not in arg_vals.keys() and arg in latents:
                arg_vals[arg] = latents[arg]

        # Now compute the parameter values
        try:
            return self.compute(**arg_vals)
        except ValueError:
            return _loop_compute(self.compute, arg_vals, meas_dims)

    def get_dependencies(self, conditions, target):
        """
        Given a list of all stress conditions, get the subset that influence this model's output value

        Parameters
        ----------
        conditions: list
            The list of all available stress conditions to filter/reduce to only those required by this model
        target: str
            Whether to get stress conditions that affect degradation, or the conditional value ('stress' or 'measure')

        Returns
        -------
        list
            The subset of stress conditions that influence this parameter's value for the given target mode
        """
        # Typically no environmental conditions will be used in the parameter compute function, but check just in case
        depends_conds = [arg for arg in self.compute_args if arg in conditions]
        # Only want the dependencies that affect either a stress phase or a measurement
        if target == 'stress':
            for mech in self.mech_mdl_list:
                depends_conds.extend([arg for arg in self.mech_mdl(mech).compute_args if arg in conditions])
        else:
            depends_conds.extend([arg for arg in self.cond_mdl.compute_args if arg in conditions])
        # There is a chance to have overlapping environmental condition dependencies, so remove any duplicates
        return [*set(depends_conds)]


class CircPrmMdl(LatentMdl):
    """
    Class for expressing higher-level parameters that are affected by degraded parameters

    This model contains a computable equation that is a function of degraded device parameters, stress conditions, and
    latent variables. The equation can more generally be any callable Python function, so long as it returns a single
    value for the parameter as the output.

    Attributes
    ----------
    name: str
        A descriptive name for the parameter model
    compute: Callable
        A callable Python function that computes the parameter output value as a function of required inputs
    compute_args: list
        A listing of the model's compute equation function signature for easy identification of required model inputs
    latent_vars: list
        A listing of the latent variables that the model's compute equation takes as inputs
    """
    def __init__(self, compute_eqn: Callable, prm_name: str = None, **latent_vars: LatentVar):
        """
        Parameters
        ----------
        compute_eqn: Callable
            The function used to calculate the parameter value from the set of inputs
        prm_name: str, optional
            The name of the circuit parameter (default None)
        **latent_vars: dict of LatentVar, optional
            Any latent variables used within the circuit parameter's compute equation
        """
        super().__init__(compute_eqn, prm_name, **latent_vars)

    def calc_circ_vals(self, num_samples: int, strs_conds: dict, degraded_prm_vals: dict, latents: dict) -> np.ndarray:
        """
        For a complete set of model equation inputs, compute the parameter values

        Parameters
        ----------
        num_samples: int
            The number of samples to calculate, used to save on computation if not all available samples are measured
        strs_conds: dict of numpy.ndarray
            Mapping from stress conditions to their values for all samples across devices, chips and lots
        degraded_prm_vals: dict of numpy.ndarray
            Mapping from other parameters to their values for all samples across devices, chips and lots
        latents: dict of numpy.ndarray
            Mapping from latent variables to their sampled values for all devices, chips and lots

        Returns
        -------
        numpy.ndarray
            A 3D array of computed parameter values, indexing style lot->chp->dev
        """
        arg_vals = {}
        for arg in self.compute_args:
            if arg in latents:
                arg_vals[arg] = latents[arg]
            elif arg in strs_conds:
                arg_vals[arg] = strs_conds[arg]
            elif arg in degraded_prm_vals:
                arg_vals[arg] = degraded_prm_vals[arg]

        circ_vals = self.compute(**arg_vals)[:][:][:num_samples]
        return circ_vals

    def get_required_prms(self, mdl_prm_list):
        """
        Given a list of all available device parameters, get the subset required to compute this model's output value

        Parameters
        ----------
        mdl_prm_list: list
            The list of device parameters to filter/reduce to only those required by this model

        Returns
        -------
        list
            The subset of device parameters that are named in the model's compute equation signature
        """
        # Determine which of the list of parameters are used to compute the value of this circuit parameter
        return [prm for prm in mdl_prm_list if prm in self.compute_args]

    def get_dependencies(self, conditions, target: str = None): # noqa: UnusedParameter
        """
        Given a list of all stress conditions, get the subset that influence this model's output value

        Parameters
        ----------
        conditions: list
            The list of all available stress conditions to filter/reduce to only those required by this model
        target: str, unused
            Argument used to provide method signature compatibility with the DegPrmMdl implementation

        Returns
        -------
        list
            The subset of stress conditions that influence this parameter's value
        """
        # Note that 'target' is an unused argument here, but needs to be kept to match the signature of get_dependencies
        # for the DegPrmMdl class
        return [arg for arg in self.compute_args if arg in conditions]


class DeviceMdl:
    """
    Top-level container for a device/product with parameters that can degrade or fail with time

    Although this class is the top-level model for the physical device, it is mostly a container for the set of
    parameters of interest that will be observed/measured as aging/wear-out progresses. These parameter models must be
    constructed and passed to this class' constructor to form the full device model.

    Attributes
    ----------
    name: str
        A descriptive name for the device model
    prm_mdl_list: list of DegPrmMdl or CircPrmMdl
        List of parameters that this device model incorporates, parameter models can be retrieved via the prm_mdl method
    """
    def __init__(self, prm_mdls: DegPrmMdl | CircPrmMdl | dict, name: str = None):
        """
        Parameters
        ----------
        prm_mdls: DegPrmMdl or CircPrmMdl or dict of DegPrmMdl or CircPrmMdl
            Single parameter model or a mapping from parameter names to parameter models
        name: str
            Descriptive name for the device model
        """
        self.name = name
        # Create attributes for the degraded parameter models based on the model names
        self.prm_mdl_list = []
        if type(prm_mdls) == dict:
            for prm in prm_mdls:
                prm_mdls[prm].name = prm
                setattr(self, f"_{prm}_mdl", prm_mdls[prm])
                self.prm_mdl_list.append(prm)
        else:
            if not prm_mdls.name:
                raise UserConfigError('Please specify a name for the device parameter.')
            setattr(self, f"_{prm_mdls.name}_mdl", prm_mdls)
            self.prm_mdl_list.append(prm_mdls.name)

    def prm_mdl(self, mdl):
        """
        Retrieve a DegPrmMdl or CircPrmMdl within the device model by name

        Parameters
        ----------
        mdl: str
            The name of the parameter model to retrieve

        Returns
        -------
        DegPrmMdl or CircPrmMdl
        """
        return getattr(self, f"_{mdl}_mdl")

    def gen_init_state(self, sample_counts: dict, num_chps: int = 1, num_lots: int = 1):
        """
        Generate sampled values for all latent variables, along with mechanism and parameter initial values

        Parameters
        ----------
        sample_counts: dict of int
            Mapping from parameter names to the number of instances of each per chip
        num_chps: int, optional
            The number of chips per lot to generate initial state for (default 1)
        num_lots: int, optional
            The number of production lots to generate initial state for (default 1)

        Returns
        -------
        SimState
            The initial state of all parameter instances across devices, chips, and lots
        """
        latents, init_mech_vals, init_prm_vals = {}, {}, {}
        for prm in self.prm_mdl_list:
            # First generate the latent variable values for each model
            latents[prm] = self.prm_mdl(prm).gen_latent_vals(sample_counts[prm], num_chps, num_lots)
            # Now initialize the degradation mechanism progression
            if type(self.prm_mdl(prm)) == DegPrmMdl:
                init_mech_vals[prm] = self.prm_mdl(prm).gen_init_mech_vals(sample_counts[prm], num_chps, num_lots)
            # Then initialize the un-aged values for the device parameters
            # In case the parameter is not directly measured and instead used in calculating other parameters,
            # ensure that any other parameter will have enough samples for now. Make more efficient in the future.
            if prm not in sample_counts:
                sample_counts[prm] = max(sample_counts.values())
            if type(self.prm_mdl(prm)) == DegPrmMdl:
                init_prm_vals[prm] = self.prm_mdl(prm).init_mdl.gen_init_vals(sample_counts[prm], num_chps, num_lots)
        # Return a simulation state containing all the generated initial information about the devices to be tested
        return SimState(init_prm_vals, init_mech_vals, latents)
