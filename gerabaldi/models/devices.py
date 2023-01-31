"""
Classes for defining degradation models for tuning or use in simulating wear-out tests.
"""

import inspect
import numpy as np
from scipy.optimize import minimize_scalar
from typing import Callable
from copy import deepcopy

from gerabaldi.models.randomvars import RandomVar
from gerabaldi.exceptions import InvalidTypeError, UserConfigError
from gerabaldi.helpers import _on_demand_import

# Optional imports are loaded using a helper function that suppresses import errors until attempted use
pymc = _on_demand_import('pymc')
pyro = _on_demand_import('pyro', 'pyro-ppl')
dist = _on_demand_import('pyro.distributions', 'pyro-ppl')
tc = _on_demand_import('torch')


__all__ = ['LatentVar', 'InitValModel', 'CondShiftModel', 'DegMechModel', 'FailMechModel',
           'DegradedParamModel', 'CircuitParamModel', 'DeviceModel']


class LatentVar:
    """
    Class that defines the value and stochastic variation of a hidden/latent variable within a model equation.
    """
    def __init__(self, dev_vrtn_mdl: RandomVar = None, chp_vrtn_mdl: RandomVar = None, lot_vrtn_mdl: RandomVar = None,
                 deter_val: float | int = None, var_name: str = None, vrtn_type: str = 'scaling'):
        self.name = var_name
        if vrtn_type not in ['scaling', 'offset']:
            raise InvalidTypeError(f"Latent {self.name} variation type can only be one of 'scaling', 'offset'.")
        self.vrtn_op = vrtn_type

        # Each variation model is a callable stochastic value generator with some distribution
        if not dev_vrtn_mdl and deter_val is None:
            raise UserConfigError(f"Latent {self.name} definition must include either a device distribution or"
                                  "deterministic value argument.")
        self.dev_vrtn_mdl = dev_vrtn_mdl
        self.chp_vrtn_mdl = chp_vrtn_mdl
        self.lot_vrtn_mdl = lot_vrtn_mdl
        self.deter_val = deter_val

    def rename(self, name):
        self.name = name
        # Give the distributions names if not manually specified
        if self.dev_vrtn_mdl and not self.dev_vrtn_mdl.name:
            # Set the distribution name to be the variable name + '_dev' suffix, unless the device variation model fully
            # specifies the variable value, in which case don't add the suffix
            if not self.chp_vrtn_mdl and not self.lot_vrtn_mdl and not self.deter_val:
                self.dev_vrtn_mdl.name = self.name
            else:
                self.dev_vrtn_mdl.name = self.name + '_dev'
        if self.chp_vrtn_mdl:
            self.chp_vrtn_mdl.name = self.name + '_chp'
        if self.lot_vrtn_mdl:
            self.lot_vrtn_mdl.name = self.name + '_lot'

    def gen_latent_vals(self, num_devs: int = 1, num_chps: int = 1, num_lots: int = 1) -> np.ndarray:
        """Generate stochastic variations for the specified number of individual samples, devices, and lots."""
        op = np.multiply if self.vrtn_op == 'scaling' else np.add

        # There are three definition cases:
        # 1. Device distribution and no base value (intended probabilistic approach)
        if self.deter_val is None:
            # The device variations are held in a 3D array, allowing for easy indexing to the value for each sample
            vals = self.dev_vrtn_mdl.sample(num_lots * num_chps * num_devs).reshape((num_lots, num_chps, num_devs))
        else:
            # 2. deterministic base value provided, no device distribution (when following 'if' evaluates to False)
            vals = np.full((num_lots, num_chps, num_devs), self.deter_val)
            if self.dev_vrtn_mdl:
                # 3. deterministic base value provided and device distribution
                vals = op(vals, self.dev_vrtn_mdl.
                           sample(num_lots * num_chps * num_devs).reshape((num_lots, num_chps, num_devs)))

        # Now include the influence of the chip and lot level distributions if specified
        if self.chp_vrtn_mdl:
            # The generated arrays have to be carefully shaped for the numpy array operators to broadcast them correctly
            vals = op(vals, self.chp_vrtn_mdl.
                      sample(num_lots * num_chps).reshape((num_lots, num_chps, 1)))
        if self.lot_vrtn_mdl:
            vals = op(vals, self.lot_vrtn_mdl.
                      sample(num_lots).reshape((num_lots, 1, 1)))
        return vals

    def cbi_export(self, target_framework: str = 'pymc'):
        # A probabilistic distribution must be defined for the device variation model. If not, raise an error
        if self.dev_vrtn_mdl:
            if target_framework == 'pymc':
                op = np.multiply if self.vrtn_op == 'scaling' else np.add
            elif target_framework == 'pyro':
                op = tc.multiply if self.vrtn_op == 'scaling' else tc.add
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
            if self.chp_vrtn_mdl and self.lot_vrtn_mdl:
                dev = self.dev_vrtn_mdl.get_cbi_form(target_framework=target_framework)
                chp = self.chp_vrtn_mdl.get_cbi_form(target_framework=target_framework)
                lot = self.lot_vrtn_mdl.get_cbi_form(target_framework=target_framework)
                if target_framework == 'pymc':
                    cbi_dist = pymc.Deterministic(self.name, op(op(dev, chp), lot))
                else:
                    cbi_dist = op(op(dev, chp), lot)
            elif self.chp_vrtn_mdl and not self.lot_vrtn_mdl:
                dev = self.dev_vrtn_mdl.get_cbi_form(target_framework=target_framework)
                chp = self.chp_vrtn_mdl.get_cbi_form(target_framework=target_framework)
                if target_framework == 'pymc':
                    cbi_dist = pymc.Deterministic(self.name, op(dev, chp))
                else:
                    cbi_dist = op(dev, chp)
            elif not self.chp_vrtn_mdl and self.lot_vrtn_mdl:
                dev = self.dev_vrtn_mdl.get_cbi_form(target_framework=target_framework)
                lot = self.lot_vrtn_mdl.get_cbi_form(target_framework=target_framework)
                if target_framework == 'pymc':
                    cbi_dist = pymc.Deterministic(self.name, op(dev, lot))
                else:
                    cbi_dist = op(dev, lot)
            else:
                cbi_dist = self.dev_vrtn_mdl.get_cbi_form(target_framework=target_framework)
            # Return the dev_vrtn_mdl centre value to previous if it was changed to avoid any side effects
            if old_centre:
                self.dev_vrtn_mdl.set_centre(old_centre)
            return cbi_dist
        else:
            raise Exception(f"Cannot export deterministic latent variable {self.name} for CBI use.")


class LatentModel:
    """Base class for modelling physical phenomenon with hidden/uncertain underlying parameters."""
    def __init__(self, mdl_eqn: Callable = None, mdl_name: str = None, **latent_vars: LatentVar):
        self.name = mdl_name
        # Add default to ensure the object attribute can always be called
        if not mdl_eqn:
            def mdl_eqn(x): return x
        self.compute = mdl_eqn
        # Define 'true'/underlying values for equation inputs used to simulate degradation
        # No latent variable values should be specified when performing model inference
        self.latent_vars = []
        for var in latent_vars:
            # If no name was specified for the latent variable, nice to set it to the same name as used in the model
            if not latent_vars[var].name:
                latent_vars[var].rename(var)
            # Indicate the latent variable attributes as private since they're supposed to be uncertain
            setattr(self, f"_latent_{var}", latent_vars[var])
            self.latent_vars.append(var)

    def latent_var(self, var):
        return getattr(self, f"_latent_{var}")

    def gen_latent_vals(self, num_devs: int = 1, num_chps: int = 1, num_lots: int = 1) -> dict:
        """Generate 'true'/underlying values for all the model's latent variables for each sample, device, and lot."""
        vals = {}
        for var in self.latent_vars:
            vals[var] = self.latent_var(var).gen_latent_vals(num_devs, num_chps, num_lots)
        return vals

    def export_to_cbi(self, target_framework: str = 'pymc', observed: dict = None):
        # First create latent variables for the hidden model parameters
        vals = {}
        for var in self.latent_vars:
            vals[var] = self.latent_var(var).cbi_export(target_framework)

        # Now add the stress conditions to the compute dictionary
        for strs in observed:
            if strs != self.name:
                vals[strs] = observed[strs]

        # Now define the compute model
        computed = self.compute(**vals)

        # Finally, define the output as a variable as well
        if target_framework == 'pymc':
            out = pymc.Normal(name=self.name, mu=computed, sigma=2, observed=observed[self.name])
        elif target_framework == 'pyro':
            return pyro.sample(self.name, dist.Normal(computed, 2), obs=observed[self.name])


class DegMechModel(LatentModel):
    """Class for modelling the effect of a degradation mechanism on some physical parameter."""
    def __init__(self, mech_eqn: Callable = None, mdl_name: str = None, unitary_val: int = 0, **latent_vars):
        if not mech_eqn:
            # Default is a non-degrading model, parameter stays the same (i.e. 'fresh') regardless of stress and time
            def always_fresh(): return unitary_val
            mech_eqn = always_fresh
        self.unitary = unitary_val
        super().__init__(mech_eqn, mdl_name, **latent_vars)

    def gen_init_vals(self, num_devs, num_chps, num_lots):
        return np.full((num_lots, num_chps, num_devs), self.unitary)

    def calc_equiv_strs_time(self, deg_val, strs_conds, init_val, latents):
        """
        This method back-calculates the length of time required to reach the current level of degradation under a given
        set of stress conditions for all the different samples in the test.
        """
        # Computes the absolute difference between the target value and the calculated value using a given time
        def residue(time, val, strs, init, prm_latents):
            return abs(val - self.calc_degraded_vals(time, strs, init, prm_latents))

        # Minimize the difference between the output and the target/observed value using scipy optimizer
        # Since we are minimizing time we use the bounded method to ensure our time doesn't go negative
        return minimize_scalar(residue, args=(deg_val, strs_conds, init_val, latents),
                               method='bounded', bounds=(0, 1e10)).x

    def calc_degraded_vals(self, times, strs, init, latents):
        """
        Calculate the underlying degraded values for the parameter given a set of stress conditions and stress duration.
        Note that this method does not calculate a measured value, only the 'true' underlying degraded parameter value.
        """
        arg_vals = deepcopy(latents)
        mech_sig = inspect.signature(self.compute)
        # Add the stress time to the argument dict
        arg_vals['time'] = times

        # Add all stress condition values to the already well formatted latents dict
        for arg in mech_sig.parameters.keys():
            if arg in strs:
                arg_vals[arg] = strs[arg]
            elif arg == 'init':
                arg_vals[arg] = init[arg]
        return self.compute(**arg_vals)


class FailMechModel(LatentModel):
    """Class for modelling the effect of a time-independent hard failure mechanism on some physical parameter."""
    def __init__(self, mech_eqn: Callable = None, mdl_name: str = None, unitary_val: int = 0, **latent_vars):
        if not mech_eqn:
            # Default is a non-degrading model, parameter stays the same (i.e. 'fresh') regardless of stress and time
            def never_fail(): return unitary_val
            mech_eqn = never_fail
        self.unitary = unitary_val
        super().__init__(mech_eqn, mdl_name, **latent_vars)

    def gen_init_vals(self, num_devs, num_chps, num_lots):
        return np.full((num_lots, num_chps, num_devs), self.unitary)

    def calc_equiv_strs_time(self, deg_val, strs_conds, init_val, latents):
        """
        Equivalent method with degradation mechanisms, except here we just return whether the failure has occurred.

        Parameters
        ----------
        deg_val
        strs_conds
        init_val
        latents

        Returns
        -------

        """
        return 0

    def calc_degraded_vals(self, times, strs, init, latents):
        """
        Calculate the failure state for the mechanism given a set of stress conditions and stress duration.
        """
        if init != self.unitary:
            return init
        else:
            arg_vals = deepcopy(latents)
            mech_sig = inspect.signature(self.compute)
            # Add the stress time to the argument dict
            arg_vals['time'] = times

            # Add all stress condition values to the already well formatted latents dict
            for arg in mech_sig.parameters.keys():
                if arg in strs:
                    arg_vals[arg] = strs[arg]
                elif arg == 'init':
                    arg_vals[arg] = init[arg]
            return self.compute(**arg_vals)


class CondShiftModel(LatentModel):
    """Class for modelling the instantaneous dependency of the value of some physical parameter on
    environmental/operating conditions."""
    def __init__(self, cond_shift_eqn: Callable = None, mdl_name: str = None, unitary_val: int = 0, **latent_vars):
        if not cond_shift_eqn:
            def unchanged(): return 0
            cond_shift_eqn = unchanged
        self.unitary = unitary_val
        super().__init__(cond_shift_eqn, mdl_name, **latent_vars)


class InitValModel(LatentModel):
    """Class for modelling the initial value of some physical parameter."""
    def __init__(self, init_val_eqn: Callable = None, mdl_name: str = None, **latent_vars: LatentVar):
        # The initial value will likely just be a specified value for most use cases, so default to that behaviour
        if not init_val_eqn:
            def basic(init_val): return init_val
            init_val_eqn = basic
        super().__init__(init_val_eqn, mdl_name, **latent_vars)

    def gen_init_vals(self, num_devs: int = 1, num_chps: int = 1, num_lots: int = 1):
        # Can either generate the dev and lot variations here, or they could be provided as method arguments
        latents = self.gen_latent_vals(num_devs, num_chps, num_lots)
        # Compute the initial values using the model's specified equation
        vals = self.compute(**latents)
        return vals


class DegradedParamModel(LatentModel):
    """
    Class for modelling parameters that degrade over time as a function of some underlying degradation mechanisms.

    Attributes
    ----------
    name: str
        The name of the parameter capable of degrading
    init_mdl: InitValModel
        The model used to determine the initial value of the parameter at t=0
    cond_mdl: CondShiftModel
        The model that specifies how the parameter shifts as an instantaneous function of conditions such as temperature
    <mech_name>_mdl: DegMechModel
        The degradation mechanism models that cause the parameter to shift or suddenly change over time
    compute: callable
        An algebraic function that specifies how the parameter value is calculated from the composing models

    Methods
    -------
    gen_latent_vals(num_inds, num_devs, num_lots)
        Determines values for the stochastic sources in the parameter model that retain their random value over time
    calc_equiv_strs_times(prm_vals, strs_conds, init_vals, latents)
        Inverse calculation to determine the input time required to obtain the parameter value under given conditions
    calc_degraded_vals(times, strs_conds, init_vals, latents)
        Calculates degraded parameter values after stress for a specific time period and set of stress conditions
    calc_cond_shifted_vals(num_samples, strs_conds, degraded_vals, latents)
        Calculates the shifted parameter values resulting from the applied conditions used for measurement
    """
    def __init__(self, deg_mech_mdls: DegMechModel | FailMechModel | dict, init_val_mdl: InitValModel = None,
                 cond_shift_mdl: CondShiftModel = None, compute_eqn: Callable = None,
                 prm_name: str = None, array_computable: bool = True, **latent_vars):
        """
        Parameters
        ----------
        deg_mech_mdls: DegMechModel or FailMechModel or dict of DegMechModel and/or FailMechModel
            The degradation mechanism models involved in the calculation of the degraded parameter
        init_val_mdl: InitValModel, optional
            The model used to generate the initial value for the parameter, defaults to an initial value of 0
        cond_shift_mdl: CondDependModel, optional
            The model used to shift the parameter value according to applied conditions, defaults to no shifts
        compute_eqn: callable, optional
            The function used to calculate the parameter value from the sub-models, defaults to a simple summation
        prm_name: str, optional
            The name of the degraded parameter, defaults to 'generic parameter'
        **latent_vars: LatentVar objects, optional
            Any latent variables used within the degraded parameters compute equation. Only used for simulation.
        """
        self.name = prm_name
        self.init_mdl = init_val_mdl if init_val_mdl else InitValModel(init_val=LatentVar(deter_val=0))
        self.cond_mdl = cond_shift_mdl if cond_shift_mdl else CondShiftModel()
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
        return getattr(self, f"_{mdl}_mdl")

    # Overload the gen_latent_vals method to add generation of latents for sub-models
    def gen_latent_vals(self, num_devs: int = 1, num_chps: int = 1, num_lots: int = 1) -> dict:
        """
        Generates the random variations for the composite parameter model that are constant through time and thus need
        to be persisted for repeated use.

        Parameters
        ----------
        num_devs: int, optional
            The number of device instances of this parameter to generate variational values for per chip, defaults to 1
        num_chps: int, optional
            The number of physical chips to generate values for per lot, defaults to 1
        num_lots: int, optional
            The number of lots of chips to generate values for, defaults to 1

        Returns
        -------
        latents: nested dict of float
            A dictionary of dictionaries of variational values for the parameter, one entry per mech and cond model
        """
        # Note that initial value variations are not generated because they don't need to be persisted, only used once
        latents = {'cond': self.cond_mdl.gen_latent_vals(num_devs, num_chps, num_lots)}
        for mech in self.mech_mdl_list:
            latents[mech] = self.mech_mdl(mech).gen_latent_vals(num_devs, num_chps, num_lots)
        # The degraded parameter compute equation can have its own latent variables if desired
        for var in self.latent_vars:
            latents[var] = getattr(self, '_latent_' + var).gen_latent_vals(num_devs, num_chps, num_lots)
        return latents

    def gen_init_mech_vals(self, num_devs, num_chps, num_lots) -> dict:
        init_mech_vals = {}
        for mech in self.mech_mdl_list:
            init_mech_vals[mech] = self.mech_mdl(mech).gen_init_vals(num_devs, num_chps, num_lots)
        return init_mech_vals

    def _set_to_single_index(self, vals: dict, i: int, j: int, k: int) -> dict:
        """
        Recursive method that obtains a single item index from a 3D list or ndarray. Takes a nested dictionary structure
        that may have arbitrarily many 3D arrays of the same shape, and substitutes the arrays with the value stored in
        the specified index location within the respective arrays.

        Parameters
        ----------
        vals: dict
            The nested dictionary structure containing any number of arrays of shape (i, j, k)
        i: int
            The first dimension index for the item to extract from the arrays.
        j: int
            The second dimension index for the item to extract from the arrays.
        k: int
            The third dimension index for the item to extract from the arrays.

        Returns
        -------
        new: dict
            The nested dictionary with all arrays replaced with respective values from the specified index location.
        """
        new = deepcopy(vals)
        for key, val in new.items():
            if type(val) == list or type(val) == np.ndarray:
                new[key] = val[i][j][k]
            elif type(val) == dict:
                new[key] = self._set_to_single_index(new[key], i, j, k)
        return new

    def _reduce_dev_dim_size(self, vals: dict, reduced_size: int) -> dict:
        new = deepcopy(vals)
        for key, val in new.items():
            if type(val) == list or type(val) == np.ndarray:
                new[key] = val[:, :, :reduced_size]
            elif type(val) == dict:
                new[key] = self._reduce_dev_dim_size(new[key], reduced_size)
        return new

    def calc_equiv_strs_times(self, mech_deg_vals: dict, strs_conds: dict,
                              init_vals: dict, latents: dict) -> np.ndarray:
        """
        This method back-calculates the length of time required to reach the passed output value given the other test
        conditions and latent parameters.

        Parameters
        ----------
        mech_deg_vals: list | ndarray of floats
            The current mechanism degradation values to determine the equivalent time to reach under the new strs_conds
        strs_conds: dict of float or ndarray
            The specific stress conditions to determine the equivalent time with respect to
        init_vals: ndarray of floats
            The initial value for each individual instance of the parameter
        latents: dict of ndarray or dict
            All the hidden values for the different variables within the parameter and mechanism models

        Returns
        -------
        equiv_times: dict of ndarray
            The times required to degrade from the init_vals to the mech_deg_vals under the given stress conditions
        """

        # Extract the number of lots, devices, and samples from the degraded mechanism values data store
        dims = np.array(next(iter(mech_deg_vals.values()))).shape
        equiv_times = {}
        for mech in self.mech_mdl_list:
            equiv_times[mech] = np.empty(dims)

        # Sadly can't run minimize_scalar on all parameter instances at once, have to loop
        # In future could either try and find a different library method or could introduce parallelization
        for lot in range(dims[0]):
            for chp in range(dims[1]):
                for dev in range(dims[2]):
                    latent = self._set_to_single_index(latents, lot, chp, dev)
                    strs_cond = self._set_to_single_index(strs_conds, lot, chp, dev)
                    # Calculate the equivalent stress time for each degradation mechanism that affects the parameter
                    for mech in self.mech_mdl_list:
                        deg_val = mech_deg_vals[mech][lot][chp][dev]
                        init_val = init_vals[mech][lot][chp][dev]
                        equiv_times[mech][lot][chp][dev] = self.mech_mdl(mech)\
                            .calc_equiv_strs_time(deg_val, strs_cond, init_val, latent[mech])
        return equiv_times

    def calc_degraded_vals(self, times: dict, strs_conds: dict,
                           init_vals: np.ndarray, latents: dict, deg_vals: dict) -> (np.ndarray, dict):
        """
        Calculate the underlying degraded values for the parameter given a set of stress conditions and stress duration.
        Note that this method does not calculate a measured value, only the 'true' underlying degraded parameter value.

        Parameters
        ----------
        times: float | ndarray of floats
            The stress time (or times) that the parameter will degrade over
        strs_conds: dict of float or ndarray
            The specific stress conditions to stress under
        init_vals: ndarray
            The initial value for each individual instance of the parameter
        latents: dict of ndarray or dict
            All the hidden values for the different variables within the parameter and mechanism models

        Returns
        -------
        ndarray
            The resulting degraded values after stress, formatted the same as the init_vals
        """
        if self.array_compute:
            arg_vals = deepcopy(latents)

            # First calculate the mechanism shift model values
            mech_vals = {}
            for mech in self.mech_mdl_list:
                mech_sig = inspect.signature(self.mech_mdl(mech).compute)
                # Add the stress time to the argument dict
                arg_vals[mech]['time'] = times[mech]
                # Add all stress condition values to the already well formatted latents dict
                for arg in mech_sig.parameters.keys():
                    if arg in strs_conds:
                        arg_vals[mech][arg] = strs_conds[arg]
                arg_vals[mech] = self.mech_mdl(mech).compute(**arg_vals[mech])
                # Also copy the mechanism degraded vals into the persistence data structure
                mech_vals[mech] = arg_vals[mech]

            # Next, set the conditional shift model value to its unitary value, i.e. won't affect our output, as the 'true'
            # underlying degradation is referenced to the standard value, specifically the conditions where the conditional
            # shift model leaves the base value unchanged
            arg_vals['cond'] = self.cond_mdl.unitary
            # Merge the initial values into the argument dictionary
            arg_vals['init'] = init_vals
            # Now compute the parameter values using all the calculated mechanism and conditional shift values
            prm_vals = self.compute(**arg_vals)
        else:
            # Extract the number of lots, devices, and samples from the time value data store
            dims = init_vals.shape
            prm_vals = np.empty(dims)
            mech_vals = {}
            for mech in self.mech_mdl_list:
                mech_vals[mech] = np.empty(dims)

            for lot in range(dims[0]):
                for chp in range(dims[1]):
                    for dev in range(dims[2]):
                        time = self._set_to_single_index(times, lot, chp, dev)
                        strs_cond = self._set_to_single_index(strs_conds, lot, chp, dev)
                        deg_val = self._set_to_single_index(deg_vals, lot, chp, dev)
                        init_val = init_vals[lot][chp][dev]
                        latent = self._set_to_single_index(latents, lot, chp, dev)
                        prm_vals[lot][chp][dev], temp = self.calc_degraded_val(time, strs_cond, init_val, latent, deg_val)
                        for mech in self.mech_mdl_list:
                            mech_vals[mech][lot][chp][dev] = temp[mech]
        return prm_vals, mech_vals

    def calc_degraded_val(self, time, strs_conds, init_val, latents, deg_val):
        arg_vals = deepcopy(latents)
        mech_val = {}

        for mech in self.mech_mdl_list:
            # This if case handles hard failure mechanisms
            if type(self.mech_mdl(mech)) == FailMechModel \
                    and deg_val[mech] != self.mech_mdl(mech).unitary:
                # If the sample has already failed, leave it failed
                arg_vals[mech] = deg_val[mech]
                mech_val[mech] = deg_val[mech]
            # This case handles mechanisms that can hit 0 or negative degradation rates during a stress phase
            # (e.g. recovery) THIS NEEDS TO BE CHANGED TO ADD RECOVERY SUPPORT
            # The comparison checks if the equivalent time needed to reach the current level of degradation is
            # effectively infinite, we just bounded the equivalent time calculation
            elif time[mech] > 9e9:
                arg_vals[mech] = deg_val[mech]
                mech_val[mech] = deg_val[mech]
            # This case handles standard degradation cases
            else:
                arg_vals[mech]['time'] = time[mech]
                mech_sig = inspect.signature(self.mech_mdl(mech).compute)
                # Add all stress condition values to the already well formatted latents dict
                for arg in mech_sig.parameters.keys():
                    if arg in strs_conds:
                        arg_vals[mech][arg] = strs_conds[arg]
                arg_vals[mech] = self.mech_mdl(mech).compute(**arg_vals[mech])
                # Also copy the mechanism degraded vals into the persistence data structure
                mech_val[mech] = arg_vals[mech]

        arg_vals['cond'] = self.cond_mdl.unitary
        arg_vals['init'] = init_val
        prm_val = self.compute(**arg_vals)
        return prm_val, mech_val

    def calc_cond_shifted_vals(self, num_samples: int, strs_conds: dict,
                               degraded_vals: np.ndarray, latents: dict) -> np.ndarray:
        """
        Calculation of the true degraded value, adjusted according to the conditional shift model

        Parameters
        ----------
        num_samples: int
            The number of individual parameter instances to measure in each device in each lot. Normally all available.
        strs_conds: dict of float or ndarray
            The specific stress conditions to measure under
        degraded_vals: ndarray
            The unshifted value for each individual instance of the parameter
        latents: dict of ndarray or dict
            All the hidden values for the different variables within the parameter and mechanism models

        Returns
        -------
        ndarray
            The shifted parameter values according to the conditional shift model, formatted the same as the init_vals
        """
        if self.array_compute:
            arg_vals = deepcopy(latents)
            arg_vals['init'] = degraded_vals

            # First check if we are measuring all the devices, if not we need to truncate some values
            if num_samples < arg_vals['init'].shape[2]:
                arg_vals = self._reduce_dev_dim_size(arg_vals, num_samples)

            # Calculate the conditional shift model value
            cond_sig = inspect.signature(self.cond_mdl.compute)
            # Add all stress condition values to the already well formatted latents dict
            for arg in cond_sig.parameters.keys():
                if arg in strs_conds:
                    arg_vals['cond'][arg] = strs_conds[arg]
            arg_vals['cond'] = self.cond_mdl.compute(**arg_vals['cond'])

            # Now set the initial values to the degraded values since they fill the same role in the parameter compute
            # equation, and set the mechanism degradation models to their unitary values
            for mech in self.mech_mdl_list:
                arg_vals[mech] = self.mech_mdl(mech).unitary
            shifted_vals = self.compute(**arg_vals)
        else:
            dims = degraded_vals.shape
            # If the number of samples to measure is less than inds, we can truncate
            if num_samples < dims[2]:
                dims[2] = num_samples
            shifted_vals = np.empty(dims)

            # Sadly can't run minimize_scalar on all parameter instances at once, have to loop
            # In future could either try and find a different library method or could introduce parallelization
            for lot in range(dims[0]):
                for chp in range(dims[1]):
                    for dev in range(dims[2]):
                        strs_cond = self._set_to_single_index(strs_conds, lot, chp, dev)
                        degraded_val = degraded_vals[lot][chp][dev]
                        latent = self._set_to_single_index(latents, lot, chp, dev)
                        shifted_vals[lot][chp][dev] = self.calc_cond_shifted_val(strs_cond, degraded_val, latent)

        # Now compute, but only return the requested number of measured samples for each device and lot
        return shifted_vals

    def calc_cond_shifted_val(self, strs_conds: dict, degraded_val: float | int, latents):
        arg_vals = deepcopy(latents)

        # First calculate the conditional shift model value
        cond_sig = inspect.signature(self.cond_mdl.compute)
        # Add all stress condition values to the already well formatted latents dict
        for arg in cond_sig.parameters.keys():
            if arg in strs_conds:
                arg_vals['cond'][arg] = strs_conds[arg]
        arg_vals['cond'] = self.cond_mdl.compute(**arg_vals['cond'])

        # Now set the initial values to the degraded values since they fill the same role in the parameter compute
        # equation, and set the mechanism degradation models to their unitary values
        arg_vals['init'] = degraded_val
        for mech in self.mech_mdl_list:
            arg_vals[mech] = self.mech_mdl(mech).unitary
        return self.compute(**arg_vals)

    def get_dependencies(self, conditions, target):
        # Typically no environmental conditions will be used in the parameter compute function, but check just in case
        all_args = inspect.signature(self.compute)
        depends_conds = [arg for arg in all_args.parameters.keys() if arg in conditions]
        # Only want the dependencies that affect either a stress phase or a measurement
        if target == 'stress':
            for mech in self.mech_mdl_list:
                all_args = inspect.signature(self.mech_mdl(mech).compute)
                depends_conds.extend([arg for arg in all_args.parameters.keys() if arg in conditions])
        else:
            all_args = inspect.signature(self.cond_mdl.compute)
            depends_conds.extend([arg for arg in all_args.parameters.keys() if arg in conditions])
        # There is a chance to have overlapping environmental condition dependencies, so remove any duplicates
        return [*set(depends_conds)]


class CircuitParamModel(LatentModel):
    """
    Class for expressing higher-level parameters that are affected by degraded parameters. Stateless other than
    sampled values for latent variables.
    """
    def __init__(self, circ_eqn, circ_name: str = None, **latent_vars):
        super().__init__(circ_eqn, circ_name, **latent_vars)

    def calc_circ_vals(self, num_samples, strs_conds, degraded_prm_vals, latents) -> np.ndarray:
        arg_vals = deepcopy(latents)

        circ_sig = inspect.signature(self.compute)
        for arg in circ_sig.parameters.keys():
            if arg in strs_conds:
                arg_vals[arg] = strs_conds[arg]
            elif arg in degraded_prm_vals:
                arg_vals[arg] = degraded_prm_vals[arg]

        circ_vals = self.compute(**arg_vals)[:][:][:num_samples]
        return circ_vals

    def get_dependencies(self, conditions, target):
        # Note that 'target' is an unused argument here, but needs to be kept to match the signature of get_dependencies
        # for the DegradedParamModel class
        all_args = inspect.signature(self.compute)
        return [arg for arg in all_args.parameters.keys() if arg in conditions]


class DeviceModel:
    """Class for composite degradation models involving multiple degradation mechanisms."""
    def __init__(self, prm_mdls: DegradedParamModel | CircuitParamModel | dict, name: str = None):
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
                raise UserConfigError('Please specify a name for the degrading parameter.')
            setattr(self, f"_{prm_mdls.name}_mdl", prm_mdls)
            self.prm_mdl_list.append(prm_mdls.name)

    def prm_mdl(self, mdl):
        return getattr(self, f"_{mdl}_mdl")

    def gen_latent_vals(self, sample_counts: dict, num_chps: int = 1, num_lots: int = 1):
        latents = {}
        for prm in self.prm_mdl_list:
            latents[prm] = self.prm_mdl(prm).gen_latent_vals(sample_counts[prm], num_chps, num_lots)
        return latents

    def gen_init_vals(self, sample_counts: dict, num_chps: int = 1, num_lots: int = 1):
        # Generate the initial values for each device parameter type in turn
        init_vals = {}
        for prm in self.prm_mdl_list:
            # In case the parameter is not directly measured and instead used in calculating other parameters, just
            # ensure that any other parameter will have enough samples for now. Make more efficient in the future.
            if not prm in sample_counts:
                sample_counts[prm] = max(sample_counts.values())
            if type(self.prm_mdl(prm)) == DegradedParamModel:
                init_vals[prm] = self.prm_mdl(prm).init_mdl.\
                    gen_init_vals(sample_counts[prm], num_chps, num_lots)
        return init_vals

    def gen_init_mech_vals(self, sample_counts: dict, num_chps: int = 1, num_lots: int = 1):
        init_mech_vals = {}
        for prm in self.prm_mdl_list:
            if type(self.prm_mdl(prm)) == DegradedParamModel:
                init_mech_vals[prm] = self.prm_mdl(prm).\
                    gen_init_mech_vals(sample_counts[prm], num_chps, num_lots)
        return init_mech_vals

    def calc_equiv_times(self, prm_vals, strs_conds, init_vals, latents):
        """
        This method back-calculates the length of time required to reach the current level of degradation under a given
        set of stress conditions for all the different sampled parameters in the test.
        """
        # Can't use a numpy array here as they can't consist of timedelta objects
        equiv_times = {}
        for prm in self.prm_mdl_list:
            if type(self.prm_mdl(prm)) == DegradedParamModel:
                equiv_times[prm] = self.prm_mdl(prm).calc_equiv_strs_times(prm_vals[prm], strs_conds[prm],
                                                                                     init_vals[prm], latents[prm])
        return equiv_times

    def calc_dev_degradation(self, times, strs_conds, init_vals, latents, deg_vals):
        """
        Calculate the degraded parameter values after a given period of stress.
        """
        degraded_vals = {}
        per_mech_deg_vals = {}
        for prm in self.prm_mdl_list:
            if type(self.prm_mdl(prm)) == DegradedParamModel:
                degraded_vals[prm], per_mech_deg_vals[prm] = self.prm_mdl(prm)\
                    .calc_degraded_vals(times[prm], strs_conds[prm], init_vals[prm], latents[prm], deg_vals[prm])
        return degraded_vals, per_mech_deg_vals
