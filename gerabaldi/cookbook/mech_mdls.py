# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from gerabaldi.models.devices import DegMechMdl, LatentVar
from gerabaldi.models.random_vars import Normal

__all__ = ['jedec_nbti_mdl']

BOLTZMANN_CONST_EV = 8.617e-5


def jedec_nbti_mdl() -> DegMechMdl:
    """
    Very simple example NBTI mechanism model with some ballpark latent variable distributions. If developing simulation
    for a specific process or with selected values a custom model instance should be used instead.

    Returns
    -------
    DegMechMdl
        Basic probabilistic NBTI model with generic latent variable distributions
    """
    # Model provided in JEDEC's JEP122H as generally used NBTI degradation model, equation 5.3.1
    def nbti_vth_shift_empirical(a_0, e_aa, temp, vdd, alpha, time, n):
        return a_0 * np.exp(e_aa / (BOLTZMANN_CONST_EV * temp)) * (vdd ** alpha) * (time ** n)

    return DegMechMdl(mdl_name='JEDEC\'s Empirical NBTI Vth Shift Model', unitary_val=0,
                      mech_eqn=nbti_vth_shift_empirical,
                      a_0=LatentVar(dev_vrtn_mdl=Normal(0.006, 0.0002)),
                      e_aa=LatentVar(dev_vrtn_mdl=Normal(0.05, 0.0002)),
                      alpha=LatentVar(dev_vrtn_mdl=Normal(3.4, 0.004)),
                      n=LatentVar(dev_vrtn_mdl=Normal(0.21, 0.005)))
