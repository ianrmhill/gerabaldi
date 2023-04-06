# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

"""
This submodule will provide common use cases as prebuilt models to aid users in quickly getting Gerabaldi simulations
working without having to custom specify all the little things.
"""

from . import test_specs, test_envs, mech_mdls
from .test_specs import *
from .test_envs import *
from .mech_mdls import *

__all__ = test_specs.__all__
__all__.extend(test_envs.__all__)
__all__.extend(mech_mdls.__all__)
