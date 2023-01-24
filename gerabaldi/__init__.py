# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

"""
Gerabaldi Wear-Out Reliability Simulator / Generative Wear-Out Model (module gerabaldi)

Description
-----------

This module allows for simulation of integrated circuit reliability tests, taking an underlying model and set of test
conditions, then generating stochastic data points to simulate the observations of some experimental test.

Interface
---------


References
----------

None yet

"""

# This determines the project version for PyPi as well
__version__ = '0.0.10'

from . import models
from . import cookbook
from .sim import *

__all__ = sim.__all__
__all__.extend(models.__all__)
__all__.extend(cookbook.__all__)
