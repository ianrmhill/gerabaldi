# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

"""
Gerabaldi Wear-Out Reliability Simulator / Generative Aging Model (module gerabaldi)

Description
-----------
The Gerabaldi module allows for generic simulation of arbitrary integrated circuit reliability tests, enabling data
generation for tests or use cases where obtaining the corresponding real-world data could require hundreds of hours or
even years to obtain. Results realism is of course governed by the quality of the physical model used to simulate some
real-world hardware device, as with any generative model.

Primary use cases for the simulator are to trial new test methods, investigate proposed physical models, or to extend
a small set of real-world measurements into a much larger dataset with comparable characteristics. The rapid generation
of such data enables computational techniques that require large datasets such as machine learning or computational
Bayesian inference (CBI).

Using Gerabaldi is straightforward, and consists of building up three models that specify a wear-out test, a physical
model of the hardware device being simulated, and a test environment model that emulates the test chamber and
instruments used to measure device parameters. Once specified, the 'simulate' procedure is called, which will execute
the test specification on the device model within the test environment.

The three models that must be specified can range in complexity from very simple to extremely detailed and nuanced
depending on the use case, and it is recommended to start simple and add the desired complexity in stages, checking that
the simulations are aligned with expectations at each stage. The models are completely arbitrary and thus the range of
devices and test specifications that can be handled by Gerabaldi is effectively limitless.

To get started, or to quickly build useful simulations without having to dig into the details of the models involved it
is helpful to use the included 'cookbook' which attempts to provide prebuilt models for common tests, integrated circuit
wear-out mechanisms using peer-reviewed empirical models, and some basic test environments for use. These cookbook
models can also be used as references to see how to construct more complex models. The 'demos' folder of the project
also contains some useful references for full working simulations with command-line interfaces and visualization of
results.

Core Interface
---------
TestSpec - Class that is instantiated to define the test specification model
PhysTestEnv - Class that is instantiated to define the physical test environment model
DeviceMdl - Class that is instantiated to define a physical device/hardware model
simulate - The main procedure for the module, uses the above three models to run a wear-out simulation and return data
"""

# This value determines the project version for PyPi as well
__version__ = '0.0.13'

from . import models
from . import cookbook
from .sim import simulate, gen_init_state

__all__ = ['simulate', 'gen_init_state', 'models', 'cookbook']
__all__.extend(models.__all__)
__all__.extend(cookbook.__all__)
