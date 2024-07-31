# Gerabaldi Wear-Out Reliability Simulator / Generative Aging Model

The Gerabaldi module allows for generic simulation of arbitrary integrated circuit reliability tests, enabling data
generation for tests or use cases where obtaining the corresponding real-world data could require hundreds of hours or
even years to obtain. Results realism is of course governed by the quality of the physical model used to simulate some
real-world hardware device, as with any generative model.

Primary use cases for the simulator are to trial new test methods, investigate proposed physical models, or to extend
a small set of real-world measurements into a much larger dataset with comparable characteristics. The rapid generation
of such data enables computational techniques that require large datasets such as deep learning (DL) or computational
Bayesian inference (CBI).

## Installation

There are two methods of installing Gerabaldi, depending on whether you simply want to use the package or whether you
want to obtain the source code for development or testing.

To install the packaged version, simply use 'pip install gerabaldi' which will install the latest release from PyPi, and
currently requires Python version 3.10 or greater.

To download the source code, clone this repository and make it available to your Python code through your PYTHONPATH.

There are minimal required dependencies in terms of other Python packages to use most features of Gerabaldi, but
some optional features require additional packages which you will be notified to install on an as-needed basis.


## Demos

The 'demos' folder of this repository contains some useful references for full working simulations with command-line
interfaces and visualization of results. These can be run directly if you have cloned the source code, or if you
installed the packaged version it is easiest to copy-paste the source code from Github (you will want to remove the
automated path add code at the very top of the demo file as the 'gerabaldi' import should already work without it).

Note that the VTS paper demos will require modification and take a significant amount of time to obtain the exact
results shown due to the size of the simulations, to run the full simulations change the three globals near the tops of
the files as follows:

For demo 1 (pre-silicon variability analysis):
`NUM_DEVICES = 10`
`NUM_CHIPS = 10`
`NUM_LOTS = 10`

For demo 2 (TDDB model sensitivity):
`TEST_LEN = 24 * 7 * 52 * 20`
`NUM_SAMPLES = 1000`
`C_LATENT = 4e-6`

Running the simulations with these values will provide similar-looking results but with far less computation and is
useful for quickly seeing these more complex simulations in action.


## Basic Use

Constructing Gerabaldi simulations is straightforward, and consists of building up three models that specify:
1. A wear-out test.
2. A physical model of the hardware device being simulated.
3. A test environment model that emulates the test chamber and instruments used to measure device parameters.

Once specified, the 'simulate' procedure is called, which will execute the test specification on the device model within
the test environment.

The three models that must be specified can range in complexity from very simple to extremely detailed and nuanced
depending on the use case, and it is recommended to start simple and add the desired complexity in stages, checking that
the simulations are aligned with expectations at each stage. The models are completely arbitrary and thus the range of
devices and test specifications that can be handled by Gerabaldi is effectively limitless.

To get started, or to quickly build useful simulations without having to dig into the details of the models involved it
is helpful to use the included 'cookbook' which attempts to provide prebuilt models for common tests, integrated circuit
wear-out mechanisms using peer-reviewed empirical models, and some basic test environments for use. These cookbook
models can also be used as references to see how to construct more complex models. 


## Documentation

The 'docs' folder of the repository contains some useful references for building simulations, and the source code has
docstrings that can provide in-editor hints and descriptions of the classes and methods.

If you are struggling to specify a model that implements the behaviour you desire or are encountering confusing
simulation results that do not align with your expectations, you are encouraged to open issues on the Github repository
or contact Ian directly. We want Gerabaldi to be easy to use, fully featured, and to contain helpful error checking and
warnings that can alert you to potential issues with your simulations. Contacting us with difficulties helps us make
improvements to achieve this goal!


## Citing Gerabaldi
If you use the Gerabaldi simulator to aid in your research it would be greatly appreciated if you could cite the
Gerabaldi 2023 IEEE VLSI Test Symposium paper in any resulting publications,
doi: https://doi.org/10.1109/VTS56346.2023.10140111.


## Code Style and Formatting
The Gerabaldi codebase uses Ruff for linting and formatting rules, the linting checks are run whenever a commit is
pushed to the Github repository. Refer to the Ruff documentation for information on different error codes that may be
raised: https://docs.astral.sh/ruff
