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

There are minimal required dependencies in terms of other Python packages to use the packaged version of Gerabaldi, but
some optional features will require additional packages which you will be notified to install if attempting to use those
features.

## Demos

The 'demos' folder of this repository contains some useful references for full working simulations with command-line
interfaces and visualization of results. These can be run directly if you have cloned the source code, or if you
installed the packaged version it is easiest to copy-paste the source code from Github (you will want to remove the
automated path add code at the very top of the demo file as the 'gerabaldi' import should already work without it).

Note that the VTS paper demos will take a significant amount of time to run due to the size of the simulations, these
can be reduced by changing the three globals near the tops of the files as follows:

For demo 1:
`NUM_SAMPLES = 5`
`NUM_DEVICES = 5`
`NUM_LOTS = 5`

For demo 2:
`TEST_LEN = 24 * 7 * 52 * 2`
`NUM_SAMPLES = 100`
`C_LATENT = 4e-5`

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
Gerabaldi 2023 IEEE VLSI Test Symposium paper in any resulting publications.
