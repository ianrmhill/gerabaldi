# Copyright (c) 2024 Ian Hill
# SPDX-License-Identifier: Apache-2.0

from gerabaldi.models.phys_envs import PhysTestEnv, MeasInstrument, EnvVrtnMdl
from gerabaldi.models.random_vars import Normal

__all__ = ['ideal_env', 'volt_and_temp_env']


def ideal_env() -> PhysTestEnv:
    """
    Construct a perfect test environment that always applies exactly the requested stress conditions and exactly
    measures all requested parameters

    Returns
    -------
    PhysTestEnv
        Idealized test environment for use
    """
    return PhysTestEnv(env_name='Idealized Test Environment')


def volt_and_temp_env() -> PhysTestEnv:
    """
    Construct a test environment that has some realistic temperature and voltage stress variability and inexpensive
    test instruments for measuring the same parameters.

    Returns
    -------
    PhysTestEnv
        Prebuilt test environment for use
    """
    # Voltage stress is applied perfectly at the batch level, but some sub-mV expected shifts between chips and devices
    volt_vrtns = EnvVrtnMdl(
        dev_vrtn_mdl=Normal(0, 0.0003),
        chp_vrtn_mdl=Normal(0, 0.0005),
        batch_vrtn_mdl=Normal(0, 0),
        vrtn_type='offset',
        name='Voltage Variation Model',
    )
    # Temperature varies little between devices, 0.2 degree shifts between chips and overall 0.5 test chamber error
    temp_vrtns = EnvVrtnMdl(
        dev_vrtn_mdl=Normal(0, 0.05),
        chp_vrtn_mdl=Normal(0, 0.2),
        batch_vrtn_mdl=Normal(0, 0.5),
        vrtn_type='offset',
        name='Temperature Variation Model',
    )

    # Voltmeter shows to nearest mV, noise error offsets of 0.2mV, range of 0V to 12V
    volt_meas = MeasInstrument(precision=3, error=Normal(0, 0.0002), meas_lims=(0, 12), name='Basic Voltmeter')
    # Temperature to nearest hundredths of a degree, error offset of 0.2K and converts to Celsius, range -40 to 180
    temp_meas = MeasInstrument(precision=2, error=Normal(-273.15, 0.2), meas_lims=(-40, 180), name='Basic Temp Sensor')

    return PhysTestEnv(
        env_vrtns={'temp': temp_vrtns, 'vdd': volt_vrtns},
        meas_instms={'temp': temp_meas, 'vdd': volt_meas},
        env_name='Stochastic Voltage and Temperature Affected Test Environment',
    )
