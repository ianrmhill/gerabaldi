# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

"""Internal helper functions used within Gerabaldi to streamline the package."""

import logging
import coloredlogs
import importlib
import numpy as np
import pandas as pd
from datetime import timedelta


### Instantiate default logger upon import of this file so that it is always configured ###
# Instantiate the logger with default settings
logger = logging.getLogger("gerabaldi")
# Log output handlers can have their own logging levels, internal logger will collect all levels
logger.setLevel(logging.DEBUG)

# Change a few colours for the logging output, making message times appear in a mid-blue and the logger name in green
custom_field_styles = coloredlogs.DEFAULT_FIELD_STYLES
custom_field_styles['asctime']['color'] = 24
custom_field_styles['name']['color'] = 22
custom_level_styles = coloredlogs.DEFAULT_LEVEL_STYLES
custom_level_styles['info']['color'] = 'white'
# Install one default colourized stream handler set to level INFO; no log files by default
coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level_styles=custom_level_styles, field_styles=custom_field_styles)


def configure_logger(logging_level: int = None,
                     file_handler: logging.FileHandler = None, stream_handler: logging.StreamHandler = None):
    """
    Configure the logger based on the user's preference.

    Parameters
    ----------
    logging_level: int, optional
        The global logging level to set for the logger. If provided, the logger's level will be set to this
        value. Default is None.
    file_handler: logging.FileHandler, optional
        A custom file handler to be added to the logger. If provided, the default file handler (if exists) will be
        removed and the custom one will be added. Default is None.
    stream_handler: logging.StreamHandler, optional
        A custom stream handler to be added to the logger. If provided, the default stream handler (if exists) will
        be removed and the custom one will be added. Default is None.

    Returns
    -------
    None

    Notes
    -----
    This function assumes that the logger has one stream and one file handler maximum.

    Example
    -------
    # Configure the logger with a custom stream handler and set the logging level to INFO
    custom_stream_handler = logging.StreamHandler()
    configure_logger(logging_level=logging.INFO, stream_handler=custom_stream_handler)
    """
    logger = logging.getLogger('gerabaldi')
    # If the user has provided a global logging level, set accordingly
    if logging_level:
        logger.setLevel(logging_level)

    # Identify any existing logging output handlers based on their types
    existing_stream_handler = None
    existing_file_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            existing_stream_handler = handler
        elif isinstance(handler, logging.FileHandler):
            existing_file_handler = handler

    # If the user provided a custom stream handler, remove the default one, add the custom one
    if stream_handler:
        if existing_stream_handler:
            logger.removeHandler(existing_stream_handler)
        logger.addHandler(stream_handler)
    # If the user provided a custom file handler, remove the default one, add the custom one
    if file_handler:
        if existing_file_handler:
            logger.removeHandler(existing_file_handler)
        logger.addHandler(file_handler)


def _convert_time(time, units, **kwargs): # noqa: UnusedParameter
    """Helper function compatible with pandas apply() function for converting between time representations."""
    if type(time) in [timedelta, pd.Timedelta]:
        return time.total_seconds() / units
    else:
        return timedelta(**{units: time})


def _on_demand_import(module: str, pypi_name: str = None):
    try:
        mod = importlib.import_module(module)
        return mod
    except ImportError:
        # Module name and pypi package name do not always match, we want to tell the user the package to install
        if not pypi_name:
            pypi_name = module
        hint = f"Trying to use a feature that requires the optional {module} module. " \
               f"Please install package '{pypi_name}' first."

        class FailedImport:
            """By returning a class that raises an error when used, we can try to import modules at the top of each file
            and only raise errors if we try to use methods of modules that failed to import"""
            def __getattr__(self, attr):
                raise ImportError(hint)

        return FailedImport()


def _get_single_index(vals: dict, i: int, j: int, k: int) -> dict:
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
    new = {}
    for key, val in vals.items():
        if type(val) == dict:
            new[key] = _get_single_index(vals[key], i, j, k)
        elif type(val) in [np.ndarray, list]:
            new[key] = val[i][j][k]
        elif type(val) in [int, float, np.float64, tuple]:
            new[key] = val
    return new


def _loop_compute(eqn, args_dict: dict, dims: tuple):
    # NOTE: This method requires all data arrays within the args dictionary to have the exact same shape
    computed = np.empty(dims)
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                args = _get_single_index(args_dict, i, j, k)
                computed[i, j, k] = eqn(**args)
    return computed
