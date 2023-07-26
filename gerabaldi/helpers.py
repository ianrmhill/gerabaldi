# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

"""Internal helper functions used within Gerabaldi to streamline the package."""

import logging
import importlib
import numpy as np
import pandas as pd
from datetime import timedelta


def _instantiate_logger():
    """Instantiate a module logger."""
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Pass all log messages to handlers, which can have their own logging levels. 
    logger.setLevel(logging.DEBUG)

    # Provide one default stream handler set to level INFO
    default_stream_handler = logging.StreamHandler()
    default_stream_handler.setLevel(logging.INFO)
    default_stream_handler.setFormatter(formatter)
    logger.addHandler(default_stream_handler)

    return logger


def configure_logger(logger: logging.Logger, logging_level: int = None, file_handler: logging.FileHandler = None, stream_handler: logging.StreamHandler = None):
    """Configure the logger based on the user's preference."""
    default_stream_handler = None
    default_file_handler = None

    # If the user has provided a global logging level, set accordingly
    if logging_level:
        logger.setLevel(logging_level)

    # Identify the default handlers based on their types
    # NOTE: The assumption here is that the logger has one stream and one file handler maximum
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            default_stream_handler = handler
        elif isinstance(handler, logging.FileHandler):
            default_file_handler = handler

    # If the user provided a custom stream handler, remove the default one, add the custom one
    if stream_handler:
        # If there's a default stream handler, remove it first
        if default_stream_handler:
            logger.removeHandler(default_stream_handler)
        logger.addHandler(stream_handler)

    # If the user provided a custom file handler, remove the default one, add the custom one
    if file_handler:
        if default_file_handler:
            logger.removeHandler(default_file_handler)
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


logger = _instantiate_logger()
