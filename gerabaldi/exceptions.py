# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

"""Custom exceptions and error handling types for Gerabaldi"""

__all__ = ['MissingParamError', 'NotYetSupportedError', 'UserConfigError', 'InvalidTypeError',
           'ParamOverwriteWarning', 'IdealWarning', 'ArgOverwriteWarning']


class MissingParamError(Exception):
    """Used to indicate when a model fails to receive all the required values to perform the requested operation."""
    pass


class NotYetSupportedError(Exception):
    """Used to indicate when a distribution, feature, or so forth is requested that is not currently supported."""
    pass


class UserConfigError(Exception):
    """Error raised when user specified options are missing, incorrectly formatted, or otherwise unsuitable."""
    pass


class InvalidTypeError(Exception):
    """Error raised when the user requests a type variable to be an invalid value."""
    pass


class ParamOverwriteWarning(UserWarning):
    """Warning used to indicate situations where a user passed parameter to a degradation model is overwritten."""
    pass


class IdealWarning(UserWarning):
    """Warning for when ideal assumptions are introduced implicitly."""
    pass


class ArgOverwriteWarning(UserWarning):
    """Warning for when unsupported arguments are overwritten implicitly."""
    pass
