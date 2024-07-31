# Copyright (c) 2024 Ian Hill
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the helper functions within Gerabaldi"""

import numpy as np

from gerabaldi.helpers import _loop_compute


def test_loop_compute():
    def sample_eqn(x, y):
        return x * y

    args = {
        'x': np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]),
        'y': np.array([[[11, 10], [9, 8]], [[7, 6], [5, 4]], [[3, 2], [1, 0]]]),
    }
    dims = (3, 2, 2)
    result = _loop_compute(sample_eqn, args, dims)
    assert result[2, 1, 0] == 11
    assert result[0, 1, 1] == 32
    args['z'] = 0.5

    def sample_eqn_2(x, y, z):
        return (x * y) - z

    result = _loop_compute(sample_eqn_2, args, dims)
    assert result[2, 1, 0] == 10.5
    assert result[0, 1, 1] == 31.5
