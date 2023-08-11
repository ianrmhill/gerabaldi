# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import pytest # noqa: PackageNotInRequirements
from gerabaldi.cookbook.test_specs import htol
from gerabaldi.exceptions import UserConfigError


def test_htol_test():
    new_test = htol({'ring_osc': 100})
    assert len(new_test.steps) == 3
    assert new_test.steps[0].conditions['vdd'] == 1.0
    assert new_test.steps[1].conditions['vdd'] == 1.2
    assert new_test.steps[1].conditions['temp'] == 398.15
    assert new_test.steps[2].conditions['temp'] == 298.15
    assert new_test.name is not None
    assert new_test.description is not None

    with pytest.raises(UserConfigError):
        _ = htol({'ring_osc': 100}, strs_meas_intrvl=43)

    new_test = htol({'param': 5}, vdd_nom=0.9, vdd_strs_mult=1.3, strs_meas_intrvl=200)
    assert len(new_test.steps) == 13
    assert new_test.steps[-1].conditions['vdd'] == 0.9
    assert round(new_test.steps[1].conditions['vdd'], 2) == 1.17
    assert new_test.num_chps == 77
    assert new_test.num_lots == 3
