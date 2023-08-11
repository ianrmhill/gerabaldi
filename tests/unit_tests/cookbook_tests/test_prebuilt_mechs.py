# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import gerabaldi.models
from gerabaldi.cookbook.mech_mdls import jedec_nbti_mdl


def test_nbti_jedec():
    new_mech = jedec_nbti_mdl()
    assert type(new_mech.latent_var('e_aa')) == gerabaldi.models.LatentVar
    assert round(new_mech.compute(a_0=1, e_aa=0.1, temp=300, vdd=0.95, alpha=0.3, time=100, n=0.2), 3) == 118.388
